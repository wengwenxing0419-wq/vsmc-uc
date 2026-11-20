from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from aggregation.vsmc_expert_aggregator import (
    merge_expert_states_into_global_state,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)

from .config import FedMoEDAConfig
from .utils import (
    blend_state_dicts,
    cosine_similarity_matrix,
    ordered_union,
    softmax_over_indices,
    top_p_neighbors,
)


TensorState = Dict[str, torch.Tensor]


class FedMoEDAExpertAggregator:
    def __init__(self, config: FedMoEDAConfig, logger=None):
        self.config = config
        self.logger = logger

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        device: torch.device,
        round_idx: int,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        del round_idx
        if not client_updates:
            raise ValueError("Cannot aggregate an empty set of client updates")

        client_states = [state_dict for state_dict, _, _ in client_updates]
        sample_counts = [int(num_samples) for _, num_samples, _ in client_updates]
        global_base_state, global_expert_groups = split_state_dict_by_expert(global_state)

        client_base_states: List[TensorState] = []
        client_expert_group_maps: List[Dict[str, Dict[str, Any]]] = []
        expert_scope_order: List[str] = list(global_expert_groups.keys())
        for state_dict in client_states:
            base_state, expert_groups = split_state_dict_by_expert(state_dict)
            client_base_states.append(base_state)
            client_expert_group_maps.append(expert_groups)
            expert_scope_order.extend(expert_groups.keys())

        shared_weights = self._compute_client_weights(sample_counts, self.config.shared_weight_mode)
        aggregated_base_state = weighted_average_state_dicts(
            client_base_states,
            shared_weights,
            fallback_state=global_base_state,
        )

        aggregated_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        per_expert_info: List[Dict[str, Any]] = []
        for expert_scope in ordered_union(expert_scope_order):
            aggregated_group, info = self._aggregate_expert_scope(
                expert_scope=expert_scope,
                global_expert_groups=global_expert_groups,
                client_states=client_states,
                client_expert_group_maps=client_expert_group_maps,
                client_updates=client_updates,
                sample_counts=sample_counts,
            )
            if aggregated_group is None:
                continue
            aggregated_expert_groups[expert_scope] = aggregated_group
            per_expert_info.append(info)

        aggregated_state = merge_expert_states_into_global_state(
            aggregated_base_state,
            aggregated_expert_groups,
        )
        aggregated_state = collections.OrderedDict(
            (name, tensor.to(device) if torch.is_tensor(tensor) else tensor)
            for name, tensor in aggregated_state.items()
        )

        mean_pairwise_similarity = self._safe_mean(
            [info.get("mean_pairwise_similarity", 0.0) for info in per_expert_info],
            default=1.0,
        )
        mean_selected_similarity = self._safe_mean(
            [info.get("mean_selected_similarity", 0.0) for info in per_expert_info],
            default=1.0,
        )
        return aggregated_state, {
            "top_p": int(self.config.top_p),
            "num_expert_groups": len(per_expert_info),
            "mean_pairwise_similarity": mean_pairwise_similarity,
            "mean_selected_similarity": mean_selected_similarity,
            "per_expert_info": per_expert_info,
        }

    def _aggregate_expert_scope(
        self,
        expert_scope: str,
        global_expert_groups: Dict[str, Dict[str, Any]],
        client_states: Sequence[TensorState],
        client_expert_group_maps: Sequence[Dict[str, Dict[str, Any]]],
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        sample_counts: Sequence[int],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        reference_group = global_expert_groups.get(expert_scope)
        reference_state = reference_group["state_dict"] if reference_group is not None else collections.OrderedDict()

        active_states: List[TensorState] = []
        active_embeddings: List[torch.Tensor] = []
        active_sample_counts: List[int] = []
        active_client_ids: List[int] = []
        expert_id: Optional[int] = None

        for client_idx, expert_group_map in enumerate(client_expert_group_maps):
            group = expert_group_map.get(expert_scope)
            if group is None:
                continue
            expert_id = int(group.get("expert_id", 0))
            active_states.append(group["state_dict"])
            active_sample_counts.append(int(sample_counts[client_idx]))
            active_client_ids.append(int(client_updates[client_idx][2].get("client_id", client_idx + 1)))
            active_embeddings.append(
                self._build_similarity_embedding(
                    client_state=client_states[client_idx],
                    expert_scope=expert_scope,
                    expert_id=expert_id,
                    expert_state=group["state_dict"],
                )
            )

        if not active_states:
            if reference_group is None:
                return None, {}
            return (
                {
                    "expert_id": int(reference_group.get("expert_id", 0)),
                    "expert_scope": expert_scope,
                    "state_dict": reference_state,
                },
                {
                    "expert_id": int(reference_group.get("expert_id", 0)),
                    "expert_scope": expert_scope,
                    "num_active_clients": 0,
                    "mean_pairwise_similarity": 1.0,
                    "mean_selected_similarity": 1.0,
                    "mean_neighbor_count": 0.0,
                    "active_client_ids": [],
                },
            )

        expert_weights = self._compute_client_weights(active_sample_counts, self.config.expert_weight_mode)
        fedavg_state = weighted_average_state_dicts(
            active_states,
            expert_weights,
            fallback_state=reference_state,
        )

        similarity_matrix = cosine_similarity_matrix(torch.stack(active_embeddings, dim=0))
        personalized_states: List[TensorState] = []
        selected_similarity_values: List[float] = []
        neighbor_counts: List[int] = []
        for row_idx in range(similarity_matrix.size(0)):
            selected_indices = top_p_neighbors(
                similarity_matrix[row_idx],
                top_p=self.config.top_p,
                min_similarity=self.config.min_similarity,
                keep_index=row_idx,
            )
            neighbor_counts.append(int(selected_indices.numel()))
            selected_similarity_values.append(
                float(similarity_matrix[row_idx].index_select(0, selected_indices).mean().item())
            )
            selected_weights = softmax_over_indices(
                similarity_matrix[row_idx],
                selected_indices,
                temperature=self.config.temperature,
            )
            personalized_states.append(
                weighted_average_state_dicts(
                    [active_states[int(index.item())] for index in selected_indices],
                    selected_weights,
                    fallback_state=reference_state,
                )
            )

        p2p_state = weighted_average_state_dicts(
            personalized_states,
            expert_weights,
            fallback_state=fedavg_state,
        )
        final_state = blend_state_dicts(
            primary_state=fedavg_state,
            secondary_state=p2p_state,
            mix=self.config.personalized_mix,
        )

        return (
            {
                "expert_id": int(expert_id if expert_id is not None else reference_group.get("expert_id", 0)),
                "expert_scope": expert_scope,
                "state_dict": final_state,
            },
            {
                "expert_id": int(expert_id if expert_id is not None else reference_group.get("expert_id", 0)),
                "expert_scope": expert_scope,
                "num_active_clients": len(active_states),
                "active_client_ids": active_client_ids,
                "mean_pairwise_similarity": self._mean_off_diagonal(similarity_matrix),
                "mean_selected_similarity": self._safe_mean(selected_similarity_values, default=1.0),
                "mean_neighbor_count": self._safe_mean(neighbor_counts, default=0.0),
            },
        )

    def _build_similarity_embedding(
        self,
        client_state: TensorState,
        expert_scope: str,
        expert_id: int,
        expert_state: TensorState,
    ) -> torch.Tensor:
        if self.config.similarity_metric == "router_row_cosine":
            block_scope = self._router_block_scope(expert_scope, expert_id)
            if block_scope is not None:
                router_weight_name = f"{block_scope}.router.weight"
                router_weight = client_state.get(router_weight_name)
                if torch.is_tensor(router_weight) and router_weight.ndim >= 2 and expert_id < router_weight.shape[0]:
                    return router_weight[expert_id].detach().float().reshape(-1)
        flat_expert_state = self._flatten_float_state(expert_state)
        if flat_expert_state.numel() > 0:
            return flat_expert_state
        return torch.zeros(1, dtype=torch.float32)

    @staticmethod
    def _router_block_scope(expert_scope: str, expert_id: int) -> Optional[str]:
        for marker in (f".experts.{expert_id}", f".expert.{expert_id}"):
            if marker in expert_scope:
                return expert_scope.split(marker)[0]
        return None

    @staticmethod
    def _flatten_float_state(state: TensorState) -> torch.Tensor:
        flat_tensors: List[torch.Tensor] = []
        for key in sorted(state.keys()):
            value = state[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                flat_tensors.append(value.detach().float().reshape(-1))
        if not flat_tensors:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(flat_tensors, dim=0)

    @staticmethod
    def _compute_client_weights(sample_counts: Sequence[int], mode: str) -> torch.Tensor:
        num_clients = len(sample_counts)
        if num_clients == 0:
            raise ValueError("Cannot compute weights for zero clients")
        normalized_mode = str(mode).lower()
        if normalized_mode == "uniform":
            return torch.full((num_clients,), 1.0 / float(num_clients), dtype=torch.float64)
        counts = torch.tensor([max(float(count), 0.0) for count in sample_counts], dtype=torch.float64)
        total = counts.sum()
        if total.item() <= 0.0 or not torch.isfinite(total):
            return torch.full((num_clients,), 1.0 / float(num_clients), dtype=torch.float64)
        return counts / total

    @staticmethod
    def _mean_off_diagonal(matrix: torch.Tensor) -> float:
        if matrix.numel() == 0:
            return 1.0
        if matrix.size(0) <= 1:
            return 1.0
        mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
        values = matrix[mask]
        if values.numel() == 0:
            return 1.0
        return float(values.mean().item())

    @staticmethod
    def _safe_mean(values: Sequence[float], default: float) -> float:
        if not values:
            return float(default)
        return float(sum(float(value) for value in values) / float(len(values)))
