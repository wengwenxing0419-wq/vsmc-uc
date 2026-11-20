from __future__ import annotations

import collections
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from .fedma_utils import (
    TensorState,
    get_expert_param_keys,
    group_expert_keys_by_moe_layer,
    is_float_tensor,
    normalize_signature,
)


class FedMAMatcher:
    def __init__(
        self,
        signature_type: str = "weight_flatten",
        similarity: str = "cosine",
        matching_method: str = "hungarian",
        keep_num_experts_fixed: bool = True,
        device: str = "cuda",
    ):
        self.signature_type = str(signature_type).lower()
        self.similarity = str(similarity).lower()
        self.matching_method = str(matching_method).lower()
        self.keep_num_experts_fixed = bool(keep_num_experts_fixed)
        self.device = torch.device(device)
        self.last_matching_diagnostics: Dict[str, Any] = {}

    def build_expert_signature(
        self,
        state_dict: MutableMapping[str, Any],
        expert_key_groups: Dict[str, Any],
        expert_id: int,
        model=None,
        data_loader=None,
        client_metrics: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Build one expert signature vector for a single MoE layer.
        """
        key_groups = expert_key_groups["key_groups"] if "key_groups" in expert_key_groups else expert_key_groups
        if int(expert_id) not in key_groups:
            raise KeyError(f"Expert id {expert_id} is missing from expert_key_groups")

        keys = list(key_groups[int(expert_id)])
        if self.signature_type == "weight_flatten":
            flat_tensors: List[torch.Tensor] = []
            for key in sorted(keys):
                value = state_dict[key]
                if is_float_tensor(value):
                    flat_tensors.append(value.detach().float().reshape(-1))
            if not flat_tensors:
                raise ValueError(f"Expert {expert_id} has no floating-point parameters for weight_flatten signature")
            return normalize_signature(torch.cat(flat_tensors, dim=0))

        if self.signature_type == "weight_mean":
            stats: List[torch.Tensor] = []
            for key in sorted(keys):
                value = state_dict[key]
                if not is_float_tensor(value):
                    continue
                flat_value = value.detach().float().reshape(-1)
                stats.extend(
                    [
                        flat_value.mean().reshape(1),
                        flat_value.std(unbiased=False).reshape(1),
                        flat_value.norm(p=2).reshape(1),
                        flat_value.abs().mean().reshape(1),
                    ]
                )
            if not stats:
                raise ValueError(f"Expert {expert_id} has no floating-point parameters for weight_mean signature")
            return torch.cat(stats, dim=0)

        if self.signature_type == "router_usage":
            if client_metrics is None or "router_usage" not in client_metrics:
                raise ValueError("router_usage signature requires client_metrics['router_usage']")
            router_usage = client_metrics["router_usage"]
            if not torch.is_tensor(router_usage):
                raise TypeError("client_metrics['router_usage'] must be a tensor")
            layer_index = int(expert_key_groups.get("layer_index", 0))
            if router_usage.ndim == 1:
                if int(expert_id) >= router_usage.numel():
                    raise ValueError("router_usage tensor is smaller than the requested expert id")
                return router_usage[int(expert_id)].detach().float().reshape(1)
            if router_usage.ndim == 2:
                if layer_index >= router_usage.size(0) or int(expert_id) >= router_usage.size(1):
                    raise ValueError("router_usage tensor shape does not match the requested layer/expert id")
                return router_usage[layer_index, int(expert_id)].detach().float().reshape(1)
            raise ValueError("router_usage tensor must be rank 1 or 2")

        if self.signature_type == "activation_mean":
            raise NotImplementedError(
                "activation_mean signature is not wired for this project yet; use weight_flatten, weight_mean, or router_usage"
            )

        raise ValueError(f"Unsupported expert signature type: {self.signature_type}")

    def _build_layer_signatures(
        self,
        state_dict: MutableMapping[str, Any],
        layer_groups: Dict[str, Dict[str, Any]],
        client_metrics: Optional[Dict[str, Any]] = None,
        model=None,
        data_loader=None,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        signatures: Dict[str, Dict[int, torch.Tensor]] = collections.OrderedDict()
        for layer_scope, layer_group in layer_groups.items():
            signatures[layer_scope] = collections.OrderedDict()
            for expert_id in layer_group["key_groups"].keys():
                signatures[layer_scope][int(expert_id)] = self.build_expert_signature(
                    state_dict=state_dict,
                    expert_key_groups=layer_group,
                    expert_id=int(expert_id),
                    model=model,
                    data_loader=data_loader,
                    client_metrics=client_metrics,
                )
        return signatures

    def build_all_signatures(
        self,
        client_states: Sequence[MutableMapping[str, Any]],
        expert_key_groups_list: Sequence[Dict[str, Dict[str, Any]]],
        client_metrics_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Dict[int, torch.Tensor]]]:
        """
        Return signatures[client_idx][layer_scope][expert_id] = tensor vector.
        """
        signatures: List[Dict[str, Dict[int, torch.Tensor]]] = []
        metrics_list = client_metrics_list or [None] * len(client_states)
        for state_dict, layer_groups, client_metrics in zip(client_states, expert_key_groups_list, metrics_list):
            signatures.append(
                self._build_layer_signatures(
                    state_dict=state_dict,
                    layer_groups=layer_groups,
                    client_metrics=client_metrics,
                )
            )
        return signatures

    def compute_similarity_matrix(
        self,
        ref_signatures: Dict[int, torch.Tensor],
        client_signatures: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute a K x K similarity matrix between one reference layer and one client layer.
        """
        ref_ids = sorted(ref_signatures.keys())
        client_ids = sorted(client_signatures.keys())
        if self.keep_num_experts_fixed and len(ref_ids) != len(client_ids):
            raise ValueError(
                f"Expert-count mismatch for fixed-size matching: reference={len(ref_ids)} client={len(client_ids)}"
            )

        matrix = torch.empty((len(ref_ids), len(client_ids)), dtype=torch.float32)
        for row_idx, ref_id in enumerate(ref_ids):
            ref_vector = ref_signatures[ref_id].detach().float().reshape(-1)
            for col_idx, client_id in enumerate(client_ids):
                client_vector = client_signatures[client_id].detach().float().reshape(-1)
                if ref_vector.numel() != client_vector.numel():
                    raise ValueError(
                        f"Signature length mismatch between reference expert {ref_id} and client expert {client_id}: "
                        f"{ref_vector.numel()} vs {client_vector.numel()}"
                    )
                if self.similarity == "cosine":
                    score = torch.dot(
                        normalize_signature(ref_vector),
                        normalize_signature(client_vector),
                    )
                elif self.similarity == "l2":
                    score = -torch.norm(ref_vector - client_vector, p=2)
                else:
                    raise ValueError(f"Unsupported similarity metric: {self.similarity}")
                matrix[row_idx, col_idx] = score
        if not torch.isfinite(matrix).all():
            raise ValueError("NaN/Inf detected in similarity matrix")
        return matrix

    @staticmethod
    def _greedy_assignment(similarity_matrix: torch.Tensor) -> Tuple[List[int], List[int]]:
        remaining_rows = list(range(similarity_matrix.size(0)))
        remaining_cols = list(range(similarity_matrix.size(1)))
        assigned_rows: List[int] = []
        assigned_cols: List[int] = []

        while remaining_rows and remaining_cols:
            best_pair = None
            best_score = float("-inf")
            for row_idx in remaining_rows:
                for col_idx in remaining_cols:
                    score = float(similarity_matrix[row_idx, col_idx].item())
                    if score > best_score:
                        best_score = score
                        best_pair = (row_idx, col_idx)
            if best_pair is None:
                break
            row_idx, col_idx = best_pair
            assigned_rows.append(row_idx)
            assigned_cols.append(col_idx)
            remaining_rows.remove(row_idx)
            remaining_cols.remove(col_idx)

        return assigned_rows, assigned_cols

    def match_client_to_reference(
        self,
        ref_signatures: Dict[int, torch.Tensor],
        client_signatures: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        Return matched_global_expert_id -> client_expert_id for one MoE layer.
        """
        ref_ids = sorted(ref_signatures.keys())
        client_ids = sorted(client_signatures.keys())
        similarity_matrix = self.compute_similarity_matrix(ref_signatures, client_signatures)

        if self.matching_method == "identity":
            if len(ref_ids) != len(client_ids):
                raise ValueError("identity matching requires equal numbers of reference/client experts")
            row_indices = list(range(len(ref_ids)))
            col_indices = list(range(len(client_ids)))
            solver = "identity"
        elif self.matching_method == "greedy":
            row_indices, col_indices = self._greedy_assignment(similarity_matrix)
            solver = "greedy"
        elif self.matching_method == "hungarian":
            try:
                from scipy.optimize import linear_sum_assignment

                cost_matrix = (-similarity_matrix).detach().cpu().numpy()
                row_indices_np, col_indices_np = linear_sum_assignment(cost_matrix)
                row_indices = [int(index) for index in row_indices_np.tolist()]
                col_indices = [int(index) for index in col_indices_np.tolist()]
                solver = "scipy_hungarian"
            except Exception:
                row_indices, col_indices = self._greedy_assignment(similarity_matrix)
                solver = "greedy_fallback"
        else:
            raise ValueError(f"Unsupported matching method: {self.matching_method}")

        matching: Dict[int, int] = collections.OrderedDict()
        matched_scores: List[float] = []
        for row_idx, col_idx in sorted(zip(row_indices, col_indices), key=lambda item: item[0]):
            global_expert_id = int(ref_ids[row_idx])
            local_expert_id = int(client_ids[col_idx])
            matching[global_expert_id] = local_expert_id
            matched_scores.append(float(similarity_matrix[row_idx, col_idx].item()))

        if len(matching) != len(ref_ids):
            raise ValueError(
                f"Incomplete expert matching produced {len(matching)} assignments for {len(ref_ids)} reference experts"
            )

        return matching, {
            "solver": solver,
            "similarity_matrix": similarity_matrix.detach().cpu(),
            "matched_scores": matched_scores,
            "reference_expert_ids": ref_ids,
            "client_expert_ids": client_ids,
        }

    def match_all_clients(
        self,
        client_states: Sequence[TensorState],
        reference_state: Optional[TensorState] = None,
        client_metrics_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[int, Dict[str, Dict[int, int]]]:
        """
        Return matching[client_idx][layer_scope][global_expert_id] = local_expert_id.
        """
        if not client_states:
            raise ValueError("client_states must not be empty")

        reference_source = "global" if reference_state is not None else "client_0"
        reference_state = client_states[0] if reference_state is None else reference_state
        reference_layers = group_expert_keys_by_moe_layer(reference_state)
        if not reference_layers:
            raise ValueError(
                "No expert parameters were found in the reference state. "
                f"Available expert-like keys: {get_expert_param_keys(reference_state)}"
            )
        reference_signatures = self._build_layer_signatures(reference_state, reference_layers, client_metrics=None)

        metrics_list = client_metrics_list or [None] * len(client_states)
        matching: Dict[int, Dict[str, Dict[int, int]]] = collections.OrderedDict()
        diagnostics: Dict[str, Any] = {
            "reference_source": reference_source,
            "per_client_layer": collections.OrderedDict(),
            "matching_solver": collections.OrderedDict(),
        }

        for client_idx, (client_state, client_metrics) in enumerate(zip(client_states, metrics_list)):
            client_layers = group_expert_keys_by_moe_layer(client_state)
            if set(client_layers.keys()) != set(reference_layers.keys()):
                raise ValueError(
                    f"MoE layer mismatch between reference state and client {client_idx}: "
                    f"reference_layers={sorted(reference_layers.keys())} client_layers={sorted(client_layers.keys())}"
                )
            client_signatures = self._build_layer_signatures(
                state_dict=client_state,
                layer_groups=client_layers,
                client_metrics=client_metrics,
            )
            matching[client_idx] = collections.OrderedDict()
            diagnostics["per_client_layer"][client_idx] = collections.OrderedDict()

            for layer_scope in reference_layers.keys():
                layer_matching, layer_diag = self.match_client_to_reference(
                    ref_signatures=reference_signatures[layer_scope],
                    client_signatures=client_signatures[layer_scope],
                )
                matching[client_idx][layer_scope] = layer_matching
                diagnostics["per_client_layer"][client_idx][layer_scope] = layer_diag
                diagnostics["matching_solver"][f"client_{client_idx}:{layer_scope}"] = layer_diag["solver"]

        self.last_matching_diagnostics = diagnostics
        return matching
