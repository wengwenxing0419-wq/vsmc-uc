from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from .fedma_config import FedMAConfig
from .fedma_matcher import FedMAMatcher
from .fedma_utils import (
    TensorState,
    clone_state_dict,
    compute_client_weights,
    group_expert_keys_by_moe_layer,
    remap_expert_params,
    split_expert_and_non_expert_state,
    weighted_average_state_dicts,
)


class FedMAAggregator:
    def __init__(self, config: FedMAConfig, matcher: FedMAMatcher, logger=None):
        self.config = config
        self.matcher = matcher
        self.logger = logger

    @staticmethod
    def _clone_state_to_cpu(state_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        cloned = collections.OrderedDict()
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().cpu().clone()
            else:
                cloned[key] = value
        return cloned

    @staticmethod
    def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
        if not values:
            return float(default)
        return float(sum(values) / len(values))

    @staticmethod
    def _build_matching_json(
        matching: Dict[int, Dict[str, Dict[int, int]]],
        global_layer_groups: Dict[str, Dict[str, Any]],
        round_idx: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"round": int(round_idx)}
        for layer_scope, layer_group in global_layer_groups.items():
            layer_label = f"moe_layer_{int(layer_group.get('layer_index', 0))}"
            payload[layer_label] = {
                "layer_scope": layer_scope,
            }
            for client_idx, client_matching in matching.items():
                payload[layer_label][f"client_{int(client_idx)}"] = {
                    f"global_{int(global_expert_id)}": f"local_{int(local_expert_id)}"
                    for global_expert_id, local_expert_id in client_matching[layer_scope].items()
                }
        return payload

    @staticmethod
    def _validate_no_nan_inf(state_dict: MutableMapping[str, Any], context: str) -> None:
        for key, value in state_dict.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                if not torch.isfinite(value).all():
                    raise ValueError(f"NaN/Inf detected in {context} for key '{key}'")

    def aggregate(
        self,
        global_state: TensorState,
        client_states: Sequence[TensorState],
        client_num_samples: Sequence[int],
        client_metrics_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        round_idx: int = 0,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        """
        Aggregate client model states with non-expert FedAvg and expert matched averaging.
        """
        if not client_states:
            raise ValueError("Cannot aggregate an empty list of client states")
        if len(client_states) != len(client_num_samples):
            raise ValueError("client_states and client_num_samples must have the same length")

        aggregate_start = time.perf_counter()
        global_state_cpu = self._clone_state_to_cpu(global_state)
        client_states_cpu = [self._clone_state_to_cpu(state_dict) for state_dict in client_states]
        metrics_list = list(client_metrics_list or [None] * len(client_states_cpu))

        shared_weights = compute_client_weights(client_num_samples, self.config.shared_weight_mode)
        expert_weights = compute_client_weights(client_num_samples, self.config.expert_weight_mode)

        global_expert_state, global_non_expert_state = split_expert_and_non_expert_state(global_state_cpu)
        client_non_expert_states: List[MutableMapping[str, Any]] = []
        client_layer_groups: List[Dict[str, Dict[str, Any]]] = []
        for state_dict in client_states_cpu:
            _, non_expert_state = split_expert_and_non_expert_state(state_dict)
            client_non_expert_states.append(non_expert_state)
            client_layer_groups.append(group_expert_keys_by_moe_layer(state_dict))

        aggregated_non_expert_state = weighted_average_state_dicts(
            client_non_expert_states,
            shared_weights,
            fallback_state=global_non_expert_state,
        )

        matching_start = time.perf_counter()
        matching = self.matcher.match_all_clients(
            client_states=client_states_cpu,
            reference_state=global_state_cpu,
            client_metrics_list=metrics_list,
        )
        matching_time_sec = time.perf_counter() - matching_start
        matching_diagnostics = dict(self.matcher.last_matching_diagnostics)

        global_layer_groups = group_expert_keys_by_moe_layer(global_state_cpu)
        aggregated_expert_state = collections.OrderedDict()
        matched_similarity_values: List[float] = []
        changed_assignments = 0
        total_assignments = 0

        for layer_scope, layer_group in global_layer_groups.items():
            global_key_groups = layer_group["key_groups"]
            for global_expert_id, template_keys in global_key_groups.items():
                remapped_states: List[MutableMapping[str, Any]] = []
                template_state = collections.OrderedDict(
                    (key, global_expert_state[key])
                    for key in template_keys
                )
                for client_idx, client_state in enumerate(client_states_cpu):
                    client_group = client_layer_groups[client_idx].get(layer_scope)
                    if client_group is None:
                        raise ValueError(f"Client {client_idx} is missing MoE layer '{layer_scope}'")
                    local_expert_id = int(matching[client_idx][layer_scope][global_expert_id])
                    changed_assignments += int(local_expert_id != int(global_expert_id))
                    total_assignments += 1
                    remapped_state = remap_expert_params(
                        client_state=client_state,
                        from_expert_id=local_expert_id,
                        to_expert_id=int(global_expert_id),
                        expert_key_groups=client_group["key_groups"],
                    )
                    if set(remapped_state.keys()) != set(template_keys):
                        raise ValueError(
                            f"Remapped expert keys do not match the global template for layer '{layer_scope}' "
                            f"expert {global_expert_id}"
                        )
                    ordered_remapped_state = collections.OrderedDict(
                        (key, remapped_state[key])
                        for key in template_keys
                    )
                    remapped_states.append(ordered_remapped_state)

                    layer_diag = matching_diagnostics["per_client_layer"][client_idx][layer_scope]
                    row_ids = layer_diag["reference_expert_ids"]
                    col_ids = layer_diag["client_expert_ids"]
                    row_index = row_ids.index(int(global_expert_id))
                    col_index = col_ids.index(int(local_expert_id))
                    matched_similarity_values.append(
                        float(layer_diag["similarity_matrix"][row_index, col_index].item())
                    )

                aggregated_expert = weighted_average_state_dicts(
                    remapped_states,
                    expert_weights,
                    fallback_state=template_state,
                )
                for key, value in aggregated_expert.items():
                    aggregated_expert_state[key] = value

        aggregated_state = clone_state_dict(global_state_cpu)
        aggregated_state.update(aggregated_non_expert_state)
        aggregated_state.update(aggregated_expert_state)
        self._validate_no_nan_inf(aggregated_state, context="FedMA aggregated state")
        aggregation_time_sec = time.perf_counter() - aggregate_start

        metrics = {
            "fedma_scope": self.config.fedma_scope,
            "expert_signature": self.config.expert_signature,
            "matching_method": self.config.matching_method,
            "similarity": self.config.similarity,
            "matching_solver": ",".join(
                sorted(set(str(value) for value in matching_diagnostics.get("matching_solver", {}).values()))
            )
            or self.config.matching_method,
            "fedma_avg_match_similarity": self._safe_mean(matched_similarity_values, default=0.0),
            "fedma_min_match_similarity": float(min(matched_similarity_values)) if matched_similarity_values else 0.0,
            "fedma_max_match_similarity": float(max(matched_similarity_values)) if matched_similarity_values else 0.0,
            "fedma_num_experts": int(self.config.switch_num_experts),
            "fedma_matching_changed_ratio": (
                float(changed_assignments) / float(total_assignments) if total_assignments > 0 else 0.0
            ),
            "num_moe_layers": int(len(global_layer_groups)),
            "num_experts_per_layer": ",".join(
                str(len(layer_group["key_groups"])) for layer_group in global_layer_groups.values()
            ),
            "matching_time_sec": float(matching_time_sec),
            "aggregation_time_sec": float(aggregation_time_sec),
        }

        round_data = {
            "round": int(round_idx),
            "metrics": metrics,
            "matching": matching,
            "matching_json": self._build_matching_json(matching, global_layer_groups, round_idx=round_idx),
            "similarity_tensors": {
                f"client_{client_idx}": {
                    layer_scope: layer_diag["similarity_matrix"]
                    for layer_scope, layer_diag in per_layer.items()
                }
                for client_idx, per_layer in matching_diagnostics.get("per_client_layer", {}).items()
            },
            "matching_solver": matching_diagnostics.get("matching_solver", {}),
        }
        return aggregated_state, round_data

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        device: torch.device,
        round_idx: int,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        if not client_updates:
            raise ValueError("Cannot aggregate an empty set of client updates")

        client_states = [state_dict for state_dict, _, _ in client_updates]
        client_num_samples = [int(num_samples) for _, num_samples, _ in client_updates]
        client_metrics = [metrics for _, _, metrics in client_updates]
        aggregated_state, round_data = self.aggregate(
            global_state=global_state,
            client_states=client_states,
            client_num_samples=client_num_samples,
            client_metrics_list=client_metrics,
            round_idx=round_idx,
        )
        aggregated_state = collections.OrderedDict(
            (key, value.to(device) if torch.is_tensor(value) else value)
            for key, value in aggregated_state.items()
        )
        return aggregated_state, round_data
