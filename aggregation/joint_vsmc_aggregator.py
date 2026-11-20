from __future__ import annotations

import collections
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from aggregation.vsmc_expert_aggregator import (
    TensorState,
    VSMCExpertAggregationConfig,
    _clone_detached_tensor,
    _merge_with_reference_state,
    _ordered_union_keys,
    _sample_count_weights,
    clone_detached_state_dict,
    compute_log_f,
    compute_log_g_values,
    compute_log_r,
    effective_sample_size,
    get_expert_id_from_name,
    merge_expert_states_into_global_state,
    safe_logsumexp_normalize,
    sample_count_fedavg_state_dicts,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)


@dataclass
class JointVSMCAggregationConfig(VSMCExpertAggregationConfig):
    expert_transition_beta: float = 1.0
    local_obs_beta: float = 1.0
    global_obs_beta: float = 1.0
    proposal_beta: float = 1.0
    router_consistency_beta: float = 0.5
    router_validation_beta: float = 0.5
    router_entropy_beta: float = 0.1
    router_aux_beta: float = 0.1
    router_load_balance_beta: float = 0.1
    proposal_ema_decay: float = 0.8
    use_client_specific_proposal: bool = True
    use_router_validation: bool = True
    route_weight_temperature: float = 1.0


@dataclass
class JointAggregationState:
    round_idx: int = 0
    expert_prev_particle_bank: Dict[str, List[TensorState]] = field(default_factory=dict)
    expert_prev_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    expert_logz_history: Dict[str, float] = field(default_factory=dict)
    client_expert_prev_states: Dict[Tuple[str, int], TensorState] = field(default_factory=dict)
    client_expert_delta_ema: Dict[Tuple[str, int], TensorState] = field(default_factory=dict)
    client_router_prev_states: Dict[Tuple[str, int], TensorState] = field(default_factory=dict)
    client_router_delta_ema: Dict[Tuple[str, int], TensorState] = field(default_factory=dict)


def _get_router_group_from_name(name: str) -> Optional[str]:
    lower_name = name.lower()
    marker = ".router."
    if marker not in lower_name:
        return None
    split_idx = lower_name.index(marker) + len(".router")
    return name[:split_idx]


def split_model_state_by_expert_and_router(
    state_dict: TensorState,
    num_experts: Optional[int] = None,
) -> Tuple[TensorState, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    base_state, expert_states = split_state_dict_by_expert(state_dict, num_experts=num_experts)
    remaining_base: TensorState = collections.OrderedDict()
    router_states: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    for name, tensor in base_state.items():
        router_group = _get_router_group_from_name(name)
        if router_group is None:
            remaining_base[name] = tensor
            continue
        router_states.setdefault(
            router_group,
            {"router_scope": router_group, "state_dict": collections.OrderedDict()},
        )
        router_states[router_group]["state_dict"][name] = tensor
    return remaining_base, expert_states, router_states


def merge_named_groups_into_model(base_state: TensorState, named_groups: Dict[str, Dict[str, Any]]) -> TensorState:
    merged = collections.OrderedDict((k, _clone_detached_tensor(v)) for k, v in base_state.items())
    for group in named_groups.values():
        for name, tensor in group["state_dict"].items():
            merged[name] = _clone_detached_tensor(tensor)
    return merged


def subtract_common_state_dict(a: TensorState, b: TensorState) -> TensorState:
    out: TensorState = collections.OrderedDict()
    for key in sorted(set(a.keys()) & set(b.keys())):
        tensor_a = a[key].detach()
        tensor_b = b[key].detach()
        if tensor_a.shape != tensor_b.shape:
            continue
        if not (torch.is_floating_point(tensor_a) and torch.is_floating_point(tensor_b)):
            continue
        out[key] = tensor_a.to(dtype=torch.float32, device="cpu") - tensor_b.to(dtype=torch.float32, device="cpu")
    return out


def add_state_dicts(base: TensorState, delta: TensorState) -> TensorState:
    out = clone_detached_state_dict(base)
    for key, tensor in delta.items():
        if key not in out:
            out[key] = tensor.detach().clone()
            continue
        base_tensor = out[key].detach()
        if base_tensor.shape != tensor.shape or not (torch.is_floating_point(base_tensor) and torch.is_floating_point(tensor)):
            continue
        out[key] = base_tensor.to(dtype=torch.float32, device="cpu") + tensor.detach().to(dtype=torch.float32, device="cpu")
    return out


def ema_state_dict(old: Optional[TensorState], new: TensorState, decay: float) -> TensorState:
    if old is None:
        return clone_detached_state_dict(new)
    out: TensorState = collections.OrderedDict()
    keep = float(max(0.0, min(1.0, decay)))
    take = 1.0 - keep
    for key in _ordered_union_keys([old, new]):
        if key not in old:
            out[key] = _clone_detached_tensor(new[key])
            continue
        if key not in new:
            out[key] = _clone_detached_tensor(old[key])
            continue
        old_tensor = old[key].detach()
        new_tensor = new[key].detach()
        if old_tensor.shape != new_tensor.shape or not (torch.is_floating_point(old_tensor) and torch.is_floating_point(new_tensor)):
            out[key] = _clone_detached_tensor(new_tensor)
            continue
        out[key] = old_tensor.to(dtype=torch.float32, device="cpu") * keep + new_tensor.to(dtype=torch.float32, device="cpu") * take
    return out


def _extract_layer_and_expert_from_scope(expert_scope: str) -> Tuple[Optional[int], Optional[int]]:
    expert_id = None
    layer_id = None
    pieces = expert_scope.split(".")
    for idx, piece in enumerate(pieces):
        if piece == "blocks" and idx + 1 < len(pieces):
            try:
                layer_id = int(pieces[idx + 1])
            except ValueError:
                pass
        if piece == "experts" and idx + 1 < len(pieces):
            try:
                expert_id = int(pieces[idx + 1])
            except ValueError:
                pass
    return layer_id, expert_id


def _route_consistency_score(metrics: Dict[str, Any], expert_scope: str, eps: float) -> float:
    layer_id, expert_id = _extract_layer_and_expert_from_scope(expert_scope)
    if layer_id is None or expert_id is None:
        return 0.0
    layer_route_counts = metrics.get("layer_route_counts", {})
    layer_counts = layer_route_counts.get(layer_id) or layer_route_counts.get(str(layer_id))
    if not isinstance(layer_counts, dict):
        return 0.0
    total = 0.0
    target = 0.0
    for key, value in layer_counts.items():
        expert_key = int(key)
        count_value = float(value)
        total += count_value
        if expert_key == expert_id:
            target = count_value
    if total <= eps:
        return 0.0
    return math.log(target / total + eps)


def _router_quality_score(
    metrics: Dict[str, Any],
    router_scope: str,
    expert_infos: Sequence[Dict[str, Any]],
    config: JointVSMCAggregationConfig,
) -> float:
    route_consistency_values = []
    for info in expert_infos:
        expert_scope = info.get("expert_scope")
        if not isinstance(expert_scope, str):
            continue
        route_consistency_values.append(_route_consistency_score(metrics, expert_scope, config.eps))
    route_consistency = sum(route_consistency_values) / max(len(route_consistency_values), 1)
    router_val_map = metrics.get("server_val_router_losses", {})
    router_val_value = 0.0
    if isinstance(router_val_map, dict):
        raw_value = router_val_map.get(router_scope)
        if raw_value is not None:
            router_val_value = float(raw_value)
    return (
        float(config.router_consistency_beta) * route_consistency
        - float(config.router_validation_beta) * router_val_value
        - float(config.router_entropy_beta) * float(metrics.get("router_entropy", 0.0))
        - float(config.router_aux_beta) * float(metrics.get("router_aux_loss", 0.0))
        - float(config.router_load_balance_beta) * float(metrics.get("load_balance_gap", 0.0))
    )


class JointVSMCExpertAggregator:
    def __init__(self, config: JointVSMCAggregationConfig, num_experts: Optional[int] = None, logger: Optional[Any] = None):
        self.config = config
        self.num_experts = num_experts
        self.logger = logger
        self.state = JointAggregationState()

    def _client_proposal_mean(
        self,
        expert_scope: str,
        client_id: int,
        previous_global_expert_state: TensorState,
        current_client_expert_state: TensorState,
    ) -> TensorState:
        if not self.config.use_client_specific_proposal:
            return previous_global_expert_state
        memory_key = (expert_scope, int(client_id))
        ema_delta = self.state.client_expert_delta_ema.get(memory_key)
        if ema_delta is None:
            prev_state = self.state.client_expert_prev_states.get(memory_key)
            if prev_state is None:
                return previous_global_expert_state
            ema_delta = subtract_common_state_dict(prev_state, previous_global_expert_state)
        proposal_state = add_state_dicts(previous_global_expert_state, ema_delta)
        return _merge_with_reference_state(current_client_expert_state, proposal_state)

    def _aggregate_expert_group(
        self,
        expert_scope: str,
        expert_id: int,
        previous_global_expert_state: TensorState,
        client_entries: List[Tuple[int, TensorState, int, Dict[str, Any]]],
    ) -> Tuple[TensorState, Dict[str, Any]]:
        log_f_values: List[float] = []
        log_g_values: List[float] = []
        log_r_values: List[float] = []
        log_router_values: List[float] = []
        route_scores: List[float] = []
        sample_counts = [entry[2] for entry in client_entries]
        client_metrics = [entry[3] for entry in client_entries]
        current_states = [entry[1] for entry in client_entries]
        local_config = copy.deepcopy(self.config)
        local_config.server_val_beta = 0.0
        log_g_tensor, log_g_info = compute_log_g_values(
            client_metrics=client_metrics,
            expert_id=expert_id,
            num_experts=int(self.num_experts or 0),
            sample_counts=sample_counts,
            config=local_config,
            expert_scope=expert_scope,
        )
        global_server_val_tensor = torch.as_tensor(log_g_info["normalized_server_val_loss"], dtype=torch.float64)
        for idx, (client_id, current_state, _, metrics) in enumerate(client_entries):
            prev_bank = self.state.expert_prev_particle_bank.get(expert_scope)
            prev_weights = self.state.expert_prev_weights.get(expert_scope)
            if (
                self.config.use_resampling
                and prev_bank is not None
                and prev_weights is not None
                and len(prev_bank) == len(client_entries)
                and prev_weights.numel() == len(client_entries)
            ):
                ancestor_state = prev_bank[idx]
            else:
                ancestor_state = previous_global_expert_state
            proposal_state = self._client_proposal_mean(
                expert_scope=expert_scope,
                client_id=client_id,
                previous_global_expert_state=previous_global_expert_state,
                current_client_expert_state=current_state,
            )
            log_f_values.append(float(compute_log_f(current_state, ancestor_state, self.config).item()))
            log_r_values.append(float(compute_log_r(current_state, proposal_state, self.config).item()))
            route_score = _route_consistency_score(metrics, expert_scope, self.config.eps)
            route_scores.append(route_score)
            log_router_values.append(float(self.config.router_consistency_beta) * route_score)
            log_g_values.append(float(log_g_tensor[idx].item()))

        log_w = (
            float(self.config.expert_transition_beta) * torch.as_tensor(log_f_values, dtype=torch.float64)
            + float(self.config.local_obs_beta) * torch.as_tensor(log_g_values, dtype=torch.float64)
            - float(self.config.global_obs_beta) * global_server_val_tensor
            - float(self.config.proposal_beta) * torch.as_tensor(log_r_values, dtype=torch.float64)
            + torch.as_tensor(log_router_values, dtype=torch.float64)
        )
        alpha = safe_logsumexp_normalize(log_w, self.config.temperature, self.config.eps)
        aggregated = weighted_average_state_dicts(current_states, alpha, fallback_state=previous_global_expert_state)
        ess = effective_sample_size(alpha, self.config.eps)
        normalized_ess = ess / max(float(len(client_entries)), 1.0)
        logZ_increment = float((torch.logsumexp(log_w, dim=0) - math.log(max(len(client_entries), 1))).item())
        info = {
            "expert_scope": expert_scope,
            "expert_id": expert_id,
            "num_particles": len(client_entries),
            "weights": alpha.tolist(),
            "log_f_values": log_f_values,
            "log_g_values": log_g_values,
            "log_r_values": log_r_values,
            "log_router_values": log_router_values,
            "route_consistency_values": route_scores,
            "loss_values": log_g_info["raw_loss"],
            "server_val_loss_values": log_g_info["server_val_loss"],
            "usage_values": log_g_info["usage"],
            "route_counts": log_g_info["route_count"],
            "ess": ess,
            "normalized_ess": normalized_ess,
            "logZ_increment": logZ_increment,
            "fallback_used": False,
            "fallback_reason": "",
            "resampled": False,
            "log_f_mean": float(torch.as_tensor(log_f_values, dtype=torch.float64).mean().item()),
            "log_g_mean": float(torch.as_tensor(log_g_values, dtype=torch.float64).mean().item()),
            "log_r_mean": float(torch.as_tensor(log_r_values, dtype=torch.float64).mean().item()),
            "min_alpha": float(alpha.min().item()),
            "max_alpha": float(alpha.max().item()),
            "mean_alpha": float(alpha.mean().item()),
            "server_val_loss_mean": float(torch.as_tensor(log_g_info["server_val_loss"], dtype=torch.float64).mean().item()),
            "router_consistency_mean": float(torch.as_tensor(route_scores, dtype=torch.float64).mean().item()),
        }
        self.state.expert_prev_particle_bank[expert_scope] = [clone_detached_state_dict(state) for state in current_states]
        self.state.expert_prev_weights[expert_scope] = alpha.detach().cpu()
        self.state.expert_logz_history[expert_scope] = float(self.state.expert_logz_history.get(expert_scope, 0.0)) + logZ_increment
        info["logZ_total"] = self.state.expert_logz_history[expert_scope]
        for client_id, current_state, _, _ in client_entries:
            memory_key = (expert_scope, int(client_id))
            prev_state = self.state.client_expert_prev_states.get(memory_key)
            if prev_state is not None:
                delta = subtract_common_state_dict(current_state, prev_state)
                self.state.client_expert_delta_ema[memory_key] = ema_state_dict(
                    self.state.client_expert_delta_ema.get(memory_key),
                    delta,
                    self.config.proposal_ema_decay,
                )
            self.state.client_expert_prev_states[memory_key] = clone_detached_state_dict(current_state)
        return aggregated, info

    def _aggregate_router_group(
        self,
        router_scope: str,
        global_router_state: TensorState,
        client_entries: List[Tuple[int, TensorState, int, Dict[str, Any]]],
        expert_infos: Sequence[Dict[str, Any]],
    ) -> Tuple[TensorState, Dict[str, Any]]:
        router_scores: List[float] = []
        router_states = [entry[1] for entry in client_entries]
        for client_id, router_state, _, metrics in client_entries:
            quality_score = _router_quality_score(metrics, router_scope, expert_infos, self.config)
            proposal_score = 0.0
            if self.config.use_client_specific_proposal:
                memory_key = (router_scope, int(client_id))
                proposal_state = self.state.client_router_prev_states.get(memory_key)
                if proposal_state is not None:
                    proposal_score = float(compute_log_r(router_state, proposal_state, self.config).item())
            router_scores.append(quality_score - float(self.config.proposal_beta) * proposal_score)
        router_alpha = safe_logsumexp_normalize(
            torch.as_tensor(router_scores, dtype=torch.float64),
            self.config.route_weight_temperature,
            self.config.eps,
        )
        aggregated = weighted_average_state_dicts(router_states, router_alpha, fallback_state=global_router_state)
        ess = effective_sample_size(router_alpha, self.config.eps)
        normalized_ess = ess / max(float(len(client_entries)), 1.0)
        info = {
            "expert_scope": router_scope,
            "router_scope": router_scope,
            "expert_id": -1,
            "weights": router_alpha.tolist(),
            "ess": ess,
            "normalized_ess": normalized_ess,
            "fallback_used": False,
            "fallback_reason": "",
            "aggregation_source": "router_aware_validation_guided",
            "router_quality_scores": router_scores,
            "router_entropy_mean": float(sum(float(entry[3].get("router_entropy", 0.0)) for entry in client_entries) / max(len(client_entries), 1)),
            "router_aux_loss_mean": float(sum(float(entry[3].get("router_aux_loss", 0.0)) for entry in client_entries) / max(len(client_entries), 1)),
            "load_balance_gap_mean": float(sum(float(entry[3].get("load_balance_gap", 0.0)) for entry in client_entries) / max(len(client_entries), 1)),
        }
        for client_id, router_state, _, _ in client_entries:
            memory_key = (router_scope, int(client_id))
            prev_state = self.state.client_router_prev_states.get(memory_key)
            if prev_state is not None:
                delta = subtract_common_state_dict(router_state, prev_state)
                self.state.client_router_delta_ema[memory_key] = ema_state_dict(
                    self.state.client_router_delta_ema.get(memory_key),
                    delta,
                    self.config.proposal_ema_decay,
                )
            self.state.client_router_prev_states[memory_key] = clone_detached_state_dict(router_state)
        return aggregated, info

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, List[Dict[str, Any]]]:
        sample_weights = _sample_count_weights([num_samples for _, num_samples, _ in client_updates])
        fedavg_state = weighted_average_state_dicts([state_dict for state_dict, _, _ in client_updates], sample_weights.tolist())
        fedavg_state = _merge_with_reference_state(global_state, fedavg_state, device=device)
        base_state, fedavg_expert_groups, fedavg_router_groups = split_model_state_by_expert_and_router(
            fedavg_state,
            num_experts=num_experts,
        )
        _, global_expert_groups, global_router_groups = split_model_state_by_expert_and_router(
            global_state,
            num_experts=num_experts,
        )
        split_client_groups = [
            split_model_state_by_expert_and_router(state_dict, num_experts=num_experts)
            for state_dict, _, _ in client_updates
        ]
        client_expert_groups = [expert_groups for _, expert_groups, _ in split_client_groups]
        client_router_groups = [router_groups for _, _, router_groups in split_client_groups]
        merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        merged_router_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        round_infos: List[Dict[str, Any]] = []

        ordered_expert_scopes = _ordered_union_keys([global_expert_groups] + client_expert_groups + [fedavg_expert_groups])
        for expert_scope in ordered_expert_scopes:
            template = global_expert_groups.get(expert_scope) or fedavg_expert_groups.get(expert_scope)
            if template is None:
                continue
            expert_id = int(template["expert_id"])
            previous_global_expert_state = template["state_dict"]
            client_entries: List[Tuple[int, TensorState, int, Dict[str, Any]]] = []
            for (_, num_samples, metrics), client_group_map in zip(client_updates, client_expert_groups):
                if expert_scope not in client_group_map:
                    continue
                client_entries.append(
                    (
                        int((metrics or {}).get("client_id", len(client_entries))),
                        client_group_map[expert_scope]["state_dict"],
                        int(num_samples),
                        metrics or {},
                    )
                )
            if not client_entries:
                continue
            aggregated, info = self._aggregate_expert_group(
                expert_scope=expert_scope,
                expert_id=expert_id,
                previous_global_expert_state=previous_global_expert_state,
                client_entries=client_entries,
            )
            merged_expert_groups[expert_scope] = {
                "expert_id": expert_id,
                "expert_scope": expert_scope,
                "state_dict": _merge_with_reference_state(previous_global_expert_state, aggregated),
            }
            round_infos.append(info)

        ordered_router_scopes = _ordered_union_keys([global_router_groups] + client_router_groups + [fedavg_router_groups])
        for router_scope in ordered_router_scopes:
            template = global_router_groups.get(router_scope) or fedavg_router_groups.get(router_scope)
            if template is None:
                continue
            client_entries: List[Tuple[int, TensorState, int, Dict[str, Any]]] = []
            for (_, num_samples, metrics), client_router_map in zip(client_updates, client_router_groups):
                if router_scope not in client_router_map:
                    continue
                client_entries.append(
                    (
                        int((metrics or {}).get("client_id", len(client_entries))),
                        client_router_map[router_scope]["state_dict"],
                        int(num_samples),
                        metrics or {},
                    )
                )
            if not client_entries:
                continue
            matched_expert_infos = [
                info for info in round_infos
                if info.get("expert_id", -1) >= 0 and str(info.get("expert_scope", "")).startswith(router_scope[: -len("router")] + "experts.")
            ]
            aggregated, info = self._aggregate_router_group(
                router_scope=router_scope,
                global_router_state=template["state_dict"],
                client_entries=client_entries,
                expert_infos=matched_expert_infos,
            )
            merged_router_groups[router_scope] = {
                "router_scope": router_scope,
                "state_dict": _merge_with_reference_state(template["state_dict"], aggregated),
            }
            round_infos.append(info)

        merged_state = merge_named_groups_into_model(base_state, merged_expert_groups)
        merged_state = merge_named_groups_into_model(merged_state, merged_router_groups)
        merged_state = _merge_with_reference_state(global_state, merged_state, device=device)
        self.state.round_idx += 1
        return merged_state, round_infos
