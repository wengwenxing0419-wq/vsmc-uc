"""
Temporal-aware VSMC-style expert aggregation for federated Switch Transformer models.

This module extends the existing server-side VSMC expert reweighting rule with a
small amount of temporal latent structure. Each round still treats client experts
as particle candidates, but the particle score now also measures whether the
client update is consistent with the recent global expert trajectory.

For each expert scope s and round t:
    delta_{i,t}^s = theta_{i,t}^s - theta_{t-1}^s
    m_{t-1}^s     = momentum memory of recent global expert updates

The temporal score adds:
    + beta_align * cos(delta_{i,t}^s, m_{t-1}^s)
    - beta_jump  * normalized(||delta_{i,t}^s - m_{t-1}^s||)

This is still a lightweight heuristic rather than a full latent-state particle
filter, but it begins to use cross-round structure instead of scoring each round
independently.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vsmc_expert_aggregator import (
    ExpertParticleStats,
    TensorState,
    VSMCAggregatorConfig,
    _clone_detached_tensor,
    _extract_expert_metric,
    _extract_metric,
    _merge_with_reference_state,
    _min_max_normalize,
    _ordered_union_keys,
    _sample_count_weights,
    effective_sample_size,
    flatten_state_dict_l2_distance,
    safe_normalize_log_weights,
    split_model_state_by_expert,
    weighted_average_state_dicts,
    merge_expert_states_into_model,
)


@dataclass
class TemporalVSMCAggregatorConfig(VSMCAggregatorConfig):
    momentum: float = 0.8
    align_beta: float = 0.5
    jump_beta: float = 0.2
    warmup_rounds: int = 2
    score_mode: str = "align_jump"
    history_size: int = 1


@dataclass
class TemporalExpertMemory:
    prev_global_state: TensorState
    prev_prev_global_state: Optional[TensorState] = None
    momentum_delta_state: Optional[TensorState] = None
    delta_history_states: Optional[List[TensorState]] = None
    last_round: int = 0


def subtract_state_dicts(a: TensorState, b: TensorState) -> TensorState:
    out: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys([a, b]):
        if key not in a or key not in b:
            continue
        tensor_a = a[key].detach()
        tensor_b = b[key].detach()
        if tensor_a.shape != tensor_b.shape:
            continue
        if not (torch.is_floating_point(tensor_a) and torch.is_floating_point(tensor_b)):
            continue
        out[key] = tensor_a.to(dtype=torch.float64, device="cpu") - tensor_b.to(dtype=torch.float64, device="cpu")
    return out


def flatten_state_dict_l2_norm(state_dict: TensorState) -> float:
    squared_sum = 0.0
    num_params = 0
    for tensor in state_dict.values():
        detached = tensor.detach()
        if not torch.is_floating_point(detached):
            continue
        value = detached.to(dtype=torch.float64, device="cpu")
        squared_sum += torch.sum(value * value).item()
        num_params += value.numel()
    if num_params == 0:
        return 0.0
    return math.sqrt(squared_sum) / math.sqrt(float(num_params))


def flatten_state_dict_cosine_similarity(a: TensorState, b: TensorState, eps: float = 1e-12) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    shared_keys = sorted(set(a.keys()) & set(b.keys()))
    for key in shared_keys:
        tensor_a = a[key].detach()
        tensor_b = b[key].detach()
        if tensor_a.shape != tensor_b.shape:
            continue
        if not (torch.is_floating_point(tensor_a) and torch.is_floating_point(tensor_b)):
            continue
        flat_a = tensor_a.to(dtype=torch.float64, device="cpu").reshape(-1)
        flat_b = tensor_b.to(dtype=torch.float64, device="cpu").reshape(-1)
        dot += torch.dot(flat_a, flat_b).item()
        norm_a += torch.dot(flat_a, flat_a).item()
        norm_b += torch.dot(flat_b, flat_b).item()
    if norm_a <= eps or norm_b <= eps:
        return 0.0
    return float(dot / (math.sqrt(norm_a) * math.sqrt(norm_b) + eps))


def blend_state_dicts(old_state: Optional[TensorState], new_state: TensorState, momentum: float) -> TensorState:
    if old_state is None:
        return collections.OrderedDict(
            (key, _clone_detached_tensor(tensor))
            for key, tensor in new_state.items()
        )

    blended: TensorState = collections.OrderedDict()
    keep = float(max(0.0, min(1.0, momentum)))
    take = 1.0 - keep
    for key in _ordered_union_keys([old_state, new_state]):
        if key not in new_state and key not in old_state:
            continue
        if key not in old_state:
            blended[key] = _clone_detached_tensor(new_state[key])
            continue
        if key not in new_state:
            blended[key] = _clone_detached_tensor(old_state[key])
            continue
        old_tensor = old_state[key].detach()
        new_tensor = new_state[key].detach()
        if old_tensor.shape != new_tensor.shape or not (
            torch.is_floating_point(old_tensor) and torch.is_floating_point(new_tensor)
        ):
            blended[key] = _clone_detached_tensor(new_tensor)
            continue
        blended[key] = (
            old_tensor.to(dtype=torch.float64, device="cpu") * keep
            + new_tensor.to(dtype=torch.float64, device="cpu") * take
        )
    return blended


def average_state_dicts(state_dicts: Sequence[TensorState]) -> Optional[TensorState]:
    if len(state_dicts) == 0:
        return None
    if len(state_dicts) == 1:
        return collections.OrderedDict(
            (key, _clone_detached_tensor(tensor))
            for key, tensor in state_dicts[0].items()
        )

    averaged: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys(state_dicts):
        available = [state_dict[key] for state_dict in state_dicts if key in state_dict]
        if not available:
            continue
        reference_tensor = available[0].detach()
        if not torch.is_floating_point(reference_tensor):
            averaged[key] = _clone_detached_tensor(reference_tensor)
            continue
        acc = torch.zeros_like(reference_tensor, dtype=torch.float64, device="cpu")
        for tensor in available:
            acc = acc + tensor.detach().to(dtype=torch.float64, device="cpu")
        averaged[key] = acc / float(len(available))
    return averaged


def _select_temporal_reference_state(
    temporal_memory: Optional[TemporalExpertMemory],
    config: TemporalVSMCAggregatorConfig,
) -> Optional[TensorState]:
    if temporal_memory is None:
        return None
    if int(config.history_size) > 1:
        history = temporal_memory.delta_history_states or []
        if history:
            selected = history[-int(config.history_size):]
            return average_state_dicts(selected)
    return temporal_memory.momentum_delta_state


def compute_temporal_features(
    client_expert_state: TensorState,
    prev_global_state: TensorState,
    temporal_reference_state: Optional[TensorState] = None,
    score_mode: str = "align_jump",
) -> Dict[str, float]:
    client_delta = subtract_state_dicts(client_expert_state, prev_global_state)
    client_delta_norm = flatten_state_dict_l2_norm(client_delta)
    reference_norm = flatten_state_dict_l2_norm(temporal_reference_state or {})

    if temporal_reference_state is None or len(temporal_reference_state) == 0:
        return {
            "alignment": 0.0,
            "jump": 0.0,
            "client_delta_norm": client_delta_norm,
            "reference_norm": reference_norm,
        }

    alignment = flatten_state_dict_cosine_similarity(client_delta, temporal_reference_state)
    if score_mode == "align_only":
        jump = 0.0
    elif score_mode == "align_cosine_jump":
        jump = max(0.0, 1.0 - alignment)
    else:
        jump_state = subtract_state_dicts(client_delta, temporal_reference_state)
        jump = flatten_state_dict_l2_norm(jump_state)
    return {
        "alignment": alignment,
        "jump": jump,
        "client_delta_norm": client_delta_norm,
        "reference_norm": reference_norm,
    }


def _compute_temporal_log_weight(
    normalized_loss: float,
    usage_value: float,
    normalized_drift: float,
    sample_count: int,
    route_count: float,
    target_usage: float,
    alignment: float,
    normalized_jump: float,
    config: TemporalVSMCAggregatorConfig,
) -> float:
    finite_values = (
        normalized_loss,
        usage_value,
        normalized_drift,
        route_count,
        target_usage,
        alignment,
        normalized_jump,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        return float("nan")

    usage_penalty = abs(float(usage_value) - float(target_usage))
    route_term = config.route_beta * math.log(max(float(route_count), 0.0) + 1.0)
    if float(route_count) < float(config.min_route_count):
        route_term -= config.route_beta * math.log(float(config.min_route_count) + 1.0)

    log_weight = (
        -config.loss_beta * float(normalized_loss)
        -config.usage_beta * usage_penalty
        -config.drift_beta * float(normalized_drift)
        + config.sample_beta * math.log(max(int(sample_count), 0) + 1.0)
        + route_term
        + config.align_beta * float(alignment)
        - config.jump_beta * float(normalized_jump)
    )
    if not math.isfinite(log_weight):
        return float("nan")
    clamp_value = abs(float(config.clamp_log_weight))
    return max(-clamp_value, min(clamp_value, log_weight))


def aggregate_expert_particles_temporal(
    global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_sample_counts: Sequence[int],
    client_metrics: Sequence[Dict[str, Any]],
    expert_id: int,
    num_experts: int,
    config: TemporalVSMCAggregatorConfig,
    round_idx: int,
    temporal_memory: Optional[TemporalExpertMemory] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TensorState, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "weights": [],
        "pre_resample_weights": [],
        "log_weights": [],
        "ess": 0.0,
        "normalized_ess": 0.0,
        "loss_values": [],
        "usage_values": [],
        "drift_values": [],
        "route_counts": [],
        "alignment_values": [],
        "jump_values": [],
        "delta_norm_values": [],
        "momentum_norm": 0.0,
        "temporal_reference_norm": 0.0,
        "temporal_score_mode": config.score_mode,
        "temporal_history_size": int(config.history_size),
        "fallback_used": False,
        "resampled": False,
        "selected_indices": [],
        "temporal_enabled": False,
        "warmup_used": False,
    }

    reference_state = collections.OrderedDict(
        (key, _clone_detached_tensor(tensor, device=device))
        for key, tensor in global_expert_state.items()
    )

    if len(client_expert_states) == 0:
        info["fallback_used"] = True
        return reference_state, info

    if len(client_expert_states) != len(client_sample_counts) or len(client_expert_states) != len(client_metrics):
        raise ValueError("client_expert_states, client_sample_counts, and client_metrics must be aligned")

    target_usage = config.target_usage if config.target_usage is not None else 1.0 / max(int(num_experts), 1)
    temporal_reference_state = _select_temporal_reference_state(temporal_memory, config)
    use_temporal = (
        temporal_memory is not None
        and temporal_reference_state is not None
        and round_idx > int(config.warmup_rounds)
    )
    info["temporal_enabled"] = bool(use_temporal)
    info["warmup_used"] = bool(round_idx <= int(config.warmup_rounds))
    if temporal_memory is not None and temporal_memory.momentum_delta_state is not None:
        info["momentum_norm"] = flatten_state_dict_l2_norm(temporal_memory.momentum_delta_state)
    if temporal_reference_state is not None:
        info["temporal_reference_norm"] = flatten_state_dict_l2_norm(temporal_reference_state)

    loss_values: List[float] = []
    usage_values: List[float] = []
    drift_values: List[float] = []
    route_counts: List[float] = []
    alignment_values: List[float] = []
    jump_values: List[float] = []
    delta_norm_values: List[float] = []
    particle_stats: List[ExpertParticleStats] = []

    for idx, (state_dict, sample_count, metrics) in enumerate(
        zip(client_expert_states, client_sample_counts, client_metrics)
    ):
        metrics = metrics or {}
        loss_value = _extract_expert_metric(metrics, ("expert_losses", "expert_loss"), expert_id)
        if loss_value is None:
            loss_value = _extract_metric(metrics, ("val_loss", "train_loss"), default=0.0) or 0.0

        usage_value = _extract_expert_metric(
            metrics,
            ("expert_usage", "expert_usages", "usage", "usages"),
            expert_id,
            default=target_usage,
        )
        if usage_value is None:
            usage_value = target_usage

        route_count = _extract_expert_metric(
            metrics,
            ("route_counts", "route_count", "expert_counts"),
            expert_id,
            default=0.0,
        )
        if route_count is None:
            route_count = 0.0

        drift_value = flatten_state_dict_l2_distance(state_dict, reference_state)
        temporal_features = compute_temporal_features(
            client_expert_state=state_dict,
            prev_global_state=reference_state,
            temporal_reference_state=temporal_reference_state if use_temporal else None,
            score_mode=config.score_mode,
        )

        loss_values.append(float(loss_value))
        usage_values.append(float(usage_value))
        drift_values.append(float(drift_value))
        route_counts.append(float(route_count))
        alignment_values.append(float(temporal_features["alignment"]))
        jump_values.append(float(temporal_features["jump"]))
        delta_norm_values.append(float(temporal_features["client_delta_norm"]))
        particle_stats.append(
            ExpertParticleStats(
                client_index=idx,
                client_id=int(metrics["client_id"]) if "client_id" in metrics else None,
                sample_count=int(sample_count),
                loss_value=float(loss_value),
                usage_value=float(usage_value),
                drift_value=float(drift_value),
                route_count=float(route_count),
            )
        )

    info["loss_values"] = loss_values
    info["usage_values"] = usage_values
    info["drift_values"] = drift_values
    info["route_counts"] = route_counts
    info["alignment_values"] = alignment_values
    info["jump_values"] = jump_values
    info["delta_norm_values"] = delta_norm_values

    if len(client_expert_states) == 1:
        info["weights"] = [1.0]
        info["pre_resample_weights"] = [1.0]
        info["log_weights"] = [0.0]
        info["ess"] = 1.0
        info["normalized_ess"] = 1.0
        client_state = weighted_average_state_dicts(client_expert_states, [1.0])
        return _merge_with_reference_state(reference_state, client_state, device=device), info

    normalized_losses = _min_max_normalize(loss_values)
    normalized_drifts = _min_max_normalize(drift_values)
    normalized_jumps = _min_max_normalize(jump_values)

    log_weights: List[float] = []
    for idx, stats in enumerate(particle_stats):
        stats.normalized_loss = float(normalized_losses[idx].item()) if normalized_losses.numel() else 0.0
        stats.normalized_drift = float(normalized_drifts[idx].item()) if normalized_drifts.numel() else 0.0
        stats.usage_penalty = abs(stats.usage_value - target_usage)
        stats.log_weight = _compute_temporal_log_weight(
            normalized_loss=stats.normalized_loss,
            usage_value=stats.usage_value,
            normalized_drift=stats.normalized_drift,
            sample_count=stats.sample_count,
            route_count=stats.route_count,
            target_usage=target_usage,
            alignment=alignment_values[idx] if use_temporal else 0.0,
            normalized_jump=float(normalized_jumps[idx].item()) if use_temporal and normalized_jumps.numel() else 0.0,
            config=config,
        )
        log_weights.append(stats.log_weight)

    log_weight_tensor = torch.as_tensor(log_weights, dtype=torch.float64)
    if not torch.isfinite(log_weight_tensor).all():
        log_weight_tensor = torch.full_like(log_weight_tensor, float("nan"))

    temperature = max(float(config.temperature), 1e-6)
    normalized_weights = safe_normalize_log_weights(log_weight_tensor / temperature)

    def _fallback_aggregate() -> Tuple[TensorState, Dict[str, Any]]:
        fallback_weights = _sample_count_weights(client_sample_counts)
        fallback_state = weighted_average_state_dicts(client_expert_states, fallback_weights.tolist())
        fallback_info = dict(info)
        fallback_info["weights"] = fallback_weights.tolist()
        fallback_info["pre_resample_weights"] = fallback_weights.tolist()
        fallback_info["log_weights"] = log_weights
        fallback_info["fallback_used"] = True
        fallback_info["ess"] = effective_sample_size(fallback_weights)
        fallback_info["normalized_ess"] = fallback_info["ess"] / max(float(len(client_expert_states)), 1.0)
        return _merge_with_reference_state(reference_state, fallback_state, device=device), fallback_info

    if normalized_weights.numel() == 0 or not torch.isfinite(normalized_weights).all():
        if config.fallback_to_fedavg_on_nan:
            return _fallback_aggregate()
        raise ValueError("Failed to normalize temporal expert log weights")

    ess = effective_sample_size(normalized_weights)
    info["log_weights"] = log_weights
    info["pre_resample_weights"] = normalized_weights.tolist()
    info["ess"] = ess
    info["normalized_ess"] = ess / max(float(len(client_expert_states)), 1.0)
    info["weights"] = normalized_weights.tolist()

    aggregated_state = weighted_average_state_dicts(client_expert_states, normalized_weights.tolist())
    return _merge_with_reference_state(reference_state, aggregated_state, device=device), info


class TemporalVSMCExpertAggregator:
    def __init__(self, config: TemporalVSMCAggregatorConfig, logger: Optional[Any] = None):
        self.config = config
        self.logger = logger

    def _update_memory(
        self,
        reference_expert_state: TensorState,
        aggregated_expert_state: TensorState,
        previous_memory: Optional[TemporalExpertMemory],
        round_idx: int,
    ) -> TemporalExpertMemory:
        new_delta = subtract_state_dicts(aggregated_expert_state, reference_expert_state)
        previous_momentum = previous_memory.momentum_delta_state if previous_memory is not None else None
        updated_momentum = blend_state_dicts(previous_momentum, new_delta, self.config.momentum)
        previous_global = previous_memory.prev_global_state if previous_memory is not None else reference_expert_state
        previous_history = list(previous_memory.delta_history_states or []) if previous_memory is not None else []
        previous_history.append(collections.OrderedDict(
            (key, _clone_detached_tensor(tensor))
            for key, tensor in new_delta.items()
        ))
        max_history = max(int(self.config.history_size), 1)
        previous_history = previous_history[-max_history:]
        return TemporalExpertMemory(
            prev_global_state=collections.OrderedDict(
                (key, _clone_detached_tensor(tensor))
                for key, tensor in aggregated_expert_state.items()
            ),
            prev_prev_global_state=collections.OrderedDict(
                (key, _clone_detached_tensor(tensor))
                for key, tensor in previous_global.items()
            ) if previous_global is not None else None,
            momentum_delta_state=updated_momentum,
            delta_history_states=previous_history,
            last_round=int(round_idx),
        )

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        round_idx: int,
        temporal_memory: Optional[Dict[str, TemporalExpertMemory]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, List[Dict[str, Any]], Dict[str, TemporalExpertMemory]]:
        if len(client_updates) == 0:
            raise ValueError("client_updates must not be empty")

        temporal_memory = temporal_memory or {}
        sample_weights = _sample_count_weights([num_samples for _, num_samples, _ in client_updates])
        fedavg_state = weighted_average_state_dicts(
            [state_dict for state_dict, _, _ in client_updates],
            sample_weights.tolist(),
        )
        fedavg_state = _merge_with_reference_state(global_state, fedavg_state, device=device)

        base_state, fedavg_expert_groups = split_model_state_by_expert(fedavg_state, num_experts=num_experts)
        _, global_expert_groups = split_model_state_by_expert(global_state, num_experts=num_experts)
        client_expert_groups = [
            split_model_state_by_expert(state_dict, num_experts=num_experts)[1]
            for state_dict, _, _ in client_updates
        ]

        ordered_group_names = _ordered_union_keys([global_expert_groups] + client_expert_groups + [fedavg_expert_groups])
        merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        round_infos: List[Dict[str, Any]] = []
        updated_memory: Dict[str, TemporalExpertMemory] = dict(temporal_memory)

        for expert_group_name in ordered_group_names:
            template_group = global_expert_groups.get(expert_group_name) or fedavg_expert_groups.get(expert_group_name)
            if template_group is None:
                continue

            expert_id = int(template_group["expert_id"])
            reference_expert_state = collections.OrderedDict(template_group["state_dict"])
            client_expert_states: List[TensorState] = []
            client_sample_counts: List[int] = []
            client_metrics: List[Dict[str, Any]] = []

            for (state_dict, num_samples, metrics), client_group_map in zip(client_updates, client_expert_groups):
                del state_dict
                if expert_group_name not in client_group_map:
                    continue
                client_expert_states.append(client_group_map[expert_group_name]["state_dict"])
                client_sample_counts.append(int(num_samples))
                client_metrics.append(metrics or {})

            aggregated_expert_state, info = aggregate_expert_particles_temporal(
                global_expert_state=reference_expert_state,
                client_expert_states=client_expert_states,
                client_sample_counts=client_sample_counts,
                client_metrics=client_metrics,
                expert_id=expert_id,
                num_experts=num_experts,
                config=self.config,
                round_idx=round_idx,
                temporal_memory=temporal_memory.get(expert_group_name),
                device=device,
            )

            merged_expert_groups[expert_group_name] = {
                "expert_id": expert_id,
                "state_dict": aggregated_expert_state,
            }
            updated_memory[expert_group_name] = self._update_memory(
                reference_expert_state=reference_expert_state,
                aggregated_expert_state=aggregated_expert_state,
                previous_memory=temporal_memory.get(expert_group_name),
                round_idx=round_idx,
            )

            info["expert_id"] = expert_id
            info["expert_scope"] = expert_group_name
            round_infos.append(info)

        merged_state = merge_expert_states_into_model(base_state, merged_expert_groups)
        merged_state = _merge_with_reference_state(global_state, merged_state, device=device)
        return merged_state, round_infos, updated_memory
