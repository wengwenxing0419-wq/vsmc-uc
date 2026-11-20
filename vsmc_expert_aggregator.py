"""
This module adapts the particle weighting idea of Variational Sequential Monte Carlo to
federated expert aggregation.

For each expert e, each client i provides a particle x_i^e, namely the local expert parameters.
Instead of FedAvg:
    theta_e = sum_i n_i / sum_j n_j * theta_{i,e}
we compute:
    alpha_{i,e} = softmax(log_w_{i,e} / tau)
    theta_e = sum_i alpha_{i,e} * theta_{i,e}

The log weight is a heuristic observation likelihood:
    log_w_{i,e}
      = - beta_loss  * normalized_loss_{i,e}
        - beta_usage * |usage_{i,e} - target_usage|
        - beta_drift * normalized_drift(theta_{i,e}, theta_e_old)
        + beta_sample * log(n_i + 1)
        + beta_route  * log(route_count_{i,e} + 1)

This is not a full reproduction of the original time-series VSMC model.
It is a VSMC-inspired server-side aggregation rule using SMC-style normalized particle weights,
ESS diagnostics, and cross-round recursive log-weight memory.

Communication round r is treated as the SMC time step.
Each client expert is treated as a particle candidate.
The incremental log weight is an engineering approximation to a sequential
log target minus log proposal ratio.
The log_weight_memory maintained by VSMCExpertAggregator corresponds to the
historical recursive particle log weights across communication rounds.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


TensorState = Dict[str, torch.Tensor]

_EXPERT_PATTERNS = [
    re.compile(r"^(?P<group>.*?experts\.(?P<id>\d+))\."),
    re.compile(r"^(?P<group>.*?expert\.(?P<id>\d+))\."),
    re.compile(r"^(?P<group>.*?moe\.experts\.(?P<id>\d+))\."),
    re.compile(r"^(?P<group>.*?block_sparse_moe\.experts\.(?P<id>\d+))\."),
]


@dataclass
class VSMCAggregatorConfig:
    enabled: bool = True
    num_particles: int = 0
    variant: str = "baseline"
    temperature: float = 1.0
    loss_beta: float = 1.0
    usage_beta: float = 0.3
    drift_beta: float = 0.05
    sample_beta: float = 0.2
    route_beta: float = 0.2
    target_usage: Optional[float] = None
    min_route_count: int = 1
    ess_threshold: float = 0.5
    use_resampling: bool = False
    resampling_noise_std: float = 0.0
    fallback_to_fedavg_on_nan: bool = True
    clamp_log_weight: float = 30.0
    history_decay: float = 0.9
    missing_decay: float = 0.9
    degeneration_ess_threshold: float = 0.2
    degeneration_temperature_scale: float = 2.0
    fallback_to_fedavg_on_degenerate: bool = False
    debug: bool = False


@dataclass
class ExpertParticleStats:
    client_index: int
    client_id: Optional[int]
    sample_count: int
    loss_value: float
    usage_value: float
    drift_value: float
    route_count: float
    normalized_loss: float = 0.0
    usage_penalty: float = 0.0
    normalized_drift: float = 0.0
    log_weight: float = 0.0
    weight: float = 0.0


def _match_expert_param(name: str) -> Optional[re.Match]:
    lower_name = name.lower()
    if ".router." in lower_name or ".gate." in lower_name:
        return None
    for pattern in _EXPERT_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            return match
    return None


def is_expert_param(name: str) -> bool:
    return _match_expert_param(name) is not None


def get_expert_id_from_name(name: str) -> Optional[int]:
    match = _match_expert_param(name)
    if match is None:
        return None
    return int(match.group("id"))


def _get_expert_group_from_name(name: str) -> Optional[str]:
    match = _match_expert_param(name)
    if match is None:
        return None
    return str(match.group("group"))


def _clone_detached_tensor(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    cloned = tensor.detach().clone()
    if device is not None:
        cloned = cloned.to(device)
    return cloned


def _ordered_union_keys(state_dicts: Iterable[TensorState]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for state_dict in state_dicts:
        for key in state_dict.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _min_max_normalize(values: Sequence[float], eps: float = 1e-12) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return tensor
    if not torch.isfinite(tensor).all():
        return torch.full_like(tensor, float("nan"))
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    scale = max_value - min_value
    if scale.abs().item() <= eps:
        return torch.zeros_like(tensor)
    return (tensor - min_value) / (scale + eps)


def _sample_count_weights(client_sample_counts: Sequence[int]) -> torch.Tensor:
    sample_counts = torch.as_tensor(client_sample_counts, dtype=torch.float64).clamp_min(0.0)
    total = sample_counts.sum()
    if sample_counts.numel() == 0:
        return sample_counts
    if total.item() <= 0.0 or not torch.isfinite(total):
        return torch.full_like(sample_counts, 1.0 / sample_counts.numel())
    return sample_counts / total


def safe_normalize_log_weights(log_w: Sequence[float], eps: float = 1e-12) -> torch.Tensor:
    log_w_tensor = torch.as_tensor(log_w, dtype=torch.float64)
    if log_w_tensor.numel() == 0:
        return log_w_tensor
    if not torch.isfinite(log_w_tensor).all():
        return torch.full_like(log_w_tensor, float("nan"))
    max_log_w = torch.max(log_w_tensor)
    weights = torch.exp(log_w_tensor - max_log_w)
    weight_sum = weights.sum()
    if (not torch.isfinite(weight_sum)) or weight_sum.item() <= eps:
        return torch.full_like(log_w_tensor, float("nan"))
    return weights / (weight_sum + eps)


def effective_sample_size(w: Sequence[float]) -> float:
    weights = torch.as_tensor(w, dtype=torch.float64)
    if weights.numel() == 0 or not torch.isfinite(weights).all():
        return 0.0
    squared_sum = torch.sum(weights * weights).item()
    if squared_sum <= 0.0:
        return 0.0
    return 1.0 / squared_sum


def stratified_resampling(
    w: Sequence[float],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    weights = torch.as_tensor(w, dtype=torch.float64).flatten()
    if weights.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if not torch.isfinite(weights).all():
        raise ValueError("Resampling weights must be finite")
    weights = weights / weights.sum().clamp_min(1e-12)
    num_particles = weights.numel()
    positions = (
        torch.arange(num_particles, dtype=torch.float64)
        + torch.rand(num_particles, generator=generator, dtype=torch.float64)
    ) / num_particles
    cumulative = torch.cumsum(weights, dim=0)
    return torch.searchsorted(cumulative, positions, right=False).clamp(max=num_particles - 1)


def weighted_average_state_dicts(state_dicts: Sequence[TensorState], weights: Sequence[float]) -> TensorState:
    if len(state_dicts) == 0:
        return collections.OrderedDict()
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")

    weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
    aggregated: TensorState = collections.OrderedDict()

    for key in _ordered_union_keys(state_dicts):
        available = [
            (state_dict[key], weight_tensor[idx])
            for idx, state_dict in enumerate(state_dicts)
            if key in state_dict
        ]
        if not available:
            continue

        reference_tensor = available[0][0].detach()
        if not torch.is_floating_point(reference_tensor):
            aggregated[key] = reference_tensor.clone()
            continue

        local_weights = torch.stack([weight for _, weight in available])
        local_weight_sum = local_weights.sum()
        if local_weight_sum.item() <= 0.0 or not torch.isfinite(local_weight_sum):
            local_weights = torch.full_like(local_weights, 1.0 / local_weights.numel())
        else:
            local_weights = local_weights / local_weight_sum

        acc = torch.zeros_like(reference_tensor)
        for (tensor, _), local_weight in zip(available, local_weights):
            acc = acc + tensor.detach().to(dtype=reference_tensor.dtype, device=reference_tensor.device) * local_weight.to(
                dtype=reference_tensor.dtype,
                device=reference_tensor.device,
            )
        aggregated[key] = acc

    return aggregated


def flatten_state_dict_l2_distance(a: TensorState, b: TensorState) -> float:
    squared_sum = 0.0
    num_params = 0
    shared_keys = sorted(set(a.keys()) & set(b.keys()))
    for key in shared_keys:
        tensor_a = a[key].detach()
        tensor_b = b[key].detach()
        if (not torch.is_floating_point(tensor_a)) or (not torch.is_floating_point(tensor_b)):
            continue
        if tensor_a.shape != tensor_b.shape:
            continue
        diff = tensor_a.to(dtype=torch.float64, device="cpu") - tensor_b.to(dtype=torch.float64, device="cpu")
        squared_sum += torch.sum(diff * diff).item()
        num_params += diff.numel()
    if num_params == 0:
        return 0.0
    return math.sqrt(squared_sum) / math.sqrt(float(num_params))


def _extract_indexed_metric(container: Any, expert_id: int) -> Optional[float]:
    if container is None:
        return None
    if isinstance(container, dict):
        for key in (expert_id, str(expert_id), f"expert_{expert_id}", f"expert{expert_id}"):
            if key in container:
                value = container[key]
                return float(value.item()) if torch.is_tensor(value) else float(value)
        return None
    if torch.is_tensor(container):
        flat = container.detach().reshape(-1)
        if 0 <= expert_id < flat.numel():
            return float(flat[expert_id].item())
        return None
    if isinstance(container, (list, tuple)):
        if 0 <= expert_id < len(container):
            value = container[expert_id]
            return float(value.item()) if torch.is_tensor(value) else float(value)
        return None
    return None


def _extract_metric(metrics: Dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        return float(value.item()) if torch.is_tensor(value) else float(value)
    return default


def _extract_expert_metric(
    metrics: Dict[str, Any],
    keys: Sequence[str],
    expert_id: int,
    default: Optional[float] = None,
) -> Optional[float]:
    for key in keys:
        if key not in metrics:
            continue
        value = _extract_indexed_metric(metrics[key], expert_id)
        if value is not None:
            return value
    return default


def compute_expert_log_weight(
    normalized_loss: float,
    usage_value: float,
    normalized_drift: float,
    sample_count: int,
    route_count: float,
    target_usage: float,
    config: VSMCAggregatorConfig,
) -> float:
    finite_values = (
        float(normalized_loss),
        float(usage_value),
        float(normalized_drift),
        float(route_count),
        float(target_usage),
    )
    if not all(math.isfinite(value) for value in finite_values):
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
    )
    if not math.isfinite(log_weight):
        return float("nan")
    clamp_value = abs(float(config.clamp_log_weight))
    return max(-clamp_value, min(clamp_value, log_weight))


def _merge_with_reference_state(
    reference_state: TensorState,
    partial_state: TensorState,
    device: Optional[torch.device] = None,
) -> TensorState:
    merged: TensorState = collections.OrderedDict()
    ordered_keys = _ordered_union_keys([reference_state, partial_state])
    for key in ordered_keys:
        if key in partial_state:
            merged[key] = _clone_detached_tensor(partial_state[key], device=device)
        elif key in reference_state:
            merged[key] = _clone_detached_tensor(reference_state[key], device=device)
    return merged


def _maybe_add_resampling_noise(
    state_dict: TensorState,
    noise_std: float,
    device: Optional[torch.device] = None,
) -> TensorState:
    copied: TensorState = collections.OrderedDict()
    for key, tensor in state_dict.items():
        out_tensor = _clone_detached_tensor(tensor, device=device)
        if noise_std > 0.0 and torch.is_floating_point(out_tensor):
            out_tensor = out_tensor + torch.randn_like(out_tensor) * noise_std
        copied[key] = out_tensor
    return copied


def aggregate_expert_particles(
    global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_sample_counts: Sequence[int],
    client_metrics: Sequence[Dict[str, Any]],
    expert_id: int,
    num_experts: int,
    config: VSMCAggregatorConfig,
    previous_log_weights: Optional[Sequence[float]] = None,
    client_ids: Optional[Sequence[Optional[int]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TensorState, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "weights": [],
        "pre_resample_weights": [],
        "incremental_log_weights": [],
        "previous_log_weights": [],
        "log_weight_totals": [],
        "log_weights": [],
        "ess": 0.0,
        "normalized_ess": 0.0,
        "loss_values": [],
        "usage_values": [],
        "drift_values": [],
        "route_counts": [],
        "logZ_increment": 0.0,
        "logZ_total": 0.0,
        "degeneration": False,
        "effective_temperature": float(config.temperature),
        "fallback_used": False,
        "resampled": False,
        "selected_indices": [],
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
    if previous_log_weights is not None and len(previous_log_weights) != len(client_expert_states):
        raise ValueError("previous_log_weights must align with client_expert_states")
    if client_ids is not None and len(client_ids) != len(client_expert_states):
        raise ValueError("client_ids must align with client_expert_states")

    target_usage = config.target_usage if config.target_usage is not None else 1.0 / max(int(num_experts), 1)

    loss_values: List[float] = []
    usage_values: List[float] = []
    drift_values: List[float] = []
    route_counts: List[float] = []
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

        loss_values.append(float(loss_value))
        usage_values.append(float(usage_value))
        drift_values.append(float(drift_value))
        route_counts.append(float(route_count))
        particle_stats.append(
            ExpertParticleStats(
                client_index=idx,
                client_id=(
                    int(client_ids[idx])
                    if client_ids is not None and client_ids[idx] is not None
                    else int(metrics["client_id"]) if "client_id" in metrics else None
                ),
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

    incremental_log_weights: List[float] = []
    total_log_weights: List[float] = []
    previous_log_weight_values: List[float] = []
    for idx, stats in enumerate(particle_stats):
        stats.normalized_loss = float(normalized_losses[idx].item()) if normalized_losses.numel() else 0.0
        stats.normalized_drift = float(normalized_drifts[idx].item()) if normalized_drifts.numel() else 0.0
        stats.usage_penalty = abs(stats.usage_value - target_usage)
        inc_log_weight = compute_expert_log_weight(
            normalized_loss=stats.normalized_loss,
            usage_value=stats.usage_value,
            normalized_drift=stats.normalized_drift,
            sample_count=stats.sample_count,
            route_count=stats.route_count,
            target_usage=target_usage,
            config=config,
        )
        prev_log_weight = 0.0
        if previous_log_weights is not None:
            candidate = float(previous_log_weights[idx])
            prev_log_weight = candidate if math.isfinite(candidate) else 0.0
        total_log_weight = self._combine_log_weight(prev_log_weight=prev_log_weight, inc_log_weight=inc_log_weight)
        if math.isfinite(total_log_weight):
            clamp_value = abs(float(config.clamp_log_weight))
            total_log_weight = max(-clamp_value, min(clamp_value, total_log_weight))
        stats.log_weight = total_log_weight
        incremental_log_weights.append(inc_log_weight)
        previous_log_weight_values.append(prev_log_weight)
        total_log_weights.append(total_log_weight)

    incremental_log_weight_tensor = torch.as_tensor(incremental_log_weights, dtype=torch.float64)
    total_log_weight_tensor = torch.as_tensor(total_log_weights, dtype=torch.float64)
    if torch.isfinite(incremental_log_weight_tensor).all() and incremental_log_weight_tensor.numel() > 0:
        max_inc_log_w = torch.max(incremental_log_weight_tensor)
        logZ_increment = (
            max_inc_log_w
            + torch.log(torch.sum(torch.exp(incremental_log_weight_tensor - max_inc_log_w)).clamp_min(1e-12))
            - math.log(max(len(client_expert_states), 1))
        )
        info["logZ_increment"] = float(logZ_increment.item())
    else:
        info["logZ_increment"] = float("nan")

    info["incremental_log_weights"] = incremental_log_weights
    info["previous_log_weights"] = previous_log_weight_values
    info["log_weight_totals"] = total_log_weights
    info["log_weights"] = total_log_weights

    log_weight_tensor = total_log_weight_tensor
    if not torch.isfinite(log_weight_tensor).all():
        log_weight_tensor = torch.full_like(log_weight_tensor, float("nan"))

    base_temperature = max(float(config.temperature), 1e-6)
    effective_temperature = base_temperature
    normalized_weights = safe_normalize_log_weights(log_weight_tensor / effective_temperature)

    def _fallback_aggregate() -> Tuple[TensorState, Dict[str, Any]]:
        fallback_weights = _sample_count_weights(client_sample_counts)
        fallback_state = weighted_average_state_dicts(client_expert_states, fallback_weights.tolist())
        fallback_info = dict(info)
        fallback_info["weights"] = fallback_weights.tolist()
        fallback_info["pre_resample_weights"] = fallback_weights.tolist()
        fallback_info["log_weights"] = total_log_weights
        fallback_info["fallback_used"] = True
        fallback_info["ess"] = effective_sample_size(fallback_weights)
        fallback_info["normalized_ess"] = (
            fallback_info["ess"] / max(float(len(client_expert_states)), 1.0)
        )
        return _merge_with_reference_state(reference_state, fallback_state, device=device), fallback_info

    if normalized_weights.numel() == 0 or not torch.isfinite(normalized_weights).all():
        if config.fallback_to_fedavg_on_nan:
            return _fallback_aggregate()
        raise ValueError("Failed to normalize expert log weights")

    ess = effective_sample_size(normalized_weights)
    normalized_ess = ess / max(float(len(client_expert_states)), 1.0)
    if normalized_ess < float(config.degeneration_ess_threshold):
        info["degeneration"] = True
        effective_temperature = base_temperature * max(float(config.degeneration_temperature_scale), 1.0)
        normalized_weights = safe_normalize_log_weights(log_weight_tensor / effective_temperature)
        if normalized_weights.numel() == 0 or not torch.isfinite(normalized_weights).all():
            if config.fallback_to_fedavg_on_nan:
                fallback_state, fallback_info = _fallback_aggregate()
                fallback_info["degeneration"] = True
                fallback_info["effective_temperature"] = effective_temperature
                return fallback_state, fallback_info
            raise ValueError("Failed to renormalize degenerate expert log weights")
        ess = effective_sample_size(normalized_weights)
        normalized_ess = ess / max(float(len(client_expert_states)), 1.0)
        if (
            normalized_ess < float(config.degeneration_ess_threshold)
            and config.fallback_to_fedavg_on_degenerate
        ):
            fallback_state, fallback_info = _fallback_aggregate()
            fallback_info["degeneration"] = True
            fallback_info["effective_temperature"] = effective_temperature
            return fallback_state, fallback_info

    info["pre_resample_weights"] = normalized_weights.tolist()
    info["ess"] = ess
    info["normalized_ess"] = normalized_ess
    info["effective_temperature"] = effective_temperature

    final_weights = normalized_weights.clone()
    aggregate_states: Sequence[TensorState] = client_expert_states

    if (
        config.use_resampling
        and len(client_expert_states) > 1
        and normalized_ess < float(config.ess_threshold)
    ):
        selected_indices = stratified_resampling(normalized_weights)
        aggregate_states = [
            _maybe_add_resampling_noise(
                client_expert_states[int(index)],
                noise_std=float(config.resampling_noise_std),
                device=device,
            )
            for index in selected_indices.tolist()
        ]
        counts = torch.bincount(selected_indices, minlength=len(client_expert_states)).to(dtype=torch.float64)
        final_weights = counts / counts.sum().clamp_min(1.0)
        info["resampled"] = True
        info["selected_indices"] = selected_indices.tolist()

    info["weights"] = final_weights.tolist()

    if not torch.isfinite(final_weights).all():
        if config.fallback_to_fedavg_on_nan:
            return _fallback_aggregate()
        raise ValueError("Final expert aggregation weights are not finite")

    if info["resampled"]:
        uniform_weights = [1.0 / len(aggregate_states)] * len(aggregate_states)
        aggregated_state = weighted_average_state_dicts(aggregate_states, uniform_weights)
    else:
        aggregated_state = weighted_average_state_dicts(client_expert_states, final_weights.tolist())

    return _merge_with_reference_state(reference_state, aggregated_state, device=device), info


def split_model_state_by_expert(
    state_dict: TensorState,
    num_experts: Optional[int] = None,
) -> Tuple[TensorState, Dict[str, Dict[str, Any]]]:
    base_state: TensorState = collections.OrderedDict()
    expert_states: Dict[str, Dict[str, Any]] = collections.OrderedDict()

    for name, tensor in state_dict.items():
        expert_id = get_expert_id_from_name(name)
        expert_group = _get_expert_group_from_name(name)
        if (
            expert_id is not None
            and expert_group is not None
            and (num_experts is None or expert_id < int(num_experts))
        ):
            if expert_group not in expert_states:
                expert_states[expert_group] = {
                    "expert_id": int(expert_id),
                    "state_dict": collections.OrderedDict(),
                }
            expert_states[expert_group]["state_dict"][name] = tensor
        else:
            base_state[name] = tensor

    return base_state, expert_states


def merge_expert_states_into_model(
    base_state_dict: TensorState,
    expert_states: Dict[str, Any],
) -> TensorState:
    merged: TensorState = collections.OrderedDict(base_state_dict)
    for value in expert_states.values():
        if isinstance(value, dict) and "state_dict" in value:
            merged.update(value["state_dict"])
        else:
            merged.update(value)
    return merged


class VSMCExpertAggregator:
    def __init__(self, config: VSMCAggregatorConfig, logger: Optional[Any] = None):
        self.config = config
        self.logger = logger
        self.log_weight_memory: Dict[Tuple[str, int, int], float] = {}
        self.logZ_memory: Dict[Tuple[str, int], float] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        supported_variants = {"baseline"}
        if self.config.variant not in supported_variants:
            raise ValueError(
                f"Unsupported vsmc_variant={self.config.variant!r}. "
                f"Supported variants: {sorted(supported_variants)}"
            )

    def _combine_log_weight(self, prev_log_weight: float, inc_log_weight: float) -> float:
        if self.config.variant == "baseline":
            return float(self.config.history_decay) * prev_log_weight + inc_log_weight
        raise ValueError(f"Unsupported vsmc_variant={self.config.variant!r}")

    @staticmethod
    def _client_identifier(metrics: Dict[str, Any], fallback_idx: int) -> int:
        if metrics is not None and "client_id" in metrics:
            return int(metrics["client_id"])
        return int(fallback_idx)

    def _memory_key(self, expert_scope: str, expert_id: int, client_id: int) -> Tuple[str, int, int]:
        return (str(expert_scope), int(expert_id), int(client_id))

    def _logz_key(self, expert_scope: str, expert_id: int) -> Tuple[str, int]:
        return (str(expert_scope), int(expert_id))

    def _decay_missing_client_memories(
        self,
        expert_scope: str,
        expert_id: int,
        active_client_ids: Sequence[int],
    ) -> None:
        active_set = {int(client_id) for client_id in active_client_ids}
        for key in list(self.log_weight_memory.keys()):
            scope, key_expert_id, client_id = key
            if scope != str(expert_scope) or int(key_expert_id) != int(expert_id):
                continue
            if int(client_id) in active_set:
                continue
            value = float(self.log_weight_memory[key])
            if not math.isfinite(value):
                value = 0.0
            self.log_weight_memory[key] = float(self.config.missing_decay) * value

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, List[Dict[str, Any]]]:
        if len(client_updates) == 0:
            raise ValueError("client_updates must not be empty")

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

        ordered_group_names = _ordered_union_keys(
            [global_expert_groups] + client_expert_groups + [fedavg_expert_groups]
        )

        merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        round_infos: List[Dict[str, Any]] = []

        for expert_group_name in ordered_group_names:
            template_group = global_expert_groups.get(expert_group_name) or fedavg_expert_groups.get(expert_group_name)
            if template_group is None:
                continue

            expert_id = int(template_group["expert_id"])
            reference_expert_state = collections.OrderedDict(template_group["state_dict"])

            client_expert_states: List[TensorState] = []
            client_sample_counts: List[int] = []
            client_metrics: List[Dict[str, Any]] = []
            client_ids: List[int] = []
            previous_log_weights: List[float] = []

            for fallback_idx, ((state_dict, num_samples, metrics), client_group_map) in enumerate(zip(client_updates, client_expert_groups)):
                del state_dict
                if expert_group_name not in client_group_map:
                    continue
                metrics = metrics or {}
                client_id = self._client_identifier(metrics, fallback_idx)
                client_expert_states.append(client_group_map[expert_group_name]["state_dict"])
                client_sample_counts.append(int(num_samples))
                client_metrics.append(metrics)
                client_ids.append(client_id)
                previous_log_weights.append(
                    float(self.log_weight_memory.get(self._memory_key(expert_group_name, expert_id, client_id), 0.0))
                )

            aggregated_expert_state, info = aggregate_expert_particles(
                global_expert_state=reference_expert_state,
                client_expert_states=client_expert_states,
                client_sample_counts=client_sample_counts,
                client_metrics=client_metrics,
                expert_id=expert_id,
                num_experts=num_experts,
                config=self.config,
                previous_log_weights=previous_log_weights,
                client_ids=client_ids,
                device=device,
            )

            for client_id, total_log_weight in zip(client_ids, info.get("log_weight_totals", [])):
                self.log_weight_memory[self._memory_key(expert_group_name, expert_id, client_id)] = float(total_log_weight)
            self._decay_missing_client_memories(expert_group_name, expert_id, client_ids)
            logz_key = self._logz_key(expert_group_name, expert_id)
            previous_logz = float(self.logZ_memory.get(logz_key, 0.0))
            logz_increment = float(info.get("logZ_increment", 0.0))
            if math.isfinite(logz_increment):
                self.logZ_memory[logz_key] = previous_logz + logz_increment
            else:
                self.logZ_memory[logz_key] = previous_logz
            info["logZ_total"] = float(self.logZ_memory.get(logz_key, previous_logz))
            info["active_client_ids"] = list(client_ids)

            merged_expert_groups[expert_group_name] = {
                "expert_id": expert_id,
                "state_dict": aggregated_expert_state,
            }
            info["expert_id"] = expert_id
            info["expert_scope"] = expert_group_name
            round_infos.append(info)

        merged_state = merge_expert_states_into_model(base_state, merged_expert_groups)
        merged_state = _merge_with_reference_state(global_state, merged_state, device=device)
        return merged_state, round_infos
