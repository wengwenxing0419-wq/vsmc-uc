"""
This module adapts the VSMC particle reweighting formula to federated Switch Transformer expert aggregation.

For expert e at federated round r:

    x_r^i = theta_{r,i,e}

is the expert parameter uploaded by client i.

The VSMC-inspired importance weight is:

    log_w_{r,i,e}
      = log f(theta_{r,i,e} | theta_{r-1,a_i,e})
        + log g(y_{r,e} | theta_{r,i,e})
        - log r(theta_{r,i,e} | theta_{r-1,a_i,e}; lambda)

where:
- f is a Gaussian target transition around the ancestor expert;
- g is an observation likelihood from expert loss, route count, usage, and sample count;
- r is a Gaussian proposal density approximated by the FedAvg local-training proposal;
- alpha is obtained by softmax(log_w / temperature);
- the global expert is the weighted average of client expert particles.

This is not a direct reuse of the original LGSS VSMC example.
It is a task-specific VSMC-style aggregation rule for FL + MoE.

Concept mapping used in the code:
- VSMC time step t -> federated communication round r
- particle i -> client i
- x_t^i -> client i's uploaded expert-e parameters theta_{r,i,e}
- x_{t-1}^{a_t^i} -> the previous-round ancestor expert state used for client i
- y_t -> round-r expert observations: expert_loss, usage, route_count, sample_count, val_loss
- f -> target transition density
- g -> observation likelihood
- r -> proposal density from the FedAvg-style local-training proposal
- w_t^i -> client-i importance weight for expert e
- ESS -> diagnostic for whether aggregation is dominated by a small subset of clients
- logZ -> per-expert VSMC-style marginal likelihood monitor
"""

from __future__ import annotations

import collections
import copy
import csv
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


TensorState = Dict[str, torch.Tensor]
_EPS = 1e-12
_EXPERT_PATTERNS = [
    re.compile(r"experts\.(\d+)\."),
    re.compile(r"expert\.(\d+)\."),
    re.compile(r"moe\.experts\.(\d+)\."),
    re.compile(r"block_sparse_moe\.experts\.(\d+)\."),
    re.compile(r"layers\.\d+\.experts\.(\d+)\."),
    re.compile(r"encoder\.block\.\d+.*experts\.(\d+)\."),
]


@dataclass
class VSMCExpertAggregationConfig:
    enabled: bool = True
    vsmc_mode: str = "baseline"
    temperature: float = 1.0
    sigma_f: float = 1.0
    sigma_r: float = 1.0
    obs_temp: float = 1.0
    loss_beta: float = 1.0
    route_beta: float = 0.2
    usage_beta: float = 0.2
    sample_beta: float = 0.1
    server_val_beta: float = 1.0
    aggregate_delta: bool = True
    expert_server_lr: float = 0.5
    use_server_validation_g: bool = True
    server_val_batches: int = 1
    server_val_batch_size: int = 128
    server_val_every_round: int = 1
    server_g_beta: float = 5.0
    server_g_clip: float = 5.0
    decouple_mu: float = 0.2
    base_alpha_source: str = "r"
    lambda_r: float = 0.3
    r_bonus_clip: float = 3.0
    lambda_g: float = 1.0
    use_local_g_if_no_server_val: bool = True
    use_ema_f_smoothing: bool = True
    expert_ema_rho: float = 0.2
    mix_with_fedavg: float = 0.1
    alpha_max: float = 0.7
    normalized_ess_min: float = 0.2
    fallback_to_fedavg_on_bad_alpha: bool = True
    use_router_usage_calibration: bool = False
    router_bias_lr: float = 0.01
    target_usage: Optional[float] = None
    replace_reward_beta: float = 0.35
    delta_reward_beta: float = 0.30
    stability_beta: float = 0.10
    novelty_beta: float = 0.10
    credit_beta: float = 0.15
    signal_temperature: float = 1.0
    reward_ema_rho: float = 0.7
    credit_momentum: float = 0.8
    uniform_mix: float = 0.05
    novelty_clip: float = 2.0
    stability_clip: float = 2.0
    signal_reward_clip: float = 5.0
    clamp_log_weight: float = 30.0
    eps: float = 1e-12
    normalize_loss: str = "zscore"
    normalize_distance: bool = True
    use_resampling: bool = True
    ess_threshold: float = 0.35
    resampling_mode: str = "stratified"
    keep_particle_bank: bool = True
    missing_client_decay: float = 0.9
    fallback_to_fedavg_on_nan: bool = True
    fallback_on_low_ess: bool = False
    low_ess_threshold: float = 0.15
    fallback_when_all_route_zero: bool = True
    debug: bool = False
    log_dir: str = "logs"
    log_csv_name: str = "vsmc_aggregation_round_metrics.csv"


@dataclass
class VSMCExpertState:
    prev_particle_bank: Dict[Any, List[TensorState]] = field(default_factory=dict)
    prev_weights: Dict[Any, torch.Tensor] = field(default_factory=dict)
    logZ_history: Dict[Any, float] = field(default_factory=dict)
    reward_replace_ema: Dict[Any, Dict[Any, float]] = field(default_factory=dict)
    reward_delta_ema: Dict[Any, Dict[Any, float]] = field(default_factory=dict)
    credit_memory: Dict[Any, Dict[Any, float]] = field(default_factory=dict)
    prev_selected_signature: Dict[Any, torch.Tensor] = field(default_factory=dict)
    round_idx: int = 0


# Backward-compatible alias used by older modules.
VSMCAggregatorConfig = VSMCExpertAggregationConfig


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


def clone_detached_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = collections.OrderedDict()
    for key, value in state.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _clone_detached_tensor(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    cloned = tensor.detach().clone()
    if device is not None:
        cloned = cloned.to(device)
    return cloned


def _ordered_union_keys(state_dicts: Iterable[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for state_dict in state_dicts:
        for key in state_dict.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def flatten_float_state_dict(state: Dict[str, Any]) -> torch.Tensor:
    flat_tensors: List[torch.Tensor] = []
    for key in sorted(state.keys()):
        value = state[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            flat_tensors.append(value.detach().float().reshape(-1))
    if not flat_tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_tensors, dim=0)


def flatten_common_float_state_dict(a: Dict[str, Any], b: Dict[str, Any]) -> torch.Tensor:
    flat_tensors: List[torch.Tensor] = []
    for key in sorted(set(a.keys()) & set(b.keys())):
        value_a = a[key]
        value_b = b[key]
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            continue
        if not (torch.is_floating_point(value_a) and torch.is_floating_point(value_b)):
            continue
        if value_a.shape != value_b.shape:
            continue
        flat_tensors.append(value_a.detach().float().reshape(-1))
        flat_tensors.append(value_b.detach().float().reshape(-1))
    if not flat_tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_tensors, dim=0)


def flatten_common_float_state_dict_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_a: List[torch.Tensor] = []
    flat_b: List[torch.Tensor] = []
    for key in sorted(set(a.keys()) & set(b.keys())):
        value_a = a[key]
        value_b = b[key]
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            continue
        if not (torch.is_floating_point(value_a) and torch.is_floating_point(value_b)):
            continue
        if value_a.shape != value_b.shape:
            continue
        flat_a.append(value_a.detach().float().reshape(-1))
        flat_b.append(value_b.detach().float().reshape(-1))
    if not flat_a:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty
    return torch.cat(flat_a, dim=0), torch.cat(flat_b, dim=0)


def gaussian_log_prob_diag(
    x: torch.Tensor,
    mean: torch.Tensor,
    sigma: float,
    normalize_dim: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    x = x.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    mean = mean.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if x.numel() != mean.numel():
        raise ValueError("x and mean must have the same flattened dimension")
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    sigma_sq = max(float(sigma), eps) ** 2
    sq = (x - mean) ** 2
    value = sq.mean() if normalize_dim else sq.sum()
    return -value / (2.0 * sigma_sq)


def safe_logsumexp_normalize(log_w: torch.Tensor, temperature: float, eps: float) -> torch.Tensor:
    log_w = torch.as_tensor(log_w, dtype=torch.float64)
    if log_w.numel() == 0:
        return log_w
    if not torch.isfinite(log_w).all():
        raise ValueError("log_w contains NaN/Inf before normalization")
    scaled = log_w / max(float(temperature), eps)
    m = torch.max(scaled)
    weights = torch.exp(scaled - m)
    denom = weights.sum() + eps
    normalized = weights / denom
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized weights contain NaN/Inf")
    return normalized


def safe_normalize_log_weights(log_w: Sequence[float], eps: float = 1e-12) -> torch.Tensor:
    return safe_logsumexp_normalize(torch.as_tensor(log_w, dtype=torch.float64), temperature=1.0, eps=eps)


def effective_sample_size(weights: Sequence[float], eps: float = 1e-12) -> float:
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if weight_tensor.numel() == 0 or not torch.isfinite(weight_tensor).all():
        return 0.0
    ess = 1.0 / torch.sum(weight_tensor ** 2).clamp_min(eps).item()
    return float(ess)


def stratified_resampling(
    weights: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).detach()
    if weight_tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(_EPS)
    n = weight_tensor.numel()
    positions = (torch.arange(n, dtype=torch.float64) + torch.rand(n, generator=generator, dtype=torch.float64)) / n
    cumulative_sum = torch.cumsum(weight_tensor, dim=0)
    return torch.searchsorted(cumulative_sum, positions, right=False).clamp(max=n - 1).long()


def _sample_count_weights(sample_counts: Sequence[int], eps: float = 1e-12) -> torch.Tensor:
    counts = torch.as_tensor(sample_counts, dtype=torch.float64).clamp_min(0.0)
    if counts.numel() == 0:
        return counts
    total = counts.sum()
    if total.item() <= eps or not torch.isfinite(total):
        return torch.full_like(counts, 1.0 / counts.numel())
    return counts / total


def weighted_average_state_dicts(
    states: List[TensorState],
    weights: torch.Tensor,
    fallback_state: Optional[TensorState] = None,
) -> TensorState:
    if not states:
        return clone_detached_state_dict(fallback_state or {})
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if weight_tensor.numel() != len(states):
        raise ValueError("states and weights must have the same length")
    aggregated: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys(states):
        available: List[Tuple[int, torch.Tensor]] = []
        for idx, state in enumerate(states):
            if key in state and torch.is_tensor(state[key]):
                available.append((idx, state[key]))
        if not available:
            if fallback_state is not None and key in fallback_state:
                aggregated[key] = _clone_detached_tensor(fallback_state[key])
            continue

        first_tensor = available[0][1].detach()
        if not torch.is_floating_point(first_tensor):
            if fallback_state is not None and key in fallback_state:
                aggregated[key] = _clone_detached_tensor(fallback_state[key])
            else:
                aggregated[key] = first_tensor.clone()
            continue

        local_weights = torch.as_tensor([weight_tensor[idx] for idx, _ in available], dtype=torch.float64)
        denom = local_weights.sum()
        if denom.item() <= _EPS or not torch.isfinite(denom):
            local_weights = torch.full_like(local_weights, 1.0 / local_weights.numel())
        else:
            local_weights = local_weights / denom

        acc = torch.zeros_like(first_tensor.detach())
        for local_weight, (_, tensor) in zip(local_weights, available):
            acc = acc + tensor.detach().to(dtype=acc.dtype, device=acc.device) * local_weight.to(
                dtype=acc.dtype,
                device=acc.device,
            )
        aggregated[key] = acc.detach()
    if fallback_state is not None:
        for key, value in fallback_state.items():
            if key not in aggregated:
                aggregated[key] = _clone_detached_tensor(value) if torch.is_tensor(value) else copy.deepcopy(value)
    return aggregated


def sample_count_fedavg_state_dicts(
    states: List[TensorState],
    sample_counts: List[int],
    fallback_state: Optional[TensorState] = None,
) -> TensorState:
    return weighted_average_state_dicts(states, _sample_count_weights(sample_counts), fallback_state=fallback_state)


def compute_state_delta(
    client_expert_state: TensorState,
    previous_global_expert_state: TensorState,
) -> TensorState:
    delta_state: TensorState = collections.OrderedDict()
    for key in sorted(set(client_expert_state.keys()) & set(previous_global_expert_state.keys())):
        client_value = client_expert_state[key]
        previous_value = previous_global_expert_state[key]
        if not (torch.is_tensor(client_value) and torch.is_tensor(previous_value)):
            continue
        if not (torch.is_floating_point(client_value) and torch.is_floating_point(previous_value)):
            continue
        if client_value.shape != previous_value.shape:
            continue
        previous_aligned = previous_value.detach().to(device=client_value.device, dtype=client_value.dtype)
        delta_state[key] = client_value.detach() - previous_aligned
    return delta_state


def apply_delta_to_state(
    previous_global_expert_state: TensorState,
    delta_state: TensorState,
    server_lr: float,
) -> TensorState:
    updated_state = clone_detached_state_dict(previous_global_expert_state)
    lr = float(server_lr)
    for key, delta_value in delta_state.items():
        if key not in previous_global_expert_state:
            continue
        previous_value = previous_global_expert_state[key]
        if not (torch.is_tensor(previous_value) and torch.is_tensor(delta_value)):
            continue
        delta_aligned = delta_value.detach().to(dtype=previous_value.dtype, device=previous_value.device)
        updated_state[key] = previous_value.detach() + lr * delta_aligned
    return updated_state


def _state_l2_norm(state: TensorState) -> float:
    flat = flatten_float_state_dict(state)
    if flat.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(flat, ord=2).item())


def _normalize_probability(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if weights.numel() == 0:
        return weights
    weights = torch.clamp(weights, min=0.0)
    total = weights.sum()
    if not torch.isfinite(total) or total.item() <= eps:
        return torch.full_like(weights, 1.0 / weights.numel())
    return weights / total


def _clamp_and_normalize_alpha(weights: torch.Tensor, alpha_max: float, eps: float) -> torch.Tensor:
    normalized = _normalize_probability(weights, eps)
    if normalized.numel() == 0:
        return normalized
    clipped = torch.clamp(normalized, max=float(alpha_max))
    return _normalize_probability(clipped, eps)


def _ema_update(previous: Optional[float], current: float, rho: float) -> float:
    current = float(current)
    if previous is None or not math.isfinite(float(previous)):
        return current
    return float(rho) * float(previous) + (1.0 - float(rho)) * current


def _get_client_memory_value(memory: Optional[Dict[Any, float]], client_key: Any) -> float:
    if not memory:
        return 0.0
    value = memory.get(client_key, 0.0)
    return float(value) if math.isfinite(float(value)) else 0.0


def _normalize_score_tensor(values: Sequence[float], eps: float = 1e-12) -> torch.Tensor:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return tensor
    if tensor.numel() == 1:
        return torch.zeros_like(tensor)
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if not torch.isfinite(std) or std.item() <= eps:
        return torch.zeros_like(tensor)
    return (tensor - mean) / (std + eps)


def _safe_nanmean(values: Sequence[float], default: float = float("nan")) -> float:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return float(default)
    finite_mask = torch.isfinite(tensor)
    if not finite_mask.any():
        return float(default)
    return float(tensor[finite_mask].mean().item())


def _flatten_common_delta_pair(a: TensorState, b: TensorState) -> Tuple[torch.Tensor, torch.Tensor]:
    return flatten_common_float_state_dict_pair(a, b)


def _merge_with_reference_state(
    reference_state: TensorState,
    partial_state: TensorState,
    device: Optional[torch.device] = None,
) -> TensorState:
    merged: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys([reference_state, partial_state]):
        if key in partial_state:
            value = partial_state[key]
        else:
            value = reference_state[key]
        merged[key] = _clone_detached_tensor(value, device=device) if torch.is_tensor(value) else copy.deepcopy(value)
    return merged


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
    if isinstance(container, (list, tuple)) and 0 <= expert_id < len(container):
        value = container[expert_id]
        return float(value.item()) if torch.is_tensor(value) else float(value)
    return None


def _extract_metric(metrics: Dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
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


def _min_max_normalize(values: Sequence[float], eps: float = 1e-12) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return tensor
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    scale = max_value - min_value
    if scale.abs().item() <= eps:
        return torch.zeros_like(tensor)
    return (tensor - min_value) / (scale + eps)


def is_expert_param(name: str) -> bool:
    lower_name = name.lower()
    if "router" in lower_name or "gate" in lower_name:
        return False
    return get_expert_id_from_name(name) is not None


def get_expert_id_from_name(name: str) -> Optional[int]:
    for pattern in _EXPERT_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            return int(match.group(1))
    return None


def _get_expert_scope_from_name(name: str) -> Optional[str]:
    if not is_expert_param(name):
        return None
    expert_id = get_expert_id_from_name(name)
    if expert_id is None:
        return None
    for marker in (
        f"experts.{expert_id}.",
        f"expert.{expert_id}.",
        f"moe.experts.{expert_id}.",
        f"block_sparse_moe.experts.{expert_id}.",
    ):
        if marker in name:
            return name.split(marker)[0] + marker[:-1]
    return f"expert_{expert_id}"


def split_state_dict_by_expert(
    state_dict: TensorState,
    num_experts: Optional[int] = None,
) -> Tuple[TensorState, Dict[str, Dict[str, Any]]]:
    base_state: TensorState = collections.OrderedDict()
    expert_states: Dict[str, Dict[str, Any]] = collections.OrderedDict()

    for name, tensor in state_dict.items():
        expert_id = get_expert_id_from_name(name)
        expert_scope = _get_expert_scope_from_name(name)
        lower_name = name.lower()
        if expert_id is not None and expert_scope is not None and (num_experts is None or expert_id < int(num_experts)):
            expert_states.setdefault(
                expert_scope,
                {"expert_id": int(expert_id), "expert_scope": expert_scope, "state_dict": collections.OrderedDict()},
            )
            expert_states[expert_scope]["state_dict"][name] = tensor
        else:
            base_state[name] = tensor
            if expert_id is None and any(token in lower_name for token in ("expert", "experts", "moe", "router", "gate")):
                warnings.warn(f"Unrecognized expert-like parameter name kept on FedAvg path: {name}")
    return base_state, expert_states


def split_model_state_by_expert(
    state_dict: TensorState,
    num_experts: Optional[int] = None,
) -> Tuple[TensorState, Dict[str, Dict[str, Any]]]:
    return split_state_dict_by_expert(state_dict, num_experts=num_experts)


def merge_expert_states_into_global_state(base_state_dict: TensorState, expert_states: Dict[str, Any]) -> TensorState:
    merged = collections.OrderedDict(base_state_dict)
    for value in expert_states.values():
        state_dict = value["state_dict"] if isinstance(value, dict) and "state_dict" in value else value
        for key, tensor in state_dict.items():
            merged[key] = tensor
    return merged


def merge_expert_states_into_model(base_state_dict: TensorState, expert_states: Dict[str, Any]) -> TensorState:
    return merge_expert_states_into_global_state(base_state_dict, expert_states)


def _get_block_scope_from_expert_scope(expert_scope: str) -> str:
    if '.experts.' in expert_scope:
        return expert_scope.split('.experts.')[0]
    if '.expert.' in expert_scope:
        return expert_scope.split('.expert.')[0]
    return expert_scope


def group_expert_groups_by_block(expert_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    for expert_scope, group in expert_groups.items():
        block_scope = _get_block_scope_from_expert_scope(expert_scope)
        grouped.setdefault(
            block_scope,
            {
                'block_scope': block_scope,
                'expert_scopes': [],
                'expert_ids': [],
                'state_dict': collections.OrderedDict(),
            },
        )
        grouped[block_scope]['expert_scopes'].append(expert_scope)
        grouped[block_scope]['expert_ids'].append(int(group.get('expert_id', 0)))
        for key, value in group['state_dict'].items():
            grouped[block_scope]['state_dict'][key] = value
    return grouped


def _normalize_values(values: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    if values.numel() == 0:
        return values
    if mode == "none":
        return values
    if mode == "minmax":
        return _min_max_normalize(values.tolist(), eps=eps)
    mean = values.mean()
    std = values.std(unbiased=False)
    return (values - mean) / (std + eps)


def compute_log_f(
    current_particle_state: TensorState,
    ancestor_particle_state: TensorState,
    config: VSMCExpertAggregationConfig,
) -> torch.Tensor:
    x, mean = flatten_common_float_state_dict_pair(current_particle_state, ancestor_particle_state)
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return gaussian_log_prob_diag(
        x=x,
        mean=mean,
        sigma=config.sigma_f,
        normalize_dim=config.normalize_distance,
        eps=config.eps,
    )


def compute_proposal_mean_state(
    previous_global_expert_state: TensorState,
    current_client_expert_states: List[TensorState],
    sample_counts: List[int],
) -> TensorState:
    # This treats the client-side local-training proposal as a FedAvg proposal:
    # proposal_mean = previous_global_expert + FedAvg_delta
    # which is equivalent to sample-count FedAvg(current_client_expert_states).
    return sample_count_fedavg_state_dicts(
        current_client_expert_states,
        sample_counts,
        fallback_state=previous_global_expert_state,
    )


def compute_log_r(
    current_particle_state: TensorState,
    proposal_mean_state: TensorState,
    config: VSMCExpertAggregationConfig,
) -> torch.Tensor:
    x, mean = flatten_common_float_state_dict_pair(current_particle_state, proposal_mean_state)
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return gaussian_log_prob_diag(
        x=x,
        mean=mean,
        sigma=config.sigma_r,
        normalize_dim=config.normalize_distance,
        eps=config.eps,
    )


def compute_log_g_values(
    client_metrics: List[dict],
    expert_id: int,
    num_experts: int,
    sample_counts: List[int],
    config: VSMCExpertAggregationConfig,
    expert_scope: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    losses: List[float] = []
    usages: List[float] = []
    route_counts: List[float] = []
    sample_count_values: List[int] = [int(v) for v in sample_counts]
    server_val_losses: List[float] = []

    target_usage = 1.0 / max(int(num_experts), 1)
    raw_loss_candidates: List[float] = []
    for metrics in client_metrics:
        metrics = metrics or {}
        loss_value = _extract_expert_metric(
            metrics,
            ("expert_losses", "expert_loss", "per_expert_loss"),
            expert_id,
            default=None,
        )
        if loss_value is None:
            fallback = _extract_metric(metrics, ("val_loss", "train_loss"), default=None)
            if fallback is not None:
                raw_loss_candidates.append(float(fallback))
        else:
            raw_loss_candidates.append(float(loss_value))
    median_loss = float(torch.median(torch.as_tensor(raw_loss_candidates, dtype=torch.float64)).item()) if raw_loss_candidates else 0.0

    for sample_count, metrics in zip(sample_count_values, client_metrics):
        metrics = metrics or {}
        loss_value = _extract_expert_metric(
            metrics,
            ("expert_losses", "expert_loss", "per_expert_loss"),
            expert_id,
            default=None,
        )
        if loss_value is None:
            loss_value = _extract_metric(metrics, ("val_loss", "train_loss"), default=median_loss)
        if loss_value is None:
            loss_value = median_loss

        route_count = _extract_expert_metric(
            metrics,
            ("route_counts", "expert_route_counts", "per_expert_route_count"),
            expert_id,
            default=0.0,
        )
        if route_count is None:
            route_count = 0.0

        usage = _extract_expert_metric(
            metrics,
            ("expert_usage", "per_expert_usage"),
            expert_id,
            default=None,
        )
        if usage is None:
            total_route_count = _extract_metric(metrics, ("total_route_count", "num_tokens"), default=0.0) or 0.0
            usage = float(route_count) / max(float(total_route_count), config.eps) if total_route_count > 0.0 else target_usage

        losses.append(float(loss_value))
        usages.append(float(usage))
        route_counts.append(float(route_count))
        server_val_loss = None
        if expert_scope is not None:
            server_val_map = metrics.get("server_val_expert_losses")
            if isinstance(server_val_map, dict) and expert_scope in server_val_map:
                raw_value = server_val_map[expert_scope]
                server_val_loss = float(raw_value.item()) if torch.is_tensor(raw_value) else float(raw_value)
        if server_val_loss is None:
            server_val_loss = float(loss_value)
        server_val_losses.append(float(server_val_loss))

    loss_tensor = torch.as_tensor(losses, dtype=torch.float64)
    normalized_loss = _normalize_values(loss_tensor, config.normalize_loss, config.eps)
    server_val_tensor = torch.as_tensor(server_val_losses, dtype=torch.float64)
    normalized_server_val_loss = _normalize_values(server_val_tensor, config.normalize_loss, config.eps)
    usage_tensor = torch.as_tensor(usages, dtype=torch.float64)
    route_tensor = torch.as_tensor(route_counts, dtype=torch.float64)
    sample_tensor = torch.as_tensor(sample_count_values, dtype=torch.float64)

    log_g = (
        -float(config.loss_beta) * normalized_loss / max(float(config.obs_temp), config.eps)
        + float(config.route_beta) * torch.log1p(route_tensor)
        + float(config.sample_beta) * torch.log1p(sample_tensor)
        - float(config.usage_beta) * torch.abs(usage_tensor - target_usage)
        - float(config.server_val_beta) * normalized_server_val_loss
    )

    return log_g, {
        "raw_loss": losses,
        "normalized_loss": normalized_loss.tolist(),
        "server_val_loss": server_val_losses,
        "normalized_server_val_loss": normalized_server_val_loss.tolist(),
        "usage": usages,
        "route_count": route_counts,
        "sample_count": sample_count_values,
        "all_route_zero": bool(route_tensor.numel() > 0 and torch.all(route_tensor == 0).item()),
        "target_usage": target_usage,
    }


def _evaluate_model_loss_on_batches(
    model: torch.nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: torch.nn.Module,
    device: torch.device,
    aux_loss_weight: float = 0.0,
) -> float:
    running_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in batches:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, router_stats = model(inputs)
            loss = loss_fn(outputs, labels)
            if isinstance(router_stats, dict):
                aux_loss = router_stats.get("aux_loss")
                if aux_loss is not None:
                    loss = loss + float(aux_loss_weight) * aux_loss
            running_loss += float(loss.item()) * int(inputs.size(0))
            total_samples += int(inputs.size(0))
    if total_samples == 0:
        return float("inf")
    return running_loss / total_samples


def _evaluate_model_loss_and_signature_on_batches(
    model: torch.nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: torch.nn.Module,
    device: torch.device,
    aux_loss_weight: float = 0.0,
) -> Tuple[float, torch.Tensor]:
    running_loss = 0.0
    total_samples = 0
    signature_sum: Optional[torch.Tensor] = None
    model.eval()
    with torch.no_grad():
        for inputs, labels in batches:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, router_stats = model(inputs)
            loss = loss_fn(outputs, labels)
            if isinstance(router_stats, dict):
                aux_loss = router_stats.get("aux_loss")
                if aux_loss is not None:
                    loss = loss + float(aux_loss_weight) * aux_loss
            batch_size = int(inputs.size(0))
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            batch_signature = outputs.detach().float().mean(dim=0).reshape(-1).cpu()
            signature_sum = batch_signature if signature_sum is None else signature_sum + batch_signature
    if total_samples == 0:
        return float("inf"), torch.empty(0, dtype=torch.float32)
    assert signature_sum is not None
    return running_loss / total_samples, signature_sum / max(len(batches), 1)


def compute_server_validation_g_for_expert(
    global_model: torch.nn.Module,
    previous_global_state_dict: TensorState,
    expert_id: int,
    client_expert_states: List[TensorState],
    val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    loss_fn: torch.nn.Module,
    device: torch.device,
    config: VSMCExpertAggregationConfig,
    expert_scope: Optional[str] = None,
    aux_loss_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    del expert_id
    if val_loader is None:
        empty = torch.zeros(len(client_expert_states), dtype=torch.float64)
        return empty, {
            "base_loss": float("nan"),
            "replaced_losses": [float("nan")] * len(client_expert_states),
            "rewards": [0.0] * len(client_expert_states),
            "used_server_validation": False,
        }

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    max_batches = max(int(config.server_val_batches), 1)
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batches.append(batch)
    if not batches:
        empty = torch.zeros(len(client_expert_states), dtype=torch.float64)
        return empty, {
            "base_loss": float("nan"),
            "replaced_losses": [float("nan")] * len(client_expert_states),
            "rewards": [0.0] * len(client_expert_states),
            "used_server_validation": False,
        }

    original_state = clone_detached_state_dict(global_model.state_dict())
    base_loss = float("inf")
    replaced_losses: List[float] = []
    rewards: List[float] = []
    try:
        with torch.no_grad():
            global_model.load_state_dict(previous_global_state_dict)
            base_loss = _evaluate_model_loss_on_batches(
                global_model,
                batches,
                loss_fn=loss_fn,
                device=device,
                aux_loss_weight=aux_loss_weight,
            )
            for client_state in client_expert_states:
                candidate_state = clone_detached_state_dict(previous_global_state_dict)
                for key, value in client_state.items():
                    candidate_state[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                global_model.load_state_dict(candidate_state)
                replaced_loss = _evaluate_model_loss_on_batches(
                    global_model,
                    batches,
                    loss_fn=loss_fn,
                    device=device,
                    aux_loss_weight=aux_loss_weight,
                )
                reward = base_loss - replaced_loss
                replaced_losses.append(float(replaced_loss))
                rewards.append(float(reward))
    finally:
        global_model.load_state_dict(original_state)

    reward_tensor = torch.as_tensor(rewards, dtype=torch.float64)
    log_g_server = torch.clamp(
        float(config.server_g_beta) * reward_tensor,
        min=-float(config.server_g_clip),
        max=float(config.server_g_clip),
    )
    return log_g_server, {
        "base_loss": float(base_loss),
        "replaced_losses": replaced_losses,
        "rewards": rewards,
        "used_server_validation": True,
        "expert_scope": expert_scope,
    }


def compute_server_validation_strong_signals_for_expert(
    global_model: torch.nn.Module,
    previous_global_state_dict: TensorState,
    previous_global_expert_state: TensorState,
    expert_id: int,
    client_expert_states: List[TensorState],
    val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    loss_fn: torch.nn.Module,
    device: torch.device,
    config: VSMCExpertAggregationConfig,
    expert_scope: Optional[str] = None,
    aux_loss_weight: float = 0.0,
    sample_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    del expert_id
    num_clients = len(client_expert_states)
    empty_signature = torch.empty(0, dtype=torch.float32)
    if val_loader is None:
        return {
            "base_loss": float("nan"),
            "base_signature": empty_signature,
            "replaced_losses": [float("nan")] * num_clients,
            "replacement_rewards": [0.0] * num_clients,
            "delta_rewards": [0.0] * num_clients,
            "best_delta_steps": [0.0] * num_clients,
            "candidate_signatures": [empty_signature.clone() for _ in range(num_clients)],
            "used_server_validation": False,
            "expert_scope": expert_scope,
        }

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    max_batches = max(int(config.server_val_batches), 1)
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batches.append(batch)
    if not batches:
        return {
            "base_loss": float("nan"),
            "base_signature": empty_signature,
            "replaced_losses": [float("nan")] * num_clients,
            "replacement_rewards": [0.0] * num_clients,
            "delta_rewards": [0.0] * num_clients,
            "best_delta_steps": [0.0] * num_clients,
            "candidate_signatures": [empty_signature.clone() for _ in range(num_clients)],
            "used_server_validation": False,
            "expert_scope": expert_scope,
        }

    original_state = clone_detached_state_dict(global_model.state_dict())
    replaced_losses: List[float] = []
    replacement_rewards: List[float] = []
    delta_rewards: List[float] = []
    best_delta_steps: List[float] = []
    candidate_signatures: List[torch.Tensor] = []
    delta_trial_steps = (0.25, 0.5, 1.0)
    coalition_topk = min(3, num_clients)
    delta_states = [
        compute_state_delta(client_state, previous_global_expert_state)
        for client_state in client_expert_states
    ]

    def _inject_expert_state(target_state: TensorState, expert_state: TensorState) -> TensorState:
        merged = clone_detached_state_dict(target_state)
        for key, value in expert_state.items():
            merged[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        return merged

    def _postprocess_candidate_expert(expert_state: TensorState) -> TensorState:
        if config.use_ema_f_smoothing:
            return weighted_average_state_dicts(
                [expert_state, previous_global_expert_state],
                torch.as_tensor([1.0 - float(config.expert_ema_rho), float(config.expert_ema_rho)], dtype=torch.float64),
                fallback_state=previous_global_expert_state,
            )
        return expert_state

    try:
        with torch.no_grad():
            global_model.load_state_dict(previous_global_state_dict)
            base_loss, base_signature = _evaluate_model_loss_and_signature_on_batches(
                global_model,
                batches,
                loss_fn=loss_fn,
                device=device,
                aux_loss_weight=aux_loss_weight,
            )

            candidate_global_rewards: List[float] = []
            candidate_global_losses: List[float] = []
            for client_state, delta_state in zip(client_expert_states, delta_states):
                candidate_expert = apply_delta_to_state(previous_global_expert_state, delta_state, config.expert_server_lr)
                candidate_expert = _postprocess_candidate_expert(candidate_expert)
                candidate_state = _inject_expert_state(previous_global_state_dict, candidate_expert)
                global_model.load_state_dict(candidate_state)
                candidate_loss, candidate_signature = _evaluate_model_loss_and_signature_on_batches(
                    global_model,
                    batches,
                    loss_fn=loss_fn,
                    device=device,
                    aux_loss_weight=aux_loss_weight,
                )
                candidate_reward = float(base_loss - candidate_loss)
                candidate_global_losses.append(float(candidate_loss))
                candidate_global_rewards.append(candidate_reward)
                replaced_losses.append(float(candidate_loss))
                replacement_rewards.append(candidate_reward)
                candidate_signatures.append(candidate_signature.detach().clone().cpu())

                best_reward = float("-inf")
                best_step = 0.0
                for step in delta_trial_steps:
                    trial_expert = apply_delta_to_state(previous_global_expert_state, delta_state, step)
                    trial_expert = _postprocess_candidate_expert(trial_expert)
                    trial_state = _inject_expert_state(previous_global_state_dict, trial_expert)
                    global_model.load_state_dict(trial_state)
                    trial_loss = _evaluate_model_loss_on_batches(
                        global_model,
                        batches,
                        loss_fn=loss_fn,
                        device=device,
                        aux_loss_weight=aux_loss_weight,
                    )
                    trial_reward = float(base_loss - trial_loss)
                    if trial_reward > best_reward:
                        best_reward = trial_reward
                        best_step = float(step)
                delta_rewards.append(0.0 if best_reward == float("-inf") else best_reward)
                best_delta_steps.append(best_step)

            if coalition_topk > 0:
                ranked_indices = sorted(range(num_clients), key=lambda idx: candidate_global_rewards[idx], reverse=True)[:coalition_topk]
                coalition_states = [delta_states[idx] for idx in ranked_indices]
                if sample_counts is None:
                    coalition_weights = torch.full((len(coalition_states),), 1.0 / max(len(coalition_states), 1), dtype=torch.float64)
                else:
                    topk_counts = [int(sample_counts[idx]) for idx in ranked_indices]
                    coalition_weights = _sample_count_weights(topk_counts, eps=config.eps)
                coalition_delta = weighted_average_state_dicts(coalition_states, coalition_weights)
                coalition_rewards: List[float] = []
                for delta_state in delta_states:
                    blended_delta = weighted_average_state_dicts(
                        [coalition_delta, delta_state],
                        torch.as_tensor([0.5, 0.5], dtype=torch.float64),
                        fallback_state=coalition_delta,
                    )
                    coalition_expert = apply_delta_to_state(previous_global_expert_state, blended_delta, config.expert_server_lr)
                    coalition_expert = _postprocess_candidate_expert(coalition_expert)
                    coalition_state = _inject_expert_state(previous_global_state_dict, coalition_expert)
                    global_model.load_state_dict(coalition_state)
                    coalition_loss = _evaluate_model_loss_on_batches(
                        global_model,
                        batches,
                        loss_fn=loss_fn,
                        device=device,
                        aux_loss_weight=aux_loss_weight,
                    )
                    coalition_rewards.append(float(base_loss - coalition_loss))
                delta_rewards = [0.5 * float(dr) + 0.5 * float(cr) for dr, cr in zip(delta_rewards, coalition_rewards)]
    finally:
        global_model.load_state_dict(original_state)

    return {
        "base_loss": float(base_loss),
        "base_signature": base_signature.detach().clone().cpu(),
        "replaced_losses": replaced_losses,
        "replacement_rewards": replacement_rewards,
        "delta_rewards": delta_rewards,
        "best_delta_steps": best_delta_steps,
        "candidate_signatures": candidate_signatures,
        "used_server_validation": True,
        "expert_scope": expert_scope,
    }


def compute_vsmc_log_weights_for_expert(
    previous_global_expert_state: TensorState,
    current_client_expert_states: List[TensorState],
    ancestor_expert_states: List[TensorState],
    client_metrics: List[dict],
    sample_counts: List[int],
    expert_id: int,
    num_experts: int,
    config: VSMCExpertAggregationConfig,
    expert_scope: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    proposal_mean_state = compute_proposal_mean_state(
        previous_global_expert_state=previous_global_expert_state,
        current_client_expert_states=current_client_expert_states,
        sample_counts=sample_counts,
    )
    log_f_values: List[float] = []
    log_r_values: List[float] = []
    for current_state, ancestor_state in zip(current_client_expert_states, ancestor_expert_states):
        log_f_values.append(float(compute_log_f(current_state, ancestor_state, config).item()))
        log_r_values.append(float(compute_log_r(current_state, proposal_mean_state, config).item()))

    log_g_values, log_g_info = compute_log_g_values(
        client_metrics=client_metrics,
        expert_id=expert_id,
        num_experts=num_experts,
        sample_counts=sample_counts,
        config=config,
        expert_scope=expert_scope,
    )
    log_f_tensor = torch.as_tensor(log_f_values, dtype=torch.float64)
    log_r_tensor = torch.as_tensor(log_r_values, dtype=torch.float64)
    mode = getattr(config, "vsmc_mode", "baseline")
    if mode == "pure_f_only":
        log_w = log_f_tensor
    elif mode == "pure_r_only":
        log_w = -log_r_tensor
    elif mode == "g_only":
        log_w = log_g_values
    elif mode == "g_minus_r":
        log_w = log_g_values - log_r_tensor
    elif mode in {"full", "baseline"}:
        log_w = log_f_tensor + log_g_values - log_r_tensor
    else:
        log_w = log_f_tensor + log_g_values - log_r_tensor
    clamp_value = abs(float(config.clamp_log_weight))
    log_w = torch.clamp(log_w, -clamp_value, clamp_value)
    if not torch.isfinite(log_w).all():
        raise ValueError("log_w contains NaN/Inf")

    info = {
        "proposal_mean_state": proposal_mean_state,
        "log_f_values": log_f_values,
        "log_g_values": log_g_values.tolist(),
        "log_r_values": log_r_values,
        "log_w_values": log_w.tolist(),
        "loss_values": log_g_info["raw_loss"],
        "normalized_loss_values": log_g_info["normalized_loss"],
        "server_val_loss_values": log_g_info["server_val_loss"],
        "normalized_server_val_loss_values": log_g_info["normalized_server_val_loss"],
        "usage_values": log_g_info["usage"],
        "route_counts": log_g_info["route_count"],
        "sample_counts": log_g_info["sample_count"],
        "all_route_zero": log_g_info["all_route_zero"],
    }
    return log_w, info


def aggregate_one_expert_strong_signal(
    previous_global_expert_state: TensorState,
    current_client_expert_states: List[TensorState],
    client_metrics: List[dict],
    sample_counts: List[int],
    expert_id: int,
    num_experts: int,
    config: VSMCExpertAggregationConfig,
    expert_scope: Optional[str] = None,
    state_key: Optional[str] = None,
    replace_reward_memory: Optional[Dict[Any, float]] = None,
    delta_reward_memory: Optional[Dict[Any, float]] = None,
    credit_memory: Optional[Dict[Any, float]] = None,
    prev_selected_signature: Optional[torch.Tensor] = None,
) -> Tuple[TensorState, Dict[str, Any]]:
    with torch.no_grad():
        info: Dict[str, Any] = {
            "expert_id": int(expert_id),
            "num_particles": len(current_client_expert_states),
            "weights": [],
            "fallback_used": False,
            "fallback_reason": "",
            "resampled": False,
            "ancestor_indices": [],
        }
        if not current_client_expert_states:
            info["fallback_used"] = True
            info["fallback_reason"] = "no_client_particles"
            return clone_detached_state_dict(previous_global_expert_state), info

        fedavg_weights = _sample_count_weights(sample_counts, eps=config.eps)
        client_ids = [
            (metrics or {}).get("_client_id", idx)
            for idx, metrics in enumerate(client_metrics)
        ]
        delta_states = [
            compute_state_delta(client_state, previous_global_expert_state)
            for client_state in current_client_expert_states
        ]
        fedavg_delta = weighted_average_state_dicts(delta_states, fedavg_weights)

        neg_log_r_values: List[float] = []
        r_clipped = 0
        for delta_state in delta_states:
            delta_vec, fedavg_vec = _flatten_common_delta_pair(delta_state, fedavg_delta)
            if delta_vec.numel() == 0:
                neg_log_r = 0.0
            else:
                dist = ((delta_vec - fedavg_vec) ** 2).mean().item()
                neg_log_r = dist / (2.0 * max(float(config.sigma_r), config.eps) ** 2)
            unclipped = max(0.0, float(neg_log_r))
            clipped = min(unclipped, float(config.r_bonus_clip))
            if clipped < unclipped:
                r_clipped += 1
            neg_log_r_values.append(clipped)
        alpha_r = safe_logsumexp_normalize(
            torch.as_tensor(neg_log_r_values, dtype=torch.float64) * float(config.lambda_r),
            temperature=1.0,
            eps=config.eps,
        )

        local_log_g, local_g_info = compute_log_g_values(
            client_metrics=client_metrics,
            expert_id=expert_id,
            num_experts=num_experts,
            sample_counts=sample_counts,
            config=config,
            expert_scope=expert_scope,
        )

        replacement_rewards_raw: List[float] = []
        delta_rewards_raw: List[float] = []
        replaced_losses: List[float] = []
        best_delta_steps: List[float] = []
        candidate_signatures: List[torch.Tensor] = []
        base_signature: Optional[torch.Tensor] = None
        base_losses: List[float] = []
        for metrics in client_metrics:
            metrics = metrics or {}
            reward_map = metrics.get("server_val_expert_rewards", {})
            delta_map = metrics.get("server_val_delta_rewards", {})
            loss_map = metrics.get("server_val_expert_losses", {})
            base_map = metrics.get("server_val_base_losses", {})
            step_map = metrics.get("server_val_best_delta_steps", {})
            signature_map = metrics.get("server_val_candidate_signatures", {})
            base_signature_map = metrics.get("server_val_base_signatures", {})
            reward_value = reward_map.get(expert_scope, 0.0) if isinstance(reward_map, dict) else 0.0
            delta_value = delta_map.get(expert_scope, 0.0) if isinstance(delta_map, dict) else 0.0
            loss_value = loss_map.get(expert_scope, float("nan")) if isinstance(loss_map, dict) else float("nan")
            base_value = base_map.get(expert_scope, float("nan")) if isinstance(base_map, dict) else float("nan")
            step_value = step_map.get(expert_scope, 0.0) if isinstance(step_map, dict) else 0.0
            signature_value = signature_map.get(expert_scope, []) if isinstance(signature_map, dict) else []
            base_signature_value = base_signature_map.get(expert_scope, []) if isinstance(base_signature_map, dict) else []
            replacement_rewards_raw.append(float(reward_value))
            delta_rewards_raw.append(float(delta_value))
            replaced_losses.append(float(loss_value))
            base_losses.append(float(base_value))
            best_delta_steps.append(float(step_value))
            signature_tensor = torch.as_tensor(signature_value, dtype=torch.float32).reshape(-1).cpu()
            candidate_signatures.append(signature_tensor)
            if base_signature is None and len(base_signature_value) > 0:
                base_signature = torch.as_tensor(base_signature_value, dtype=torch.float32).reshape(-1).cpu()

        if base_signature is None:
            base_signature = torch.empty(0, dtype=torch.float32)

        replacement_rewards_ema: List[float] = []
        delta_rewards_ema: List[float] = []
        credit_values: List[float] = []
        novelty_values: List[float] = []
        stability_values: List[float] = []
        for client_key, replace_reward, delta_reward, signature_tensor in zip(
            client_ids,
            replacement_rewards_raw,
            delta_rewards_raw,
            candidate_signatures,
        ):
            previous_replace = _get_client_memory_value(replace_reward_memory, client_key)
            previous_delta = _get_client_memory_value(delta_reward_memory, client_key)
            previous_credit = _get_client_memory_value(credit_memory, client_key)
            replacement_rewards_ema.append(_ema_update(previous_replace, replace_reward, config.reward_ema_rho))
            delta_rewards_ema.append(_ema_update(previous_delta, delta_reward, config.reward_ema_rho))
            credit_values.append(previous_credit)

            if signature_tensor.numel() > 0 and base_signature.numel() == signature_tensor.numel():
                novelty = float(torch.mean((signature_tensor - base_signature) ** 2).item())
            else:
                novelty = 0.0
            novelty_values.append(min(novelty, float(config.novelty_clip)))

            if (
                prev_selected_signature is not None
                and torch.is_tensor(prev_selected_signature)
                and prev_selected_signature.numel() > 0
                and signature_tensor.numel() == prev_selected_signature.numel()
            ):
                stability = -float(torch.mean((signature_tensor - prev_selected_signature.cpu()) ** 2).item())
            else:
                stability = 0.0
            stability_values.append(max(-float(config.stability_clip), min(stability, float(config.stability_clip))))

        replace_signal = _normalize_score_tensor(replacement_rewards_ema, config.eps)
        delta_signal = _normalize_score_tensor(delta_rewards_ema, config.eps)
        credit_signal = _normalize_score_tensor(credit_values, config.eps)
        novelty_signal = _normalize_score_tensor(novelty_values, config.eps)
        stability_signal = _normalize_score_tensor(stability_values, config.eps)
        r_signal = _normalize_score_tensor(neg_log_r_values, config.eps)

        combined_signal = (
            float(config.replace_reward_beta) * replace_signal
            + float(config.delta_reward_beta) * delta_signal
            + float(config.credit_beta) * credit_signal
            + float(config.novelty_beta) * novelty_signal
            + float(config.stability_beta) * stability_signal
            + float(config.lambda_r) * r_signal
        )
        combined_signal = torch.clamp(
            combined_signal,
            min=-float(config.signal_reward_clip),
            max=float(config.signal_reward_clip),
        )

        if torch.allclose(combined_signal, torch.zeros_like(combined_signal)) and config.use_local_g_if_no_server_val:
            alpha_signal = safe_logsumexp_normalize(
                local_log_g * float(config.lambda_g),
                temperature=max(float(config.signal_temperature), config.eps),
                eps=config.eps,
            )
        else:
            alpha_signal = safe_logsumexp_normalize(
                combined_signal,
                temperature=max(float(config.signal_temperature), config.eps),
                eps=config.eps,
            )

        alpha = (1.0 - float(config.mix_with_fedavg) - float(config.uniform_mix)) * alpha_signal
        alpha = alpha + float(config.mix_with_fedavg) * fedavg_weights
        alpha = alpha + float(config.uniform_mix) * torch.full_like(alpha_signal, 1.0 / max(alpha_signal.numel(), 1))
        alpha = _clamp_and_normalize_alpha(alpha, config.alpha_max, config.eps)

        ess = effective_sample_size(alpha, config.eps)
        normalized_ess = ess / max(float(len(current_client_expert_states)), 1.0)
        fallback_reason = ""
        if not torch.isfinite(alpha).all():
            fallback_reason = "alpha_nan_or_inf"
        elif normalized_ess < float(config.normalized_ess_min):
            fallback_reason = "low_final_ess"
        if fallback_reason and config.fallback_to_fedavg_on_bad_alpha:
            alpha = fedavg_weights.clone()
            ess = effective_sample_size(alpha, config.eps)
            normalized_ess = ess / max(float(len(current_client_expert_states)), 1.0)
            info["fallback_used"] = True
            info["fallback_reason"] = fallback_reason

        aggregated_delta = weighted_average_state_dicts(delta_states, alpha, fallback_state=fedavg_delta)
        new_expert = (
            apply_delta_to_state(previous_global_expert_state, aggregated_delta, config.expert_server_lr)
            if config.aggregate_delta
            else weighted_average_state_dicts(
                current_client_expert_states,
                alpha,
                fallback_state=previous_global_expert_state,
            )
        )
        if config.use_ema_f_smoothing:
            final_expert = weighted_average_state_dicts(
                [new_expert, previous_global_expert_state],
                torch.as_tensor([1.0 - float(config.expert_ema_rho), float(config.expert_ema_rho)], dtype=torch.float64),
                fallback_state=previous_global_expert_state,
            )
        else:
            final_expert = new_expert

        weighted_signature = torch.empty(0, dtype=torch.float32)
        if any(sig.numel() > 0 for sig in candidate_signatures):
            valid_pairs = [
                (float(weight), sig)
                for weight, sig in zip(alpha.tolist(), candidate_signatures)
                if sig.numel() > 0
            ]
            if valid_pairs:
                target_dim = valid_pairs[0][1].numel()
                signature_acc = torch.zeros(target_dim, dtype=torch.float32)
                total_weight = 0.0
                for weight, sig in valid_pairs:
                    if sig.numel() != target_dim:
                        continue
                    signature_acc = signature_acc + sig.to(dtype=torch.float32) * float(weight)
                    total_weight += float(weight)
                if total_weight > 0.0:
                    weighted_signature = signature_acc / total_weight

        blended_reward = [
            0.5 * float(replace_reward) + 0.5 * float(delta_reward)
            for replace_reward, delta_reward in zip(replacement_rewards_ema, delta_rewards_ema)
        ]
        next_credit_values = [
            _ema_update(previous_credit, reward_value, config.credit_momentum)
            for previous_credit, reward_value in zip(credit_values, blended_reward)
        ]

        delta_norms = [_state_l2_norm(delta_state) for delta_state in delta_states]
        fedavg_delta_norm = _state_l2_norm(fedavg_delta)
        info.update(
            {
                "weights": alpha.tolist(),
                "alpha": alpha.tolist(),
                "ess": float(ess),
                "normalized_ess": float(normalized_ess),
                "alpha_g_server_ess": float(effective_sample_size(alpha_signal, config.eps) / max(len(current_client_expert_states), 1)),
                "alpha_r_ess": float(effective_sample_size(alpha_r, config.eps) / max(len(current_client_expert_states), 1)),
                "alpha_final_ess": float(normalized_ess),
                "min_alpha": float(alpha.min().item()),
                "max_alpha": float(alpha.max().item()),
                "mean_alpha": float(alpha.mean().item()),
                "log_f_mean": float(_safe_nanmean(stability_values, default=0.0)),
                "log_g_mean": float(_safe_nanmean(replacement_rewards_ema, default=0.0)),
                "log_r_mean": float(_safe_nanmean(neg_log_r_values, default=0.0)),
                "log_w_min": float(combined_signal.min().item()) if combined_signal.numel() else 0.0,
                "log_w_max": float(combined_signal.max().item()) if combined_signal.numel() else 0.0,
                "logZ_increment": float("nan"),
                "loss_values": local_g_info["raw_loss"],
                "normalized_loss_values": local_g_info["normalized_loss"],
                "usage_values": local_g_info["usage"],
                "route_counts": local_g_info["route_count"],
                "sample_counts": local_g_info["sample_count"],
                "server_val_loss_values": replaced_losses,
                "server_val_base_loss": _safe_nanmean(base_losses),
                "server_val_replaced_loss_mean": _safe_nanmean(replaced_losses),
                "server_val_reward_mean": float(_safe_nanmean(replacement_rewards_raw, default=0.0)),
                "server_val_reward_std": float(torch.as_tensor(replacement_rewards_raw, dtype=torch.float64).std(unbiased=False).item()) if replacement_rewards_raw else 0.0,
                "server_val_loss_mean": _safe_nanmean(replaced_losses),
                "delta_reward_mean": float(_safe_nanmean(delta_rewards_raw, default=0.0)),
                "delta_reward_std": float(torch.as_tensor(delta_rewards_raw, dtype=torch.float64).std(unbiased=False).item()) if delta_rewards_raw else 0.0,
                "credit_mean": float(_safe_nanmean(next_credit_values, default=0.0)),
                "credit_std": float(torch.as_tensor(next_credit_values, dtype=torch.float64).std(unbiased=False).item()) if next_credit_values else 0.0,
                "novelty_mean": float(_safe_nanmean(novelty_values, default=0.0)),
                "stability_mean": float(_safe_nanmean(stability_values, default=0.0)),
                "usage_mean": float(torch.as_tensor(local_g_info["usage"], dtype=torch.float64).mean().item()),
                "route_count_sum": float(torch.as_tensor(local_g_info["route_count"], dtype=torch.float64).sum().item()),
                "delta_norm_mean": float(sum(delta_norms) / max(len(delta_norms), 1)),
                "delta_norm_max": float(max(delta_norms) if delta_norms else 0.0),
                "fedavg_delta_norm": float(fedavg_delta_norm),
                "expert_server_lr": float(config.expert_server_lr),
                "expert_ema_rho": float(config.expert_ema_rho),
                "r_clip_fraction": float(r_clipped / max(len(neg_log_r_values), 1)),
                "neg_log_r_values": neg_log_r_values,
                "replace_reward_values": replacement_rewards_raw,
                "replace_reward_ema_values": replacement_rewards_ema,
                "delta_reward_values": delta_rewards_raw,
                "delta_reward_ema_values": delta_rewards_ema,
                "credit_values": next_credit_values,
                "client_ids": list(client_ids),
                "weighted_signature": weighted_signature.tolist(),
                "base_signature": base_signature.tolist() if base_signature.numel() > 0 else [],
                "state_key": state_key,
                "best_delta_steps": best_delta_steps,
            }
        )
        return final_expert, info


def aggregate_one_expert_delta_decoupled(
    previous_global_expert_state: TensorState,
    current_client_expert_states: List[TensorState],
    client_metrics: List[dict],
    sample_counts: List[int],
    expert_id: int,
    num_experts: int,
    config: VSMCExpertAggregationConfig,
    expert_scope: Optional[str] = None,
) -> Tuple[TensorState, Dict[str, Any]]:
    with torch.no_grad():
        info: Dict[str, Any] = {
            "expert_id": int(expert_id),
            "num_particles": len(current_client_expert_states),
            "weights": [],
            "fallback_used": False,
            "fallback_reason": "",
            "resampled": False,
            "ancestor_indices": [],
        }
        if not current_client_expert_states:
            info["fallback_used"] = True
            info["fallback_reason"] = "no_client_particles"
            return clone_detached_state_dict(previous_global_expert_state), info

        fedavg_weights = _sample_count_weights(sample_counts, eps=config.eps)
        if len(current_client_expert_states) == 1:
            delta_state = compute_state_delta(current_client_expert_states[0], previous_global_expert_state)
            new_expert = (
                apply_delta_to_state(previous_global_expert_state, delta_state, config.expert_server_lr)
                if config.aggregate_delta
                else clone_detached_state_dict(current_client_expert_states[0])
            )
            if config.use_ema_f_smoothing:
                new_expert = weighted_average_state_dicts(
                    [new_expert, previous_global_expert_state],
                    torch.as_tensor([1.0 - float(config.expert_ema_rho), float(config.expert_ema_rho)], dtype=torch.float64),
                    fallback_state=previous_global_expert_state,
                )
            info.update(
                {
                    "weights": [1.0],
                    "alpha": [1.0],
                    "ess": 1.0,
                    "normalized_ess": 1.0,
                    "logZ_increment": 0.0,
                    "min_alpha": 1.0,
                    "max_alpha": 1.0,
                    "mean_alpha": 1.0,
                    "alpha_g_server_ess": 1.0,
                    "alpha_r_ess": 1.0,
                    "alpha_final_ess": 1.0,
                }
            )
            return new_expert, info

        delta_states = [
            compute_state_delta(client_state, previous_global_expert_state)
            for client_state in current_client_expert_states
        ]
        fedavg_delta = weighted_average_state_dicts(delta_states, fedavg_weights)
        neg_log_r_values: List[float] = []
        r_clipped = 0
        for delta_state in delta_states:
            delta_vec, fedavg_vec = _flatten_common_delta_pair(delta_state, fedavg_delta)
            if delta_vec.numel() == 0:
                neg_log_r = 0.0
            else:
                dist = ((delta_vec - fedavg_vec) ** 2).mean().item()
                neg_log_r = dist / (2.0 * max(float(config.sigma_r), config.eps) ** 2)
            unclipped = max(0.0, float(neg_log_r))
            clipped = min(unclipped, float(config.r_bonus_clip))
            if clipped < unclipped:
                r_clipped += 1
            neg_log_r_values.append(clipped)
        alpha_r = safe_logsumexp_normalize(
            torch.as_tensor(neg_log_r_values, dtype=torch.float64) * float(config.lambda_r),
            temperature=1.0,
            eps=config.eps,
        )

        local_log_g, local_g_info = compute_log_g_values(
            client_metrics=client_metrics,
            expert_id=expert_id,
            num_experts=num_experts,
            sample_counts=sample_counts,
            config=config,
            expert_scope=expert_scope,
        )
        has_server_reward = any(
            isinstance((metrics or {}).get("server_val_expert_rewards"), dict)
            and expert_scope in (metrics or {}).get("server_val_expert_rewards", {})
            for metrics in client_metrics
        )
        server_reward_values: List[float] = []
        server_base_losses: List[float] = []
        server_replaced_losses: List[float] = []
        if config.use_server_validation_g and expert_scope is not None and has_server_reward:
            for metrics in client_metrics:
                metrics = metrics or {}
                reward_map = metrics.get("server_val_expert_rewards", {})
                base_map = metrics.get("server_val_base_losses", {})
                replaced_map = metrics.get("server_val_expert_losses", {})
                reward_value = reward_map.get(expert_scope, 0.0) if isinstance(reward_map, dict) else 0.0
                base_value = base_map.get(expert_scope, float("nan")) if isinstance(base_map, dict) else float("nan")
                replaced_value = replaced_map.get(expert_scope, float("nan")) if isinstance(replaced_map, dict) else float("nan")
                server_reward_values.append(float(reward_value))
                server_base_losses.append(float(base_value))
                server_replaced_losses.append(float(replaced_value))
            log_g_server = torch.clamp(
                float(config.server_g_beta) * torch.as_tensor(server_reward_values, dtype=torch.float64),
                min=-float(config.server_g_clip),
                max=float(config.server_g_clip),
            )
        else:
            server_reward_values = [0.0] * len(current_client_expert_states)
            server_base_losses = [float("nan")] * len(current_client_expert_states)
            server_replaced_losses = [float("nan")] * len(current_client_expert_states)
            log_g_server = torch.zeros(len(current_client_expert_states), dtype=torch.float64)

        if has_server_reward and config.use_server_validation_g:
            alpha_g = safe_logsumexp_normalize(log_g_server, temperature=1.0, eps=config.eps)
        elif config.use_local_g_if_no_server_val:
            alpha_g = safe_logsumexp_normalize(local_log_g * float(config.lambda_g), temperature=1.0, eps=config.eps)
        else:
            alpha_g = fedavg_weights.clone()

        alpha_f = safe_logsumexp_normalize(
            torch.as_tensor(
                [float(compute_log_f(state, previous_global_expert_state, config).item()) for state in current_client_expert_states],
                dtype=torch.float64,
            ),
            temperature=1.0,
            eps=config.eps,
        )
        if config.base_alpha_source == "fedavg":
            alpha_base = fedavg_weights.clone()
        elif config.base_alpha_source == "f":
            alpha_base = alpha_f
        elif config.base_alpha_source == "fr":
            alpha_base = _normalize_probability(alpha_f + alpha_r, config.eps)
        else:
            alpha_base = alpha_r

        alpha = (1.0 - float(config.decouple_mu)) * alpha_base + float(config.decouple_mu) * alpha_g
        if float(config.mix_with_fedavg) > 0.0:
            alpha = (1.0 - float(config.mix_with_fedavg)) * alpha + float(config.mix_with_fedavg) * fedavg_weights
        alpha = _clamp_and_normalize_alpha(alpha, config.alpha_max, config.eps)

        ess = effective_sample_size(alpha, config.eps)
        normalized_ess = ess / max(float(len(current_client_expert_states)), 1.0)
        alpha_g_ess = effective_sample_size(alpha_g, config.eps)
        alpha_r_ess = effective_sample_size(alpha_r, config.eps)
        fallback_reason = ""
        if not torch.isfinite(alpha).all():
            fallback_reason = "alpha_nan_or_inf"
        elif normalized_ess < float(config.normalized_ess_min):
            fallback_reason = "low_final_ess"

        if fallback_reason and config.fallback_to_fedavg_on_bad_alpha:
            alpha = fedavg_weights.clone()
            ess = effective_sample_size(alpha, config.eps)
            normalized_ess = ess / max(float(len(current_client_expert_states)), 1.0)
            info["fallback_used"] = True
            info["fallback_reason"] = fallback_reason

        if config.aggregate_delta:
            aggregated_delta = weighted_average_state_dicts(delta_states, alpha, fallback_state=fedavg_delta)
            new_expert = apply_delta_to_state(previous_global_expert_state, aggregated_delta, config.expert_server_lr)
        else:
            aggregated_delta = weighted_average_state_dicts(delta_states, alpha, fallback_state=fedavg_delta)
            new_expert = weighted_average_state_dicts(
                current_client_expert_states,
                alpha,
                fallback_state=previous_global_expert_state,
            )
        if config.use_ema_f_smoothing:
            final_expert = weighted_average_state_dicts(
                [new_expert, previous_global_expert_state],
                torch.as_tensor([1.0 - float(config.expert_ema_rho), float(config.expert_ema_rho)], dtype=torch.float64),
                fallback_state=previous_global_expert_state,
            )
        else:
            final_expert = new_expert

        delta_norms = [_state_l2_norm(delta_state) for delta_state in delta_states]
        fedavg_delta_norm = _state_l2_norm(fedavg_delta)
        info.update(
            {
                "weights": alpha.tolist(),
                "alpha": alpha.tolist(),
                "ess": float(ess),
                "normalized_ess": float(normalized_ess),
                "alpha_g_server_ess": float(alpha_g_ess / max(len(current_client_expert_states), 1)),
                "alpha_r_ess": float(alpha_r_ess / max(len(current_client_expert_states), 1)),
                "alpha_final_ess": float(normalized_ess),
                "min_alpha": float(alpha.min().item()),
                "max_alpha": float(alpha.max().item()),
                "mean_alpha": float(alpha.mean().item()),
                "log_f_mean": float(torch.as_tensor(alpha_f, dtype=torch.float64).mean().item()),
                "log_g_mean": float(log_g_server.mean().item()) if has_server_reward and config.use_server_validation_g else float(local_log_g.mean().item()),
                "log_r_mean": float(torch.as_tensor(neg_log_r_values, dtype=torch.float64).mean().item()),
                "log_w_min": float("nan"),
                "log_w_max": float("nan"),
                "logZ_increment": float("nan"),
                "loss_values": local_g_info["raw_loss"],
                "normalized_loss_values": local_g_info["normalized_loss"],
                "usage_values": local_g_info["usage"],
                "route_counts": local_g_info["route_count"],
                "sample_counts": local_g_info["sample_count"],
                "server_val_loss_values": server_replaced_losses,
                "server_val_base_loss": float(torch.as_tensor(server_base_losses, dtype=torch.float64).nanmean().item()) if server_base_losses else float("nan"),
                "server_val_replaced_loss_mean": float(torch.as_tensor(server_replaced_losses, dtype=torch.float64).nanmean().item()) if server_replaced_losses else float("nan"),
                "server_val_reward_mean": float(torch.as_tensor(server_reward_values, dtype=torch.float64).mean().item()),
                "server_val_reward_std": float(torch.as_tensor(server_reward_values, dtype=torch.float64).std(unbiased=False).item()),
                "server_val_loss_mean": float(torch.as_tensor(server_replaced_losses, dtype=torch.float64).nanmean().item()) if server_replaced_losses else float("nan"),
                "usage_mean": float(torch.as_tensor(local_g_info["usage"], dtype=torch.float64).mean().item()),
                "route_count_sum": float(torch.as_tensor(local_g_info["route_count"], dtype=torch.float64).sum().item()),
                "delta_norm_mean": float(sum(delta_norms) / max(len(delta_norms), 1)),
                "delta_norm_max": float(max(delta_norms) if delta_norms else 0.0),
                "fedavg_delta_norm": float(fedavg_delta_norm),
                "expert_server_lr": float(config.expert_server_lr),
                "expert_ema_rho": float(config.expert_ema_rho),
                "r_clip_fraction": float(r_clipped / max(len(neg_log_r_values), 1)),
                "neg_log_r_values": neg_log_r_values,
            }
        )
        return final_expert, info


def aggregate_one_expert_vsmc(
    previous_global_expert_state: TensorState,
    current_client_expert_states: List[TensorState],
    client_metrics: List[dict],
    sample_counts: List[int],
    expert_id: int,
    num_experts: int,
    config: VSMCExpertAggregationConfig,
    prev_particle_bank_for_expert: Optional[List[TensorState]] = None,
    prev_weights_for_expert: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    expert_scope: Optional[str] = None,
    state_key: Optional[str] = None,
    replace_reward_memory: Optional[Dict[Any, float]] = None,
    delta_reward_memory: Optional[Dict[Any, float]] = None,
    credit_memory: Optional[Dict[Any, float]] = None,
    prev_selected_signature: Optional[torch.Tensor] = None,
) -> Tuple[TensorState, Dict[str, Any]]:
    with torch.no_grad():
        if config.vsmc_mode in {"strong_signal_joint", "block_particle_gain_filter"}:
            return aggregate_one_expert_strong_signal(
                previous_global_expert_state=previous_global_expert_state,
                current_client_expert_states=current_client_expert_states,
                client_metrics=client_metrics,
                sample_counts=sample_counts,
                expert_id=expert_id,
                num_experts=num_experts,
                config=config,
                expert_scope=expert_scope,
                state_key=state_key,
                replace_reward_memory=replace_reward_memory,
                delta_reward_memory=delta_reward_memory,
                credit_memory=credit_memory,
                prev_selected_signature=prev_selected_signature,
            )
        if config.vsmc_mode == "server_validated_delta_decoupled":
            return aggregate_one_expert_delta_decoupled(
                previous_global_expert_state=previous_global_expert_state,
                current_client_expert_states=current_client_expert_states,
                client_metrics=client_metrics,
                sample_counts=sample_counts,
                expert_id=expert_id,
                num_experts=num_experts,
                config=config,
                expert_scope=expert_scope,
            )
        info: Dict[str, Any] = {
            "expert_id": int(expert_id),
            "num_particles": len(current_client_expert_states),
            "weights": [],
            "fallback_used": False,
            "fallback_reason": "",
            "resampled": False,
            "ancestor_indices": [],
        }
        if not current_client_expert_states:
            info["fallback_used"] = True
            info["fallback_reason"] = "no_client_particles"
            return clone_detached_state_dict(previous_global_expert_state), info

        if len(current_client_expert_states) == 1:
            info.update(
                {
                    "weights": [1.0],
                    "alpha": [1.0],
                    "ess": 1.0,
                    "normalized_ess": 1.0,
                    "logZ_increment": 0.0,
                    "min_alpha": 1.0,
                    "max_alpha": 1.0,
                    "mean_alpha": 1.0,
                    "loss_mean": 0.0,
                    "usage_mean": 0.0,
                    "route_count_sum": 0.0,
                }
            )
            return clone_detached_state_dict(current_client_expert_states[0]), info

        ancestor_states = [previous_global_expert_state for _ in current_client_expert_states]
        if (
            config.use_resampling
            and prev_particle_bank_for_expert is not None
            and prev_weights_for_expert is not None
            and len(prev_particle_bank_for_expert) == len(current_client_expert_states)
            and prev_weights_for_expert.numel() == len(current_client_expert_states)
        ):
            if config.resampling_mode != "stratified":
                raise ValueError(f"Unsupported resampling_mode={config.resampling_mode!r}")
            ancestor_indices = stratified_resampling(prev_weights_for_expert, generator=generator)
            ancestor_states = [clone_detached_state_dict(prev_particle_bank_for_expert[int(idx)]) for idx in ancestor_indices.tolist()]
            info["resampled"] = True
            info["ancestor_indices"] = ancestor_indices.tolist()

        try:
            log_w, weight_info = compute_vsmc_log_weights_for_expert(
                previous_global_expert_state=previous_global_expert_state,
                current_client_expert_states=current_client_expert_states,
                ancestor_expert_states=ancestor_states,
                client_metrics=client_metrics,
                sample_counts=sample_counts,
                expert_id=expert_id,
                num_experts=num_experts,
                config=config,
                expert_scope=expert_scope,
            )
            alpha = safe_logsumexp_normalize(log_w, config.temperature, config.eps)
            ess = effective_sample_size(alpha, config.eps)
            normalized_ess = ess / max(float(len(current_client_expert_states)), 1.0)
            alpha_sum = float(alpha.sum().item())
            fallback_reason = ""
            if not torch.isfinite(alpha).all():
                fallback_reason = "alpha_nan_or_inf"
            elif abs(alpha_sum - 1.0) > 1e-4:
                fallback_reason = "alpha_sum_not_one"
            elif weight_info["all_route_zero"] and config.fallback_when_all_route_zero:
                fallback_reason = "all_route_zero"
            elif normalized_ess < float(config.low_ess_threshold) and config.fallback_on_low_ess:
                fallback_reason = "low_ess"

            if fallback_reason:
                raise ValueError(fallback_reason)

            aggregated = weighted_average_state_dicts(
                current_client_expert_states,
                alpha,
                fallback_state=previous_global_expert_state,
            )
            log_w_tensor = torch.as_tensor(weight_info["log_w_values"], dtype=torch.float64)
            logZ_increment = torch.logsumexp(log_w_tensor, dim=0) - math.log(max(len(current_client_expert_states), 1))
            info.update(weight_info)
            info.update(
                {
                    "weights": alpha.tolist(),
                    "alpha": alpha.tolist(),
                    "ess": float(ess),
                    "normalized_ess": float(normalized_ess),
                    "logZ_increment": float(logZ_increment.item()),
                    "fallback_used": False,
                    "fallback_reason": "",
                    "log_f_mean": float(torch.as_tensor(weight_info["log_f_values"], dtype=torch.float64).mean().item()),
                    "log_g_mean": float(torch.as_tensor(weight_info["log_g_values"], dtype=torch.float64).mean().item()),
                    "log_r_mean": float(torch.as_tensor(weight_info["log_r_values"], dtype=torch.float64).mean().item()),
                    "log_w_min": float(log_w_tensor.min().item()),
                    "log_w_max": float(log_w_tensor.max().item()),
                    "min_alpha": float(alpha.min().item()),
                    "max_alpha": float(alpha.max().item()),
                    "mean_alpha": float(alpha.mean().item()),
                    "loss_mean": float(torch.as_tensor(weight_info["loss_values"], dtype=torch.float64).mean().item()),
                    "server_val_loss_mean": float(
                        torch.as_tensor(weight_info["server_val_loss_values"], dtype=torch.float64).mean().item()
                    ),
                    "usage_mean": float(torch.as_tensor(weight_info["usage_values"], dtype=torch.float64).mean().item()),
                    "route_count_sum": float(torch.as_tensor(weight_info["route_counts"], dtype=torch.float64).sum().item()),
                }
            )
            return aggregated, info
        except ValueError as exc:
            if not config.fallback_to_fedavg_on_nan and str(exc) not in {"all_route_zero", "low_ess"}:
                raise
            fallback_alpha = _sample_count_weights(sample_counts, eps=config.eps)
            aggregated = sample_count_fedavg_state_dicts(
                current_client_expert_states,
                sample_counts,
                fallback_state=previous_global_expert_state,
            )
            ess = effective_sample_size(fallback_alpha, config.eps)
            normalized_ess = ess / max(float(len(current_client_expert_states)), 1.0)
            info.update(
                {
                    "weights": fallback_alpha.tolist(),
                    "alpha": fallback_alpha.tolist(),
                    "ess": float(ess),
                    "normalized_ess": float(normalized_ess),
                    "logZ_increment": float("nan"),
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                    "min_alpha": float(fallback_alpha.min().item()),
                    "max_alpha": float(fallback_alpha.max().item()),
                    "mean_alpha": float(fallback_alpha.mean().item()),
                }
            )
            return aggregated, info


def aggregate_expert_particles(
    global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_sample_counts: Sequence[int],
    client_metrics: Sequence[Dict[str, Any]],
    expert_id: int,
    num_experts: int,
    config: VSMCExpertAggregationConfig,
    previous_log_weights: Optional[Sequence[float]] = None,
    client_ids: Optional[Sequence[Optional[int]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TensorState, Dict[str, Any]]:
    del previous_log_weights, client_ids
    aggregated, info = aggregate_one_expert_vsmc(
        previous_global_expert_state=global_expert_state,
        current_client_expert_states=list(client_expert_states),
        client_metrics=list(client_metrics),
        sample_counts=list(client_sample_counts),
        expert_id=expert_id,
        num_experts=num_experts,
        config=config,
        expert_scope=None,
    )
    return _merge_with_reference_state(global_expert_state, aggregated, device=device), info


class VSMCExpertAggregator:
    def __init__(self, config: VSMCExpertAggregationConfig, num_experts: Optional[int] = None, logger: Optional[Any] = None):
        self.config = config
        self.num_experts = num_experts
        self.logger = logger
        self.state = VSMCExpertState(
            prev_particle_bank={},
            prev_weights={},
            logZ_history=collections.defaultdict(float),
            round_idx=0,
        )
        self.last_full_info: Dict[str, Any] = {}

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)
        else:
            warnings.warn(message)

    @staticmethod
    def _state_key(expert_scope: str, expert_id: int) -> str:
        return f"{expert_scope}::{int(expert_id)}"

    def aggregate_model(
        self,
        previous_global_state_dict: TensorState,
        client_state_dicts: List[TensorState],
        client_metrics: List[dict],
        sample_counts: List[int],
        num_experts: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        with torch.no_grad():
            if not client_state_dicts:
                raise ValueError("client_state_dicts must not be empty")
            local_num_experts = int(num_experts if num_experts is not None else self.num_experts or 0)
            if local_num_experts <= 0:
                raise ValueError("num_experts must be provided")

            fedavg_state = sample_count_fedavg_state_dicts(
                client_state_dicts,
                sample_counts,
                fallback_state=previous_global_state_dict,
            )
            base_state, fedavg_expert_groups = split_state_dict_by_expert(fedavg_state, num_experts=local_num_experts)
            _, previous_expert_groups = split_state_dict_by_expert(previous_global_state_dict, num_experts=local_num_experts)
            client_expert_groups = [
                split_state_dict_by_expert(state_dict, num_experts=local_num_experts)[1]
                for state_dict in client_state_dicts
            ]

            use_block_mode = self.config.vsmc_mode == "block_particle_gain_filter"
            if use_block_mode:
                fedavg_grouped = group_expert_groups_by_block(fedavg_expert_groups)
                previous_grouped = group_expert_groups_by_block(previous_expert_groups)
                client_grouped = [group_expert_groups_by_block(group_map) for group_map in client_expert_groups]
            else:
                fedavg_grouped = fedavg_expert_groups
                previous_grouped = previous_expert_groups
                client_grouped = client_expert_groups

            merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
            per_expert_info: List[Dict[str, Any]] = []

            ordered_scopes = _ordered_union_keys([previous_grouped, fedavg_grouped] + client_grouped)
            for expert_scope in ordered_scopes:
                previous_group = previous_grouped.get(expert_scope)
                if previous_group is None:
                    previous_group = fedavg_grouped.get(expert_scope)
                if previous_group is None:
                    continue
                if use_block_mode:
                    expert_id = int(previous_group.get("expert_ids", [0])[0])
                    previous_global_expert_state = previous_group["state_dict"]
                else:
                    expert_id = int(previous_group["expert_id"])
                    previous_global_expert_state = previous_group["state_dict"]
                state_key = self._state_key(expert_scope, expert_id)

                current_states: List[TensorState] = []
                current_metrics: List[dict] = []
                current_sample_counts: List[int] = []
                for client_group_map, metrics, sample_count in zip(client_grouped, client_metrics, sample_counts):
                    if expert_scope not in client_group_map:
                        continue
                    current_states.append(client_group_map[expert_scope]["state_dict"])
                    current_metrics.append(metrics or {})
                    current_sample_counts.append(int(sample_count))

                aggregated_state, info = aggregate_one_expert_vsmc(
                    previous_global_expert_state=previous_global_expert_state,
                    current_client_expert_states=current_states,
                    client_metrics=current_metrics,
                    sample_counts=current_sample_counts,
                    expert_id=expert_id,
                    num_experts=local_num_experts,
                    config=self.config,
                    prev_particle_bank_for_expert=self.state.prev_particle_bank.get(state_key),
                    prev_weights_for_expert=self.state.prev_weights.get(state_key),
                    generator=generator,
                    expert_scope=expert_scope,
                    state_key=state_key,
                    replace_reward_memory=self.state.reward_replace_ema.get(state_key),
                    delta_reward_memory=self.state.reward_delta_ema.get(state_key),
                    credit_memory=self.state.credit_memory.get(state_key),
                    prev_selected_signature=self.state.prev_selected_signature.get(state_key),
                )

                logz_increment = info.get("logZ_increment", 0.0)
                if math.isfinite(float(logz_increment)):
                    self.state.logZ_history[state_key] = float(self.state.logZ_history.get(state_key, 0.0)) + float(logz_increment)
                info["expert_scope"] = expert_scope
                info["logZ_total"] = float(self.state.logZ_history.get(state_key, 0.0))
                merged_expert_groups[expert_scope] = {
                    "expert_id": expert_id,
                    "expert_scope": expert_scope,
                    "state_dict": _merge_with_reference_state(previous_global_expert_state, aggregated_state),
                }
                per_expert_info.append(info)

                if self.config.keep_particle_bank:
                    self.state.prev_particle_bank[state_key] = [clone_detached_state_dict(state) for state in current_states]
                    self.state.prev_weights[state_key] = torch.as_tensor(info.get("weights", []), dtype=torch.float64).detach().cpu()
                if self.config.vsmc_mode in {"strong_signal_joint", "block_particle_gain_filter"}:
                    client_ids = info.get("client_ids", [])
                    replace_reward_emas = info.get("replace_reward_ema_values", [])
                    delta_reward_emas = info.get("delta_reward_ema_values", [])
                    credit_values = info.get("credit_values", [])
                    if client_ids:
                        self.state.reward_replace_ema[state_key] = {
                            client_id: float(value)
                            for client_id, value in zip(client_ids, replace_reward_emas)
                        }
                        self.state.reward_delta_ema[state_key] = {
                            client_id: float(value)
                            for client_id, value in zip(client_ids, delta_reward_emas)
                        }
                        self.state.credit_memory[state_key] = {
                            client_id: float(value)
                            for client_id, value in zip(client_ids, credit_values)
                        }
                    weighted_signature = torch.as_tensor(info.get("weighted_signature", []), dtype=torch.float32).reshape(-1)
                    if weighted_signature.numel() > 0:
                        self.state.prev_selected_signature[state_key] = weighted_signature.detach().clone().cpu()

                if float(info.get("normalized_ess", 1.0)) < float(self.config.ess_threshold):
                    self._warn(
                        f"Low normalized ESS for expert_scope={expert_scope} expert_id={expert_id}: "
                        f"{float(info.get('normalized_ess', 0.0)):.4f}"
                    )

            new_global_state_dict = merge_expert_states_into_global_state(base_state, merged_expert_groups)
            new_global_state_dict = _merge_with_reference_state(previous_global_state_dict, new_global_state_dict)
            self.state.round_idx += 1
            full_info = {
                "round_idx": int(self.state.round_idx),
                "per_expert_info": per_expert_info,
                "mean_normalized_ess": float(
                    sum(float(info.get("normalized_ess", 0.0)) for info in per_expert_info) / max(len(per_expert_info), 1)
                ),
                "mean_logZ_increment": float(
                    sum(float(info.get("logZ_increment", 0.0)) for info in per_expert_info if math.isfinite(float(info.get("logZ_increment", 0.0))))
                    / max(sum(1 for info in per_expert_info if math.isfinite(float(info.get("logZ_increment", 0.0)))), 1)
                ),
                "fallback_expert_count": int(sum(1 for info in per_expert_info if info.get("fallback_used"))),
            }
            self.last_full_info = full_info
            return new_global_state_dict, full_info

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, List[Dict[str, Any]]]:
        client_state_dicts = [state_dict for state_dict, _, _ in client_updates]
        sample_counts = [num_samples for _, num_samples, _ in client_updates]
        client_metrics = [metrics for _, _, metrics in client_updates]
        aggregated_state, full_info = self.aggregate_model(
            previous_global_state_dict=global_state,
            client_state_dicts=client_state_dicts,
            client_metrics=client_metrics,
            sample_counts=sample_counts,
            num_experts=num_experts,
        )
        if device is not None:
            aggregated_state = collections.OrderedDict(
                (key, value.to(device) if torch.is_tensor(value) else value)
                for key, value in aggregated_state.items()
            )
        return aggregated_state, full_info["per_expert_info"]


def ensure_vsmc_log_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        return
    fieldnames = [
        "round",
        "expert_id",
        "expert_scope",
        "num_particles",
        "log_f_mean",
        "log_f_min",
        "log_f_max",
        "log_g_mean",
        "log_g_min",
        "log_g_max",
        "log_r_mean",
        "log_r_min",
        "log_r_max",
        "log_w_min",
        "log_w_max",
        "log_w_mean",
        "min_alpha",
        "max_alpha",
        "mean_alpha",
        "ess",
        "normalized_ess",
        "logZ_increment",
        "logZ_total",
        "loss_mean",
        "loss_min",
        "loss_max",
        "server_val_base_loss",
        "server_val_replaced_loss_mean",
        "server_val_reward_mean",
        "server_val_reward_std",
        "delta_reward_mean",
        "delta_reward_std",
        "credit_mean",
        "credit_std",
        "novelty_mean",
        "stability_mean",
        "usage_mean",
        "route_count_sum",
        "alpha_g_server_ess",
        "alpha_r_ess",
        "alpha_final_ess",
        "delta_norm_mean",
        "delta_norm_max",
        "fedavg_delta_norm",
        "expert_server_lr",
        "expert_ema_rho",
        "router_usage_entropy",
        "router_usage_min",
        "router_usage_max",
        "fallback_used",
        "fallback_reason",
        "resampled",
    ]
    with open(csv_path, "w", newline="") as csvfile:
        csv.DictWriter(csvfile, fieldnames=fieldnames).writeheader()
