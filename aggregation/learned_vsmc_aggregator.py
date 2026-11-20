from __future__ import annotations

import collections
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    functional_call = None

from aggregation.vsmc_expert_aggregator import (
    TensorState,
    _extract_expert_metric,
    _merge_with_reference_state,
    _ordered_union_keys,
    _sample_count_weights,
    clone_detached_state_dict,
    effective_sample_size,
    merge_expert_states_into_global_state,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
    group_expert_groups_by_block,
)

_LOG_2PI = math.log(2.0 * math.pi)
_SAFE_LOG_MIN = -50.0
_SAFE_LOG_MAX = 50.0
_NA = "NA"


@dataclass
class LearnedVSMCAggregationConfig:
    num_particles: int = 4
    proposal_lr: float = 1e-4
    proposal_train_steps: int = 1
    sigma_f: float = 1.0
    server_g_beta: float = 1.0
    use_reward_g: bool = True
    val_batches: int = 4
    grad_clip: float = 1.0
    use_reinforce_if_not_differentiable: bool = True
    require_differentiable_g: bool = False
    reinforce_baseline_momentum: float = 0.9
    total_rounds: int = 1
    latent_dim: int = 4
    block_latent_dim: int = 8
    use_block_joint: bool = False
    latent_prior_sigma: float = 1.0
    latent_mean_scale: float = 0.25
    latent_residual_max: float = 0.75
    final_residual_max: float = 0.15
    anneal_start_ratio: float = 0.6
    include_zero_particle: bool = False
    mean_a_scale: float = 0.5
    mean_b_scale: float = 0.2
    init_logvar: float = -4.0
    logvar_scale: float = 2.0
    a_min: float = 0.0
    a_max: float = 1.5
    b_min: float = -0.3
    b_max: float = 0.3
    reward_clip: float = 5.0
    large_loss: float = 20.0
    feature_clip: float = 10.0
    logq_coef: float = 0.03
    normalize_reward: bool = True
    reward_norm_eps: float = 1e-8
    reward_type: str = "loss_plus_confidence"
    confidence_reward_weight: float = 0.3
    loss_reward_weight: float = 0.7
    include_fedavg_particle: bool = True
    mix_fedavg_candidate: float = 0.2
    final_mix_fedavg_candidate: float = 0.5
    mix_uniform: float = 0.2
    alpha_max: float = 0.7
    min_normalized_ess: float = 0.25
    mode: str = "default"
    debug: bool = False


@dataclass
class LearnedVSMCState:
    previous_round_ess: Dict[str, float] = field(default_factory=dict)
    reinforce_baseline: Dict[str, float] = field(default_factory=dict)
    logz_history: Dict[str, float] = field(default_factory=dict)
    last_grad_norm: float = 0.0
    last_param_delta_norm: float = 0.0
    round_idx: int = 0


class ProposalNet(nn.Module):
    def __init__(self, input_dim: int, config: LearnedVSMCAggregationConfig):
        super().__init__()
        self.config = config
        output_dim = max(int(config.latent_dim), int(config.block_latent_dim), 1) * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, features: torch.Tensor, base_coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(features)
        latent_dim = int(base_coeffs.numel()) if base_coeffs.numel() > 0 else max(int(self.config.latent_dim), 1)
        raw_mean_full, raw_logvar_full = raw.chunk(2, dim=-1)
        raw_mean = raw_mean_full[..., :latent_dim]
        raw_logvar = raw_logvar_full[..., :latent_dim]
        base_coeffs = base_coeffs.to(device=raw.device, dtype=raw.dtype).reshape(-1)
        mean_coeffs = base_coeffs + float(self.config.latent_mean_scale) * torch.tanh(raw_mean)
        logvar_coeffs = float(self.config.init_logvar) + float(self.config.logvar_scale) * torch.tanh(raw_logvar)
        first_mean = mean_coeffs[..., 0]
        second_mean = mean_coeffs[..., 1] if latent_dim > 1 else torch.zeros_like(first_mean)
        first_logvar = logvar_coeffs[..., 0]
        second_logvar = logvar_coeffs[..., 1] if latent_dim > 1 else torch.full_like(first_logvar, float(self.config.init_logvar))
        return {
            "mean_coeffs": mean_coeffs,
            "logvar_coeffs": logvar_coeffs,
            "mean_a": first_mean,
            "logvar_a": first_logvar,
            "mean_b": second_mean,
            "logvar_b": second_logvar,
        }


def _state_to_device(state: TensorState, device: torch.device, detach: bool = False) -> TensorState:
    moved: TensorState = collections.OrderedDict()
    for key, value in state.items():
        if torch.is_tensor(value):
            tensor = value.detach() if detach else value
            moved[key] = tensor.to(device)
        else:
            moved[key] = copy.deepcopy(value)
    return moved


def _zero_like_float_state(reference_state: TensorState) -> TensorState:
    zero_state: TensorState = collections.OrderedDict()
    for key, value in reference_state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            zero_state[key] = torch.zeros_like(value)
    return zero_state


def _state_has_non_finite(state: TensorState) -> bool:
    for value in state.values():
        if torch.is_tensor(value) and torch.is_floating_point(value) and not torch.isfinite(value).all():
            return True
    return False


def _state_sq_sum_and_count(state: TensorState) -> Tuple[float, int]:
    sq_sum = 0.0
    count = 0
    for value in state.values():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            tensor = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            sq_sum += float(tensor.pow(2).sum().item())
            count += tensor.numel()
    return sq_sum, count


def _state_l2_norm(state: TensorState) -> float:
    sq_sum, _ = _state_sq_sum_and_count(state)
    return math.sqrt(max(sq_sum, 0.0))


def _state_mse(state: TensorState) -> float:
    sq_sum, count = _state_sq_sum_and_count(state)
    return sq_sum / max(count, 1)


def _sanitize_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _subtract_float_states(a: TensorState, b: TensorState, device: torch.device, detach: bool = False) -> TensorState:
    out: TensorState = collections.OrderedDict()
    for key in sorted(set(a.keys()) & set(b.keys())):
        tensor_a = a[key]
        tensor_b = b[key]
        if not (torch.is_tensor(tensor_a) and torch.is_tensor(tensor_b)):
            continue
        if not (torch.is_floating_point(tensor_a) and torch.is_floating_point(tensor_b)):
            continue
        if tensor_a.shape != tensor_b.shape:
            continue
        lhs = tensor_a.detach() if detach else tensor_a
        rhs = tensor_b.detach() if detach else tensor_b
        diff = lhs.to(device=device, dtype=torch.float32) - rhs.to(device=device, dtype=torch.float32)
        out[key] = _sanitize_float_tensor(diff)
    return out


def _weighted_average_float_states(states: Sequence[TensorState], weights: torch.Tensor, device: torch.device, detach: bool = False) -> TensorState:
    out: TensorState = collections.OrderedDict()
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device).reshape(-1)
    if weight_tensor.numel() == 0:
        return out
    weight_tensor = torch.nan_to_num(weight_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if weight_tensor.sum().item() <= 0.0:
        weight_tensor = torch.full_like(weight_tensor, 1.0 / max(weight_tensor.numel(), 1))
    for key in _ordered_union_keys(states):
        tensors: List[torch.Tensor] = []
        local_weights: List[torch.Tensor] = []
        for idx, state in enumerate(states):
            value = state.get(key)
            if not (torch.is_tensor(value) and torch.is_floating_point(value)):
                continue
            tensor = value.detach() if detach else value
            tensors.append(_sanitize_float_tensor(tensor.to(device=device, dtype=torch.float32)))
            local_weights.append(weight_tensor[idx])
        if not tensors:
            continue
        stacked = torch.stack(tensors, dim=0)
        stacked_weights = torch.stack(local_weights, dim=0)
        normalized = stacked_weights / stacked_weights.sum().clamp_min(1e-12)
        view_shape = [normalized.shape[0]] + [1] * (stacked.dim() - 1)
        out[key] = (stacked * normalized.view(*view_shape)).sum(dim=0)
    return out


def _blend_expert_states(a: TensorState, b: TensorState, mix_b: float, reference_state: TensorState) -> TensorState:
    mix_b = float(max(0.0, min(1.0, mix_b)))
    if mix_b <= 0.0:
        return _merge_with_reference_state(reference_state, a)
    if mix_b >= 1.0:
        return _merge_with_reference_state(reference_state, b)
    out: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys([reference_state, a, b]):
        if key in a and key in b and torch.is_tensor(a[key]) and torch.is_tensor(b[key]) and torch.is_floating_point(a[key]) and torch.is_floating_point(b[key]) and a[key].shape == b[key].shape:
            tensor_a = a[key]
            tensor_b = b[key].to(device=tensor_a.device, dtype=tensor_a.dtype)
            out[key] = (1.0 - mix_b) * tensor_a + mix_b * tensor_b
        elif key in a:
            out[key] = a[key]
        elif key in b:
            out[key] = b[key]
        else:
            out[key] = reference_state[key]
    return _merge_with_reference_state(reference_state, out)


def _tensor_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    tensor = torch.as_tensor(values, dtype=torch.float32)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    return mean, std


def _sanitize_feature_tensor(features: torch.Tensor, clip_value: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    has_nan = bool((~torch.isfinite(features)).any().item())
    sanitized = torch.nan_to_num(features, nan=0.0, posinf=clip_value, neginf=-clip_value)
    sanitized = torch.clamp(sanitized, -clip_value, clip_value)
    return sanitized, {
        "feature_min": float(sanitized.min().item()) if sanitized.numel() else 0.0,
        "feature_max": float(sanitized.max().item()) if sanitized.numel() else 0.0,
        "feature_mean": float(sanitized.mean().item()) if sanitized.numel() else 0.0,
        "feature_has_nan": has_nan,
    }


def _state_to_na_dict() -> Dict[str, Any]:
    return {}


def _effective_latent_dim(config: LearnedVSMCAggregationConfig, is_block_scope: bool) -> int:
    return max(int(config.block_latent_dim if is_block_scope else config.latent_dim), 1)



def _current_round_ratio(round_idx: int, total_rounds: int) -> float:
    if total_rounds <= 1:
        return 1.0
    return min(max(float(round_idx - 1) / float(total_rounds - 1), 0.0), 1.0)



def _annealed_value(start: float, final: float, round_ratio: float, start_ratio: float) -> float:
    if round_ratio <= start_ratio:
        return float(start)
    progress = min(max((round_ratio - start_ratio) / max(1.0 - start_ratio, 1e-12), 0.0), 1.0)
    return float(start) + progress * (float(final) - float(start))



def _expert_ids_list(expert_ref: Any) -> List[int]:
    if isinstance(expert_ref, (list, tuple)):
        return [int(v) for v in expert_ref]
    return [int(expert_ref)]



def _extract_metric_for_ids(metrics: Dict[str, Any], names: Tuple[str, ...], expert_ref: Any, default: float = 0.0) -> float:
    values = []
    for expert_id in _expert_ids_list(expert_ref):
        value = _extract_expert_metric(metrics, names, expert_id, default=None)
        if value is None:
            continue
        values.append(float(value))
    if values:
        return float(sum(values) / len(values))
    return float(default)


def _build_feature_tensor(
    previous_global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_metrics: Sequence[Dict[str, Any]],
    sample_counts: Sequence[int],
    expert_ref: Any,
    previous_round_ess: float,
    round_idx: int,
    total_rounds: int,
    device: torch.device,
    config: LearnedVSMCAggregationConfig,
) -> Tuple[torch.Tensor, Dict[str, Any], TensorState, List[TensorState]]:
    loss_values: List[float] = []
    route_counts: List[float] = []
    usages: List[float] = []
    client_delta_norms: List[float] = []
    client_delta_states = [
        _subtract_float_states(client_state, previous_global_expert_state, device=device, detach=False)
        for client_state in client_expert_states
    ]
    for metrics, delta_state in zip(client_metrics, client_delta_states):
        loss_values.append(_extract_metric_for_ids(metrics, ("expert_losses", "expert_loss"), expert_ref, default=metrics.get("val_loss", 0.0)))
        route_counts.append(_extract_metric_for_ids(metrics, ("route_counts", "route_count", "expert_counts"), expert_ref, default=0.0))
        usages.append(_extract_metric_for_ids(metrics, ("expert_usage", "expert_usages", "usage", "usages"), expert_ref, default=0.0))
        client_delta_norms.append(_state_l2_norm(delta_state))

    sample_weight_tensor = _sample_count_weights(sample_counts).to(dtype=torch.float32, device=device)
    delta_fedavg = _weighted_average_float_states(client_delta_states, sample_weight_tensor, device=device, detach=False)
    fedavg_delta_norm = _state_l2_norm(delta_fedavg)
    fedavg_delta_mse = _state_mse(delta_fedavg)
    mean_client_loss, std_client_loss = _tensor_mean_std(loss_values)
    mean_route_count, std_route_count = _tensor_mean_std(route_counts)
    mean_usage, std_usage = _tensor_mean_std(usages)
    client_delta_norm_mean, client_delta_norm_std = _tensor_mean_std(client_delta_norms)
    round_idx_normalized = float(round_idx) / max(float(total_rounds), 1.0)
    feature_values = torch.tensor(
        [
            math.log1p(max(float(len(client_expert_states)), 0.0)),
            min(max(mean_client_loss, 0.0), 10.0),
            min(max(std_client_loss, 0.0), 10.0),
            math.log1p(max(mean_route_count, 0.0)),
            math.log1p(max(std_route_count, 0.0)),
            min(max(mean_usage, 0.0), 10.0),
            min(max(std_usage, 0.0), 10.0),
            math.log1p(max(fedavg_delta_norm, 0.0)),
            math.log1p(max(client_delta_norm_mean, 0.0)),
            math.log1p(max(client_delta_norm_std, 0.0)),
            math.log1p(max(previous_round_ess, 0.0)),
            round_idx_normalized,
        ],
        dtype=torch.float32,
        device=device,
    )
    feature_values, feature_stats = _sanitize_feature_tensor(feature_values, clip_value=float(config.feature_clip))
    feature_stats.update(
        {
            "fedavg_delta_norm": float(fedavg_delta_norm),
            "fedavg_delta_mse": float(fedavg_delta_mse),
        }
    )
    return feature_values, feature_stats, delta_fedavg, client_delta_states


def _state_mse_between(a: TensorState, b: TensorState) -> float:
    diff = _subtract_float_states(a, b, device=next((v.device for v in a.values() if torch.is_tensor(v)), torch.device("cpu")), detach=True)
    return _state_mse(diff)


def _particle_norms_and_mses(
    particles: Sequence[TensorState],
    previous_global_expert_state: TensorState,
    fedavg_expert_state: TensorState,
    device: torch.device,
) -> Dict[str, Any]:
    particle_delta_norms: List[float] = []
    particle_to_prev_mses: List[float] = []
    particle_to_fedavg_mses: List[float] = []
    any_nan = False
    any_inf = False
    for particle in particles:
        diff_prev = _subtract_float_states(particle, previous_global_expert_state, device=device, detach=True)
        diff_fedavg = _subtract_float_states(particle, fedavg_expert_state, device=device, detach=True)
        particle_delta_norms.append(_state_l2_norm(diff_prev))
        particle_to_prev_mses.append(_state_mse(diff_prev))
        particle_to_fedavg_mses.append(_state_mse(diff_fedavg))
        for value in particle.values():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                any_nan = any_nan or bool(torch.isnan(value).any().item())
                any_inf = any_inf or bool(torch.isinf(value).any().item())
    norm_mean, _ = _tensor_mean_std(particle_delta_norms)
    prev_mse_mean, _ = _tensor_mean_std(particle_to_prev_mses)
    fedavg_mse_mean, _ = _tensor_mean_std(particle_to_fedavg_mses)
    return {
        "particle_delta_norm_mean": norm_mean,
        "particle_delta_norm_min": min(particle_delta_norms) if particle_delta_norms else 0.0,
        "particle_delta_norm_max": max(particle_delta_norms) if particle_delta_norms else 0.0,
        "particle_to_prev_mse_mean": prev_mse_mean,
        "particle_to_fedavg_mse_mean": fedavg_mse_mean,
        "particle_to_fedavg_mse_min": min(particle_to_fedavg_mses) if particle_to_fedavg_mses else 0.0,
        "particle_to_fedavg_mse_max": max(particle_to_fedavg_mses) if particle_to_fedavg_mses else 0.0,
        "any_particle_nan": any_nan,
        "any_particle_inf": any_inf,
    }


def _add_float_states(reference_state: TensorState, delta_state: TensorState, device: torch.device) -> TensorState:
    merged: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys([reference_state, delta_state]):
        reference_value = reference_state.get(key)
        if not torch.is_tensor(reference_value):
            if key in reference_state:
                merged[key] = copy.deepcopy(reference_value)
            continue
        if not torch.is_floating_point(reference_value):
            merged[key] = reference_value
            continue
        delta_value = delta_state.get(key, torch.zeros_like(reference_value))
        merged[key] = _sanitize_float_tensor(reference_value + delta_value.to(device=reference_value.device, dtype=reference_value.dtype))
    return merged



def _flatten_float_state(state: TensorState, keys: Sequence[str], device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for key in keys:
        value = state.get(key)
        if not (torch.is_tensor(value) and torch.is_floating_point(value)):
            continue
        chunks.append(_sanitize_float_tensor(value.to(device=device, dtype=torch.float32)).reshape(-1))
    if not chunks:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(chunks, dim=0)



def _delta_state_from_vector(reference_state: TensorState, keys: Sequence[str], vector: torch.Tensor, device: torch.device) -> TensorState:
    delta_state: TensorState = collections.OrderedDict()
    offset = 0
    for key in keys:
        reference_value = reference_state.get(key)
        if not (torch.is_tensor(reference_value) and torch.is_floating_point(reference_value)):
            continue
        numel = reference_value.numel()
        piece = vector[offset: offset + numel].reshape(reference_value.shape)
        delta_state[key] = piece.to(device=reference_value.device, dtype=reference_value.dtype)
        offset += numel
    return delta_state



def _build_latent_basis_from_client_deltas(
    previous_global_expert_state: TensorState,
    client_delta_states: Sequence[TensorState],
    sample_counts: Sequence[int],
    latent_dim: int,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, TensorState, Dict[str, Any]]:
    reference_state = _state_to_device(previous_global_expert_state, device=device, detach=False)
    candidate_keys: List[str] = []
    for key, value in reference_state.items():
        if not (torch.is_tensor(value) and torch.is_floating_point(value)):
            continue
        if any(key not in delta_state or not torch.is_tensor(delta_state[key]) or delta_state[key].shape != value.shape for delta_state in client_delta_states):
            continue
        candidate_keys.append(key)

    if not candidate_keys:
        empty_basis = torch.empty(max(int(latent_dim), 1), 0, dtype=torch.float32, device=device)
        empty_coeffs = torch.zeros(max(int(latent_dim), 1), dtype=torch.float32, device=device)
        return candidate_keys, empty_basis, empty_coeffs, _zero_like_float_state(reference_state), {
            "latent_basis_rank": 0,
            "latent_basis_dim": 0,
            "latent_basis_energy": 0.0,
        }

    delta_matrix = torch.stack([
        _flatten_float_state(delta_state, candidate_keys, device=device)
        for delta_state in client_delta_states
    ], dim=0)
    weights = _sample_count_weights(sample_counts).to(device=device, dtype=torch.float32)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum().item() <= 0.0:
        weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
    fedavg_vector = (weights.unsqueeze(1) * delta_matrix).sum(dim=0)
    weighted_matrix = torch.sqrt(weights.unsqueeze(1).clamp_min(1e-12)) * delta_matrix
    if weighted_matrix.numel() == 0 or weighted_matrix.shape[1] == 0:
        basis_matrix = torch.empty(max(int(latent_dim), 1), 0, dtype=torch.float32, device=device)
        coeffs = torch.zeros(max(int(latent_dim), 1), dtype=torch.float32, device=device)
    else:
        _, singular_values, vh = torch.linalg.svd(weighted_matrix, full_matrices=False)
        rank = min(max(int(latent_dim), 1), int(vh.shape[0]))
        basis_matrix = vh[:rank]
        if rank < max(int(latent_dim), 1):
            pad = torch.zeros(max(int(latent_dim), 1) - rank, basis_matrix.shape[1], dtype=basis_matrix.dtype, device=device)
            basis_matrix = torch.cat([basis_matrix, pad], dim=0)
        coeffs = basis_matrix @ fedavg_vector
        energy = float((singular_values[:rank].pow(2).sum() / singular_values.pow(2).sum().clamp_min(1e-12)).item()) if singular_values.numel() else 0.0
        stats = {
            "latent_basis_rank": int(rank),
            "latent_basis_dim": int(basis_matrix.shape[0]),
            "latent_basis_energy": energy,
        }
        fedavg_delta_state = _delta_state_from_vector(reference_state, candidate_keys, fedavg_vector, device=device)
        return candidate_keys, basis_matrix, coeffs, fedavg_delta_state, stats

    stats = {
        "latent_basis_rank": 0,
        "latent_basis_dim": int(basis_matrix.shape[0]),
        "latent_basis_energy": 0.0,
    }
    fedavg_delta_state = _delta_state_from_vector(reference_state, candidate_keys, fedavg_vector, device=device)
    return candidate_keys, basis_matrix, coeffs, fedavg_delta_state, stats



def sample_vsmc_particles_for_expert(
    previous_global_expert_state: TensorState,
    client_delta_states: Sequence[TensorState],
    sample_counts: Sequence[int],
    proposal_net: ProposalNet,
    features: torch.Tensor,
    num_particles: int,
    latent_dim: int,
    residual_max: float,
    device: torch.device,
    config: LearnedVSMCAggregationConfig,
) -> Tuple[List[TensorState], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], TensorState, Dict[str, Any]]:
    delta_div = _zero_like_float_state(previous_global_expert_state)
    reference_state = _state_to_device(previous_global_expert_state, device=device, detach=False)
    state_keys, basis_matrix, fedavg_coeffs, fedavg_delta, basis_stats = _build_latent_basis_from_client_deltas(
        previous_global_expert_state=previous_global_expert_state,
        client_delta_states=client_delta_states,
        sample_counts=sample_counts,
        latent_dim=latent_dim,
        device=device,
    )
    fedavg_particle = _add_float_states(reference_state, fedavg_delta, device=device)

    proposal_stats = proposal_net(features, fedavg_coeffs)
    mean = proposal_stats["mean_coeffs"].reshape(-1)
    logvar = proposal_stats["logvar_coeffs"].reshape(-1)
    logvar = torch.nan_to_num(logvar, nan=float(config.init_logvar), posinf=-2.0, neginf=-6.0)
    logvar = torch.clamp(logvar, -6.0, -2.0)
    std = torch.exp(0.5 * logvar)

    deterministic_particles = int(bool(config.include_fedavg_particle)) + int(bool(config.include_zero_particle))
    sampled_count = max(int(num_particles) - deterministic_particles, 0)
    eps = torch.randn(sampled_count, latent_dim, device=device, dtype=features.dtype) if sampled_count > 0 else torch.empty(0, latent_dim, device=device, dtype=features.dtype)
    raw_residuals = mean.unsqueeze(0) + std.unsqueeze(0) * eps if sampled_count > 0 else torch.empty(0, latent_dim, device=device, dtype=features.dtype)
    if sampled_count > 0:
        residuals = torch.clamp(raw_residuals - fedavg_coeffs.unsqueeze(0), -float(residual_max), float(residual_max))
        sampled_coeffs = fedavg_coeffs.unsqueeze(0) + residuals
        log_q_terms = -0.5 * (((residuals - (mean - fedavg_coeffs).unsqueeze(0)) ** 2) / torch.exp(logvar).unsqueeze(0) + logvar.unsqueeze(0) + _LOG_2PI)
        sampled_log_q = torch.clamp(torch.nan_to_num(log_q_terms.sum(dim=-1), nan=0.0, posinf=_SAFE_LOG_MAX, neginf=_SAFE_LOG_MIN), _SAFE_LOG_MIN, _SAFE_LOG_MAX)
    else:
        sampled_coeffs = torch.empty(0, latent_dim, device=device, dtype=features.dtype)
        sampled_log_q = torch.empty(0, device=device, dtype=features.dtype)

    particles: List[TensorState] = []
    coeffs_list: List[torch.Tensor] = []
    raw_coeffs_list: List[torch.Tensor] = []
    log_q_list: List[torch.Tensor] = []

    if config.include_fedavg_particle:
        particles.append(fedavg_particle)
        coeffs_list.append(fedavg_coeffs)
        raw_coeffs_list.append(fedavg_coeffs)
        log_q_list.append(torch.tensor(0.0, device=device, dtype=features.dtype))

    if config.include_zero_particle:
        particles.append(reference_state)
        zero_coeffs = torch.zeros(latent_dim, device=device, dtype=features.dtype)
        coeffs_list.append(zero_coeffs)
        raw_coeffs_list.append(zero_coeffs)
        log_q_list.append(torch.tensor(0.0, device=device, dtype=features.dtype))

    for particle_idx in range(sampled_count):
        coeff_vector = sampled_coeffs[particle_idx]
        delta_vector = torch.matmul(coeff_vector.to(dtype=basis_matrix.dtype), basis_matrix)
        particle_delta = _delta_state_from_vector(reference_state, state_keys, delta_vector, device=device)
        particle_state = _add_float_states(reference_state, particle_delta, device=device)
        particles.append(particle_state)
        coeffs_list.append(coeff_vector)
        raw_coeffs_list.append(raw_residuals[particle_idx])
        log_q_list.append(sampled_log_q[particle_idx])

    coeffs = torch.stack(coeffs_list, dim=0) if coeffs_list else torch.empty(0, latent_dim, device=device, dtype=features.dtype)
    raw_coeffs_full = torch.stack(raw_coeffs_list, dim=0) if raw_coeffs_list else torch.empty(0, latent_dim, device=device, dtype=features.dtype)
    log_q = torch.stack(log_q_list, dim=0) if log_q_list else torch.empty(0, device=device, dtype=features.dtype)
    stats = {
        "delta_div_norm": float(_state_l2_norm(delta_div)),
        "delta_div_enabled": False,
        "latent_dim": int(latent_dim),
        "latent_coeff_norm_mean": float(coeffs.detach().norm(dim=-1).mean().item()) if coeffs.numel() else 0.0,
        "latent_residual_max": float(residual_max),
        **basis_stats,
    }
    return particles, coeffs, raw_coeffs_full, log_q, proposal_stats, fedavg_delta, stats

def compute_particle_log_f(
    particle_expert_states: Sequence[TensorState],
    previous_global_expert_state: TensorState,
    sigma_f: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    sigma_sq = max(float(sigma_f), 1e-12) ** 2
    log_f_values: List[torch.Tensor] = []
    particle_to_prev_mses: List[float] = []
    previous_state = _state_to_device(previous_global_expert_state, device=device, detach=False)
    for particle_state in particle_expert_states:
        particle_state = _state_to_device(particle_state, device=device, detach=False)
        mse_terms: List[torch.Tensor] = []
        for key in sorted(set(previous_state.keys()) & set(particle_state.keys())):
            prev_val = previous_state[key]
            part_val = particle_state[key]
            if not (torch.is_tensor(prev_val) and torch.is_tensor(part_val)):
                continue
            if not (torch.is_floating_point(prev_val) and torch.is_floating_point(part_val)):
                continue
            if prev_val.shape != part_val.shape:
                continue
            mse_terms.append(((part_val - prev_val.to(dtype=part_val.dtype)).float() ** 2).mean())
        mse = torch.stack(mse_terms).mean() if mse_terms else torch.zeros((), device=device, dtype=torch.float32)
        particle_to_prev_mses.append(float(mse.detach().item()))
        log_f = -(mse / (2.0 * sigma_sq))
        log_f = torch.clamp(torch.nan_to_num(log_f, nan=_SAFE_LOG_MIN, posinf=0.0, neginf=_SAFE_LOG_MIN), min=_SAFE_LOG_MIN, max=0.0)
        log_f_values.append(log_f)
    log_f_tensor = torch.stack(log_f_values, dim=0) if log_f_values else torch.empty(0, device=device, dtype=torch.float32)
    return log_f_tensor, {
        "particle_to_prev_mse_mean": float(sum(particle_to_prev_mses) / max(len(particle_to_prev_mses), 1)) if particle_to_prev_mses else 0.0,
        "log_f_min": float(log_f_tensor.min().detach().item()) if log_f_tensor.numel() else _SAFE_LOG_MIN,
        "log_f_max": float(log_f_tensor.max().detach().item()) if log_f_tensor.numel() else 0.0,
        "log_f_requires_grad": bool(log_f_tensor.requires_grad),
    }


def _build_functional_state(global_model: nn.Module, base_global_state: TensorState, particle_expert_state: TensorState, device: torch.device) -> Dict[str, torch.Tensor]:
    functional_state: Dict[str, torch.Tensor] = {}
    for name, param in global_model.named_parameters():
        if name in particle_expert_state:
            value = particle_expert_state[name]
        elif name in base_global_state:
            value = base_global_state[name]
        else:
            value = param.detach()
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=param.dtype)
            functional_state[name] = _sanitize_float_tensor(tensor) if torch.is_floating_point(tensor) else tensor
        else:
            functional_state[name] = value
    for name, buffer in global_model.named_buffers():
        if name in particle_expert_state:
            value = particle_expert_state[name]
        elif name in base_global_state:
            value = base_global_state[name]
        else:
            value = buffer.detach()
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=buffer.dtype)
            functional_state[name] = _sanitize_float_tensor(tensor) if torch.is_floating_point(tensor) else tensor
        else:
            functional_state[name] = value
    return functional_state


def _true_label_confidence(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    gathered = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    return gathered.mean()


def _evaluate_particle_metrics_functional(global_model: nn.Module, functional_state: Dict[str, torch.Tensor], val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], loss_fn: nn.Module, device: torch.device, aux_loss_weight: float, large_loss: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if functional_call is None:
        raise RuntimeError("functional_call is unavailable in this PyTorch build")
    total_loss = None
    total_confidence = None
    total_samples = 0
    for inputs, labels in val_batches:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs, router_stats = functional_call(global_model, functional_state, (inputs,))
        if not torch.isfinite(outputs).all():
            batch_loss = torch.full((), float(large_loss), dtype=torch.float32, device=device)
            batch_confidence = torch.zeros((), dtype=torch.float32, device=device)
        else:
            safe_outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)
            batch_loss = loss_fn(safe_outputs, labels)
            if isinstance(router_stats, dict) and router_stats.get("aux_loss") is not None:
                batch_loss = batch_loss + float(aux_loss_weight) * router_stats["aux_loss"]
            batch_loss = torch.nan_to_num(batch_loss, nan=float(large_loss), posinf=float(large_loss), neginf=float(large_loss))
            batch_confidence = _true_label_confidence(safe_outputs, labels)
            batch_confidence = torch.nan_to_num(batch_confidence, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size = int(inputs.size(0))
        total_loss = batch_loss * batch_size if total_loss is None else total_loss + batch_loss * batch_size
        total_confidence = batch_confidence * batch_size if total_confidence is None else total_confidence + batch_confidence * batch_size
        total_samples += batch_size
    if total_loss is None or total_confidence is None or total_samples == 0:
        return (
            torch.tensor(float(large_loss), dtype=torch.float32, device=device),
            torch.zeros((), dtype=torch.float32, device=device),
        )
    mean_loss = torch.nan_to_num(total_loss / float(total_samples), nan=float(large_loss), posinf=float(large_loss), neginf=float(large_loss))
    mean_confidence = torch.nan_to_num(total_confidence / float(total_samples), nan=0.0, posinf=0.0, neginf=0.0)
    return mean_loss, mean_confidence


def _evaluate_particle_metrics_black_box(global_model: nn.Module, base_global_state: TensorState, particle_expert_state: TensorState, val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], loss_fn: nn.Module, device: torch.device, aux_loss_weight: float, large_loss: float) -> Tuple[float, float]:
    original_state = clone_detached_state_dict(global_model.state_dict())
    try:
        candidate_state = clone_detached_state_dict(base_global_state)
        for key, value in particle_expert_state.items():
            candidate_state[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        global_model.load_state_dict(candidate_state)
        global_model.eval()
        total_loss = 0.0
        total_confidence = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in val_batches:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, router_stats = global_model(inputs)
                if not torch.isfinite(outputs).all():
                    loss = torch.tensor(float(large_loss), device=device)
                    batch_confidence = torch.tensor(0.0, device=device)
                else:
                    safe_outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)
                    loss = loss_fn(safe_outputs, labels)
                    if isinstance(router_stats, dict) and router_stats.get("aux_loss") is not None:
                        loss = loss + float(aux_loss_weight) * router_stats["aux_loss"]
                    loss = torch.nan_to_num(loss, nan=float(large_loss), posinf=float(large_loss), neginf=float(large_loss))
                    batch_confidence = _true_label_confidence(safe_outputs, labels)
                    batch_confidence = torch.nan_to_num(batch_confidence, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss += float(loss.item()) * int(inputs.size(0))
                total_confidence += float(batch_confidence.item()) * int(inputs.size(0))
                total_samples += int(inputs.size(0))
        if total_samples == 0:
            return float(large_loss), 0.0
        mean_loss = min(max(total_loss / float(total_samples), 0.0), float(large_loss))
        mean_confidence = min(max(total_confidence / float(total_samples), 0.0), 1.0)
        return mean_loss, mean_confidence
    finally:
        global_model.load_state_dict(original_state)


def _reward_to_log_g(raw_reward_tensor: torch.Tensor, config: LearnedVSMCAggregationConfig) -> Tuple[torch.Tensor, Dict[str, Any]]:
    reward_clip = float(config.reward_clip)
    raw_reward_tensor = torch.nan_to_num(raw_reward_tensor, nan=-reward_clip, posinf=reward_clip, neginf=-reward_clip)
    raw_reward_tensor = torch.clamp(raw_reward_tensor, -reward_clip, reward_clip)
    raw_mean, raw_std = _tensor_mean_std(raw_reward_tensor.detach().cpu().tolist())
    reward_std_tensor = raw_reward_tensor.std(unbiased=False) if raw_reward_tensor.numel() > 1 else torch.zeros((), device=raw_reward_tensor.device)
    reward_zero_std = bool(reward_std_tensor.item() <= float(config.reward_norm_eps))

    if config.reward_type == "loss":
        normalized_reward = raw_reward_tensor
    else:
        if reward_std_tensor.item() > float(config.reward_norm_eps):
            normalized_reward = (raw_reward_tensor - raw_reward_tensor.mean()) / (reward_std_tensor + float(config.reward_norm_eps))
        else:
            normalized_reward = torch.zeros_like(raw_reward_tensor)
    normalized_mean, normalized_std = _tensor_mean_std(normalized_reward.detach().cpu().tolist())
    log_g = float(config.server_g_beta) * normalized_reward
    log_g = torch.clamp(torch.nan_to_num(log_g, nan=-10.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    return log_g, {
        "raw_reward_mean": raw_mean,
        "raw_reward_std": raw_std,
        "raw_reward_min": float(raw_reward_tensor.min().detach().item()) if raw_reward_tensor.numel() else 0.0,
        "raw_reward_max": float(raw_reward_tensor.max().detach().item()) if raw_reward_tensor.numel() else 0.0,
        "normalized_reward_mean": normalized_mean,
        "normalized_reward_std": normalized_std,
        "normalized_reward_min": float(normalized_reward.min().detach().item()) if normalized_reward.numel() else 0.0,
        "normalized_reward_max": float(normalized_reward.max().detach().item()) if normalized_reward.numel() else 0.0,
        "log_g_std": float(log_g.detach().std(unbiased=False).item()) if log_g.numel() > 1 else 0.0,
        "log_g_min": float(log_g.detach().min().item()) if log_g.numel() else 0.0,
        "log_g_max": float(log_g.detach().max().item()) if log_g.numel() else 0.0,
        "reward_zero_std": reward_zero_std,
        "normalized_rewards": [float(v) for v in normalized_reward.detach().cpu().tolist()],
    }


def compute_particle_log_g_server_val(global_model: nn.Module, base_global_state: TensorState, particle_expert_states: Sequence[TensorState], val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], loss_fn: nn.Module, device: torch.device, config: LearnedVSMCAggregationConfig, aux_loss_weight: float = 0.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
    particle_requires_grad = any(
        torch.is_tensor(value) and bool(value.requires_grad)
        for particle_state in particle_expert_states
        for value in particle_state.values()
    )
    if not val_batches:
        zeros = torch.zeros(len(particle_expert_states), dtype=torch.float32, device=device)
        return zeros, {
            "base_loss": float(config.large_loss),
            "base_correct_confidence": 0.0,
            "particle_losses": [float(config.large_loss)] * len(particle_expert_states),
            "particle_confidences": [0.0] * len(particle_expert_states),
            "rewards": [0.0] * len(particle_expert_states),
            "confidence_rewards": [0.0] * len(particle_expert_states),
            "normalized_rewards": [0.0] * len(particle_expert_states),
            "differentiable": True,
            "particle_requires_grad": particle_requires_grad,
            "all_invalid_particles": False,
            "reward_stats": {
                "raw_reward_mean": 0.0,
                "raw_reward_std": 0.0,
                "raw_reward_min": 0.0,
                "raw_reward_max": 0.0,
                "normalized_reward_mean": 0.0,
                "normalized_reward_std": 0.0,
                "normalized_reward_min": 0.0,
                "normalized_reward_max": 0.0,
                "log_g_std": 0.0,
                "log_g_min": 0.0,
                "log_g_max": 0.0,
                "reward_zero_std": True,
                "normalized_rewards": [0.0] * len(particle_expert_states),
            },
        }

    base_functional_state = _build_functional_state(global_model, base_global_state, {}, device=device)
    base_loss, base_confidence = _evaluate_particle_metrics_functional(global_model, base_functional_state, val_batches, loss_fn, device, aux_loss_weight, float(config.large_loss))
    particle_losses = []
    particle_confidences = []
    for particle_state in particle_expert_states:
        functional_state = _build_functional_state(global_model, base_global_state, particle_state, device=device)
        particle_loss, particle_confidence = _evaluate_particle_metrics_functional(global_model, functional_state, val_batches, loss_fn, device, aux_loss_weight, float(config.large_loss))
        particle_losses.append(particle_loss)
        particle_confidences.append(particle_confidence)
    particle_loss_tensor = torch.stack(particle_losses, dim=0) if particle_losses else torch.empty(0, device=device)
    particle_conf_tensor = torch.stack(particle_confidences, dim=0) if particle_confidences else torch.empty(0, device=device)
    particle_loss_tensor = torch.nan_to_num(particle_loss_tensor, nan=float(config.large_loss), posinf=float(config.large_loss), neginf=float(config.large_loss))
    particle_conf_tensor = torch.nan_to_num(particle_conf_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    loss_reward = base_loss.unsqueeze(0) - particle_loss_tensor
    confidence_reward = particle_conf_tensor - base_confidence.unsqueeze(0)

    if config.reward_type == "loss_plus_confidence":
        raw_reward_tensor = float(config.loss_reward_weight) * loss_reward + float(config.confidence_reward_weight) * confidence_reward
    else:
        raw_reward_tensor = loss_reward

    log_g, reward_stats = _reward_to_log_g(raw_reward_tensor, config)
    invalid_tensor = particle_loss_tensor >= float(config.large_loss)
    log_g = torch.where(invalid_tensor, torch.full_like(log_g, -10.0), log_g)
    differentiable = bool(log_g.requires_grad and particle_requires_grad)
    return log_g, {
        "base_loss": float(base_loss.detach().item()),
        "base_correct_confidence": float(base_confidence.detach().item()),
        "particle_losses": [float(v.detach().item()) for v in particle_loss_tensor],
        "particle_confidences": [float(v.detach().item()) for v in particle_conf_tensor],
        "rewards": [float(v.detach().item()) for v in raw_reward_tensor],
        "confidence_rewards": [float(v.detach().item()) for v in confidence_reward],
        "normalized_rewards": list(reward_stats.get("normalized_rewards", [])),
        "differentiable": differentiable,
        "particle_requires_grad": particle_requires_grad,
        "all_invalid_particles": bool(invalid_tensor.all().item()),
        "reward_stats": {
            **reward_stats,
            "server_val_particle_loss_mean": float(particle_loss_tensor.detach().mean().item()) if particle_loss_tensor.numel() else float(config.large_loss),
            "particle_correct_confidence_mean": float(particle_conf_tensor.detach().mean().item()) if particle_conf_tensor.numel() else 0.0,
            "particle_correct_confidence_min": float(particle_conf_tensor.detach().min().item()) if particle_conf_tensor.numel() else 0.0,
            "particle_correct_confidence_max": float(particle_conf_tensor.detach().max().item()) if particle_conf_tensor.numel() else 0.0,
            "confidence_reward_mean": float(confidence_reward.detach().mean().item()) if confidence_reward.numel() else 0.0,
            "confidence_reward_std": float(confidence_reward.detach().std(unbiased=False).item()) if confidence_reward.numel() > 1 else 0.0,
        },
    }


class LearnedVSMCExpertAggregator:
    def __init__(self, config: LearnedVSMCAggregationConfig, num_experts: Optional[int] = None, logger: Optional[Any] = None):
        self.config = config
        self.num_experts = num_experts
        self.logger = logger
        self.state = LearnedVSMCState()
        self.proposal_net = ProposalNet(input_dim=12, config=config)
        self.optimizer = torch.optim.Adam(self.proposal_net.parameters(), lr=float(config.proposal_lr))

    def _materialize_val_batches(self, val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]], device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches = []
        if val_loader is None:
            return batches
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max(int(self.config.val_batches), 1):
                break
            inputs, labels = batch
            batches.append((inputs.to(device), labels.to(device)))
        return batches

    def _reinforce_objective(self, expert_scope: str, log_q: torch.Tensor, log_f: torch.Tensor, log_g: torch.Tensor, logq_coef: float) -> torch.Tensor:
        reward = torch.nan_to_num(log_f.detach() + log_g.detach() - float(logq_coef) * log_q.detach(), nan=_SAFE_LOG_MIN, posinf=_SAFE_LOG_MAX, neginf=_SAFE_LOG_MIN)
        baseline_prev = float(self.state.reinforce_baseline.get(expert_scope, 0.0))
        reward_mean = float(reward.mean().item())
        baseline = float(self.config.reinforce_baseline_momentum) * baseline_prev + (1.0 - float(self.config.reinforce_baseline_momentum)) * reward_mean
        self.state.reinforce_baseline[expert_scope] = baseline
        advantage = reward - baseline
        return -(advantage * log_q).mean()

    def _compute_proposal_grad_norm(self) -> float:
        sq_sum = 0.0
        for param in self.proposal_net.parameters():
            if param.grad is None:
                continue
            grad = torch.nan_to_num(param.grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            sq_sum += float(grad.pow(2).sum().item())
        return math.sqrt(max(sq_sum, 0.0))

    def _proposal_has_non_finite_params(self) -> bool:
        return any(not torch.isfinite(param).all() for param in self.proposal_net.parameters())

    def _build_fedavg_expert_state(self, previous_global_expert_state: TensorState, client_expert_states: Sequence[TensorState], sample_counts: Sequence[int]) -> TensorState:
        if not client_expert_states:
            return clone_detached_state_dict(previous_global_expert_state)
        sample_weights = _sample_count_weights(sample_counts)
        aggregated = weighted_average_state_dicts(list(client_expert_states), sample_weights, fallback_state=previous_global_expert_state)
        return _merge_with_reference_state(previous_global_expert_state, aggregated)

    def _normalize_alpha(self, alpha_vsmc: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        n = int(alpha_vsmc.numel())
        if n == 0:
            return alpha_vsmc, {
                "alpha_max_before_clip": _NA,
                "alpha_max_after_clip": _NA,
                "normalized_ess_before_clip": _NA,
                "normalized_ess_after_clip": _NA,
                "alpha_entropy": _NA,
                "alpha_uniform_l1": _NA,
                "alpha_mixed_with_uniform": False,
                "bad_alpha_fallback": False,
            }
        uniform = torch.full_like(alpha_vsmc, 1.0 / max(n, 1))
        alpha = alpha_vsmc
        ess_before = effective_sample_size(alpha.detach().cpu().tolist())
        normalized_ess_before = ess_before / max(float(n), 1.0)
        alpha_max_before = float(alpha.max().item())
        alpha = (1.0 - float(self.config.mix_uniform)) * alpha + float(self.config.mix_uniform) * uniform
        alpha = torch.clamp(alpha, max=float(self.config.alpha_max))
        alpha = alpha / alpha.sum().clamp_min(1e-12)
        ess_after = effective_sample_size(alpha.detach().cpu().tolist())
        normalized_ess_after = ess_after / max(float(n), 1.0)
        bad_alpha_fallback = False
        if normalized_ess_after < float(self.config.min_normalized_ess):
            alpha = 0.5 * alpha + 0.5 * uniform
            alpha = alpha / alpha.sum().clamp_min(1e-12)
            ess_after = effective_sample_size(alpha.detach().cpu().tolist())
            normalized_ess_after = ess_after / max(float(n), 1.0)
            bad_alpha_fallback = True
        alpha_entropy = float((-(alpha * torch.log(alpha.clamp_min(1e-12))).sum()).item())
        alpha_uniform_l1 = float(torch.abs(alpha - uniform).sum().item())
        return alpha, {
            "alpha_max_before_clip": alpha_max_before,
            "alpha_max_after_clip": float(alpha.max().item()),
            "normalized_ess_before_clip": normalized_ess_before,
            "normalized_ess_after_clip": normalized_ess_after,
            "alpha_entropy": alpha_entropy,
            "alpha_uniform_l1": alpha_uniform_l1,
            "alpha_mixed_with_uniform": bool(self.config.mix_uniform > 0.0),
            "bad_alpha_fallback": bad_alpha_fallback,
        }

    def _compute_param_delta_norm(self, before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
        sq_sum = 0.0
        for key, value_before in before.items():
            value_after = after[key]
            if torch.is_tensor(value_before) and torch.is_tensor(value_after):
                diff = torch.nan_to_num((value_after.detach().float() - value_before.detach().float()), nan=0.0, posinf=0.0, neginf=0.0)
                sq_sum += float(diff.pow(2).sum().item())
        return math.sqrt(max(sq_sum, 0.0))

    def learned_vsmc_lower_bound_for_expert(
        self,
        global_model: nn.Module,
        base_global_state: TensorState,
        previous_global_expert_state: TensorState,
        client_expert_states: Sequence[TensorState],
        client_metrics: Sequence[Dict[str, Any]],
        sample_counts: Sequence[int],
        expert_scope: str,
        expert_ref: Any,
        val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        device: torch.device,
        round_idx: int,
        aux_loss_weight: float,
        update_history: bool = True,
    ) -> Dict[str, Any]:
        fedavg_expert_state = self._build_fedavg_expert_state(previous_global_expert_state, client_expert_states, sample_counts)
        features, feature_stats, computed_fedavg_delta, client_delta_states = _build_feature_tensor(
            previous_global_expert_state=previous_global_expert_state,
            client_expert_states=client_expert_states,
            client_metrics=client_metrics,
            sample_counts=sample_counts,
            expert_ref=expert_ref,
            previous_round_ess=float(self.state.previous_round_ess.get(expert_scope, 0.0)),
            round_idx=round_idx,
            total_rounds=int(self.config.total_rounds),
            device=device,
            config=self.config,
        )
        fedavg_delta = computed_fedavg_delta
        is_block_scope = bool(self.config.use_block_joint and "block" in str(expert_scope))
        latent_dim = _effective_latent_dim(self.config, is_block_scope=is_block_scope)
        round_ratio = _current_round_ratio(round_idx, int(self.config.total_rounds))
        residual_max = _annealed_value(self.config.latent_residual_max, self.config.final_residual_max, round_ratio, self.config.anneal_start_ratio)
        current_mix_fedavg_candidate = _annealed_value(self.config.mix_fedavg_candidate, self.config.final_mix_fedavg_candidate, round_ratio, self.config.anneal_start_ratio)
        fedavg_delta_norm = float(feature_stats.get("fedavg_delta_norm", 0.0))
        fallback_used = False
        fallback_reason = ""
        if fedavg_delta_norm < 1e-12:
            fallback_used = True
            fallback_reason = "near_zero_fedavg_delta"
            if self.logger is not None:
                self.logger.warning("fedavg_delta_norm is almost zero; learned_vsmc cannot update expert")

        particles: List[TensorState] = []
        coeffs = torch.empty(0, latent_dim, device=device)
        raw_coeffs = torch.empty(0, latent_dim, device=device)
        log_q = torch.empty(0, device=device)
        base_coeffs = torch.zeros(latent_dim, device=device)
        base_logvars = torch.full((latent_dim,), float(self.config.init_logvar), device=device)
        proposal_stats = {
            "mean_coeffs": base_coeffs,
            "logvar_coeffs": base_logvars,
            "mean_a": base_coeffs[0],
            "logvar_a": base_logvars[0],
            "mean_b": base_coeffs[1] if latent_dim > 1 else torch.tensor(0.0, device=device),
            "logvar_b": base_logvars[1] if latent_dim > 1 else torch.tensor(float(self.config.init_logvar), device=device),
        }
        delta_div_stats = {"delta_div_norm": 0.0, "delta_div_enabled": False, "latent_dim": latent_dim, "latent_basis_rank": 0, "latent_basis_dim": 0, "latent_basis_energy": 0.0, "latent_coeff_norm_mean": 0.0}
        if self.config.mode == "sanity_fedavg_particle":
            particles = [fedavg_expert_state]
            coeffs = torch.zeros(1, latent_dim, device=device)
            raw_coeffs = coeffs.clone()
            log_q = torch.zeros(1, device=device)
        elif not fallback_used:
            particles, coeffs, raw_coeffs, log_q, proposal_stats, _, delta_div_stats = sample_vsmc_particles_for_expert(
                previous_global_expert_state=previous_global_expert_state,
                client_delta_states=client_delta_states,
                sample_counts=sample_counts,
                proposal_net=self.proposal_net,
                features=features,
                num_particles=int(self.config.num_particles),
                latent_dim=latent_dim,
                residual_max=residual_max,
                device=device,
                config=self.config,
            )
        else:
            particles = [fedavg_expert_state]
            coeffs = torch.zeros(1, latent_dim, device=device)
            raw_coeffs = coeffs.clone()
            log_q = torch.zeros(1, device=device)

        particle_stats = _particle_norms_and_mses(particles, previous_global_expert_state, fedavg_expert_state, device=device)
        log_f, log_f_stats = compute_particle_log_f(particles, previous_global_expert_state, sigma_f=self.config.sigma_f, device=device)

        used_reinforce = False
        try:
            log_g, log_g_info = compute_particle_log_g_server_val(
                global_model,
                base_global_state,
                particles,
                val_batches,
                loss_fn,
                device,
                self.config,
                aux_loss_weight,
            )
        except Exception:
            used_reinforce = True
            base_loss, base_confidence = _evaluate_particle_metrics_black_box(global_model, base_global_state, {}, val_batches, loss_fn, device, aux_loss_weight, float(self.config.large_loss))
            particle_metrics = [_evaluate_particle_metrics_black_box(global_model, base_global_state, particle, val_batches, loss_fn, device, aux_loss_weight, float(self.config.large_loss)) for particle in particles]
            particle_losses = [metric[0] for metric in particle_metrics]
            particle_confidences = [metric[1] for metric in particle_metrics]
            particle_loss_tensor = torch.as_tensor(particle_losses, dtype=torch.float32, device=device)
            particle_conf_tensor = torch.as_tensor(particle_confidences, dtype=torch.float32, device=device)
            loss_reward = torch.tensor(base_loss, dtype=torch.float32, device=device) - particle_loss_tensor
            confidence_reward = particle_conf_tensor - torch.tensor(base_confidence, dtype=torch.float32, device=device)
            if self.config.reward_type == "loss_plus_confidence":
                raw_reward_tensor = float(self.config.loss_reward_weight) * loss_reward + float(self.config.confidence_reward_weight) * confidence_reward
            else:
                raw_reward_tensor = loss_reward
            log_g, reward_stats = _reward_to_log_g(raw_reward_tensor, self.config)
            log_g_info = {
                "base_loss": float(base_loss),
                "base_correct_confidence": float(base_confidence),
                "particle_losses": [float(v) for v in particle_losses],
                "particle_confidences": [float(v) for v in particle_confidences],
                "rewards": [float(v) for v in raw_reward_tensor.detach().cpu().tolist()],
                "confidence_rewards": [float(v) for v in confidence_reward.detach().cpu().tolist()],
                "normalized_rewards": list(reward_stats.get("normalized_rewards", [])),
                "differentiable": False,
                "particle_requires_grad": False,
                "all_invalid_particles": False,
                "reward_stats": {
                    **reward_stats,
                    "server_val_particle_loss_mean": float(particle_loss_tensor.detach().mean().item()) if particle_loss_tensor.numel() else float(self.config.large_loss),
                    "particle_correct_confidence_mean": float(particle_conf_tensor.detach().mean().item()) if particle_conf_tensor.numel() else 0.0,
                    "particle_correct_confidence_min": float(particle_conf_tensor.detach().min().item()) if particle_conf_tensor.numel() else 0.0,
                    "particle_correct_confidence_max": float(particle_conf_tensor.detach().max().item()) if particle_conf_tensor.numel() else 0.0,
                    "confidence_reward_mean": float(confidence_reward.detach().mean().item()) if confidence_reward.numel() else 0.0,
                    "confidence_reward_std": float(confidence_reward.detach().std(unbiased=False).item()) if confidence_reward.numel() > 1 else 0.0,
                },
            }

        log_g = torch.nan_to_num(log_g, nan=-10.0, posinf=10.0, neginf=-10.0)
        log_g_requires_grad = bool(log_g.requires_grad)
        particle_requires_grad = bool(log_g_info.get("particle_requires_grad", False))
        differentiable_g = bool(log_g_info.get("differentiable", False) and log_g_requires_grad and particle_requires_grad)
        if bool(log_g_info.get("all_invalid_particles", False)):
            fallback_used = True
            fallback_reason = "all_nan_val_loss"
        if bool(log_g_info.get("reward_stats", {}).get("reward_zero_std", False)) and self.logger is not None:
            self.logger.warning("server reward has near-zero variance; log_g cannot distinguish particles")

        if self.config.use_reward_g and not used_reinforce and self.config.mode != "sanity_fedavg_particle" and not differentiable_g:
            if self.logger is not None:
                self.logger.warning("server validation log_g is not differentiable; proposal cannot receive reparameterized validation gradient")
            if self.config.require_differentiable_g:
                fallback_used = True
                fallback_reason = fallback_reason or "nondifferentiable_g_required"
            elif self.config.use_reinforce_if_not_differentiable:
                used_reinforce = True

        logq_coef = 0.0 if self.config.mode in {"sanity_fedavg_particle", "g_only_particle_selector"} else float(self.config.logq_coef)
        log_f_contribution = torch.zeros_like(log_f) if self.config.mode == "g_only_particle_selector" else log_f
        log_g_contribution = log_g
        log_q_contribution = torch.zeros_like(log_q) if self.config.mode == "g_only_particle_selector" else (-logq_coef * log_q)
        log_w = log_g_contribution if self.config.mode == "g_only_particle_selector" else (log_f_contribution + log_g_contribution + log_q_contribution)
        log_w = torch.clamp(torch.nan_to_num(log_w, nan=_SAFE_LOG_MIN, posinf=_SAFE_LOG_MAX, neginf=_SAFE_LOG_MIN), _SAFE_LOG_MIN, _SAFE_LOG_MAX)
        logZ = torch.logsumexp(log_w, dim=0) - math.log(max(int(log_w.numel()), 1))
        logZ = torch.nan_to_num(logZ, nan=_SAFE_LOG_MIN, posinf=_SAFE_LOG_MAX, neginf=_SAFE_LOG_MIN)
        logZ_requires_grad = bool(logZ.requires_grad)
        differentiable_g = bool(differentiable_g and logZ_requires_grad)
        if self.config.use_reward_g and not used_reinforce and self.config.mode != "sanity_fedavg_particle" and not differentiable_g:
            if self.logger is not None:
                self.logger.warning("server validation log_g is not differentiable; proposal cannot receive reparameterized validation gradient")
            if self.config.require_differentiable_g:
                fallback_used = True
                fallback_reason = fallback_reason or "nondifferentiable_g_required"
            elif self.config.use_reinforce_if_not_differentiable:
                used_reinforce = True

        alpha_vsmc = torch.softmax(log_w.detach(), dim=0) if log_w.numel() > 0 else torch.empty(0, device=device)
        alpha, alpha_stats = self._normalize_alpha(alpha_vsmc) if alpha_vsmc.numel() > 0 else (alpha_vsmc, {
            "alpha_max_before_clip": _NA,
            "alpha_max_after_clip": _NA,
            "normalized_ess_before_clip": _NA,
            "normalized_ess_after_clip": _NA,
            "alpha_entropy": _NA,
            "alpha_uniform_l1": _NA,
            "alpha_mixed_with_uniform": False,
            "bad_alpha_fallback": False,
        })
        if fallback_used and self.config.include_fedavg_particle and alpha.numel() > 0:
            alpha = torch.zeros_like(alpha)
            alpha[0] = 1.0
        if alpha.numel() == 0 or (not torch.isfinite(alpha).all().item()) or alpha.sum().item() <= 0.0:
            fallback_used = True
            fallback_reason = fallback_reason or "bad_alpha"
            alpha = torch.tensor([1.0] + [0.0] * max(len(particles) - 1, 0), device=device, dtype=torch.float32)
        ess = effective_sample_size(alpha.detach().cpu().tolist()) if alpha.numel() > 0 else 0.0
        normalized_ess = ess / max(float(alpha.numel()), 1.0)
        self.state.previous_round_ess[expert_scope] = ess
        if update_history and torch.isfinite(logZ).item():
            self.state.logz_history[expert_scope] = float(self.state.logz_history.get(expert_scope, 0.0)) + float(logZ.detach().item())

        if fallback_used and fallback_reason == "nondifferentiable_g_required":
            proposal_loss = torch.tensor(float("nan"), device=device)
        elif used_reinforce:
            proposal_loss = self._reinforce_objective(expert_scope, log_q=log_q, log_f=log_f_contribution, log_g=log_g_contribution, logq_coef=logq_coef)
        else:
            proposal_loss = -logZ

        raw_rewards = list(log_g_info.get("rewards", []))
        normalized_rewards = list(log_g_info.get("normalized_rewards", []))
        particle_losses = list(log_g_info.get("particle_losses", []))
        particle_confidences = list(log_g_info.get("particle_confidences", []))
        confidence_rewards = list(log_g_info.get("confidence_rewards", []))
        best_particle_index = int(torch.argmax(alpha).item()) if alpha.numel() > 0 else -1
        best_particle_is_fedavg = bool(self.config.include_fedavg_particle and best_particle_index == 0)
        fedavg_particle_alpha = float(alpha[0].item()) if self.config.include_fedavg_particle and alpha.numel() > 0 else _NA
        fedavg_particle_raw_reward = raw_rewards[0] if (self.config.include_fedavg_particle and raw_rewards) else _NA
        fedavg_particle_normalized_reward = normalized_rewards[0] if (self.config.include_fedavg_particle and normalized_rewards) else _NA
        fedavg_particle_log_g = float(log_g[0].detach().item()) if (self.config.include_fedavg_particle and log_g.numel() > 0) else _NA

        log_f_contribution_mean = float(log_f_contribution.detach().mean().item()) if log_f_contribution.numel() else 0.0
        log_g_contribution_mean = float(log_g_contribution.detach().mean().item()) if log_g_contribution.numel() else 0.0
        log_q_contribution_mean = float(log_q_contribution.detach().mean().item()) if log_q_contribution.numel() else 0.0
        log_q_contribution_abs_mean = float(log_q_contribution.detach().abs().mean().item()) if log_q_contribution.numel() else 0.0
        dominates_threshold = 2.0 * max(abs(log_f_contribution_mean), abs(log_g_contribution_mean), 1e-8)
        log_q_dominates = bool(abs(log_q_contribution_mean) > dominates_threshold)
        if not fallback_used and log_q_dominates and self.logger is not None:
            self.logger.warning("log_q still dominates log_w")

        diagnostics = {
            "expert_id": expert_ref if isinstance(expert_ref, int) else _expert_ids_list(expert_ref)[0],
            "expert_scope": expert_scope,
            "num_particles": int(len(particles)),
            "logZ_increment": float(logZ.detach().item()) if torch.isfinite(logZ).item() else float("nan"),
            "logZ_total": float(self.state.logz_history.get(expert_scope, 0.0)),
            "proposal_loss": float(proposal_loss.detach().item()) if torch.is_tensor(proposal_loss) and torch.isfinite(proposal_loss).item() else float("nan"),
            "proposal_grad_norm": _NA,
            "proposal_param_delta_norm": _NA,
            "logZ_requires_grad": bool(logZ_requires_grad),
            "log_f_requires_grad": bool(log_f.requires_grad),
            "log_g_requires_grad": bool(log_g_requires_grad),
            "log_q_requires_grad": bool(log_q.requires_grad),
            "particle_requires_grad": bool(particle_requires_grad),
            "differentiable_g": bool(differentiable_g),
            "log_f_values": [float(v.detach().item()) for v in log_f_contribution],
            "log_g_values": [float(v.detach().item()) for v in log_g_contribution],
            "log_r_values": [float(v.detach().item()) for v in log_q],
            "log_q_values": [float(v.detach().item()) for v in log_q],
            "log_w_values": [float(v.detach().item()) for v in log_w],
            "weights": [float(v) for v in alpha.detach().cpu().tolist()],
            "ess": float(ess),
            "normalized_ess": float(normalized_ess),
            "min_alpha": float(alpha.min().item()) if alpha.numel() else 0.0,
            "max_alpha": float(alpha.max().item()) if alpha.numel() else 0.0,
            "mean_alpha": float(alpha.mean().item()) if alpha.numel() else 0.0,
            "mean_a": float(proposal_stats["mean_a"].detach().item()) if torch.is_tensor(proposal_stats["mean_a"]) else 1.0,
            "std_a": float(torch.exp(0.5 * proposal_stats["logvar_a"]).detach().item()) if torch.is_tensor(proposal_stats["logvar_a"]) else math.exp(0.5 * self.config.init_logvar),
            "mean_b": float(proposal_stats["mean_b"].detach().item()) if torch.is_tensor(proposal_stats["mean_b"]) else 0.0,
            "std_b": float(torch.exp(0.5 * proposal_stats["logvar_b"]).detach().item()) if torch.is_tensor(proposal_stats["logvar_b"]) else math.exp(0.5 * self.config.init_logvar),
            "latent_coeff0_mean": float(proposal_stats["mean_coeffs"][0].detach().item()) if proposal_stats.get("mean_coeffs") is not None and proposal_stats["mean_coeffs"].numel() > 0 else 0.0,
            "latent_coeff0_std": float(torch.exp(0.5 * proposal_stats["logvar_coeffs"][0]).detach().item()) if proposal_stats.get("logvar_coeffs") is not None and proposal_stats["logvar_coeffs"].numel() > 0 else math.exp(0.5 * self.config.init_logvar),
            "latent_coeff1_mean": float(proposal_stats["mean_coeffs"][1].detach().item()) if proposal_stats.get("mean_coeffs") is not None and proposal_stats["mean_coeffs"].numel() > 1 else 0.0,
            "latent_coeff1_std": float(torch.exp(0.5 * proposal_stats["logvar_coeffs"][1]).detach().item()) if proposal_stats.get("logvar_coeffs") is not None and proposal_stats["logvar_coeffs"].numel() > 1 else math.exp(0.5 * self.config.init_logvar),
            "coeff_a_values": [float(v) for v in coeffs[:, 0].detach().cpu().tolist()] if coeffs.numel() else [],
            "coeff_b_values": [float(v) for v in coeffs[:, 1].detach().cpu().tolist()] if coeffs.dim() == 2 and coeffs.shape[1] > 1 else [],
            "latent_coeff0_values": [float(v) for v in coeffs[:, 0].detach().cpu().tolist()] if coeffs.numel() else [],
            "latent_coeff1_values": [float(v) for v in coeffs[:, 1].detach().cpu().tolist()] if coeffs.dim() == 2 and coeffs.shape[1] > 1 else [],
            "latent_dim": latent_dim,
            "use_block_joint": bool(is_block_scope),
            "round_ratio": float(round_ratio),
            "latent_residual_max": float(residual_max),
            "learned_vsmc_mix_fedavg_candidate": float(current_mix_fedavg_candidate),
            "latent_coeff_norm_mean": float(coeffs.detach().norm(dim=-1).mean().item()) if coeffs.numel() else 0.0,
            "server_val_base_loss": float(log_g_info.get("base_loss", float(self.config.large_loss))),
            "server_val_loss_mean": float(sum(particle_losses) / max(len(particle_losses), 1)) if particle_losses else float(self.config.large_loss),
            "server_val_replaced_loss_mean": float(sum(particle_losses) / max(len(particle_losses), 1)) if particle_losses else float(self.config.large_loss),
            "server_val_particle_loss_mean": float(sum(particle_losses) / max(len(particle_losses), 1)) if particle_losses else float(self.config.large_loss),
            "server_val_particle_loss_min": min(particle_losses) if particle_losses else float(self.config.large_loss),
            "server_val_particle_loss_max": max(particle_losses) if particle_losses else float(self.config.large_loss),
            "server_val_reward_mean": float(sum(raw_rewards) / max(len(raw_rewards), 1)) if raw_rewards else 0.0,
            "server_val_reward_std": float(torch.as_tensor(raw_rewards, dtype=torch.float32).std(unbiased=False).item()) if raw_rewards else 0.0,
            "val_reward_mean": float(sum(raw_rewards) / max(len(raw_rewards), 1)) if raw_rewards else 0.0,
            "val_reward_max": max(raw_rewards) if raw_rewards else 0.0,
            "base_correct_confidence": float(log_g_info.get("base_correct_confidence", 0.0)),
            "particle_correct_confidence_mean": float(sum(particle_confidences) / max(len(particle_confidences), 1)) if particle_confidences else 0.0,
            "particle_correct_confidence_min": min(particle_confidences) if particle_confidences else 0.0,
            "particle_correct_confidence_max": max(particle_confidences) if particle_confidences else 0.0,
            "confidence_reward_mean": float(sum(confidence_rewards) / max(len(confidence_rewards), 1)) if confidence_rewards else 0.0,
            "confidence_reward_std": float(torch.as_tensor(confidence_rewards, dtype=torch.float32).std(unbiased=False).item()) if confidence_rewards else 0.0,
            "fedavg_particle_alpha": fedavg_particle_alpha,
            "fedavg_particle_val_reward": fedavg_particle_raw_reward,
            "fedavg_particle_raw_reward": fedavg_particle_raw_reward,
            "fedavg_particle_normalized_reward": fedavg_particle_normalized_reward,
            "fedavg_particle_log_g": fedavg_particle_log_g,
            "best_particle_index": best_particle_index,
            "best_particle_is_fedavg": best_particle_is_fedavg,
            "learned_vsmc_logq_coef": logq_coef,
            "log_f_contribution_mean": log_f_contribution_mean,
            "log_g_contribution_mean": log_g_contribution_mean,
            "log_q_contribution_mean": log_q_contribution_mean,
            "log_q_contribution_abs_mean": log_q_contribution_abs_mean,
            "log_q_dominates": log_q_dominates,
            "log_w_std": float(log_w.detach().std(unbiased=False).item()) if log_w.numel() > 1 else 0.0,
            "alpha_final_ess": float(normalized_ess),
            "alpha_g_server_ess": _NA,
            "alpha_r_ess": _NA,
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "resampled": False,
            "used_reinforce": bool(used_reinforce),
            "bad_alpha_fallback": bool(alpha_stats.get("bad_alpha_fallback", False) or fallback_reason == "bad_alpha"),
            "any_aggregated_expert_nan": False,
            "any_global_nan": False,
            "loss_values": [_extract_metric_for_ids(metrics, ("expert_losses", "expert_loss"), expert_ref, default=metrics.get("val_loss", 0.0)) for metrics in client_metrics],
            "usage_values": [_extract_metric_for_ids(metrics, ("expert_usage", "expert_usages", "usage", "usages"), expert_ref, default=0.0) for metrics in client_metrics],
            "route_counts": [_extract_metric_for_ids(metrics, ("route_counts", "route_count", "expert_counts"), expert_ref, default=0.0) for metrics in client_metrics],
            "log_f_mean": float(log_f_contribution.detach().mean().item()) if log_f_contribution.numel() else 0.0,
            "log_g_mean": float(log_g_contribution.detach().mean().item()) if log_g_contribution.numel() else 0.0,
            "log_q_mean": float(log_q.detach().mean().item()) if log_q.numel() else 0.0,
            "log_r_mean": float(log_q.detach().mean().item()) if log_q.numel() else 0.0,
            "log_w_mean": float(log_w.detach().mean().item()) if log_w.numel() else 0.0,
            **feature_stats,
            **delta_div_stats,
            **particle_stats,
            **log_f_stats,
            **log_g_info.get("reward_stats", {}),
            **alpha_stats,
        }
        return {
            "loss_for_proposal": proposal_loss,
            "weights": alpha,
            "particles": particles,
            "diagnostics": diagnostics,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "fallback_expert_state": fedavg_expert_state,
            "fedavg_expert_state": fedavg_expert_state,
        }

    def aggregate_model_state(self, global_model: nn.Module, global_state: TensorState, client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]], num_experts: int, device: torch.device, val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]], loss_fn: nn.Module, round_idx: int, aux_loss_weight: float = 0.0) -> Tuple[TensorState, List[Dict[str, Any]]]:
        self.state.round_idx = round_idx
        self.proposal_net.to(device)
        self.proposal_net.train()
        val_batches = self._materialize_val_batches(val_loader, device=device)
        original_training = global_model.training
        global_model.eval()

        sample_weights = _sample_count_weights([num_samples for _, num_samples, _ in client_updates])
        fedavg_state = weighted_average_state_dicts([state_dict for state_dict, _, _ in client_updates], sample_weights, fallback_state=global_state)
        fedavg_state = _merge_with_reference_state(global_state, fedavg_state, device=device)
        base_state, global_expert_groups = split_state_dict_by_expert(global_state, num_experts=num_experts)
        fedavg_base_state, fedavg_expert_groups = split_state_dict_by_expert(fedavg_state, num_experts=num_experts)
        client_expert_groups = [split_state_dict_by_expert(state_dict, num_experts=num_experts)[1] for state_dict, _, _ in client_updates]
        if self.config.use_block_joint:
            global_group_map = group_expert_groups_by_block(global_expert_groups)
            fedavg_group_map = group_expert_groups_by_block(fedavg_expert_groups)
            client_group_maps = [group_expert_groups_by_block(group_map) for group_map in client_expert_groups]
        else:
            global_group_map = global_expert_groups
            fedavg_group_map = fedavg_expert_groups
            client_group_maps = client_expert_groups
        ordered_expert_scopes = _ordered_union_keys([global_group_map, fedavg_group_map] + client_group_maps)

        if self.config.mode != "sanity_fedavg_particle" and int(self.config.proposal_train_steps) > 0:
            for _ in range(max(int(self.config.proposal_train_steps), 0)):
                expert_losses = []
                for expert_scope in ordered_expert_scopes:
                    previous_group = global_group_map.get(expert_scope) or fedavg_group_map.get(expert_scope)
                    if previous_group is None:
                        continue
                    client_expert_states = []
                    client_metrics = []
                    sample_counts = []
                    for (_, num_samples, metrics), group_map in zip(client_updates, client_group_maps):
                        if expert_scope not in group_map:
                            continue
                        client_expert_states.append(group_map[expert_scope]["state_dict"])
                        client_metrics.append(metrics)
                        sample_counts.append(num_samples)
                    if not client_expert_states:
                        continue
                    result = self.learned_vsmc_lower_bound_for_expert(global_model, fedavg_state, previous_group["state_dict"], client_expert_states, client_metrics, sample_counts, expert_scope, previous_group.get("expert_ids", previous_group.get("expert_id", 0)), val_batches, loss_fn, device, round_idx, aux_loss_weight, update_history=False)
                    loss = result["loss_for_proposal"]
                    if torch.is_tensor(loss) and torch.isfinite(loss).all().item():
                        expert_losses.append(loss)
                if not expert_losses:
                    self.state.last_grad_norm = 0.0
                    self.state.last_param_delta_norm = 0.0
                    break
                total_loss = torch.stack(expert_losses, dim=0).mean()
                if not torch.isfinite(total_loss).all().item():
                    self.state.last_grad_norm = 0.0
                    self.state.last_param_delta_norm = 0.0
                    break
                proposal_backup = {name: tensor.detach().clone() for name, tensor in self.proposal_net.state_dict().items()}
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = self._compute_proposal_grad_norm()
                self.state.last_grad_norm = grad_norm
                clip_grad_norm_(self.proposal_net.parameters(), max_norm=float(self.config.grad_clip))
                self.optimizer.step()
                param_delta_norm = self._compute_param_delta_norm(proposal_backup, self.proposal_net.state_dict())
                self.state.last_param_delta_norm = param_delta_norm
                if self._proposal_has_non_finite_params():
                    self.proposal_net.load_state_dict(proposal_backup)
                    self.state.last_param_delta_norm = 0.0
                if self.logger is not None and (grad_norm == 0.0 or param_delta_norm == 0.0):
                    self.logger.warning("ProposalNet did not update")

        final_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        round_infos: List[Dict[str, Any]] = []
        self.proposal_net.eval()
        with torch.enable_grad():
            for expert_scope in ordered_expert_scopes:
                previous_group = global_group_map.get(expert_scope) or fedavg_group_map.get(expert_scope)
                if previous_group is None:
                    continue
                client_expert_states = []
                client_metrics = []
                sample_counts = []
                for (_, num_samples, metrics), group_map in zip(client_updates, client_group_maps):
                    if expert_scope not in group_map:
                        continue
                    client_expert_states.append(group_map[expert_scope]["state_dict"])
                    client_metrics.append(metrics)
                    sample_counts.append(num_samples)
                if not client_expert_states:
                    final_expert_groups[expert_scope] = {
                        "expert_id": previous_group.get("expert_ids", previous_group.get("expert_id", 0)),
                        "expert_scope": expert_scope,
                        "state_dict": clone_detached_state_dict(previous_group["state_dict"]),
                    }
                    continue
                result = self.learned_vsmc_lower_bound_for_expert(global_model, fedavg_state, previous_group["state_dict"], client_expert_states, client_metrics, sample_counts, expert_scope, previous_group.get("expert_ids", previous_group.get("expert_id", 0)), val_batches, loss_fn, device, round_idx, aux_loss_weight, update_history=True)
                result["diagnostics"]["proposal_grad_norm"] = float(self.state.last_grad_norm)
                result["diagnostics"]["proposal_param_delta_norm"] = float(self.state.last_param_delta_norm)
                if result["fallback_used"]:
                    aggregated_expert_state = result["fallback_expert_state"]
                else:
                    vsmc_expert_state = weighted_average_state_dicts(result["particles"], result["weights"], fallback_state=previous_group["state_dict"])
                    vsmc_expert_state = _merge_with_reference_state(previous_group["state_dict"], vsmc_expert_state, device=device)
                    aggregated_expert_state = _blend_expert_states(vsmc_expert_state, result["fedavg_expert_state"], mix_b=float(result["diagnostics"].get("learned_vsmc_mix_fedavg_candidate", self.config.mix_fedavg_candidate)), reference_state=previous_group["state_dict"])
                    if _state_has_non_finite(aggregated_expert_state):
                        aggregated_expert_state = result["fallback_expert_state"]
                        result["diagnostics"]["fallback_used"] = True
                        result["diagnostics"]["fallback_reason"] = "nan_aggregated_expert"
                        result["diagnostics"]["any_aggregated_expert_nan"] = True
                final_expert_groups[expert_scope] = {"expert_id": previous_group.get("expert_ids", previous_group.get("expert_id", 0)), "expert_scope": expert_scope, "state_dict": aggregated_expert_state}
                round_infos.append(result["diagnostics"])

        merged_state = merge_expert_states_into_global_state(fedavg_base_state, final_expert_groups)
        merged_state = _merge_with_reference_state(fedavg_state, merged_state, device=device)
        any_global_nan = _state_has_non_finite(merged_state)
        if any_global_nan:
            if self.logger is not None:
                self.logger.error("NaN detected in global model, restored previous global state")
            merged_state = _merge_with_reference_state(global_state, global_state, device=device)
        for info in round_infos:
            info["any_global_nan"] = any_global_nan
        if original_training:
            global_model.train()
        return merged_state, round_infos
