"""
Temporal latent Gaussian expert aggregation for federated Switch Transformer models.

This module introduces a low-dimensional latent-state model for each expert scope.
For expert scope s at round t:

    delta_{i,t}^s = theta_{i,t}^s - theta_{t-1}^s
    u_{i,t}^s     = Phi_s(delta_{i,t}^s)      in R^d
    z_t^s         = latent global expert update in R^d

Transition model:
    z_t^s ~ N(rho * z_{t-1}^s, Q)

Observation model:
    u_{i,t}^s ~ N(z_t^s, R_{i,t}^s)

The server fuses the client observations with a diagonal Gaussian update, then
turns observation-posterior compatibility into expert aggregation weights:

    alpha_{i,s,t} propto exp( -0.5 * ||u_{i,t}^s - z_t^s||^2 / sigma_{i,t}^{2,s} )

This is a first practical approximation to a temporal latent VSMC-style expert
aggregator. It keeps server-side aggregation deterministic and lightweight while
using cross-round latent memory more structurally than a heuristic score.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vsmc_expert_aggregator import (
    TensorState,
    _clone_detached_tensor,
    _extract_expert_metric,
    _extract_metric,
    _merge_with_reference_state,
    _min_max_normalize,
    _ordered_union_keys,
    _sample_count_weights,
    effective_sample_size,
    flatten_state_dict_l2_distance,
    merge_expert_states_into_model,
    safe_normalize_log_weights,
    split_model_state_by_expert,
    weighted_average_state_dicts,
)


@dataclass
class TemporalLatentAggregatorConfig:
    temperature: float = 1.0
    loss_beta: float = 0.5
    usage_beta: float = 0.0
    drift_beta: float = 0.05
    route_beta: float = 0.2
    target_usage: Optional[float] = None
    min_route_count: int = 1
    fallback_to_fedavg_on_nan: bool = True
    clamp_log_weight: float = 30.0
    latent_dim: int = 16
    transition_rho: float = 0.9
    process_noise: float = 0.1
    obs_noise_base: float = 1.0
    loss_noise_beta: float = 1.0
    route_noise_beta: float = 0.2
    sample_noise_beta: float = 0.2
    warmup_rounds: int = 3
    projection_seed: int = 0
    fedavg_mix: float = 0.0
    min_weight: float = 0.0
    debug: bool = False


@dataclass
class LatentExpertMemory:
    projection_matrix: torch.Tensor
    z_mean: torch.Tensor
    z_cov_diag: torch.Tensor
    flat_keys: List[str]
    last_round: int = 0


def _floating_flat_keys(state_dict: TensorState) -> List[str]:
    return [
        key
        for key in state_dict.keys()
        if torch.is_floating_point(state_dict[key].detach())
    ]


def flatten_state_dict_to_vector(state_dict: TensorState, flat_keys: Sequence[str]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for key in flat_keys:
        if key not in state_dict:
            chunks.append(torch.zeros(0, dtype=torch.float64))
            continue
        tensor = state_dict[key].detach()
        if not torch.is_floating_point(tensor):
            continue
        chunks.append(tensor.to(dtype=torch.float64, device="cpu").reshape(-1))
    if not chunks:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(chunks, dim=0)


def _build_projection_matrix(
    latent_dim: int,
    input_dim: int,
    seed: int,
    expert_scope: str,
) -> torch.Tensor:
    if input_dim <= 0:
        return torch.zeros((int(latent_dim), 0), dtype=torch.float64)
    scope_seed = abs(hash(expert_scope)) % (2**31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(scope_seed))
    matrix = torch.randn(int(latent_dim), int(input_dim), generator=generator, dtype=torch.float64)
    matrix = matrix / math.sqrt(float(input_dim))
    return matrix


def _project_delta(delta_vector: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
    if delta_vector.numel() == 0 or projection_matrix.numel() == 0:
        return torch.zeros(projection_matrix.shape[0], dtype=torch.float64)
    return projection_matrix @ delta_vector


def _kalman_fuse_diagonal(
    prior_mean: torch.Tensor,
    prior_cov_diag: torch.Tensor,
    observations: Sequence[torch.Tensor],
    observation_vars: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = prior_mean.clone()
    cov = prior_cov_diag.clone()
    for obs, obs_var in zip(observations, observation_vars):
        variance = max(float(obs_var), 1e-6)
        innovation = obs - mean
        gain = cov / (cov + variance)
        mean = mean + gain * innovation
        cov = (1.0 - gain) * cov
    return mean, cov.clamp_min(1e-9)


def _compute_observation_variances(
    loss_values: Sequence[float],
    route_counts: Sequence[float],
    sample_counts: Sequence[int],
    config: TemporalLatentAggregatorConfig,
) -> List[float]:
    normalized_losses = _min_max_normalize(loss_values)
    variances: List[float] = []
    for idx, (route_count, sample_count) in enumerate(zip(route_counts, sample_counts)):
        normalized_loss = float(normalized_losses[idx].item()) if normalized_losses.numel() else 0.0
        route_bonus = config.route_noise_beta * math.log(max(float(route_count), 0.0) + 1.0)
        sample_bonus = config.sample_noise_beta * math.log(max(int(sample_count), 0) + 1.0)
        log_variance = (
            math.log(max(float(config.obs_noise_base), 1e-6))
            + config.loss_noise_beta * normalized_loss
            - route_bonus
            - sample_bonus
        )
        variances.append(max(math.exp(log_variance), 1e-6))
    return variances


def _stabilize_latent_weights(
    latent_weights: torch.Tensor,
    fedavg_weights: torch.Tensor,
    mix: float,
    min_weight: float,
) -> torch.Tensor:
    if latent_weights.numel() == 0:
        return latent_weights

    stabilized = latent_weights.to(dtype=torch.float64)
    fedavg = fedavg_weights.to(dtype=torch.float64)

    mix_value = float(max(0.0, min(1.0, mix)))
    if fedavg.numel() == stabilized.numel() and mix_value > 0.0:
        stabilized = (1.0 - mix_value) * stabilized + mix_value * fedavg

    floor_value = float(max(0.0, min_weight))
    if floor_value > 0.0:
        max_floor = 1.0 / max(int(stabilized.numel()), 1)
        floor_value = min(floor_value, max_floor)
        low_mask = stabilized < floor_value
        if torch.any(low_mask):
            stabilized = stabilized.clone()
            stabilized[low_mask] = floor_value
            remaining_mass = 1.0 - floor_value * float(low_mask.sum().item())
            high_mask = ~low_mask
            if torch.any(high_mask):
                high_values = stabilized[high_mask]
                high_sum = high_values.sum()
                if torch.isfinite(high_sum) and high_sum.item() > 0.0:
                    stabilized[high_mask] = high_values / high_sum * remaining_mass
                else:
                    stabilized[high_mask] = remaining_mass / float(high_mask.sum().item())
            total_after_floor = stabilized.sum()
            if torch.isfinite(total_after_floor) and total_after_floor.item() > 0.0:
                stabilized = stabilized / total_after_floor

    total = stabilized.sum()
    if not torch.isfinite(total) or total.item() <= 0.0:
        return torch.full_like(stabilized, 1.0 / stabilized.numel())
    return stabilized / total


def _compute_usage_penalty(usage_value: float, target_usage: float) -> float:
    return abs(float(usage_value) - float(target_usage))


def _compute_latent_log_weight(
    latent_distance: float,
    obs_variance: float,
    normalized_drift: float,
    usage_penalty: float,
    route_count: float,
    latent_dim: int,
    config: TemporalLatentAggregatorConfig,
) -> float:
    if not all(
        math.isfinite(float(value))
        for value in (latent_distance, obs_variance, normalized_drift, usage_penalty, route_count)
    ):
        return float("nan")

    route_term = config.route_beta * math.log(max(float(route_count), 0.0) + 1.0)
    if float(route_count) < float(config.min_route_count):
        route_term -= config.route_beta * math.log(float(config.min_route_count) + 1.0)

    likelihood_term = -0.5 * (
        float(latent_distance) / max(float(obs_variance), 1e-6)
        + int(latent_dim) * math.log(max(float(obs_variance), 1e-6))
    )
    log_weight = (
        likelihood_term
        - config.drift_beta * float(normalized_drift)
        - config.usage_beta * float(usage_penalty)
        + route_term
    )
    clamp_value = abs(float(config.clamp_log_weight))
    return max(-clamp_value, min(clamp_value, log_weight))


def _initialize_memory(
    reference_state: TensorState,
    config: TemporalLatentAggregatorConfig,
    expert_scope: str,
) -> LatentExpertMemory:
    flat_keys = _floating_flat_keys(reference_state)
    flat_vector = flatten_state_dict_to_vector(reference_state, flat_keys)
    projection_matrix = _build_projection_matrix(
        latent_dim=config.latent_dim,
        input_dim=int(flat_vector.numel()),
        seed=config.projection_seed,
        expert_scope=expert_scope,
    )
    latent_dim = int(projection_matrix.shape[0])
    return LatentExpertMemory(
        projection_matrix=projection_matrix,
        z_mean=torch.zeros(latent_dim, dtype=torch.float64),
        z_cov_diag=torch.ones(latent_dim, dtype=torch.float64),
        flat_keys=flat_keys,
        last_round=0,
    )


def aggregate_expert_particles_temporal_latent(
    global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_sample_counts: Sequence[int],
    client_metrics: Sequence[Dict[str, Any]],
    expert_id: int,
    expert_scope: str,
    num_experts: int,
    config: TemporalLatentAggregatorConfig,
    round_idx: int,
    latent_memory: Optional[LatentExpertMemory] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TensorState, Dict[str, Any], LatentExpertMemory]:
    info: Dict[str, Any] = {
        "weights": [],
        "pre_resample_weights": [],
        "raw_latent_weights": [],
        "fedavg_mixture_weights": [],
        "log_weights": [],
        "ess": 0.0,
        "normalized_ess": 0.0,
        "loss_values": [],
        "usage_values": [],
        "drift_values": [],
        "route_counts": [],
        "latent_distance_values": [],
        "obs_noise_values": [],
        "latent_enabled": False,
        "posterior_cov_trace": 0.0,
        "prior_cov_trace": 0.0,
        "latent_dim": int(config.latent_dim),
        "fedavg_mix": float(config.fedavg_mix),
        "latent_min_weight": float(config.min_weight),
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
        memory = latent_memory or _initialize_memory(reference_state, config, expert_scope)
        return reference_state, info, memory

    if len(client_expert_states) != len(client_sample_counts) or len(client_expert_states) != len(client_metrics):
        raise ValueError("client_expert_states, client_sample_counts, and client_metrics must be aligned")

    target_usage = config.target_usage if config.target_usage is not None else 1.0 / max(int(num_experts), 1)
    memory = latent_memory or _initialize_memory(reference_state, config, expert_scope)
    flat_keys = memory.flat_keys
    reference_vector = flatten_state_dict_to_vector(reference_state, flat_keys)
    client_delta_vectors: List[torch.Tensor] = []
    client_latent_vectors: List[torch.Tensor] = []
    loss_values: List[float] = []
    usage_values: List[float] = []
    drift_values: List[float] = []
    route_counts: List[float] = []

    for state_dict, sample_count, metrics in zip(client_expert_states, client_sample_counts, client_metrics):
        del sample_count
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
        delta_vector = flatten_state_dict_to_vector(state_dict, flat_keys) - reference_vector
        latent_vector = _project_delta(delta_vector, memory.projection_matrix)

        loss_values.append(float(loss_value))
        usage_values.append(float(usage_value))
        drift_values.append(float(drift_value))
        route_counts.append(float(route_count))
        client_delta_vectors.append(delta_vector)
        client_latent_vectors.append(latent_vector)

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
        aggregated_state = weighted_average_state_dicts(client_expert_states, [1.0])
        updated_delta = client_delta_vectors[0]
        updated_latent = client_latent_vectors[0]
        updated_memory = LatentExpertMemory(
            projection_matrix=memory.projection_matrix,
            z_mean=updated_latent.clone(),
            z_cov_diag=torch.full_like(updated_latent, max(float(config.process_noise), 1e-6) ** 2),
            flat_keys=flat_keys,
            last_round=int(round_idx),
        )
        return _merge_with_reference_state(reference_state, aggregated_state, device=device), info, updated_memory

    observation_variances = _compute_observation_variances(
        loss_values=loss_values,
        route_counts=route_counts,
        sample_counts=client_sample_counts,
        config=config,
    )
    normalized_drifts = _min_max_normalize(drift_values)

    prior_mean = float(config.transition_rho) * memory.z_mean
    prior_cov_diag = (float(config.transition_rho) ** 2) * memory.z_cov_diag + float(config.process_noise) ** 2
    info["prior_cov_trace"] = float(prior_cov_diag.sum().item())

    latent_enabled = round_idx > int(config.warmup_rounds)
    info["latent_enabled"] = bool(latent_enabled)

    if latent_enabled:
        posterior_mean, posterior_cov_diag = _kalman_fuse_diagonal(
            prior_mean=prior_mean,
            prior_cov_diag=prior_cov_diag,
            observations=client_latent_vectors,
            observation_vars=observation_variances,
        )
    else:
        posterior_mean = torch.stack(client_latent_vectors, dim=0).mean(dim=0)
        posterior_cov_diag = torch.full_like(prior_cov_diag, float(config.process_noise) ** 2 + 1e-6)

    info["posterior_cov_trace"] = float(posterior_cov_diag.sum().item())
    info["obs_noise_values"] = [float(value) for value in observation_variances]

    log_weights: List[float] = []
    latent_distance_values: List[float] = []
    latent_dim = int(posterior_mean.numel())
    for idx, latent_vector in enumerate(client_latent_vectors):
        diff = latent_vector - posterior_mean
        latent_distance = float(torch.dot(diff, diff).item())
        latent_distance_values.append(latent_distance)
        usage_penalty = _compute_usage_penalty(usage_values[idx], target_usage)
        log_weight = _compute_latent_log_weight(
            latent_distance=latent_distance,
            obs_variance=observation_variances[idx],
            normalized_drift=float(normalized_drifts[idx].item()) if normalized_drifts.numel() else 0.0,
            usage_penalty=usage_penalty,
            route_count=route_counts[idx],
            latent_dim=latent_dim,
            config=config,
        )
        log_weights.append(log_weight)

    info["latent_distance_values"] = latent_distance_values
    info["log_weights"] = log_weights
    normalized_weights = safe_normalize_log_weights(
        torch.as_tensor(log_weights, dtype=torch.float64) / max(float(config.temperature), 1e-6)
    )

    def _fallback_aggregate() -> Tuple[TensorState, Dict[str, Any], LatentExpertMemory]:
        fallback_weights = _sample_count_weights(client_sample_counts)
        fallback_state = weighted_average_state_dicts(client_expert_states, fallback_weights.tolist())
        fallback_info = dict(info)
        fallback_info["weights"] = fallback_weights.tolist()
        fallback_info["pre_resample_weights"] = fallback_weights.tolist()
        fallback_info["raw_latent_weights"] = fallback_weights.tolist()
        fallback_info["fedavg_mixture_weights"] = fallback_weights.tolist()
        fallback_info["fallback_used"] = True
        fallback_info["ess"] = effective_sample_size(fallback_weights)
        fallback_info["normalized_ess"] = fallback_info["ess"] / max(float(len(client_expert_states)), 1.0)
        updated_memory = LatentExpertMemory(
            projection_matrix=memory.projection_matrix,
            z_mean=posterior_mean.clone(),
            z_cov_diag=posterior_cov_diag.clone(),
            flat_keys=flat_keys,
            last_round=int(round_idx),
        )
        return _merge_with_reference_state(reference_state, fallback_state, device=device), fallback_info, updated_memory

    if normalized_weights.numel() == 0 or not torch.isfinite(normalized_weights).all():
        if config.fallback_to_fedavg_on_nan:
            return _fallback_aggregate()
        raise ValueError("Failed to normalize temporal latent expert log weights")

    fedavg_weights = _sample_count_weights(client_sample_counts)
    stabilized_weights = _stabilize_latent_weights(
        latent_weights=normalized_weights,
        fedavg_weights=fedavg_weights,
        mix=float(config.fedavg_mix),
        min_weight=float(config.min_weight),
    )

    info["raw_latent_weights"] = normalized_weights.tolist()
    info["fedavg_mixture_weights"] = stabilized_weights.tolist()
    info["pre_resample_weights"] = normalized_weights.tolist()
    info["weights"] = stabilized_weights.tolist()
    info["ess"] = effective_sample_size(stabilized_weights)
    info["normalized_ess"] = info["ess"] / max(float(len(client_expert_states)), 1.0)

    aggregated_state = weighted_average_state_dicts(client_expert_states, stabilized_weights.tolist())
    updated_memory = LatentExpertMemory(
        projection_matrix=memory.projection_matrix,
        z_mean=posterior_mean.clone(),
        z_cov_diag=posterior_cov_diag.clone(),
        flat_keys=flat_keys,
        last_round=int(round_idx),
    )
    return _merge_with_reference_state(reference_state, aggregated_state, device=device), info, updated_memory


class TemporalLatentExpertAggregator:
    def __init__(self, config: TemporalLatentAggregatorConfig, logger: Optional[Any] = None):
        self.config = config
        self.logger = logger

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        round_idx: int,
        latent_memory: Optional[Dict[str, LatentExpertMemory]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, List[Dict[str, Any]], Dict[str, LatentExpertMemory]]:
        if len(client_updates) == 0:
            raise ValueError("client_updates must not be empty")

        latent_memory = latent_memory or {}
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
        updated_memory: Dict[str, LatentExpertMemory] = dict(latent_memory)

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

            aggregated_expert_state, info, group_memory = aggregate_expert_particles_temporal_latent(
                global_expert_state=reference_expert_state,
                client_expert_states=client_expert_states,
                client_sample_counts=client_sample_counts,
                client_metrics=client_metrics,
                expert_id=expert_id,
                expert_scope=expert_group_name,
                num_experts=num_experts,
                config=self.config,
                round_idx=round_idx,
                latent_memory=latent_memory.get(expert_group_name),
                device=device,
            )

            merged_expert_groups[expert_group_name] = {
                "expert_id": expert_id,
                "state_dict": aggregated_expert_state,
            }
            updated_memory[expert_group_name] = group_memory
            info["expert_id"] = expert_id
            info["expert_scope"] = expert_group_name
            round_infos.append(info)

        merged_state = merge_expert_states_into_model(base_state, merged_expert_groups)
        merged_state = _merge_with_reference_state(global_state, merged_state, device=device)
        return merged_state, round_infos, updated_memory
