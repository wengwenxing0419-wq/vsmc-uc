"""
Full-VSMC aggregation path for federated expert aggregation.

This module moves beyond the heuristic baseline by explicitly separating:

    log p(z_t | z_{t-1})      prior term
  + log p(y_t | z_t)          observation term
  - log q(z_t | z_{t-1}, y_t) proposal term

for client-provided latent particles at each communication round. The current
implementation is still a practical approximation for the federated setting,
but it now follows a much closer importance-ratio structure than the baseline
heuristic VSMC weights.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from temporal_latent_expert_aggregator import (
    LatentExpertMemory,
    _compute_observation_variances,
    _compute_usage_penalty,
    _initialize_memory,
    _kalman_fuse_diagonal,
    _project_delta,
    _stabilize_latent_weights,
    flatten_state_dict_to_vector,
)
from vsmc_expert_aggregator import (
    TensorState,
    _maybe_add_resampling_noise,
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
    stratified_resampling,
    weighted_average_state_dicts,
)


@dataclass
class FullVSMCAggregatorConfig:
    temperature: float = 1.0
    loss_beta: float = 0.5
    usage_beta: float = 0.0
    drift_beta: float = 0.05
    route_beta: float = 0.2
    target_usage: Optional[float] = None
    min_route_count: int = 1
    ess_threshold: float = 0.5
    use_resampling: bool = False
    resampling_noise_std: float = 0.0
    fallback_to_fedavg_on_nan: bool = True
    fallback_to_fedavg_on_degenerate: bool = False
    clamp_log_weight: float = 30.0
    degeneration_ess_threshold: float = 0.2
    degeneration_temperature_scale: float = 2.0
    latent_dim: int = 16
    transition_rho: float = 0.9
    process_noise: float = 0.1
    obs_noise_base: float = 1.0
    loss_noise_beta: float = 1.0
    route_noise_beta: float = 0.2
    sample_noise_beta: float = 0.2
    proposal_mode: str = "posterior_diag"
    proposal_blend: float = 0.5
    proposal_var_scale: float = 1.0
    use_proposal_particles: bool = False
    num_particles: int = 8
    responsibility_temperature: float = 1.0
    particle_temperature: float = 1.0
    metric_ema_beta: float = 0.8
    responsibility_floor: float = 0.0
    prev_weight_decay: float = 1.0
    warmup_rounds: int = 3
    projection_seed: int = 0
    fedavg_mix: float = 0.0
    min_weight: float = 0.0
    debug: bool = False


@dataclass
class FullVSMCExpertMemory:
    latent_memory: LatentExpertMemory
    particle_log_weights: Optional[torch.Tensor] = None
    particle_latents: Optional[torch.Tensor] = None
    metric_ema_by_client: Dict[int, Dict[str, float]] = field(default_factory=dict)


def _gaussian_diag_log_prob(x: torch.Tensor, mean: torch.Tensor, cov_diag: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    variance = cov_diag.to(dtype=torch.float64).clamp_min(1e-9)
    diff = x.to(dtype=torch.float64) - mean.to(dtype=torch.float64)
    dim = int(diff.numel())
    quadratic = torch.sum(diff.pow(2) / variance)
    log_det = torch.sum(torch.log(variance))
    value = -0.5 * (quadratic + log_det + dim * math.log(2.0 * math.pi))
    return float(value.item())


def _gaussian_isotropic_log_prob(x: torch.Tensor, mean: torch.Tensor, variance: float) -> float:
    if x.numel() == 0:
        return 0.0
    var = max(float(variance), 1e-9)
    diff = x.to(dtype=torch.float64) - mean.to(dtype=torch.float64)
    dim = int(diff.numel())
    quadratic = torch.sum(diff.pow(2)).item() / var
    return float(-0.5 * (quadratic + dim * math.log(var) + dim * math.log(2.0 * math.pi)))


def _route_log_bonus(route_count: float, config: FullVSMCAggregatorConfig) -> float:
    bonus = float(config.route_beta) * math.log(max(float(route_count), 0.0) + 1.0)
    if float(route_count) < float(config.min_route_count):
        bonus -= float(config.route_beta) * math.log(float(config.min_route_count) + 1.0)
    return bonus


def _sample_diag_gaussian_particles(
    mean: torch.Tensor,
    cov_diag: torch.Tensor,
    num_particles: int,
    seed: int,
) -> torch.Tensor:
    particle_count = max(int(num_particles), 1)
    if mean.numel() == 0:
        return torch.zeros((particle_count, 0), dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    std = cov_diag.to(dtype=torch.float64).clamp_min(1e-9).sqrt()
    eps = torch.randn((particle_count, mean.numel()), generator=generator, dtype=torch.float64)
    return mean.to(dtype=torch.float64).unsqueeze(0) + eps * std.unsqueeze(0)


def _client_identifier(metrics: Dict[str, Any], fallback_idx: int) -> int:
    if metrics is not None and "client_id" in metrics:
        return int(metrics["client_id"])
    return int(fallback_idx)


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
    base_state, expert_states = split_model_state_by_expert(state_dict, num_experts=num_experts)
    remaining_base: TensorState = collections.OrderedDict()
    router_states: Dict[str, Dict[str, Any]] = collections.OrderedDict()

    for name, tensor in base_state.items():
        router_group = _get_router_group_from_name(name)
        if router_group is not None:
            if router_group not in router_states:
                router_states[router_group] = {
                    "router_scope": router_group,
                    "state_dict": collections.OrderedDict(),
                }
            router_states[router_group]["state_dict"][name] = tensor
        else:
            remaining_base[name] = tensor

    return remaining_base, expert_states, router_states


def merge_named_groups_into_model(
    base_state: TensorState,
    named_groups: Dict[str, Dict[str, Any]],
) -> TensorState:
    merged_state: TensorState = collections.OrderedDict(
        (name, tensor.clone()) for name, tensor in base_state.items()
    )
    for group in named_groups.values():
        for name, tensor in group["state_dict"].items():
            merged_state[name] = tensor.clone()
    return merged_state


def _build_full_vsmc_proposal(
    *,
    prior_mean: torch.Tensor,
    prior_cov_diag: torch.Tensor,
    observations: Sequence[torch.Tensor],
    observation_variances: Sequence[float],
    config: FullVSMCAggregatorConfig,
    latent_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    mode = str(config.proposal_mode)
    if (not latent_enabled) or len(observations) == 0:
        fallback_mean = torch.stack(observations, dim=0).mean(dim=0) if observations else prior_mean.clone()
        fallback_cov = torch.full_like(
            prior_cov_diag,
            max(float(config.process_noise), 1e-6) ** 2 * max(float(config.proposal_var_scale), 1e-6),
        )
        return fallback_mean, fallback_cov, "warmup_empirical"

    if mode == "prior_only":
        return (
            prior_mean.clone(),
            prior_cov_diag.clone() * max(float(config.proposal_var_scale), 1e-6),
            "prior_only",
        )

    posterior_mean, posterior_cov_diag = _kalman_fuse_diagonal(
        prior_mean=prior_mean,
        prior_cov_diag=prior_cov_diag,
        observations=observations,
        observation_vars=observation_variances,
    )
    posterior_cov_diag = posterior_cov_diag * max(float(config.proposal_var_scale), 1e-6)

    if mode == "posterior_diag":
        return posterior_mean, posterior_cov_diag, "posterior_diag"

    if mode == "blend":
        obs_mean = torch.stack(observations, dim=0).mean(dim=0)
        blend = min(max(float(config.proposal_blend), 0.0), 1.0)
        blended_mean = (1.0 - blend) * prior_mean + blend * obs_mean
        blended_cov = ((1.0 - blend) * prior_cov_diag + blend * posterior_cov_diag).clamp_min(1e-9)
        return blended_mean, blended_cov, "blend"

    raise ValueError(f"Unsupported full_vsmc proposal_mode: {mode}")


def aggregate_expert_particles_full_vsmc(
    global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_sample_counts: Sequence[int],
    client_metrics: Sequence[Dict[str, Any]],
    expert_id: int,
    expert_scope: str,
    num_experts: int,
    config: FullVSMCAggregatorConfig,
    round_idx: int,
    latent_memory: Optional[FullVSMCExpertMemory] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TensorState, Dict[str, Any], FullVSMCExpertMemory]:
    info: Dict[str, Any] = {
        "weights": [],
        "pre_resample_weights": [],
        "particle_weights": [],
        "particle_state_weights": [],
        "log_weights": [],
        "incremental_log_weights": [],
        "previous_log_weights": [],
        "log_weight_totals": [],
        "prior_log_probs": [],
        "observation_log_probs": [],
        "proposal_log_probs": [],
        "aux_log_scores": [],
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
        "proposal_cov_trace": 0.0,
        "proposal_mode": str(config.proposal_mode),
        "proposal_source": "",
        "proposal_particle_count": 0,
        "proposal_particle_mode": "client_latent",
        "particle_aggregation_mode": "client_weight_projection",
        "particle_temperature": float(config.particle_temperature),
        "metric_ema_beta": float(config.metric_ema_beta),
        "responsibility_floor": float(config.responsibility_floor),
        "prev_weight_decay": float(config.prev_weight_decay),
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
        latent_state = latent_memory.latent_memory if latent_memory is not None else _initialize_memory(
            reference_state, config, expert_scope
        )
        return reference_state, info, FullVSMCExpertMemory(latent_memory=latent_state)

    if len(client_expert_states) != len(client_sample_counts) or len(client_expert_states) != len(client_metrics):
        raise ValueError("client_expert_states, client_sample_counts, and client_metrics must be aligned")

    target_usage = config.target_usage if config.target_usage is not None else 1.0 / max(int(num_experts), 1)
    memory = latent_memory.latent_memory if latent_memory is not None else _initialize_memory(
        reference_state, config, expert_scope
    )
    previous_log_weights: Optional[torch.Tensor] = None
    if latent_memory is not None and latent_memory.particle_log_weights is not None:
        previous_log_weights = latent_memory.particle_log_weights.to(dtype=torch.float64).clone()
    previous_metric_ema_by_client = (
        dict(latent_memory.metric_ema_by_client) if latent_memory is not None else {}
    )
    updated_metric_ema_by_client: Dict[int, Dict[str, float]] = dict(previous_metric_ema_by_client)
    flat_keys = memory.flat_keys
    reference_vector = flatten_state_dict_to_vector(reference_state, flat_keys)

    client_latent_vectors: List[torch.Tensor] = []
    client_delta_vectors: List[torch.Tensor] = []
    loss_values: List[float] = []
    usage_values: List[float] = []
    drift_values: List[float] = []
    route_counts: List[float] = []

    ema_beta = min(max(float(config.metric_ema_beta), 0.0), 0.999)
    for fallback_idx, (state_dict, sample_count, metrics) in enumerate(
        zip(client_expert_states, client_sample_counts, client_metrics)
    ):
        del sample_count
        metrics = metrics or {}
        client_id = _client_identifier(metrics, fallback_idx)
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

        previous_metric_state = previous_metric_ema_by_client.get(client_id, {})
        smoothed_metrics = {
            "loss": float(loss_value),
            "usage": float(usage_value),
            "route_count": float(route_count),
            "drift": float(drift_value),
        }
        if previous_metric_state:
            smoothed_metrics = {
                key: ema_beta * float(previous_metric_state.get(key, value)) + (1.0 - ema_beta) * float(value)
                for key, value in smoothed_metrics.items()
            }
        updated_metric_ema_by_client[client_id] = smoothed_metrics

        loss_values.append(smoothed_metrics["loss"])
        usage_values.append(smoothed_metrics["usage"])
        drift_values.append(smoothed_metrics["drift"])
        route_counts.append(smoothed_metrics["route_count"])
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
        info["incremental_log_weights"] = [0.0]
        info["log_weight_totals"] = [0.0]
        info["ess"] = 1.0
        info["normalized_ess"] = 1.0
        aggregated_state = weighted_average_state_dicts(client_expert_states, [1.0])
        updated_latent = client_latent_vectors[0]
        updated_memory = FullVSMCExpertMemory(
            latent_memory=LatentExpertMemory(
                projection_matrix=memory.projection_matrix,
                z_mean=updated_latent.clone(),
                z_cov_diag=torch.full_like(updated_latent, max(float(config.process_noise), 1e-6) ** 2),
                flat_keys=flat_keys,
                last_round=int(round_idx),
            ),
            particle_log_weights=torch.zeros(1, dtype=torch.float64),
            particle_latents=updated_latent.unsqueeze(0).clone(),
            metric_ema_by_client=updated_metric_ema_by_client,
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

    proposal_mean, proposal_cov_diag, proposal_source = _build_full_vsmc_proposal(
        prior_mean=prior_mean,
        prior_cov_diag=prior_cov_diag,
        observations=client_latent_vectors,
        observation_variances=observation_variances,
        config=config,
        latent_enabled=latent_enabled,
    )

    info["posterior_cov_trace"] = float(proposal_cov_diag.sum().item())
    info["proposal_cov_trace"] = float(proposal_cov_diag.sum().item())
    info["proposal_source"] = proposal_source
    info["obs_noise_values"] = [float(value) for value in observation_variances]

    client_aux_scores: List[float] = []
    for idx in range(len(client_latent_vectors)):
        usage_penalty = _compute_usage_penalty(usage_values[idx], target_usage)
        drift_penalty = float(normalized_drifts[idx].item()) if normalized_drifts.numel() else 0.0
        client_aux_scores.append(
            - float(config.drift_beta) * drift_penalty
            - float(config.usage_beta) * usage_penalty
            + _route_log_bonus(route_counts[idx], config)
        )

    use_proposal_particles = bool(config.use_proposal_particles)
    particle_latents: List[torch.Tensor]
    if use_proposal_particles:
        particle_seed = int(round_idx) * 10007 + abs(hash(expert_scope)) % 10007 + int(expert_id)
        sampled_particles = _sample_diag_gaussian_particles(
            mean=proposal_mean,
            cov_diag=proposal_cov_diag,
            num_particles=int(config.num_particles),
            seed=particle_seed,
        )
        particle_latents = [sampled_particles[idx] for idx in range(sampled_particles.shape[0])]
        info["proposal_particle_count"] = int(sampled_particles.shape[0])
        info["proposal_particle_mode"] = "sampled_proposal"
    else:
        particle_latents = client_latent_vectors
        info["proposal_particle_count"] = len(client_latent_vectors)
        info["proposal_particle_mode"] = "client_latent"

    prior_log_probs: List[float] = []
    observation_log_probs: List[float] = []
    proposal_log_probs: List[float] = []
    aux_log_scores: List[float] = []
    previous_log_weight_values: List[float] = []
    total_log_weights: List[float] = []
    log_weights: List[float] = []
    latent_distance_values: List[float] = []
    latent_dim = int(proposal_mean.numel())
    responsibility_rows: List[torch.Tensor] = []

    for idx, particle_latent in enumerate(particle_latents):
        prior_log_prob = _gaussian_diag_log_prob(particle_latent, prior_mean, prior_cov_diag)
        obs_log_prob = 0.0
        responsibility_logits: List[float] = []
        for obs_latent, obs_var in zip(client_latent_vectors, observation_variances):
            log_prob = _gaussian_isotropic_log_prob(obs_latent, particle_latent, obs_var)
            obs_log_prob += log_prob
            responsibility_logits.append(log_prob)
        obs_log_prob /= max(len(client_latent_vectors), 1)

        responsibility_row = safe_normalize_log_weights(
            torch.as_tensor(responsibility_logits, dtype=torch.float64)
            / max(float(config.responsibility_temperature), 1e-6)
        )
        if responsibility_row.numel() == 0 or not torch.isfinite(responsibility_row).all():
            responsibility_row = torch.full(
                (len(client_latent_vectors),),
                1.0 / max(len(client_latent_vectors), 1),
                dtype=torch.float64,
            )
        responsibility_floor = max(float(config.responsibility_floor), 0.0)
        if responsibility_floor > 0.0 and responsibility_row.numel() > 0:
            max_floor = (1.0 - 1e-6) / max(int(responsibility_row.numel()), 1)
            applied_floor = min(responsibility_floor, max_floor)
            responsibility_row = responsibility_row * (1.0 - applied_floor * responsibility_row.numel()) + applied_floor
            responsibility_row = responsibility_row / responsibility_row.sum().clamp_min(1e-12)
        responsibility_rows.append(responsibility_row)

        proposal_log_prob = _gaussian_diag_log_prob(particle_latent, proposal_mean, proposal_cov_diag)
        latent_distance = float(torch.sum((particle_latent - proposal_mean).pow(2)).item())
        latent_distance_values.append(latent_distance)

        aux_log_score = float(torch.dot(responsibility_row, torch.as_tensor(client_aux_scores, dtype=torch.float64)).item())
        inc_log_weight = prior_log_prob + obs_log_prob - proposal_log_prob + aux_log_score
        prev_log_weight = 0.0
        if previous_log_weights is not None and idx < int(previous_log_weights.numel()):
            candidate = float(previous_log_weights[idx].item())
            prev_log_weight = candidate if math.isfinite(candidate) else 0.0
        log_weight = float(config.prev_weight_decay) * prev_log_weight + inc_log_weight
        clamp_value = abs(float(config.clamp_log_weight))
        log_weight = max(-clamp_value, min(clamp_value, log_weight))

        prior_log_probs.append(prior_log_prob)
        observation_log_probs.append(obs_log_prob)
        proposal_log_probs.append(proposal_log_prob)
        aux_log_scores.append(aux_log_score)
        previous_log_weight_values.append(prev_log_weight)
        total_log_weights.append(log_weight)
        log_weights.append(inc_log_weight)

    info["prior_log_probs"] = prior_log_probs
    info["observation_log_probs"] = observation_log_probs
    info["proposal_log_probs"] = proposal_log_probs
    info["aux_log_scores"] = aux_log_scores
    info["latent_distance_values"] = latent_distance_values
    info["log_weights"] = total_log_weights
    info["incremental_log_weights"] = log_weights
    info["previous_log_weights"] = previous_log_weight_values
    info["log_weight_totals"] = total_log_weights

    particle_temperature = (
        max(float(config.particle_temperature), 1e-6)
        if use_proposal_particles
        else max(float(config.temperature), 1e-6)
    )
    raw_weights = safe_normalize_log_weights(
        torch.as_tensor(total_log_weights, dtype=torch.float64) / particle_temperature
    )

    def _fallback_aggregate() -> Tuple[TensorState, Dict[str, Any], FullVSMCExpertMemory]:
        fallback_weights = _sample_count_weights(client_sample_counts)
        fallback_state = weighted_average_state_dicts(client_expert_states, fallback_weights.tolist())
        fallback_info = dict(info)
        fallback_info["weights"] = fallback_weights.tolist()
        fallback_info["pre_resample_weights"] = fallback_weights.tolist()
        fallback_info["fallback_used"] = True
        fallback_info["ess"] = effective_sample_size(fallback_weights)
        fallback_info["normalized_ess"] = fallback_info["ess"] / max(float(len(client_expert_states)), 1.0)
        updated_memory = FullVSMCExpertMemory(
            latent_memory=LatentExpertMemory(
                projection_matrix=memory.projection_matrix,
                z_mean=proposal_mean.clone(),
                z_cov_diag=proposal_cov_diag.clone(),
                flat_keys=flat_keys,
                last_round=int(round_idx),
            ),
            particle_log_weights=torch.log(fallback_weights.clamp_min(1e-12)),
            particle_latents=torch.stack(client_latent_vectors, dim=0) if client_latent_vectors else None,
            metric_ema_by_client=updated_metric_ema_by_client,
        )
        return _merge_with_reference_state(reference_state, fallback_state, device=device), fallback_info, updated_memory

    if raw_weights.numel() == 0 or not torch.isfinite(raw_weights).all():
        if config.fallback_to_fedavg_on_nan:
            return _fallback_aggregate()
        raise ValueError("Failed to normalize full-VSMC expert log weights")

    base_temperature = max(float(config.temperature), 1e-6)
    effective_temperature = base_temperature
    ess = effective_sample_size(raw_weights)
    normalized_ess = ess / max(float(len(client_expert_states)), 1.0)
    if normalized_ess < float(config.degeneration_ess_threshold):
        info["degeneration"] = True
        effective_temperature = base_temperature * max(float(config.degeneration_temperature_scale), 1.0)
        effective_particle_temperature = (
            effective_temperature if not use_proposal_particles else particle_temperature * max(float(config.degeneration_temperature_scale), 1.0)
        )
        raw_weights = safe_normalize_log_weights(
            torch.as_tensor(total_log_weights, dtype=torch.float64) / effective_particle_temperature
        )
        if raw_weights.numel() == 0 or not torch.isfinite(raw_weights).all():
            if config.fallback_to_fedavg_on_nan:
                fallback_state, fallback_info, fallback_memory = _fallback_aggregate()
                fallback_info["degeneration"] = True
                fallback_info["effective_temperature"] = effective_temperature
                fallback_info["effective_particle_temperature"] = effective_particle_temperature
                return fallback_state, fallback_info, fallback_memory
            raise ValueError("Failed to renormalize degenerate full-VSMC expert log weights")
        ess = effective_sample_size(raw_weights)
        normalized_ess = ess / max(float(len(client_expert_states)), 1.0)
        if normalized_ess < float(config.degeneration_ess_threshold) and config.fallback_to_fedavg_on_degenerate:
            fallback_state, fallback_info, fallback_memory = _fallback_aggregate()
            fallback_info["degeneration"] = True
            fallback_info["effective_temperature"] = effective_temperature
            fallback_info["effective_particle_temperature"] = effective_particle_temperature
            return fallback_state, fallback_info, fallback_memory

    fedavg_weights = _sample_count_weights(client_sample_counts)
    particle_weights = raw_weights.clone()
    particle_state_dicts: Optional[List[TensorState]] = None
    projected_client_weights: Optional[torch.Tensor] = None
    if use_proposal_particles:
        particle_state_dicts = [
            weighted_average_state_dicts(client_expert_states, responsibility_row.tolist())
            for responsibility_row in responsibility_rows
        ]
        latent_weights_for_aggregation = particle_weights
        info["particle_aggregation_mode"] = "particle_state_mixture"
    else:
        latent_weights_for_aggregation = particle_weights
        info["particle_aggregation_mode"] = "client_weight_projection"

    stabilized_weights = _stabilize_latent_weights(
        latent_weights=latent_weights_for_aggregation,
        fedavg_weights=fedavg_weights,
        mix=float(config.fedavg_mix),
        min_weight=float(config.min_weight),
    )

    info["pre_resample_weights"] = raw_weights.tolist()
    info["particle_weights"] = particle_weights.tolist()
    effective_count = len(particle_state_dicts) if particle_state_dicts is not None else len(client_expert_states)
    info["ess"] = effective_sample_size(stabilized_weights)
    info["normalized_ess"] = info["ess"] / max(float(effective_count), 1.0)
    info["effective_temperature"] = effective_temperature
    info["effective_particle_temperature"] = particle_temperature

    final_weights = stabilized_weights.clone()
    aggregate_states: Sequence[TensorState] = (
        particle_state_dicts if particle_state_dicts is not None else client_expert_states
    )
    if particle_state_dicts is not None:
        projected_client_weights = torch.zeros(len(client_expert_states), dtype=torch.float64)
        for particle_weight, responsibility_row in zip(final_weights, responsibility_rows):
            projected_client_weights = projected_client_weights + float(particle_weight.item()) * responsibility_row
        info["weights"] = projected_client_weights.tolist()
        info["particle_state_weights"] = final_weights.tolist()
    stored_particle_log_weights = torch.log(final_weights.clamp_min(1e-12))

    if (
        config.use_resampling
        and len(aggregate_states) > 1
        and info["normalized_ess"] < float(config.ess_threshold)
    ):
        selected_indices = stratified_resampling(final_weights)
        aggregate_states = [
            _maybe_add_resampling_noise(
                aggregate_states[int(index)],
                noise_std=float(config.resampling_noise_std),
                device=device,
            )
            for index in selected_indices.tolist()
        ]
        info["resampled"] = True
        info["selected_indices"] = selected_indices.tolist()
        counts = torch.bincount(selected_indices, minlength=int(final_weights.numel())).to(dtype=torch.float64)
        final_weights = counts / counts.sum().clamp_min(1.0)
        stored_particle_log_weights = torch.log(final_weights.clamp_min(1e-12))

    if particle_state_dicts is None:
        info["weights"] = final_weights.tolist()
    else:
        info["particle_state_weights"] = final_weights.tolist()

    if info["resampled"]:
        aggregated_state = weighted_average_state_dicts(
            aggregate_states,
            [1.0 / len(aggregate_states)] * len(aggregate_states),
        )
    else:
        aggregated_state = weighted_average_state_dicts(aggregate_states, final_weights.tolist())

    updated_memory = FullVSMCExpertMemory(
        latent_memory=LatentExpertMemory(
            projection_matrix=memory.projection_matrix,
            z_mean=proposal_mean.clone(),
            z_cov_diag=proposal_cov_diag.clone(),
            flat_keys=flat_keys,
            last_round=int(round_idx),
        ),
        particle_log_weights=stored_particle_log_weights.clone(),
        particle_latents=torch.stack(particle_latents, dim=0) if particle_latents else None,
        metric_ema_by_client=updated_metric_ema_by_client,
    )
    return _merge_with_reference_state(reference_state, aggregated_state, device=device), info, updated_memory


class FullVSMCExpertAggregator:
    def __init__(self, config: FullVSMCAggregatorConfig, logger: Optional[Any] = None):
        self.config = config
        self.logger = logger
        self.round_idx = 0
        self.latent_memory: Dict[str, FullVSMCExpertMemory] = {}

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        num_experts: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[TensorState, List[Dict[str, Any]]]:
        if len(client_updates) == 0:
            raise ValueError("client_updates must not be empty")

        self.round_idx += 1
        sample_weights = _sample_count_weights([num_samples for _, num_samples, _ in client_updates])
        fedavg_state = weighted_average_state_dicts(
            [state_dict for state_dict, _, _ in client_updates],
            sample_weights.tolist(),
        )
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

        ordered_group_names = _ordered_union_keys([global_expert_groups] + client_expert_groups + [fedavg_expert_groups])
        merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        merged_router_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
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

            for (state_dict, num_samples, metrics), client_group_map in zip(client_updates, client_expert_groups):
                del state_dict
                if expert_group_name not in client_group_map:
                    continue
                client_expert_states.append(client_group_map[expert_group_name]["state_dict"])
                client_sample_counts.append(int(num_samples))
                client_metrics.append(metrics or {})

            aggregated_expert_state, info, group_memory = aggregate_expert_particles_full_vsmc(
                global_expert_state=reference_expert_state,
                client_expert_states=client_expert_states,
                client_sample_counts=client_sample_counts,
                client_metrics=client_metrics,
                expert_id=expert_id,
                expert_scope=expert_group_name,
                num_experts=num_experts,
                config=self.config,
                round_idx=self.round_idx,
                latent_memory=self.latent_memory.get(expert_group_name),
                device=device,
            )

            merged_expert_groups[expert_group_name] = {
                "expert_id": expert_id,
                "state_dict": aggregated_expert_state,
            }
            self.latent_memory[expert_group_name] = group_memory
            info["expert_id"] = expert_id
            info["expert_scope"] = expert_group_name
            info["vsmc_variant"] = "full_vsmc"
            info["full_vsmc_stage"] = "sequential_importance_v3"
            round_infos.append(info)

        ordered_router_names = _ordered_union_keys([global_router_groups] + client_router_groups + [fedavg_router_groups])
        per_expert_weight_map: Dict[str, torch.Tensor] = {}
        for info in round_infos:
            weight_values = info.get("weights", [])
            if isinstance(weight_values, list) and len(weight_values) == len(client_updates):
                per_expert_weight_map[str(info["expert_scope"])] = torch.as_tensor(weight_values, dtype=torch.float64)

        for router_group_name in ordered_router_names:
            template_group = global_router_groups.get(router_group_name) or fedavg_router_groups.get(router_group_name)
            if template_group is None:
                continue

            router_prefix = router_group_name[: -len("router")] + "experts."
            matched_weight_tensors = [
                weights
                for expert_scope, weights in per_expert_weight_map.items()
                if expert_scope.startswith(router_prefix)
            ]
            router_weights = (
                torch.stack(matched_weight_tensors, dim=0).mean(dim=0)
                if matched_weight_tensors
                else sample_weights.to(dtype=torch.float64)
            )

            router_client_states: List[TensorState] = []
            router_weight_values: List[float] = []
            router_sample_counts: List[int] = []
            for client_idx, ((_, num_samples, _), client_router_map) in enumerate(zip(client_updates, client_router_groups)):
                if router_group_name not in client_router_map:
                    continue
                router_client_states.append(client_router_map[router_group_name]["state_dict"])
                router_weight_values.append(float(router_weights[client_idx].item()))
                router_sample_counts.append(int(num_samples))

            if not router_client_states:
                continue

            router_weight_tensor = torch.as_tensor(router_weight_values, dtype=torch.float64)
            if not torch.isfinite(router_weight_tensor).all() or float(router_weight_tensor.sum().item()) <= 0.0:
                router_weight_tensor = _sample_count_weights(router_sample_counts)
            else:
                router_weight_tensor = router_weight_tensor / router_weight_tensor.sum().clamp_min(1e-12)

            merged_router_groups[router_group_name] = {
                "router_scope": router_group_name,
                "state_dict": weighted_average_state_dicts(router_client_states, router_weight_tensor.tolist()),
            }
            round_infos.append(
                {
                    "expert_id": -1,
                    "expert_scope": router_group_name,
                    "router_scope": router_group_name,
                    "weights": router_weight_tensor.tolist(),
                    "aggregation_source": "expert_weight_mean",
                    "vsmc_variant": "full_vsmc",
                    "full_vsmc_stage": "router_follow_expert_v1",
                }
            )

        merged_state = merge_named_groups_into_model(base_state, merged_expert_groups)
        merged_state = merge_named_groups_into_model(merged_state, merged_router_groups)
        merged_state = _merge_with_reference_state(global_state, merged_state, device=device)
        return merged_state, round_infos
