from __future__ import annotations

import collections
import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from aggregation.vsmc_expert_aggregator import (
    TensorState,
    _clone_detached_tensor,
    _merge_with_reference_state,
    _ordered_union_keys,
    clone_detached_state_dict,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)


_EPS = 1e-12
_BLOCK_EXPERT_PARAM_RE = re.compile(r"blocks\.(\d+)\.switch_ffn\.experts\.(\d+)\.")
_BLOCK_EXPERT_SCOPE_RE = re.compile(r"blocks\.(\d+)\.switch_ffn\.experts\.(\d+)$")


@dataclass
class HAVSMCETFConfig:
    enabled: bool = True
    num_particles: int = 8
    state_level: str = "block_expert"
    expert_velocity_beta: float = 0.90
    second_moment_beta: float = 0.99
    client_reliability_beta: float = 0.90
    usage_beta: float = 0.90
    temperature_init: float = 1.0
    temperature_min: float = 0.5
    temperature_max: float = 2.0
    lambda_max: float = 0.20
    lambda_warmup_rounds: int = 10
    anchor_mode: str = "fedavg"
    pure_vsmc: bool = False
    ess_threshold_ratio: float = 0.50
    trust_update_lr: float = 0.20
    reliability_momentum_lr: float = 0.10
    prediction_noise_std: float = 0.03
    alpha_usage: float = 0.10
    alpha_history_direction: float = 0.25
    alpha_consensus: float = 0.25
    alpha_drift: float = 0.20
    alpha_function_drift: float = 0.20
    collapse_penalty: float = 0.05
    probe_batch_size: int = 128
    use_function_drift: bool = True
    function_drift_metric: str = "cosine"
    function_drift_interval: int = 1
    require_function_drift: bool = False
    use_validation_loss: bool = False
    normalize_observation_components: bool = True
    min_router_usage: float = 1e-6
    eps: float = 1e-8
    log_every_round: bool = True
    deterministic: bool = True
    shared_weight_mode: str = "uniform"
    expert_weight_mode: str = "uniform"
    clamp_trust_logits: float = 10.0
    drift_clamp: float = 10.0
    function_drift_clamp: float = 2.0
    small_entropy_bonus: float = 0.0
    reliability_weighted_update: bool = False
    score_transform: str = "zscore"
    score_clip_low: float = -2.0
    score_clip_high: float = 2.0
    temperature_scale: float = 1.0
    prob_ema_rho: float = 0.0
    reliability_floor_eps: float = 0.0
    reliability_floor_source: str = "none"
    top_k_clients: int = 0
    adaptive_temperature: bool = False
    entropy_target: str = "none"
    entropy_target_min_max_weight: float = 0.25
    entropy_target_max_max_weight: float = 0.55
    save_client_weights: bool = True
    anti_collapse_enabled: bool = False
    anti_collapse_max_weight_thresh: float = 0.75
    anti_collapse_neff_thresh: float = 1.8
    anti_collapse_entropy_thresh: float = 0.5
    anti_collapse_eps: float = 0.05
    anti_collapse_prior: str = "usage"
    coverage_bonus_beta: float = 0.0
    client_coverage_scores: Optional[Sequence[float]] = None
    proxy_gain_gamma: float = 0.0
    seed: int = 0
    current_round_idx: int = 0


class HAVSMCETFState:
    def __init__(
        self,
        num_blocks: int,
        num_experts_per_block: int,
        num_clients: int,
        num_particles: int,
        expert_param_shapes: Dict[Tuple[int, int], TensorState],
        device: torch.device,
        config: HAVSMCETFConfig,
    ):
        self.num_blocks = int(num_blocks)
        self.num_experts_per_block = int(num_experts_per_block)
        self.num_clients = int(num_clients)
        self.num_particles = int(num_particles)
        self.device = device
        self.trust_logits = torch.zeros(
            self.num_blocks,
            self.num_experts_per_block,
            self.num_particles,
            self.num_clients,
            dtype=torch.float64,
            device=device,
        )
        self.particle_log_weights = torch.zeros(
            self.num_blocks,
            self.num_experts_per_block,
            self.num_particles,
            dtype=torch.float64,
            device=device,
        )
        self.client_reliability = torch.zeros(
            self.num_blocks,
            self.num_experts_per_block,
            self.num_clients,
            dtype=torch.float64,
            device=device,
        )
        self.usage_ema = torch.zeros(
            self.num_blocks,
            self.num_experts_per_block,
            self.num_clients,
            dtype=torch.float64,
            device=device,
        )
        self.prev_client_weights = torch.full(
            (self.num_blocks, self.num_experts_per_block, self.num_clients),
            1.0 / float(max(self.num_clients, 1)),
            dtype=torch.float64,
            device=device,
        )
        self.temperature = torch.full(
            (self.num_blocks, self.num_experts_per_block),
            float(config.temperature_init),
            dtype=torch.float64,
            device=device,
        )
        self.expert_velocity_ema = {
            key: _zeros_like_state(template, device=device)
            for key, template in expert_param_shapes.items()
        }
        self.expert_second_moment = {
            key: _zeros_like_state(template, device=device)
            for key, template in expert_param_shapes.items()
        }
        self.prev_global_experts = {
            key: _clone_state_to_device(template, device=device)
            for key, template in expert_param_shapes.items()
        }
        self.round_idx = 0


def is_expert_param(name: str) -> bool:
    name_l = str(name).lower()
    if "router" in name_l:
        return False
    return (
        "expert" in name_l
        or "experts" in name_l
        or ("moe" in name_l and "router" not in name_l)
    )


def get_expert_param_groups(model) -> Dict[Tuple[int, int], List[Tuple[str, torch.Tensor]]]:
    groups: Dict[Tuple[int, int], List[Tuple[str, torch.Tensor]]] = collections.OrderedDict()
    for name, param in model.named_parameters():
        block_expert = _parse_block_expert_from_name(name)
        if block_expert is None:
            continue
        groups.setdefault(block_expert, []).append((name, param))
    return groups


def get_non_expert_param_names(model) -> List[str]:
    names: List[str] = []
    for name, _ in model.named_parameters():
        if not is_expert_param(name):
            names.append(name)
    return names


def collect_probe_hidden_states(model, probe_batch, block_id: int):
    if probe_batch is None:
        return None
    inputs = probe_batch[0] if isinstance(probe_batch, (tuple, list)) else probe_batch
    if not torch.is_tensor(inputs):
        return None
    was_training = bool(model.training)
    try:
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            _, aux = model(inputs, return_aux=True, return_moe_inputs=True)
            moe_inputs = aux.get("moe_inputs_by_layer")
            if moe_inputs is None or int(block_id) >= int(moe_inputs.size(0)):
                return None
            return moe_inputs[int(block_id)].detach()
    finally:
        model.train(was_training)


def compute_expert_function_drift(
    old_expert_module,
    candidate_expert_module,
    probe_hidden_states,
    metric: str = "cosine",
):
    del metric
    with torch.no_grad():
        old_output = old_expert_module(probe_hidden_states)
        candidate_output = candidate_expert_module(probe_hidden_states)
        old_flat = old_output.reshape(1, -1)
        candidate_flat = candidate_output.reshape(1, -1)
        similarity = F.cosine_similarity(old_flat, candidate_flat, dim=-1, eps=1e-12)
        drift = 1.0 - similarity.mean()
        return torch.clamp(drift, min=0.0, max=2.0)


def build_ha_vsmc_etf_state(
    global_model: nn.Module,
    global_state: TensorState,
    num_clients: int,
    config: HAVSMCETFConfig,
    device: torch.device,
) -> HAVSMCETFState:
    del global_model
    _, expert_groups = split_state_dict_by_expert(global_state, num_experts=None)
    expert_templates: Dict[Tuple[int, int], TensorState] = collections.OrderedDict()
    max_block_id = -1
    max_expert_id = -1
    for expert_scope, group in expert_groups.items():
        block_expert = _parse_block_expert_from_scope(expert_scope)
        if block_expert is None:
            continue
        block_id, expert_id = block_expert
        expert_templates[(block_id, expert_id)] = group["state_dict"]
        max_block_id = max(max_block_id, block_id)
        max_expert_id = max(max_expert_id, expert_id)
    if max_block_id < 0 or max_expert_id < 0:
        raise ValueError("Failed to initialize HA-VSMC-ETF state because no expert parameters were found")
    return HAVSMCETFState(
        num_blocks=max_block_id + 1,
        num_experts_per_block=max_expert_id + 1,
        num_clients=num_clients,
        num_particles=config.num_particles,
        expert_param_shapes=expert_templates,
        device=device,
        config=config,
    )


def aggregate_with_ha_vsmc_etf(
    global_model,
    client_updates,
    state: HAVSMCETFState,
    config: HAVSMCETFConfig,
    probe_batch=None,
):
    assert bool(config.use_validation_loss) is False
    current_round = int(config.current_round_idx) if int(config.current_round_idx) > 0 else int(state.round_idx) + 1
    num_clients = len(client_updates)
    if num_clients != int(state.num_clients):
        raise ValueError(
            f"HA-VSMC-ETF state expects {state.num_clients} clients, but received {num_clients}"
        )

    global_state = collections.OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in global_model.state_dict().items()
    )
    shared_weights = _client_weights_from_mode(client_updates, config.shared_weight_mode)
    expert_weights = _client_weights_from_mode(client_updates, config.expert_weight_mode)
    fedavg_state = weighted_average_state_dicts(
        [state_dict for state_dict, _, _ in client_updates],
        shared_weights,
        fallback_state=global_state,
    )

    _, global_expert_groups = split_state_dict_by_expert(global_state, num_experts=None)
    split_client_groups = [
        split_state_dict_by_expert(state_dict, num_experts=None)[1]
        for state_dict, _, _ in client_updates
    ]
    probe_hidden_cache: Dict[int, Optional[torch.Tensor]] = {}
    expert_summaries: List[Dict[str, Any]] = []
    particle_summaries: List[Dict[str, Any]] = []
    merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()

    for expert_scope, global_group in global_expert_groups.items():
        block_expert = _parse_block_expert_from_scope(expert_scope)
        if block_expert is None:
            continue
        block_id, expert_id = block_expert
        group_key = (block_id, expert_id)
        client_group_states: List[TensorState] = []
        client_metrics: List[Dict[str, Any]] = []
        sample_counts: List[int] = []
        for (state_dict, num_samples, metrics), client_group_map in zip(client_updates, split_client_groups):
            del state_dict
            group = client_group_map.get(expert_scope)
            if group is None:
                client_group_states.append(clone_detached_state_dict(global_group["state_dict"]))
            else:
                client_group_states.append(group["state_dict"])
            sample_counts.append(int(num_samples))
            client_metrics.append(metrics or {})

        aggregated_expert_state, info, particle_rows = _aggregate_single_expert(
            global_model=global_model,
            global_expert_state=global_group["state_dict"],
            client_expert_states=client_group_states,
            client_metrics=client_metrics,
            sample_counts=sample_counts,
            shared_weights=shared_weights,
            expert_weights=expert_weights,
            state=state,
            config=config,
            group_key=group_key,
            expert_scope=expert_scope,
            probe_batch=probe_batch,
            probe_hidden_cache=probe_hidden_cache,
            round_idx=current_round,
        )
        merged_expert_groups[expert_scope] = {
            "expert_id": int(expert_id),
            "expert_scope": expert_scope,
            "state_dict": aggregated_expert_state,
        }
        expert_summaries.append(info)
        particle_summaries.extend(particle_rows)

    new_global_state = _merge_with_reference_state(
        fedavg_state,
        _merge_expert_groups(merged_expert_groups),
        device=None,
    )
    state.round_idx = current_round
    round_logs = {
        "round": int(current_round),
        "anchor_mode": str(config.anchor_mode),
        "pure_vsmc": bool(config.pure_vsmc),
        "fedavg_blend_enabled": not bool(config.pure_vsmc) and str(config.anchor_mode) == "fedavg",
        "expert_output": "theta_vsmc" if bool(config.pure_vsmc) or str(config.anchor_mode) == "none" else "blend(theta_fedavg,theta_vsmc)",
        "expert_summaries": expert_summaries,
        "particle_summaries": particle_summaries,
        "mean_ess_ratio": _safe_mean([info.get("ESS_ratio") for info in expert_summaries]),
        "fallback_count": int(sum(1 for info in expert_summaries if bool(info.get("fallback_used", False)))),
    }
    return new_global_state, state, round_logs


class HAVSMCETFExpertAggregator:
    def __init__(self, config: HAVSMCETFConfig, logger=None):
        self.config = config
        self.logger = logger
        self.state: Optional[HAVSMCETFState] = None

    def aggregate_model_state(
        self,
        global_model: nn.Module,
        global_state: TensorState,
        client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
        device: torch.device,
        round_idx: int,
        probe_batch=None,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        if (
            self.state is None
            or int(self.state.num_clients) != len(client_updates)
            or int(self.state.num_particles) != int(self.config.num_particles)
        ):
            self.state = build_ha_vsmc_etf_state(
                global_model=global_model,
                global_state=global_state,
                num_clients=len(client_updates),
                config=self.config,
                device=device,
            )
        self.config.current_round_idx = int(round_idx)
        aggregated_state, updated_state, round_logs = aggregate_with_ha_vsmc_etf(
            global_model=global_model,
            client_updates=client_updates,
            state=self.state,
            config=self.config,
            probe_batch=probe_batch,
        )
        self.state = updated_state
        return aggregated_state, round_logs


def _aggregate_single_expert(
    global_model: nn.Module,
    global_expert_state: TensorState,
    client_expert_states: Sequence[TensorState],
    client_metrics: Sequence[Dict[str, Any]],
    sample_counts: Sequence[int],
    shared_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    state: HAVSMCETFState,
    config: HAVSMCETFConfig,
    group_key: Tuple[int, int],
    expert_scope: str,
    probe_batch,
    probe_hidden_cache: Dict[int, Optional[torch.Tensor]],
    round_idx: int,
) -> Tuple[TensorState, Dict[str, Any], List[Dict[str, Any]]]:
    del shared_weights
    block_id, expert_id = group_key
    device = state.trust_logits.device
    eps = float(config.eps)
    num_clients = len(client_expert_states)
    particle_count = int(config.num_particles)
    pure_vsmc = bool(config.pure_vsmc)

    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        expert_weights,
        fallback_state=global_expert_state,
    )
    fallback_state = clone_detached_state_dict(theta_fedavg if not pure_vsmc else global_expert_state)
    fallback_info = _build_base_info(
        round_idx=round_idx,
        block_id=block_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        num_particles=particle_count,
    )

    try:
        usage_tensor = torch.stack(
            [
                _extract_router_usage_tensor(
                    metrics=metrics,
                    num_blocks=state.num_blocks,
                    num_experts=state.num_experts_per_block,
                    device=device,
                )[block_id, expert_id]
                for metrics in client_metrics
            ],
            dim=0,
        ).to(dtype=torch.float64, device=device)
        usage_tensor = torch.clamp(usage_tensor, min=0.0)
        prev_usage_ema = state.usage_ema[block_id, expert_id].clone()
        state.usage_ema[block_id, expert_id] = (
            float(config.usage_beta) * state.usage_ema[block_id, expert_id]
            + (1.0 - float(config.usage_beta)) * usage_tensor
        )

        delta_states = [
            _subtract_state_dict(client_state, global_expert_state)
            for client_state in client_expert_states
        ]
        delta_vecs = [
            _flatten_float_state(delta_state).to(dtype=torch.float64, device=device)
            for delta_state in delta_states
        ]
        if all(vec.numel() == 0 or torch.allclose(vec, torch.zeros_like(vec), atol=1e-12) for vec in delta_vecs):
            if not pure_vsmc:
                raise _ExpertFallback("all_client_delta_zero")
        if not bool((usage_tensor > 0.0).any().item()):
            raise _ExpertFallback("all_router_usage_zero")

        robust_center = _coordinate_median(delta_vecs, device=device)
        consensus_cos = torch.stack(
            [_cosine_with_reference(vec, robust_center, eps=eps) for vec in delta_vecs],
            dim=0,
        )
        velocity_vec = _flatten_float_state(state.expert_velocity_ema[group_key]).to(dtype=torch.float64, device=device)
        history_cos = torch.stack(
            [_cosine_with_reference(vec, velocity_vec, eps=eps) for vec in delta_vecs],
            dim=0,
        )

        drift_values = torch.tensor(
            [
                _normalized_drift(
                    delta_state,
                    state.expert_second_moment[group_key],
                    eps=eps,
                    clamp_value=float(config.drift_clamp),
                )
                for delta_state in delta_states
            ],
            dtype=torch.float64,
            device=device,
        )

        function_drift_available = False
        function_drift_values = torch.zeros(num_clients, dtype=torch.float64, device=device)
        function_drift_failure = ""
        if bool(config.use_function_drift) and int(config.function_drift_interval) > 0:
            should_run = (int(round_idx) - 1) % max(int(config.function_drift_interval), 1) == 0
            if should_run:
                try:
                    if block_id not in probe_hidden_cache:
                        probe_hidden_cache[block_id] = collect_probe_hidden_states(global_model, probe_batch, block_id)
                    probe_hidden_states = probe_hidden_cache.get(block_id)
                    if probe_hidden_states is not None:
                        function_drift_values = _compute_function_drifts_for_clients(
                            global_model=global_model,
                            block_id=block_id,
                            expert_id=expert_id,
                            client_expert_states=client_expert_states,
                            probe_hidden_states=probe_hidden_states,
                            clamp_value=float(config.function_drift_clamp),
                            metric=str(config.function_drift_metric),
                        ).to(dtype=torch.float64, device=device)
                        function_drift_available = True
                except Exception as exc:  # pragma: no cover - defensive fallback path
                    function_drift_failure = str(exc)
                    function_drift_available = False
        if bool(config.require_function_drift) and bool(config.use_function_drift) and not function_drift_available:
            raise _ExpertFallback("function_drift_unavailable")

        usage_score = torch.log(usage_tensor.clamp_min(max(float(config.min_router_usage), eps)))
        history_scale = (
            min(1.0, float(round_idx) / float(max(int(config.lambda_warmup_rounds), 1)))
            if int(config.lambda_warmup_rounds) > 0
            else 1.0
        )
        effective_alpha_history = float(config.alpha_history_direction) * history_scale

        if str(getattr(config, "score_transform", "zscore")) == "rank_based":
            usage_norm = _rank_score(usage_score, descending=True)
            history_norm = _rank_score(history_cos, descending=True)
            consensus_norm = _rank_score(consensus_cos, descending=True)
            drift_norm = _rank_score(drift_values, descending=False)
            function_norm = _rank_score(function_drift_values, descending=False)
            observation_scores = (
                float(config.alpha_usage) * usage_norm
                + float(effective_alpha_history) * history_norm
                + float(config.alpha_consensus) * consensus_norm
                + float(config.alpha_drift) * drift_norm
                + float(config.alpha_function_drift) * function_norm
            )
        else:
            usage_norm = _normalize_component(usage_score, config)
            history_norm = _normalize_component(history_cos, config)
            consensus_norm = _normalize_component(consensus_cos, config)
            drift_norm = _normalize_component(drift_values, config)
            function_norm = _normalize_component(function_drift_values, config)

            observation_scores = (
                float(config.alpha_usage) * usage_norm
                + float(effective_alpha_history) * history_norm
                + float(config.alpha_consensus) * consensus_norm
                - float(config.alpha_drift) * drift_norm
                - float(config.alpha_function_drift) * function_norm
            )

        coverage_bonus_beta = float(getattr(config, "coverage_bonus_beta", 0.0))
        coverage_scores = getattr(config, "client_coverage_scores", None)
        coverage_bonus_vec = torch.zeros(num_clients, dtype=torch.float64, device=device)
        if coverage_bonus_beta > 0.0 and coverage_scores is not None:
            coverage_bonus_vec = torch.tensor(list(coverage_scores), dtype=torch.float64, device=device)
            coverage_bonus_vec = _normalize_component(coverage_bonus_vec, config)
            observation_scores = observation_scores + coverage_bonus_beta * coverage_bonus_vec

        proxy_gain_gamma = float(getattr(config, "proxy_gain_gamma", 0.0))
        proxy_gain_vec = torch.zeros(num_clients, dtype=torch.float64, device=device)
        if proxy_gain_gamma > 0.0:
            proxy_gain_values = [float((metrics or {}).get("proxy_gain", 0.0)) for metrics in client_metrics]
            proxy_gain_vec = torch.tensor(proxy_gain_values, dtype=torch.float64, device=device)
            proxy_gain_vec = _normalize_component(proxy_gain_vec, config)
            observation_scores = observation_scores + proxy_gain_gamma * proxy_gain_vec

        if torch.isnan(observation_scores).any():
            raise _ExpertFallback("nan_detected")

        trust_logits_prev = state.trust_logits[block_id, expert_id].clone()
        reliability_vec = state.client_reliability[block_id, expert_id].clone()
        trust_logits_pred = (
            trust_logits_prev
            + float(config.reliability_momentum_lr) * reliability_vec.unsqueeze(0)
            + _prediction_noise_like(
                trust_logits_prev,
                config=config,
                offset=block_id * 1000 + expert_id * 10 + int(round_idx),
            )
        )
        trust_logits = trust_logits_pred + float(config.trust_update_lr) * observation_scores.unsqueeze(0)
        trust_logits = trust_logits.clamp(
            min=-abs(float(config.clamp_trust_logits)),
            max=abs(float(config.clamp_trust_logits)),
        )
        if torch.isnan(trust_logits).any():
            raise _ExpertFallback("nan_detected")

        temperature = float(state.temperature[block_id, expert_id].item()) * max(float(getattr(config, "temperature_scale", 1.0)), eps)
        client_weight_rows = torch.softmax(
            trust_logits / max(temperature, eps),
            dim=-1,
        )
        if not torch.allclose(
            client_weight_rows.sum(dim=-1),
            torch.ones(particle_count, dtype=client_weight_rows.dtype, device=client_weight_rows.device),
            atol=1e-5,
        ):
            raise _ExpertFallback("particle_weights_invalid")

        theta_particles = [
            weighted_average_state_dicts(list(client_expert_states), client_weight_rows[particle_idx], fallback_state=global_expert_state)
            for particle_idx in range(particle_count)
        ]
        particle_scores = torch.sum(
            client_weight_rows * observation_scores.unsqueeze(0),
            dim=-1,
        ) - float(config.collapse_penalty) * client_weight_rows.max(dim=-1).values
        if float(config.small_entropy_bonus) > 0.0 and num_clients > 1:
            entropy = _probability_entropy(client_weight_rows)
            entropy_norm = entropy / math.log(float(num_clients))
            particle_scores = particle_scores + float(config.small_entropy_bonus) * entropy_norm

        particle_log_weights = state.particle_log_weights[block_id, expert_id].clone() + particle_scores
        particle_log_weights = particle_log_weights - torch.logsumexp(particle_log_weights, dim=0)
        particle_weights = torch.softmax(particle_log_weights, dim=0)
        if not torch.allclose(
            particle_weights.sum(),
            torch.tensor(1.0, dtype=particle_weights.dtype, device=particle_weights.device),
            atol=1e-5,
        ):
            raise _ExpertFallback("particle_weights_invalid")
        if torch.isnan(particle_weights).any():
            raise _ExpertFallback("nan_detected")

        ess = 1.0 / particle_weights.pow(2).sum().clamp_min(eps)
        ess_ratio = float((ess / float(max(particle_count, 1))).item())
        theta_vsmc = weighted_average_state_dicts(theta_particles, particle_weights, fallback_state=global_expert_state)
        confidence = float(
            particle_scores.mean().item()
            - particle_scores.std(unbiased=False).item()
        )
        anchor_mode = str(config.anchor_mode)
        if pure_vsmc:
            if anchor_mode != "none":
                raise ValueError("HA-VSMC pure_vsmc requires anchor_mode='none'")
            theta_new = clone_detached_state_dict(theta_vsmc)
            logged_lambda_expert = 1.0
            fedavg_blend_enabled = False
        else:
            warmup = (
                min(1.0, float(round_idx) / float(max(int(config.lambda_warmup_rounds), 1)))
                if int(config.lambda_warmup_rounds) > 0
                else 1.0
            )
            lambda_expert = float(config.lambda_max) * warmup * float(torch.sigmoid(torch.tensor(confidence)).item()) * ess_ratio
            lambda_expert = max(0.0, min(float(config.lambda_max), lambda_expert))
            if anchor_mode not in {"fedavg", "none"}:
                raise ValueError(f"Unsupported HA-VSMC anchor mode: {anchor_mode}")
            if anchor_mode == "none":
                theta_new = clone_detached_state_dict(theta_vsmc)
                logged_lambda_expert = 1.0
                fedavg_blend_enabled = False
            else:
                theta_new = _blend_state_dicts(theta_fedavg, theta_vsmc, lambda_expert)
                logged_lambda_expert = float(lambda_expert)
                fedavg_blend_enabled = True
        if _state_has_nan(theta_new):
            raise _ExpertFallback("nan_detected")

        client_weight_final_raw = torch.sum(
            particle_weights.unsqueeze(-1) * client_weight_rows,
            dim=0,
        )
        client_weight_final = _normalize_probability_vector(client_weight_final_raw, eps)
        topk_active_count = int((client_weight_final > 0).sum().item())
        postprocess_enabled = any(
            [
                float(getattr(config, "prob_ema_rho", 0.0)) > 0.0,
                float(getattr(config, "reliability_floor_eps", 0.0)) > 0.0 and str(getattr(config, "reliability_floor_source", "none")) != "none",
                int(getattr(config, "top_k_clients", 0)) > 0,
                str(getattr(config, "entropy_target", "none")) != "none",
                bool(getattr(config, "adaptive_temperature", False)),
            ]
        )
        if pure_vsmc and postprocess_enabled:
            if float(getattr(config, "prob_ema_rho", 0.0)) > 0.0:
                rho = max(0.0, min(1.0, float(getattr(config, "prob_ema_rho", 0.0))))
                prev_weights = state.prev_client_weights[block_id, expert_id].clone()
                client_weight_final = _normalize_probability_vector(rho * prev_weights + (1.0 - rho) * client_weight_final, eps)
            floor_eps = max(0.0, min(1.0, float(getattr(config, "reliability_floor_eps", 0.0))))
            floor_source = str(getattr(config, "reliability_floor_source", "none"))
            if floor_eps > 0.0 and floor_source != "none":
                active_mask = (usage_tensor > 0.0).to(dtype=torch.float64)
                if floor_source == "uniform_active":
                    floor_q = _normalize_probability_vector(active_mask, eps)
                else:
                    floor_q = _normalize_probability_vector(usage_tensor * active_mask, eps)
                client_weight_final = _normalize_probability_vector((1.0 - floor_eps) * client_weight_final + floor_eps * floor_q, eps)
            client_weight_final, topk_active_count = _apply_top_k(client_weight_final, int(getattr(config, "top_k_clients", 0)), eps)
            client_weight_final = _apply_entropy_target_temperature(client_weight_final, config, eps)
            if bool(getattr(config, "adaptive_temperature", False)):
                max_client_weight = float(client_weight_final.max().item()) if client_weight_final.numel() > 0 else 0.0
                if max_client_weight < float(getattr(config, "entropy_target_min_max_weight", 0.25)):
                    temperature *= 0.95
                elif max_client_weight > float(getattr(config, "entropy_target_max_max_weight", 0.55)):
                    temperature *= 1.05
            theta_new = weighted_average_state_dicts(list(client_expert_states), client_weight_final, fallback_state=global_expert_state)
        if pure_vsmc and bool(getattr(config, "anti_collapse_enabled", False)):
            current_entropy = float(_probability_entropy(client_weight_final.unsqueeze(0))[0].item()) if client_weight_final.numel() > 0 else 0.0
            current_neff = _effective_num_clients(client_weight_final, eps)
            current_max_weight = float(client_weight_final.max().item()) if client_weight_final.numel() > 0 else 0.0
            guard_triggered = (
                current_max_weight > float(getattr(config, "anti_collapse_max_weight_thresh", 0.75))
                or current_neff < float(getattr(config, "anti_collapse_neff_thresh", 1.8))
                or current_entropy < float(getattr(config, "anti_collapse_entropy_thresh", 0.5))
            )
            if guard_triggered:
                guard_eps = max(0.0, min(0.5, float(getattr(config, "anti_collapse_eps", 0.05))))
                prior_q = _build_anti_collapse_prior(config, usage_tensor, client_weight_final, eps)
                client_weight_final = _normalize_probability_vector((1.0 - guard_eps) * client_weight_final + guard_eps * prior_q, eps)
                theta_new = weighted_average_state_dicts(list(client_expert_states), client_weight_final, fallback_state=global_expert_state)
        state.prev_client_weights[block_id, expert_id] = client_weight_final.clone()
        reliability_target = observation_scores
        if bool(config.reliability_weighted_update):
            reliability_target = client_weight_final * reliability_target
        state.client_reliability[block_id, expert_id] = (
            float(config.client_reliability_beta) * state.client_reliability[block_id, expert_id]
            + (1.0 - float(config.client_reliability_beta)) * reliability_target
        )

        global_delta = _subtract_state_dict(theta_new, global_expert_state)
        state.expert_velocity_ema[group_key] = _ema_state_update(
            state.expert_velocity_ema[group_key],
            global_delta,
            beta=float(config.expert_velocity_beta),
        )
        state.expert_second_moment[group_key] = _ema_state_update_squared(
            state.expert_second_moment[group_key],
            global_delta,
            beta=float(config.second_moment_beta),
        )
        state.prev_global_experts[group_key] = _clone_state_to_device(theta_new, device=state.device)

        if ess_ratio < 0.4:
            temperature *= 1.05
        elif ess_ratio > 0.8 and confidence > 0.0:
            temperature *= 0.98
        temperature = max(float(config.temperature_min), min(float(config.temperature_max), temperature))
        state.temperature[block_id, expert_id] = temperature

        resampled = False
        if float(ess.item()) < float(config.ess_threshold_ratio) * float(particle_count):
            resampled = True
            selected = _systematic_resampling(
                particle_weights,
                config=config,
                offset=block_id * 1000 + expert_id * 10 + int(round_idx),
            )
            resampled_logits = trust_logits.index_select(0, selected)
            resampled_logits = resampled_logits + _prediction_noise_like(
                resampled_logits,
                config=config,
                offset=block_id * 2000 + expert_id * 20 + int(round_idx),
            )
            state.trust_logits[block_id, expert_id] = resampled_logits.clamp(
                min=-abs(float(config.clamp_trust_logits)),
                max=abs(float(config.clamp_trust_logits)),
            )
            state.particle_log_weights[block_id, expert_id] = torch.zeros_like(particle_log_weights)
        else:
            state.trust_logits[block_id, expert_id] = trust_logits
            state.particle_log_weights[block_id, expert_id] = particle_log_weights

        velocity_norm = _state_rms(state.expert_velocity_ema[group_key])
        second_moment_norm = _state_rms(state.expert_second_moment[group_key])
        vsmc_fedavg_l2_distance = _state_l2_distance(theta_vsmc, theta_fedavg)
        final_fedavg_l2_distance = _state_l2_distance(theta_new, theta_fedavg)
        final_residual_ratio = (
            final_fedavg_l2_distance / max(vsmc_fedavg_l2_distance, eps)
            if vsmc_fedavg_l2_distance > 0.0
            else 0.0
        )

        info = dict(fallback_info)
        info.update(
            {
                "lambda_expert": float(logged_lambda_expert),
                "anchor_mode": anchor_mode,
                "pure_vsmc": pure_vsmc,
                "fedavg_blend_enabled": bool(fedavg_blend_enabled),
                "expert_output": "theta_vsmc" if pure_vsmc or anchor_mode == "none" else "blend(theta_fedavg,theta_vsmc)",
                "temperature": float(temperature),
                "ESS": float(ess.item()),
                "ESS_ratio": float(ess_ratio),
                "resampled": bool(resampled),
                "particle_weight_entropy": float(_probability_entropy(particle_weights.unsqueeze(0))[0].item()),
                "particle_max_weight": float(particle_weights.max().item()),
                "client_weight_entropy": float(_probability_entropy(client_weight_final.unsqueeze(0))[0].item()),
                "expert_weight_entropy": float(_probability_entropy(client_weight_final.unsqueeze(0))[0].item()),
                "client_max_weight": float(client_weight_final.max().item()),
                "client_min_weight": float(client_weight_final.min().item()),
                "max_client_weight_mean": float(client_weight_final.max().item()),
                "min_client_weight_mean": float(client_weight_final.min().item()),
                "active_client_count": int(topk_active_count),
                "active_client_count_per_expert": int(topk_active_count),
                "usage_score_mean": float(usage_score.mean().item()),
                "usage_score_std": float(usage_score.std(unbiased=False).item()),
                "history_cos_mean": float(history_cos.mean().item()),
                "history_cos_std": float(history_cos.std(unbiased=False).item()),
                "consensus_cos_mean": float(consensus_cos.mean().item()),
                "consensus_cos_std": float(consensus_cos.std(unbiased=False).item()),
                "normalized_drift_mean": float(drift_values.mean().item()),
                "normalized_drift_std": float(drift_values.std(unbiased=False).item()),
                "function_drift_mean": float(function_drift_values.mean().item()),
                "function_drift_std": float(function_drift_values.std(unbiased=False).item()),
                "observation_score_mean": float(observation_scores.mean().item()),
                "observation_score_std": float(observation_scores.std(unbiased=False).item()),
                "particle_score_mean": float(particle_scores.mean().item()),
                "particle_score_std": float(particle_scores.std(unbiased=False).item()),
                "max_particle_weight": float(particle_weights.max().item()),
                "vsmc_fedavg_l2_distance": float(vsmc_fedavg_l2_distance),
                "final_fedavg_l2_distance": float(final_fedavg_l2_distance),
                "final_residual_ratio": float(final_residual_ratio),
                "velocity_norm": float(velocity_norm),
                "second_moment_norm": float(second_moment_norm),
                "function_drift_enabled": bool(config.use_function_drift),
                "function_drift_available": bool(function_drift_available),
                "fallback_used": False,
                "fallback_reason": function_drift_failure if function_drift_failure else "",
                "fallback_type": function_drift_failure if function_drift_failure else "none",
                "confidence": float(confidence),
                "weights": client_weight_final.detach().cpu().tolist(),
                "particle_weights": particle_weights.detach().cpu().tolist(),
                "usage_score_values": usage_score.detach().cpu().tolist(),
                "history_cos_values": history_cos.detach().cpu().tolist(),
                "consensus_cos_values": consensus_cos.detach().cpu().tolist(),
                "normalized_drift_values": drift_values.detach().cpu().tolist(),
                "function_drift_values": function_drift_values.detach().cpu().tolist(),
                "usage_signal_mean": float(usage_norm.mean().item()),
                "usage_signal_std": float(usage_norm.std(unbiased=False).item()),
                "history_signal_mean": float(history_norm.mean().item()),
                "history_signal_std": float(history_norm.std(unbiased=False).item()),
                "consensus_signal_mean": float(consensus_norm.mean().item()),
                "consensus_signal_std": float(consensus_norm.std(unbiased=False).item()),
                "drift_signal_mean": float(drift_norm.mean().item()),
                "drift_signal_std": float(drift_norm.std(unbiased=False).item()),
                "function_signal_mean": float(function_norm.mean().item()),
                "function_signal_std": float(function_norm.std(unbiased=False).item()),
                "reliability_score_mean": float(observation_scores.mean().item()),
                "reliability_score_std": float(observation_scores.std(unbiased=False).item()),
                "proxy_val_loss_delta": float(sum(float((metrics or {}).get("proxy_val_loss_delta", 0.0)) for metrics in client_metrics) / max(len(client_metrics), 1)),
                "proxy_val_acc_delta": float(sum(float((metrics or {}).get("proxy_val_acc_delta", 0.0)) for metrics in client_metrics) / max(len(client_metrics), 1)),
                "client_weight_final_raw_entropy": float(_probability_entropy(client_weight_final_raw.unsqueeze(0))[0].item()),
                "observation_score_values": observation_scores.detach().cpu().tolist(),
                "particle_score_values": particle_scores.detach().cpu().tolist(),
                "usage_ema_mean": float(prev_usage_ema.mean().item()) if prev_usage_ema.numel() > 0 else 0.0,
                "score_transform": str(getattr(config, "score_transform", "zscore")),
            }
        )

        particle_rows: List[Dict[str, Any]] = []
        for particle_idx in range(particle_count):
            row_weights = client_weight_rows[particle_idx]
            particle_rows.append(
                {
                    "round": int(round_idx),
                    "block_id": int(block_id),
                    "expert_id": int(expert_id),
                    "particle_id": int(particle_idx),
                    "particle_weight": float(particle_weights[particle_idx].item()),
                    "particle_score": float(particle_scores[particle_idx].item()),
                    "particle_entropy": float(_probability_entropy(row_weights.unsqueeze(0))[0].item()),
                    "particle_max_client_weight": float(row_weights.max().item()),
                    "particle_min_client_weight": float(row_weights.min().item()),
                    "dominant_client_idx": int(torch.argmax(row_weights).item()),
                    "resampled": bool(resampled),
                }
            )
        return theta_new, info, particle_rows
    except _ExpertFallback as exc:
        if pure_vsmc:
            raise RuntimeError(
                f"Pure VSMC expert aggregation encountered forbidden fallback for {expert_scope}: {exc}"
            ) from exc
        state.trust_logits[block_id, expert_id] = torch.zeros_like(state.trust_logits[block_id, expert_id])
        state.particle_log_weights[block_id, expert_id] = torch.zeros_like(state.particle_log_weights[block_id, expert_id])
        info = dict(fallback_info)
        info.update(
            {
                "lambda_expert": 0.0,
                "anchor_mode": str(config.anchor_mode),
                "pure_vsmc": pure_vsmc,
                "fedavg_blend_enabled": False,
                "expert_output": "theta_fedavg_fallback",
                "temperature": float(state.temperature[block_id, expert_id].item()),
                "function_drift_enabled": bool(config.use_function_drift),
                "ESS": float(config.num_particles),
                "ESS_ratio": 1.0,
                "resampled": False,
                "particle_weight_entropy": float(math.log(float(max(config.num_particles, 1)))) if int(config.num_particles) > 1 else 0.0,
                "particle_max_weight": 1.0 / float(max(config.num_particles, 1)),
                "client_weight_entropy": float(math.log(float(max(num_clients, 1)))) if num_clients > 1 else 0.0,
                "client_max_weight": float(expert_weights.max().item()) if expert_weights.numel() > 0 else 1.0,
                "client_min_weight": float(expert_weights.min().item()) if expert_weights.numel() > 0 else 1.0,
                "max_client_weight_mean": float(expert_weights.max().item()) if expert_weights.numel() > 0 else 1.0,
                "min_client_weight_mean": float(expert_weights.min().item()) if expert_weights.numel() > 0 else 1.0,
                "active_client_count_per_expert": int((expert_weights > 0).sum().item()) if expert_weights.numel() > 0 else 0,
                "usage_score_mean": 0.0,
                "usage_score_std": 0.0,
                "history_cos_mean": 0.0,
                "history_cos_std": 0.0,
                "consensus_cos_mean": 0.0,
                "consensus_cos_std": 0.0,
                "normalized_drift_mean": 0.0,
                "normalized_drift_std": 0.0,
                "function_drift_mean": 0.0,
                "function_drift_std": 0.0,
                "observation_score_mean": 0.0,
                "observation_score_std": 0.0,
                "particle_score_mean": 0.0,
                "particle_score_std": 0.0,
                "max_particle_weight": 1.0 / float(max(config.num_particles, 1)),
                "vsmc_fedavg_l2_distance": 0.0,
                "final_fedavg_l2_distance": 0.0,
                "final_residual_ratio": 0.0,
                "velocity_norm": _state_rms(state.expert_velocity_ema[group_key]),
                "second_moment_norm": _state_rms(state.expert_second_moment[group_key]),
                "function_drift_available": False,
                "fallback_used": True,
                "fallback_reason": str(exc),
                "fallback_type": str(exc),
                "confidence": 0.0,
                "weights": expert_weights.detach().cpu().tolist(),
                "particle_weights": [1.0 / float(max(config.num_particles, 1))] * int(config.num_particles),
            }
        )
        return fallback_state, info, []


def _build_base_info(
    round_idx: int,
    block_id: int,
    expert_id: int,
    expert_scope: str,
    num_particles: int,
) -> Dict[str, Any]:
    return {
        "round": int(round_idx),
        "block_id": int(block_id),
        "expert_id": int(expert_id),
        "expert_scope": str(expert_scope),
        "num_particles": int(num_particles),
    }


class _ExpertFallback(RuntimeError):
    pass


def _parse_block_expert_from_name(name: str) -> Optional[Tuple[int, int]]:
    match = _BLOCK_EXPERT_PARAM_RE.search(str(name))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_block_expert_from_scope(expert_scope: str) -> Optional[Tuple[int, int]]:
    match = _BLOCK_EXPERT_SCOPE_RE.search(str(expert_scope))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _clone_state_to_device(state_dict: TensorState, device: torch.device) -> TensorState:
    cloned: TensorState = collections.OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = _clone_detached_tensor(value, device=device)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _zeros_like_state(state_dict: TensorState, device: torch.device) -> TensorState:
    zeros: TensorState = collections.OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            zeros[key] = torch.zeros_like(
                value.detach(),
                dtype=torch.float64 if torch.is_floating_point(value) else value.dtype,
                device=device,
            )
        else:
            zeros[key] = copy.deepcopy(value)
    return zeros


def _subtract_state_dict(a: TensorState, b: TensorState) -> TensorState:
    delta: TensorState = collections.OrderedDict()
    for key in sorted(set(a.keys()) & set(b.keys())):
        value_a = a[key]
        value_b = b[key]
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            continue
        if not (torch.is_floating_point(value_a) and torch.is_floating_point(value_b)):
            continue
        if value_a.shape != value_b.shape:
            continue
        tensor_a = value_a.detach().to(dtype=torch.float64, device="cpu")
        tensor_b = value_b.detach().to(dtype=torch.float64, device="cpu")
        delta[key] = tensor_a - tensor_b
    return delta


def _flatten_float_state(state_dict: TensorState) -> torch.Tensor:
    flat_tensors: List[torch.Tensor] = []
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            flat_tensors.append(value.detach().to(dtype=torch.float64, device="cpu").reshape(-1))
    if not flat_tensors:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(flat_tensors, dim=0)


def _coordinate_median(vectors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    valid = [vec.detach().to(dtype=torch.float64, device=device) for vec in vectors if vec.numel() > 0]
    if not valid:
        return torch.empty(0, dtype=torch.float64, device=device)
    stacked = torch.stack(valid, dim=0)
    return torch.median(stacked, dim=0).values


def _cosine_with_reference(vector: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    if vector.numel() == 0 or reference.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=vector.device if vector.numel() > 0 else reference.device)
    aligned_vector = vector.detach().to(dtype=torch.float64, device=reference.device)
    aligned_reference = reference.detach().to(dtype=torch.float64, device=reference.device)
    if aligned_vector.numel() != aligned_reference.numel():
        min_dim = min(aligned_vector.numel(), aligned_reference.numel())
        aligned_vector = aligned_vector[:min_dim]
        aligned_reference = aligned_reference[:min_dim]
    if aligned_vector.norm().item() <= eps or aligned_reference.norm().item() <= eps:
        return torch.zeros((), dtype=torch.float64, device=reference.device)
    return F.cosine_similarity(aligned_vector.view(1, -1), aligned_reference.view(1, -1), dim=-1, eps=eps)[0]


def _normalized_drift(
    delta_state: TensorState,
    second_moment_state: TensorState,
    eps: float,
    clamp_value: float,
) -> float:
    squared_sum = 0.0
    total_count = 0
    used_second_moment = False
    for key, delta_value in delta_state.items():
        second_moment_value = second_moment_state.get(key)
        if not torch.is_tensor(delta_value) or not torch.is_floating_point(delta_value):
            continue
        term = delta_value.detach().to(dtype=torch.float64, device="cpu")
        if torch.is_tensor(second_moment_value) and torch.is_floating_point(second_moment_value):
            moment = second_moment_value.detach().to(dtype=torch.float64, device="cpu")
            if moment.numel() == term.numel() and float(moment.abs().mean().item()) > eps:
                term = term / (torch.sqrt(moment.clamp_min(0.0)) + eps)
                used_second_moment = True
        squared_sum += float(torch.sum(term * term).item())
        total_count += int(term.numel())
    if total_count <= 0:
        return 0.0
    drift = math.sqrt(squared_sum / float(total_count))
    if not used_second_moment:
        drift = min(drift, clamp_value)
    return max(0.0, min(float(clamp_value), drift))


def _normalize_component(values: torch.Tensor, config: HAVSMCETFConfig) -> torch.Tensor:
    values = values.to(dtype=torch.float64)
    if not bool(config.normalize_observation_components):
        return values
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or float(std.item()) <= float(config.eps):
        return torch.zeros_like(values)
    normalized = (values - values.mean()) / (std + float(config.eps))
    transform = str(getattr(config, "score_transform", "zscore"))
    if transform == "zscore":
        return normalized
    if transform == "zscore_clip":
        return torch.clamp(
            normalized,
            min=float(getattr(config, "score_clip_low", -2.0)),
            max=float(getattr(config, "score_clip_high", 2.0)),
        )
    if transform == "rank_based":
        order = torch.argsort(normalized, dim=0)
        ranks = torch.empty_like(normalized)
        ranks[order] = torch.arange(normalized.numel(), device=normalized.device, dtype=normalized.dtype)
        denom = float(max(normalized.numel() - 1, 1))
        return ranks / denom
    return normalized


def _rank_score(values: torch.Tensor, descending: bool = True) -> torch.Tensor:
    values = values.to(dtype=torch.float64)
    if values.numel() <= 1:
        return torch.ones_like(values)
    order = torch.argsort(values, descending=descending)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    denom = float(max(values.numel() - 1, 1))
    return 1.0 - ranks / denom


def _normalize_probability_vector(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.to(dtype=torch.float64)
    weights = torch.clamp(weights, min=0.0)
    total = float(weights.sum().item())
    if total <= eps:
        if weights.numel() <= 0:
            return weights
        return torch.full_like(weights, 1.0 / float(weights.numel()))
    return weights / total


def _apply_top_k(weights: torch.Tensor, top_k: int, eps: float) -> Tuple[torch.Tensor, int]:
    if int(top_k) <= 0 or weights.numel() <= int(top_k):
        return _normalize_probability_vector(weights, eps), int(weights.numel())
    selected = torch.topk(weights, int(top_k)).indices
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask[selected] = True
    filtered = torch.where(mask, weights, torch.zeros_like(weights))
    return _normalize_probability_vector(filtered, eps), int(mask.sum().item())


def _apply_entropy_target_temperature(weights: torch.Tensor, config: HAVSMCETFConfig, eps: float) -> torch.Tensor:
    mode = str(getattr(config, "entropy_target", "none"))
    if mode != "moderate":
        return _normalize_probability_vector(weights, eps)
    max_w = float(weights.max().item()) if weights.numel() > 0 else 0.0
    target_lo = float(getattr(config, "entropy_target_min_max_weight", 0.25))
    target_hi = float(getattr(config, "entropy_target_max_max_weight", 0.55))
    logits = torch.log(weights.clamp_min(eps))
    if max_w < target_lo:
        adjusted = torch.softmax(logits / 0.7, dim=0)
        return _normalize_probability_vector(adjusted, eps)
    if max_w > target_hi:
        adjusted = torch.softmax(logits / 1.3, dim=0)
        return _normalize_probability_vector(adjusted, eps)
    return _normalize_probability_vector(weights, eps)


def _effective_num_clients(weights: torch.Tensor, eps: float) -> float:
    weights = _normalize_probability_vector(weights, eps)
    return float((1.0 / weights.pow(2).sum().clamp_min(eps)).item()) if weights.numel() > 0 else 0.0


def _build_rank_prior(num_clients: int, device: torch.device) -> torch.Tensor:
    if num_clients <= 0:
        return torch.empty(0, dtype=torch.float64, device=device)
    scores = torch.arange(num_clients, 0, -1, dtype=torch.float64, device=device)
    return _normalize_probability_vector(scores, _EPS)


def _build_anti_collapse_prior(config: HAVSMCETFConfig, usage_tensor: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    prior_mode = str(getattr(config, "anti_collapse_prior", "usage"))
    active_mask = (usage_tensor > 0.0).to(dtype=torch.float64)
    if prior_mode == "uniform_active":
        return _normalize_probability_vector(active_mask, eps)
    if prior_mode == "rank_prior":
        return _build_rank_prior(int(weights.numel()), device=weights.device)
    return _normalize_probability_vector(torch.clamp(usage_tensor, min=0.0), eps)


def _prediction_noise_like(tensor: torch.Tensor, config: HAVSMCETFConfig, offset: int) -> torch.Tensor:
    if float(config.prediction_noise_std) <= 0.0:
        return torch.zeros_like(tensor, dtype=torch.float64)
    generator = None
    if bool(config.deterministic):
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(int(config.seed) + int(offset))
    return torch.randn(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=torch.float64,
    ) * float(config.prediction_noise_std)


def _probability_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.to(dtype=torch.float64)
    return -(probs.clamp_min(_EPS) * torch.log(probs.clamp_min(_EPS))).sum(dim=-1)


def _blend_state_dicts(base_state: TensorState, residual_state: TensorState, residual_weight: float) -> TensorState:
    mix = float(max(0.0, min(1.0, residual_weight)))
    out: TensorState = collections.OrderedDict()
    for key in _ordered_union_keys([base_state, residual_state]):
        if key not in base_state:
            out[key] = _clone_detached_tensor(residual_state[key]) if torch.is_tensor(residual_state[key]) else copy.deepcopy(residual_state[key])
            continue
        if key not in residual_state:
            out[key] = _clone_detached_tensor(base_state[key]) if torch.is_tensor(base_state[key]) else copy.deepcopy(base_state[key])
            continue
        value_a = base_state[key]
        value_b = residual_state[key]
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            out[key] = copy.deepcopy(value_b)
            continue
        if not (torch.is_floating_point(value_a) and torch.is_floating_point(value_b)) or value_a.shape != value_b.shape:
            out[key] = _clone_detached_tensor(value_b)
            continue
        tensor_a = value_a.detach().to(dtype=torch.float64)
        tensor_b = value_b.detach().to(dtype=torch.float64)
        out[key] = ((1.0 - mix) * tensor_a + mix * tensor_b).to(dtype=value_a.dtype)
    return out


def _state_has_nan(state_dict: TensorState) -> bool:
    for value in state_dict.values():
        if torch.is_tensor(value) and torch.is_floating_point(value) and torch.isnan(value).any():
            return True
    return False


def _ema_state_update(old_state: TensorState, delta_state: TensorState, beta: float) -> TensorState:
    updated: TensorState = collections.OrderedDict()
    keep = float(beta)
    take = 1.0 - keep
    for key in _ordered_union_keys([old_state, delta_state]):
        if key not in old_state:
            updated[key] = _clone_detached_tensor(delta_state[key]) if torch.is_tensor(delta_state[key]) else copy.deepcopy(delta_state[key])
            continue
        if key not in delta_state:
            updated[key] = _clone_detached_tensor(old_state[key]) if torch.is_tensor(old_state[key]) else copy.deepcopy(old_state[key])
            continue
        old_value = old_state[key]
        delta_value = delta_state[key]
        if not (torch.is_tensor(old_value) and torch.is_tensor(delta_value)):
            updated[key] = copy.deepcopy(delta_value)
            continue
        if not (torch.is_floating_point(old_value) and torch.is_floating_point(delta_value)) or old_value.shape != delta_value.shape:
            updated[key] = _clone_detached_tensor(delta_value)
            continue
        updated[key] = (
            keep * old_value.detach().to(dtype=torch.float64, device=old_value.device)
            + take * delta_value.detach().to(dtype=torch.float64, device=old_value.device)
        )
    return updated


def _ema_state_update_squared(old_state: TensorState, delta_state: TensorState, beta: float) -> TensorState:
    squared_delta: TensorState = collections.OrderedDict()
    for key, value in delta_state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            squared_delta[key] = value.detach().to(dtype=torch.float64, device=value.device) ** 2
    return _ema_state_update(old_state, squared_delta, beta=beta)


def _state_rms(state_dict: TensorState) -> float:
    flat = _flatten_float_state(state_dict)
    if flat.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(flat * flat)).item())


def _state_l2_distance(a: TensorState, b: TensorState) -> float:
    squared_sum = 0.0
    for key in sorted(set(a.keys()) & set(b.keys())):
        value_a = a[key]
        value_b = b[key]
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            continue
        if not (torch.is_floating_point(value_a) and torch.is_floating_point(value_b)):
            continue
        if value_a.shape != value_b.shape:
            continue
        diff = value_a.detach().to(dtype=torch.float64, device="cpu") - value_b.detach().to(dtype=torch.float64, device="cpu")
        squared_sum += float(torch.sum(diff * diff).item())
    return math.sqrt(squared_sum)


def _client_weights_from_mode(
    client_updates: Sequence[Tuple[TensorState, int, Dict[str, Any]]],
    mode: str,
) -> torch.Tensor:
    normalized_mode = str(mode).lower()
    num_clients = len(client_updates)
    if num_clients <= 0:
        raise ValueError("Cannot compute client weights for an empty update list")
    if normalized_mode == "sample_count":
        counts = torch.tensor([max(int(num_samples), 0) for _, num_samples, _ in client_updates], dtype=torch.float64)
        total = counts.sum()
        if float(total.item()) <= 0.0:
            return torch.full((num_clients,), 1.0 / float(num_clients), dtype=torch.float64)
        return counts / total
    return torch.full((num_clients,), 1.0 / float(num_clients), dtype=torch.float64)


def _extract_router_usage_tensor(
    metrics: Dict[str, Any],
    num_blocks: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    metrics = metrics or {}
    if "router_usage" in metrics and metrics["router_usage"] is not None:
        usage = metrics["router_usage"]
        if torch.is_tensor(usage):
            usage_tensor = usage.detach().to(dtype=torch.float64, device=device)
            if usage_tensor.dim() == 2:
                return usage_tensor
    usage_tensor = torch.zeros(num_blocks, num_experts, dtype=torch.float64, device=device)
    layer_route_counts = metrics.get("layer_route_counts", {}) or {}
    for raw_layer_idx, expert_counts in layer_route_counts.items():
        try:
            layer_idx = int(raw_layer_idx)
        except (TypeError, ValueError):
            continue
        if layer_idx < 0 or layer_idx >= num_blocks:
            continue
        if not isinstance(expert_counts, dict):
            continue
        for raw_expert_idx, value in expert_counts.items():
            try:
                expert_idx = int(raw_expert_idx)
            except (TypeError, ValueError):
                continue
            if expert_idx < 0 or expert_idx >= num_experts:
                continue
            usage_tensor[layer_idx, expert_idx] = float(value)
    return usage_tensor


def _systematic_resampling(
    weights: torch.Tensor,
    config: HAVSMCETFConfig,
    offset: int,
) -> torch.Tensor:
    normalized = weights.detach().to(dtype=torch.float64)
    normalized = normalized / normalized.sum().clamp_min(_EPS)
    n = int(normalized.numel())
    if n <= 0:
        return torch.empty(0, dtype=torch.long, device=weights.device)
    if bool(config.deterministic):
        start = torch.tensor(0.5 / float(n), dtype=torch.float64, device=weights.device)
    else:
        generator = torch.Generator(device=weights.device)
        generator.manual_seed(int(config.seed) + int(offset))
        start = torch.rand((), generator=generator, dtype=torch.float64, device=weights.device) / float(n)
    positions = start + torch.arange(n, dtype=torch.float64, device=weights.device) / float(n)
    cumulative = torch.cumsum(normalized, dim=0)
    return torch.searchsorted(cumulative, positions, right=False).clamp(max=n - 1).long()


def _merge_expert_groups(expert_groups: Dict[str, Dict[str, Any]]) -> TensorState:
    merged: TensorState = collections.OrderedDict()
    for group in expert_groups.values():
        for key, value in group["state_dict"].items():
            merged[key] = value
    return merged


def _safe_mean(values: Iterable[Any]) -> float:
    numeric: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            cast_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cast_value):
            numeric.append(cast_value)
    if not numeric:
        return 0.0
    return sum(numeric) / float(len(numeric))


def _compute_function_drifts_for_clients(
    global_model: nn.Module,
    block_id: int,
    expert_id: int,
    client_expert_states: Sequence[TensorState],
    probe_hidden_states: torch.Tensor,
    clamp_value: float,
    metric: str,
) -> torch.Tensor:
    device = probe_hidden_states.device
    old_expert = global_model.blocks[block_id].switch_ffn.experts[expert_id]
    with torch.no_grad():
        old_expert = old_expert.to(device)
        old_output = old_expert(probe_hidden_states)
        old_flat = old_output.reshape(1, -1)
    drifts: List[torch.Tensor] = []
    for client_state in client_expert_states:
        candidate_expert = copy.deepcopy(old_expert).to(device)
        local_state = _extract_local_expert_state(client_state, block_id, expert_id, device=device)
        candidate_expert.load_state_dict(local_state, strict=False)
        with torch.no_grad():
            new_output = candidate_expert(probe_hidden_states)
            similarity = F.cosine_similarity(
                old_flat,
                new_output.reshape(1, -1),
                dim=-1,
                eps=1e-12,
            )
            drift = 1.0 - similarity.mean()
            drifts.append(torch.clamp(drift, min=0.0, max=float(clamp_value)))
    if not drifts:
        return torch.empty(0, dtype=torch.float64, device=device)
    return torch.stack(drifts).to(dtype=torch.float64, device=device)


def _extract_local_expert_state(
    expert_state: TensorState,
    block_id: int,
    expert_id: int,
    device: torch.device,
) -> TensorState:
    prefix = f"blocks.{block_id}.switch_ffn.experts.{expert_id}."
    local_state: TensorState = collections.OrderedDict()
    for name, value in expert_state.items():
        if not str(name).startswith(prefix):
            continue
        local_name = str(name)[len(prefix):]
        if torch.is_tensor(value):
            local_state[local_name] = value.detach().clone().to(device=device)
        else:
            local_state[local_name] = copy.deepcopy(value)
    return local_state
