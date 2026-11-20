#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from args import parse
import server as server_module
from aggregation import history_aware_vsmc_expert_aggregator as ha
from aggregation.vsmc_expert_aggregator import (
    _merge_with_reference_state,
    clone_detached_state_dict,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)


_ABALATION_METHODS = (
    "freq_weighted",
    "freq_vsmc",
    "multi_signal_weighted",
    "uc_weighted",
    "uc_vsmc",
)
_EPS = 1e-12


def _build_custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reliability_ablation_method",
        type=str,
        required=True,
        choices=_ABALATION_METHODS,
    )
    return parser


def _extract_custom_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    custom_parser = _build_custom_parser()
    custom_args, remaining = custom_parser.parse_known_args(list(argv))
    return custom_args, remaining


def _usage_fraction_for_expert(
    metrics: Dict[str, Any],
    block_id: int,
    expert_id: int,
    num_blocks: int,
    num_experts: int,
    device: torch.device,
) -> Tuple[float, float]:
    usage_tensor = ha._extract_router_usage_tensor(
        metrics=metrics,
        num_blocks=num_blocks,
        num_experts=num_experts,
        device=device,
    ).to(dtype=torch.float64, device=device)
    if block_id < 0 or block_id >= int(usage_tensor.shape[0]):
        return 0.0, 0.0
    block_usage = usage_tensor[block_id]
    if expert_id < 0 or expert_id >= int(block_usage.numel()):
        return 0.0, 0.0
    raw_count = float(block_usage[expert_id].item())
    total_count = float(block_usage.sum().item())
    if total_count <= _EPS:
        return 0.0, raw_count
    return raw_count / total_count, raw_count


def _collect_usage_fractions(
    client_metrics: Sequence[Dict[str, Any]],
    block_id: int,
    expert_id: int,
    state: Optional[ha.HAVSMCETFState],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if state is not None:
        num_blocks = int(state.num_blocks)
        num_experts = int(state.num_experts_per_block)
    else:
        max_block = block_id + 1
        max_expert = expert_id + 1
        for metrics in client_metrics:
            usage = metrics.get("router_usage")
            if torch.is_tensor(usage) and usage.dim() == 2:
                max_block = max(max_block, int(usage.shape[0]))
                max_expert = max(max_expert, int(usage.shape[1]))
        num_blocks = max_block
        num_experts = max_expert

    fractions: List[float] = []
    counts: List[float] = []
    for metrics in client_metrics:
        fraction, count = _usage_fraction_for_expert(
            metrics=metrics or {},
            block_id=block_id,
            expert_id=expert_id,
            num_blocks=num_blocks,
            num_experts=num_experts,
            device=device,
        )
        fractions.append(float(fraction))
        counts.append(float(count))
    return (
        torch.tensor(fractions, dtype=torch.float64, device=device),
        torch.tensor(counts, dtype=torch.float64, device=device),
    )


def _compute_shared_components(
    global_model,
    global_expert_state,
    client_expert_states,
    client_metrics,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
    block_id: int,
    expert_id: int,
    group_key: Tuple[int, int],
    probe_batch,
    probe_hidden_cache: Dict[int, Optional[torch.Tensor]],
    round_idx: int,
) -> Dict[str, Any]:
    device = state.trust_logits.device
    usage_fractions, raw_usage_counts = _collect_usage_fractions(
        client_metrics=client_metrics,
        block_id=block_id,
        expert_id=expert_id,
        state=state,
        device=device,
    )
    delta_states = [
        ha._subtract_state_dict(client_state, global_expert_state)
        for client_state in client_expert_states
    ]
    delta_vecs = [
        ha._flatten_float_state(delta_state).to(dtype=torch.float64, device=device)
        for delta_state in delta_states
    ]
    if all(
        vec.numel() == 0 or torch.allclose(vec, torch.zeros_like(vec), atol=1e-12)
        for vec in delta_vecs
    ):
        raise ha._ExpertFallback("all_client_delta_zero")

    robust_center = ha._coordinate_median(delta_vecs, device=device)
    consensus_cos = torch.stack(
        [ha._cosine_with_reference(vec, robust_center, eps=float(config.eps)) for vec in delta_vecs],
        dim=0,
    )
    velocity_vec = ha._flatten_float_state(state.expert_velocity_ema[group_key]).to(
        dtype=torch.float64,
        device=device,
    )
    history_cos = torch.stack(
        [ha._cosine_with_reference(vec, velocity_vec, eps=float(config.eps)) for vec in delta_vecs],
        dim=0,
    )
    drift_values = torch.tensor(
        [
            ha._normalized_drift(
                delta_state,
                state.expert_second_moment[group_key],
                eps=float(config.eps),
                clamp_value=float(config.drift_clamp),
            )
            for delta_state in delta_states
        ],
        dtype=torch.float64,
        device=device,
    )

    function_drift_available = False
    function_drift_values = torch.zeros(len(client_expert_states), dtype=torch.float64, device=device)
    function_drift_failure = ""
    if bool(config.use_function_drift) and int(config.function_drift_interval) > 0:
        should_run = (int(round_idx) - 1) % max(int(config.function_drift_interval), 1) == 0
        if should_run:
            try:
                if block_id not in probe_hidden_cache:
                    probe_hidden_cache[block_id] = ha.collect_probe_hidden_states(global_model, probe_batch, block_id)
                probe_hidden_states = probe_hidden_cache.get(block_id)
                if probe_hidden_states is not None:
                    function_drift_values = ha._compute_function_drifts_for_clients(
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
        raise ha._ExpertFallback("function_drift_unavailable")

    history_scale = (
        min(1.0, float(round_idx) / float(max(int(config.lambda_warmup_rounds), 1)))
        if int(config.lambda_warmup_rounds) > 0
        else 1.0
    )
    return {
        "usage_fractions": usage_fractions,
        "raw_usage_counts": raw_usage_counts,
        "delta_states": delta_states,
        "delta_vecs": delta_vecs,
        "consensus_cos": consensus_cos,
        "history_cos": history_cos,
        "drift_values": drift_values,
        "function_drift_values": function_drift_values,
        "function_drift_available": bool(function_drift_available),
        "function_drift_failure": function_drift_failure,
        "history_scale": float(history_scale),
    }


def _finalize_state_update(
    state: ha.HAVSMCETFState,
    group_key: Tuple[int, int],
    global_expert_state,
    theta_new,
    config: ha.HAVSMCETFConfig,
) -> None:
    global_delta = ha._subtract_state_dict(theta_new, global_expert_state)
    state.expert_velocity_ema[group_key] = ha._ema_state_update(
        state.expert_velocity_ema[group_key],
        global_delta,
        beta=float(config.expert_velocity_beta),
    )
    state.expert_second_moment[group_key] = ha._ema_state_update_squared(
        state.expert_second_moment[group_key],
        global_delta,
        beta=float(config.second_moment_beta),
    )
    state.prev_global_experts[group_key] = ha._clone_state_to_device(theta_new, device=state.device)


def _direct_weighted_info(
    round_idx: int,
    block_id: int,
    expert_id: int,
    expert_scope: str,
    weights: torch.Tensor,
    usage_fractions: torch.Tensor,
    observation_scores: torch.Tensor,
    theta_new,
    theta_fedavg,
    fallback_used: bool,
    fallback_reason: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnostics = diagnostics or {}
    final_fedavg_l2_distance = ha._state_l2_distance(theta_new, theta_fedavg)
    info = ha._build_base_info(
        round_idx=round_idx,
        block_id=block_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        num_particles=0,
    )
    info.update(
        {
            "lambda_expert": 1.0,
            "anchor_mode": "none",
            "temperature": 0.0,
            "ESS": 0.0,
            "ESS_ratio": 0.0,
            "resampled": False,
            "particle_weight_entropy": 0.0,
            "particle_max_weight": 0.0,
            "client_weight_entropy": float(ha._probability_entropy(weights.unsqueeze(0))[0].item()),
            "client_max_weight": float(weights.max().item()) if weights.numel() > 0 else 0.0,
            "client_min_weight": float(weights.min().item()) if weights.numel() > 0 else 0.0,
            "usage_score_mean": float(usage_fractions.mean().item()) if usage_fractions.numel() > 0 else 0.0,
            "usage_score_std": float(usage_fractions.std(unbiased=False).item()) if usage_fractions.numel() > 0 else 0.0,
            "history_cos_mean": float(diagnostics.get("history_cos_mean", 0.0)),
            "history_cos_std": float(diagnostics.get("history_cos_std", 0.0)),
            "consensus_cos_mean": float(diagnostics.get("consensus_cos_mean", 0.0)),
            "consensus_cos_std": float(diagnostics.get("consensus_cos_std", 0.0)),
            "normalized_drift_mean": float(diagnostics.get("normalized_drift_mean", 0.0)),
            "normalized_drift_std": float(diagnostics.get("normalized_drift_std", 0.0)),
            "function_drift_mean": float(diagnostics.get("function_drift_mean", 0.0)),
            "function_drift_std": float(diagnostics.get("function_drift_std", 0.0)),
            "observation_score_mean": float(observation_scores.mean().item()) if observation_scores.numel() > 0 else 0.0,
            "observation_score_std": float(observation_scores.std(unbiased=False).item()) if observation_scores.numel() > 0 else 0.0,
            "particle_score_mean": 0.0,
            "particle_score_std": 0.0,
            "vsmc_fedavg_l2_distance": float(final_fedavg_l2_distance),
            "final_fedavg_l2_distance": float(final_fedavg_l2_distance),
            "final_residual_ratio": 1.0 if final_fedavg_l2_distance > 0.0 else 0.0,
            "velocity_norm": float(diagnostics.get("velocity_norm", 0.0)),
            "second_moment_norm": float(diagnostics.get("second_moment_norm", 0.0)),
            "function_drift_available": bool(diagnostics.get("function_drift_available", False)),
            "fallback_used": bool(fallback_used),
            "fallback_reason": str(fallback_reason),
            "confidence": 0.0,
            "weights": weights.detach().cpu().tolist(),
            "particle_weights": [],
            "usage_score_values": usage_fractions.detach().cpu().tolist(),
            "history_cos_values": list(diagnostics.get("history_cos_values", [])),
            "consensus_cos_values": list(diagnostics.get("consensus_cos_values", [])),
            "normalized_drift_values": list(diagnostics.get("normalized_drift_values", [])),
            "function_drift_values": list(diagnostics.get("function_drift_values", [])),
            "observation_score_values": observation_scores.detach().cpu().tolist(),
            "particle_score_values": [],
        }
    )
    return info


def _aggregate_freq_weighted_expert(
    global_expert_state,
    client_expert_states,
    client_metrics,
    expert_weights: torch.Tensor,
    block_id: int,
    expert_id: int,
    expert_scope: str,
    round_idx: int,
    num_blocks: int,
    num_experts: int,
    device: torch.device,
):
    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        expert_weights,
        fallback_state=global_expert_state,
    )
    usage_fractions, _ = _collect_usage_fractions(
        client_metrics=client_metrics,
        block_id=block_id,
        expert_id=expert_id,
        state=None,
        device=device,
    )
    if usage_fractions.sum().item() <= _EPS:
        weights = expert_weights.clone()
        theta_new = clone_detached_state_dict(theta_fedavg)
        info = _direct_weighted_info(
            round_idx=round_idx,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            weights=weights,
            usage_fractions=usage_fractions,
            observation_scores=usage_fractions,
            theta_new=theta_new,
            theta_fedavg=theta_fedavg,
            fallback_used=True,
            fallback_reason="all_router_usage_zero",
        )
        return theta_new, info, []

    weights = usage_fractions / usage_fractions.sum().clamp_min(_EPS)
    theta_new = weighted_average_state_dicts(
        list(client_expert_states),
        weights,
        fallback_state=theta_fedavg,
    )
    info = _direct_weighted_info(
        round_idx=round_idx,
        block_id=block_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        weights=weights,
        usage_fractions=usage_fractions,
        observation_scores=usage_fractions,
        theta_new=theta_new,
        theta_fedavg=theta_fedavg,
        fallback_used=False,
        fallback_reason="",
    )
    return theta_new, info, []


def _aggregate_multi_signal_weighted_expert(
    global_model,
    global_expert_state,
    client_expert_states,
    client_metrics,
    expert_weights: torch.Tensor,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
    group_key: Tuple[int, int],
    expert_scope: str,
    probe_batch,
    probe_hidden_cache: Dict[int, Optional[torch.Tensor]],
    round_idx: int,
):
    block_id, expert_id = group_key
    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        expert_weights,
        fallback_state=global_expert_state,
    )
    try:
        components = _compute_shared_components(
            global_model=global_model,
            global_expert_state=global_expert_state,
            client_expert_states=client_expert_states,
            client_metrics=client_metrics,
            state=state,
            config=config,
            block_id=block_id,
            expert_id=expert_id,
            group_key=group_key,
            probe_batch=probe_batch,
            probe_hidden_cache=probe_hidden_cache,
            round_idx=round_idx,
        )
        usage_norm = ha._normalize_component(components["usage_fractions"], config)
        history_norm = ha._normalize_component(components["history_cos"], config)
        consensus_norm = ha._normalize_component(components["consensus_cos"], config)
        drift_norm = ha._normalize_component(components["drift_values"], config)
        function_norm = ha._normalize_component(components["function_drift_values"], config)
        effective_alpha_history = float(config.alpha_history_direction) * float(components["history_scale"])
        observation_scores = (
            float(config.alpha_usage) * usage_norm
            + float(effective_alpha_history) * history_norm
            + float(config.alpha_consensus) * consensus_norm
            - float(config.alpha_drift) * drift_norm
            - float(config.alpha_function_drift) * function_norm
        )
        if torch.isnan(observation_scores).any():
            raise ha._ExpertFallback("nan_detected")
        weights = torch.softmax(observation_scores, dim=0)
        theta_new = weighted_average_state_dicts(
            list(client_expert_states),
            weights,
            fallback_state=theta_fedavg,
        )
        if ha._state_has_nan(theta_new):
            raise ha._ExpertFallback("nan_detected")
        _finalize_state_update(
            state=state,
            group_key=group_key,
            global_expert_state=global_expert_state,
            theta_new=theta_new,
            config=config,
        )
        diagnostics = {
            "history_cos_mean": float(components["history_cos"].mean().item()),
            "history_cos_std": float(components["history_cos"].std(unbiased=False).item()),
            "consensus_cos_mean": float(components["consensus_cos"].mean().item()),
            "consensus_cos_std": float(components["consensus_cos"].std(unbiased=False).item()),
            "normalized_drift_mean": float(components["drift_values"].mean().item()),
            "normalized_drift_std": float(components["drift_values"].std(unbiased=False).item()),
            "function_drift_mean": float(components["function_drift_values"].mean().item()),
            "function_drift_std": float(components["function_drift_values"].std(unbiased=False).item()),
            "history_cos_values": components["history_cos"].detach().cpu().tolist(),
            "consensus_cos_values": components["consensus_cos"].detach().cpu().tolist(),
            "normalized_drift_values": components["drift_values"].detach().cpu().tolist(),
            "function_drift_values": components["function_drift_values"].detach().cpu().tolist(),
            "velocity_norm": float(ha._state_rms(state.expert_velocity_ema[group_key])),
            "second_moment_norm": float(ha._state_rms(state.expert_second_moment[group_key])),
            "function_drift_available": bool(components["function_drift_available"]),
        }
        info = _direct_weighted_info(
            round_idx=round_idx,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            weights=weights,
            usage_fractions=components["usage_fractions"],
            observation_scores=observation_scores,
            theta_new=theta_new,
            theta_fedavg=theta_fedavg,
            fallback_used=False,
            fallback_reason=components["function_drift_failure"],
            diagnostics=diagnostics,
        )
        return theta_new, info, []
    except ha._ExpertFallback as exc:
        info = _direct_weighted_info(
            round_idx=round_idx,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            weights=expert_weights,
            usage_fractions=torch.zeros_like(expert_weights, dtype=torch.float64),
            observation_scores=torch.zeros_like(expert_weights, dtype=torch.float64),
            theta_new=theta_fedavg,
            theta_fedavg=theta_fedavg,
            fallback_used=True,
            fallback_reason=str(exc),
        )
        return clone_detached_state_dict(theta_fedavg), info, []


def _aggregate_uc_weighted_expert(
    global_model,
    global_expert_state,
    client_expert_states,
    client_metrics,
    expert_weights: torch.Tensor,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
    group_key: Tuple[int, int],
    expert_scope: str,
    probe_batch,
    probe_hidden_cache: Dict[int, Optional[torch.Tensor]],
    round_idx: int,
):
    block_id, expert_id = group_key
    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        expert_weights,
        fallback_state=global_expert_state,
    )
    try:
        components = _compute_shared_components(
            global_model=global_model,
            global_expert_state=global_expert_state,
            client_expert_states=client_expert_states,
            client_metrics=client_metrics,
            state=state,
            config=config,
            block_id=block_id,
            expert_id=expert_id,
            group_key=group_key,
            probe_batch=probe_batch,
            probe_hidden_cache=probe_hidden_cache,
            round_idx=round_idx,
        )
        usage_norm = ha._normalize_component(components["usage_fractions"], config)
        consensus_norm = ha._normalize_component(components["consensus_cos"], config)
        observation_scores = (
            float(config.alpha_usage) * usage_norm
            + float(config.alpha_consensus) * consensus_norm
        )
        if torch.isnan(observation_scores).any():
            raise ha._ExpertFallback("nan_detected")
        weights = torch.softmax(observation_scores, dim=0)
        theta_new = weighted_average_state_dicts(
            list(client_expert_states),
            weights,
            fallback_state=theta_fedavg,
        )
        if ha._state_has_nan(theta_new):
            raise ha._ExpertFallback("nan_detected")
        _finalize_state_update(
            state=state,
            group_key=group_key,
            global_expert_state=global_expert_state,
            theta_new=theta_new,
            config=config,
        )
        diagnostics = {
            "history_cos_mean": 0.0,
            "history_cos_std": 0.0,
            "consensus_cos_mean": float(components["consensus_cos"].mean().item()),
            "consensus_cos_std": float(components["consensus_cos"].std(unbiased=False).item()),
            "normalized_drift_mean": 0.0,
            "normalized_drift_std": 0.0,
            "function_drift_mean": 0.0,
            "function_drift_std": 0.0,
            "history_cos_values": [],
            "consensus_cos_values": components["consensus_cos"].detach().cpu().tolist(),
            "normalized_drift_values": [],
            "function_drift_values": [],
            "velocity_norm": float(ha._state_rms(state.expert_velocity_ema[group_key])),
            "second_moment_norm": float(ha._state_rms(state.expert_second_moment[group_key])),
            "function_drift_available": False,
        }
        info = _direct_weighted_info(
            round_idx=round_idx,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            weights=weights,
            usage_fractions=components["usage_fractions"],
            observation_scores=observation_scores,
            theta_new=theta_new,
            theta_fedavg=theta_fedavg,
            fallback_used=False,
            fallback_reason="",
            diagnostics=diagnostics,
        )
        return theta_new, info, []
    except ha._ExpertFallback as exc:
        info = _direct_weighted_info(
            round_idx=round_idx,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            weights=expert_weights,
            usage_fractions=torch.zeros_like(expert_weights, dtype=torch.float64),
            observation_scores=torch.zeros_like(expert_weights, dtype=torch.float64),
            theta_new=theta_fedavg,
            theta_fedavg=theta_fedavg,
            fallback_used=True,
            fallback_reason=str(exc),
        )
        return clone_detached_state_dict(theta_fedavg), info, []


def _aggregate_freq_vsmc_expert(
    global_expert_state,
    client_expert_states,
    client_metrics,
    expert_weights: torch.Tensor,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
    group_key: Tuple[int, int],
    expert_scope: str,
    round_idx: int,
):
    block_id, expert_id = group_key
    device = state.trust_logits.device
    particle_count = int(config.num_particles)
    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        expert_weights,
        fallback_state=global_expert_state,
    )
    fallback_state = clone_detached_state_dict(theta_fedavg)
    fallback_info = ha._build_base_info(
        round_idx=round_idx,
        block_id=block_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        num_particles=particle_count,
    )
    try:
        usage_fractions, _ = _collect_usage_fractions(
            client_metrics=client_metrics,
            block_id=block_id,
            expert_id=expert_id,
            state=state,
            device=device,
        )
        if not bool((usage_fractions > 0.0).any().item()):
            raise ha._ExpertFallback("all_router_usage_zero")
        observation_scores = usage_fractions.to(dtype=torch.float64, device=device)

        trust_logits_prev = state.trust_logits[block_id, expert_id].clone()
        reliability_vec = state.client_reliability[block_id, expert_id].clone()
        trust_logits_pred = (
            trust_logits_prev
            + float(config.reliability_momentum_lr) * reliability_vec.unsqueeze(0)
            + ha._prediction_noise_like(
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
            raise ha._ExpertFallback("nan_detected")

        temperature = float(state.temperature[block_id, expert_id].item())
        client_weight_rows = torch.softmax(
            trust_logits / max(temperature, float(config.eps)),
            dim=-1,
        )
        if not torch.allclose(
            client_weight_rows.sum(dim=-1),
            torch.ones(
                particle_count,
                dtype=client_weight_rows.dtype,
                device=client_weight_rows.device,
            ),
            atol=1e-5,
        ):
            raise ha._ExpertFallback("particle_weights_invalid")

        theta_particles = [
            weighted_average_state_dicts(
                list(client_expert_states),
                client_weight_rows[particle_idx],
                fallback_state=global_expert_state,
            )
            for particle_idx in range(particle_count)
        ]
        particle_scores = torch.sum(
            client_weight_rows * observation_scores.unsqueeze(0),
            dim=-1,
        ) - float(config.collapse_penalty) * client_weight_rows.max(dim=-1).values
        if float(config.small_entropy_bonus) > 0.0 and int(client_weight_rows.shape[-1]) > 1:
            entropy = ha._probability_entropy(client_weight_rows)
            entropy_norm = entropy / math.log(float(client_weight_rows.shape[-1]))
            particle_scores = particle_scores + float(config.small_entropy_bonus) * entropy_norm

        particle_log_weights = state.particle_log_weights[block_id, expert_id].clone() + particle_scores
        particle_log_weights = particle_log_weights - torch.logsumexp(particle_log_weights, dim=0)
        particle_weights = torch.softmax(particle_log_weights, dim=0)
        if not torch.allclose(
            particle_weights.sum(),
            torch.tensor(1.0, dtype=particle_weights.dtype, device=particle_weights.device),
            atol=1e-5,
        ):
            raise ha._ExpertFallback("particle_weights_invalid")
        if torch.isnan(particle_weights).any():
            raise ha._ExpertFallback("nan_detected")

        ess = 1.0 / particle_weights.pow(2).sum().clamp_min(float(config.eps))
        ess_ratio = float((ess / float(max(particle_count, 1))).item())
        theta_vsmc = weighted_average_state_dicts(
            theta_particles,
            particle_weights,
            fallback_state=theta_fedavg,
        )
        confidence = float(
            particle_scores.mean().item()
            - particle_scores.std(unbiased=False).item()
        )
        warmup = (
            min(1.0, float(round_idx) / float(max(int(config.lambda_warmup_rounds), 1)))
            if int(config.lambda_warmup_rounds) > 0
            else 1.0
        )
        lambda_expert = (
            float(config.lambda_max)
            * warmup
            * float(torch.sigmoid(torch.tensor(confidence)).item())
            * ess_ratio
        )
        lambda_expert = max(0.0, min(float(config.lambda_max), lambda_expert))
        anchor_mode = str(config.anchor_mode)
        if anchor_mode == "none":
            theta_new = clone_detached_state_dict(theta_vsmc)
            logged_lambda_expert = 1.0
        elif anchor_mode == "fedavg":
            theta_new = ha._blend_state_dicts(theta_fedavg, theta_vsmc, lambda_expert)
            logged_lambda_expert = float(lambda_expert)
        else:
            raise ValueError(f"Unsupported HA-VSMC anchor mode: {anchor_mode}")
        if ha._state_has_nan(theta_new):
            raise ha._ExpertFallback("nan_detected")

        client_weight_final = torch.sum(
            particle_weights.unsqueeze(-1) * client_weight_rows,
            dim=0,
        )
        reliability_target = observation_scores
        if bool(config.reliability_weighted_update):
            reliability_target = client_weight_final * reliability_target
        state.client_reliability[block_id, expert_id] = (
            float(config.client_reliability_beta) * state.client_reliability[block_id, expert_id]
            + (1.0 - float(config.client_reliability_beta)) * reliability_target
        )

        _finalize_state_update(
            state=state,
            group_key=group_key,
            global_expert_state=global_expert_state,
            theta_new=theta_new,
            config=config,
        )

        if ess_ratio < 0.4:
            temperature *= 1.05
        elif ess_ratio > 0.8 and confidence > 0.0:
            temperature *= 0.98
        temperature = max(float(config.temperature_min), min(float(config.temperature_max), temperature))
        state.temperature[block_id, expert_id] = temperature

        resampled = False
        if float(ess.item()) < float(config.ess_threshold_ratio) * float(particle_count):
            resampled = True
            selected = ha._systematic_resampling(
                particle_weights,
                config=config,
                offset=block_id * 1000 + expert_id * 10 + int(round_idx),
            )
            resampled_logits = trust_logits.index_select(0, selected)
            resampled_logits = resampled_logits + ha._prediction_noise_like(
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

        velocity_norm = ha._state_rms(state.expert_velocity_ema[group_key])
        second_moment_norm = ha._state_rms(state.expert_second_moment[group_key])
        vsmc_fedavg_l2_distance = ha._state_l2_distance(theta_vsmc, theta_fedavg)
        final_fedavg_l2_distance = ha._state_l2_distance(theta_new, theta_fedavg)
        final_residual_ratio = (
            final_fedavg_l2_distance / max(vsmc_fedavg_l2_distance, float(config.eps))
            if vsmc_fedavg_l2_distance > 0.0
            else 0.0
        )

        info = dict(fallback_info)
        info.update(
            {
                "lambda_expert": float(logged_lambda_expert),
                "anchor_mode": anchor_mode,
                "temperature": float(temperature),
                "ESS": float(ess.item()),
                "ESS_ratio": float(ess_ratio),
                "resampled": bool(resampled),
                "particle_weight_entropy": float(ha._probability_entropy(particle_weights.unsqueeze(0))[0].item()),
                "particle_max_weight": float(particle_weights.max().item()),
                "client_weight_entropy": float(ha._probability_entropy(client_weight_final.unsqueeze(0))[0].item()),
                "client_max_weight": float(client_weight_final.max().item()),
                "client_min_weight": float(client_weight_final.min().item()),
                "usage_score_mean": float(observation_scores.mean().item()),
                "usage_score_std": float(observation_scores.std(unbiased=False).item()),
                "history_cos_mean": 0.0,
                "history_cos_std": 0.0,
                "consensus_cos_mean": 0.0,
                "consensus_cos_std": 0.0,
                "normalized_drift_mean": 0.0,
                "normalized_drift_std": 0.0,
                "function_drift_mean": 0.0,
                "function_drift_std": 0.0,
                "observation_score_mean": float(observation_scores.mean().item()),
                "observation_score_std": float(observation_scores.std(unbiased=False).item()),
                "particle_score_mean": float(particle_scores.mean().item()),
                "particle_score_std": float(particle_scores.std(unbiased=False).item()),
                "vsmc_fedavg_l2_distance": float(vsmc_fedavg_l2_distance),
                "final_fedavg_l2_distance": float(final_fedavg_l2_distance),
                "final_residual_ratio": float(final_residual_ratio),
                "velocity_norm": float(velocity_norm),
                "second_moment_norm": float(second_moment_norm),
                "function_drift_available": False,
                "fallback_used": False,
                "fallback_reason": "",
                "confidence": float(confidence),
                "weights": client_weight_final.detach().cpu().tolist(),
                "particle_weights": particle_weights.detach().cpu().tolist(),
                "usage_score_values": observation_scores.detach().cpu().tolist(),
                "history_cos_values": [],
                "consensus_cos_values": [],
                "normalized_drift_values": [],
                "function_drift_values": [],
                "observation_score_values": observation_scores.detach().cpu().tolist(),
                "particle_score_values": particle_scores.detach().cpu().tolist(),
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
                    "particle_entropy": float(ha._probability_entropy(row_weights.unsqueeze(0))[0].item()),
                    "particle_max_client_weight": float(row_weights.max().item()),
                    "particle_min_client_weight": float(row_weights.min().item()),
                    "dominant_client_idx": int(torch.argmax(row_weights).item()),
                    "resampled": bool(resampled),
                }
            )
        return theta_new, info, particle_rows
    except ha._ExpertFallback as exc:
        state.trust_logits[block_id, expert_id] = torch.zeros_like(state.trust_logits[block_id, expert_id])
        state.particle_log_weights[block_id, expert_id] = torch.zeros_like(state.particle_log_weights[block_id, expert_id])
        info = dict(fallback_info)
        info.update(
            {
                "lambda_expert": 0.0,
                "anchor_mode": str(config.anchor_mode),
                "temperature": float(state.temperature[block_id, expert_id].item()),
                "ESS": float(config.num_particles),
                "ESS_ratio": 1.0,
                "resampled": False,
                "particle_weight_entropy": float(math.log(float(max(config.num_particles, 1)))) if int(config.num_particles) > 1 else 0.0,
                "particle_max_weight": 1.0 / float(max(config.num_particles, 1)),
                "client_weight_entropy": float(math.log(float(max(len(client_expert_states), 1)))) if len(client_expert_states) > 1 else 0.0,
                "client_max_weight": float(expert_weights.max().item()) if expert_weights.numel() > 0 else 1.0,
                "client_min_weight": float(expert_weights.min().item()) if expert_weights.numel() > 0 else 1.0,
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
                "vsmc_fedavg_l2_distance": 0.0,
                "final_fedavg_l2_distance": 0.0,
                "final_residual_ratio": 0.0,
                "velocity_norm": ha._state_rms(state.expert_velocity_ema[group_key]),
                "second_moment_norm": ha._state_rms(state.expert_second_moment[group_key]),
                "function_drift_available": False,
                "fallback_used": True,
                "fallback_reason": str(exc),
                "confidence": 0.0,
                "weights": expert_weights.detach().cpu().tolist(),
                "particle_weights": [1.0 / float(max(config.num_particles, 1))] * int(config.num_particles),
            }
        )
        return fallback_state, info, []


def _aggregate_uc_vsmc_expert(
    global_model,
    global_expert_state,
    client_expert_states,
    client_metrics,
    expert_weights: torch.Tensor,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
    group_key: Tuple[int, int],
    expert_scope: str,
    probe_batch,
    probe_hidden_cache: Dict[int, Optional[torch.Tensor]],
    round_idx: int,
):
    block_id, expert_id = group_key
    device = state.trust_logits.device
    particle_count = int(config.num_particles)
    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        expert_weights,
        fallback_state=global_expert_state,
    )
    fallback_state = clone_detached_state_dict(theta_fedavg)
    fallback_info = ha._build_base_info(
        round_idx=round_idx,
        block_id=block_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        num_particles=particle_count,
    )
    try:
        components = _compute_shared_components(
            global_model=global_model,
            global_expert_state=global_expert_state,
            client_expert_states=client_expert_states,
            client_metrics=client_metrics,
            state=state,
            config=config,
            block_id=block_id,
            expert_id=expert_id,
            group_key=group_key,
            probe_batch=probe_batch,
            probe_hidden_cache=probe_hidden_cache,
            round_idx=round_idx,
        )
        usage_norm = ha._normalize_component(components["usage_fractions"], config)
        consensus_norm = ha._normalize_component(components["consensus_cos"], config)
        observation_scores = (
            float(config.alpha_usage) * usage_norm
            + float(config.alpha_consensus) * consensus_norm
        ).to(dtype=torch.float64, device=device)
        if torch.isnan(observation_scores).any():
            raise ha._ExpertFallback("nan_detected")

        trust_logits_prev = state.trust_logits[block_id, expert_id].clone()
        reliability_vec = state.client_reliability[block_id, expert_id].clone()
        trust_logits_pred = (
            trust_logits_prev
            + float(config.reliability_momentum_lr) * reliability_vec.unsqueeze(0)
            + ha._prediction_noise_like(
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
            raise ha._ExpertFallback("nan_detected")

        temperature = float(state.temperature[block_id, expert_id].item())
        client_weight_rows = torch.softmax(
            trust_logits / max(temperature, float(config.eps)),
            dim=-1,
        )
        if not torch.allclose(
            client_weight_rows.sum(dim=-1),
            torch.ones(
                particle_count,
                dtype=client_weight_rows.dtype,
                device=client_weight_rows.device,
            ),
            atol=1e-5,
        ):
            raise ha._ExpertFallback("particle_weights_invalid")

        theta_particles = [
            weighted_average_state_dicts(
                list(client_expert_states),
                client_weight_rows[particle_idx],
                fallback_state=global_expert_state,
            )
            for particle_idx in range(particle_count)
        ]
        particle_scores = torch.sum(
            client_weight_rows * observation_scores.unsqueeze(0),
            dim=-1,
        ) - float(config.collapse_penalty) * client_weight_rows.max(dim=-1).values
        if float(config.small_entropy_bonus) > 0.0 and int(client_weight_rows.shape[-1]) > 1:
            entropy = ha._probability_entropy(client_weight_rows)
            entropy_norm = entropy / math.log(float(client_weight_rows.shape[-1]))
            particle_scores = particle_scores + float(config.small_entropy_bonus) * entropy_norm

        particle_log_weights = state.particle_log_weights[block_id, expert_id].clone() + particle_scores
        particle_log_weights = particle_log_weights - torch.logsumexp(particle_log_weights, dim=0)
        particle_weights = torch.softmax(particle_log_weights, dim=0)
        if not torch.allclose(
            particle_weights.sum(),
            torch.tensor(1.0, dtype=particle_weights.dtype, device=particle_weights.device),
            atol=1e-5,
        ):
            raise ha._ExpertFallback("particle_weights_invalid")
        if torch.isnan(particle_weights).any():
            raise ha._ExpertFallback("nan_detected")

        ess = 1.0 / particle_weights.pow(2).sum().clamp_min(float(config.eps))
        ess_ratio = float((ess / float(max(particle_count, 1))).item())
        theta_vsmc = weighted_average_state_dicts(
            theta_particles,
            particle_weights,
            fallback_state=theta_fedavg,
        )
        confidence = float(
            particle_scores.mean().item()
            - particle_scores.std(unbiased=False).item()
        )
        warmup = (
            min(1.0, float(round_idx) / float(max(int(config.lambda_warmup_rounds), 1)))
            if int(config.lambda_warmup_rounds) > 0
            else 1.0
        )
        lambda_expert = (
            float(config.lambda_max)
            * warmup
            * float(torch.sigmoid(torch.tensor(confidence)).item())
            * ess_ratio
        )
        lambda_expert = max(0.0, min(float(config.lambda_max), lambda_expert))
        anchor_mode = str(config.anchor_mode)
        if anchor_mode == "none":
            theta_new = clone_detached_state_dict(theta_vsmc)
            logged_lambda_expert = 1.0
        elif anchor_mode == "fedavg":
            theta_new = ha._blend_state_dicts(theta_fedavg, theta_vsmc, lambda_expert)
            logged_lambda_expert = float(lambda_expert)
        else:
            raise ValueError(f"Unsupported HA-VSMC anchor mode: {anchor_mode}")
        if ha._state_has_nan(theta_new):
            raise ha._ExpertFallback("nan_detected")

        client_weight_final = torch.sum(
            particle_weights.unsqueeze(-1) * client_weight_rows,
            dim=0,
        )
        reliability_target = observation_scores
        if bool(config.reliability_weighted_update):
            reliability_target = client_weight_final * reliability_target
        state.client_reliability[block_id, expert_id] = (
            float(config.client_reliability_beta) * state.client_reliability[block_id, expert_id]
            + (1.0 - float(config.client_reliability_beta)) * reliability_target
        )

        _finalize_state_update(
            state=state,
            group_key=group_key,
            global_expert_state=global_expert_state,
            theta_new=theta_new,
            config=config,
        )

        if ess_ratio < 0.4:
            temperature *= 1.05
        elif ess_ratio > 0.8 and confidence > 0.0:
            temperature *= 0.98
        temperature = max(float(config.temperature_min), min(float(config.temperature_max), temperature))
        state.temperature[block_id, expert_id] = temperature

        resampled = False
        if float(ess.item()) < float(config.ess_threshold_ratio) * float(particle_count):
            resampled = True
            selected = ha._systematic_resampling(
                particle_weights,
                config=config,
                offset=block_id * 1000 + expert_id * 10 + int(round_idx),
            )
            resampled_logits = trust_logits.index_select(0, selected)
            resampled_logits = resampled_logits + ha._prediction_noise_like(
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

        velocity_norm = ha._state_rms(state.expert_velocity_ema[group_key])
        second_moment_norm = ha._state_rms(state.expert_second_moment[group_key])
        vsmc_fedavg_l2_distance = ha._state_l2_distance(theta_vsmc, theta_fedavg)
        final_fedavg_l2_distance = ha._state_l2_distance(theta_new, theta_fedavg)
        final_residual_ratio = (
            final_fedavg_l2_distance / max(vsmc_fedavg_l2_distance, float(config.eps))
            if vsmc_fedavg_l2_distance > 0.0
            else 0.0
        )

        info = dict(fallback_info)
        info.update(
            {
                "lambda_expert": float(logged_lambda_expert),
                "anchor_mode": anchor_mode,
                "temperature": float(temperature),
                "ESS": float(ess.item()),
                "ESS_ratio": float(ess_ratio),
                "resampled": bool(resampled),
                "particle_weight_entropy": float(ha._probability_entropy(particle_weights.unsqueeze(0))[0].item()),
                "particle_max_weight": float(particle_weights.max().item()),
                "client_weight_entropy": float(ha._probability_entropy(client_weight_final.unsqueeze(0))[0].item()),
                "client_max_weight": float(client_weight_final.max().item()),
                "client_min_weight": float(client_weight_final.min().item()),
                "usage_score_mean": float(usage_norm.mean().item()),
                "usage_score_std": float(usage_norm.std(unbiased=False).item()),
                "history_cos_mean": 0.0,
                "history_cos_std": 0.0,
                "consensus_cos_mean": float(components["consensus_cos"].mean().item()),
                "consensus_cos_std": float(components["consensus_cos"].std(unbiased=False).item()),
                "normalized_drift_mean": 0.0,
                "normalized_drift_std": 0.0,
                "function_drift_mean": 0.0,
                "function_drift_std": 0.0,
                "observation_score_mean": float(observation_scores.mean().item()),
                "observation_score_std": float(observation_scores.std(unbiased=False).item()),
                "particle_score_mean": float(particle_scores.mean().item()),
                "particle_score_std": float(particle_scores.std(unbiased=False).item()),
                "vsmc_fedavg_l2_distance": float(vsmc_fedavg_l2_distance),
                "final_fedavg_l2_distance": float(final_fedavg_l2_distance),
                "final_residual_ratio": float(final_residual_ratio),
                "velocity_norm": float(velocity_norm),
                "second_moment_norm": float(second_moment_norm),
                "function_drift_available": False,
                "fallback_used": False,
                "fallback_reason": "",
                "confidence": float(confidence),
                "weights": client_weight_final.detach().cpu().tolist(),
                "particle_weights": particle_weights.detach().cpu().tolist(),
                "usage_score_values": usage_norm.detach().cpu().tolist(),
                "history_cos_values": [],
                "consensus_cos_values": components["consensus_cos"].detach().cpu().tolist(),
                "normalized_drift_values": [],
                "function_drift_values": [],
                "observation_score_values": observation_scores.detach().cpu().tolist(),
                "particle_score_values": particle_scores.detach().cpu().tolist(),
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
                    "particle_entropy": float(ha._probability_entropy(row_weights.unsqueeze(0))[0].item()),
                    "particle_max_client_weight": float(row_weights.max().item()),
                    "particle_min_client_weight": float(row_weights.min().item()),
                    "dominant_client_idx": int(torch.argmax(row_weights).item()),
                    "resampled": bool(resampled),
                }
            )
        return theta_new, info, particle_rows
    except ha._ExpertFallback as exc:
        state.trust_logits[block_id, expert_id] = torch.zeros_like(state.trust_logits[block_id, expert_id])
        state.particle_log_weights[block_id, expert_id] = torch.zeros_like(state.particle_log_weights[block_id, expert_id])
        info = dict(fallback_info)
        info.update(
            {
                "lambda_expert": 0.0,
                "anchor_mode": str(config.anchor_mode),
                "temperature": float(state.temperature[block_id, expert_id].item()),
                "ESS": float(config.num_particles),
                "ESS_ratio": 1.0,
                "resampled": False,
                "particle_weight_entropy": float(math.log(float(max(config.num_particles, 1)))) if int(config.num_particles) > 1 else 0.0,
                "particle_max_weight": 1.0 / float(max(config.num_particles, 1)),
                "client_weight_entropy": float(math.log(float(max(len(client_expert_states), 1)))) if len(client_expert_states) > 1 else 0.0,
                "client_max_weight": float(expert_weights.max().item()) if expert_weights.numel() > 0 else 1.0,
                "client_min_weight": float(expert_weights.min().item()) if expert_weights.numel() > 0 else 1.0,
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
                "vsmc_fedavg_l2_distance": 0.0,
                "final_fedavg_l2_distance": 0.0,
                "final_residual_ratio": 0.0,
                "velocity_norm": ha._state_rms(state.expert_velocity_ema[group_key]),
                "second_moment_norm": ha._state_rms(state.expert_second_moment[group_key]),
                "function_drift_available": False,
                "fallback_used": True,
                "fallback_reason": str(exc),
                "confidence": 0.0,
                "weights": expert_weights.detach().cpu().tolist(),
                "particle_weights": [1.0 / float(max(config.num_particles, 1))] * int(config.num_particles),
            }
        )
        return fallback_state, info, []


def _aggregate_mode_with_state(
    mode: str,
    global_model,
    client_updates,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
    probe_batch=None,
):
    current_round = int(config.current_round_idx) if int(config.current_round_idx) > 0 else int(state.round_idx) + 1
    num_clients = len(client_updates)
    if num_clients != int(state.num_clients):
        raise ValueError(
            f"HA-VSMC-ETF state expects {state.num_clients} clients, but received {num_clients}"
        )

    device = state.trust_logits.device
    global_state = collections.OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in global_model.state_dict().items()
    )
    shared_weights = ha._client_weights_from_mode(client_updates, config.shared_weight_mode)
    expert_weights = ha._client_weights_from_mode(client_updates, config.expert_weight_mode)
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
        block_expert = ha._parse_block_expert_from_scope(expert_scope)
        if block_expert is None:
            continue
        block_id, expert_id = block_expert
        group_key = (block_id, expert_id)
        client_group_states = []
        client_metrics = []
        for (_, _, metrics), client_group_map in zip(client_updates, split_client_groups):
            group = client_group_map.get(expert_scope)
            if group is None:
                client_group_states.append(clone_detached_state_dict(global_group["state_dict"]))
            else:
                client_group_states.append(group["state_dict"])
            client_metrics.append(metrics or {})

        if mode == "freq_vsmc":
            aggregated_expert_state, info, particle_rows = _aggregate_freq_vsmc_expert(
                global_expert_state=global_group["state_dict"],
                client_expert_states=client_group_states,
                client_metrics=client_metrics,
                expert_weights=expert_weights,
                state=state,
                config=config,
                group_key=group_key,
                expert_scope=expert_scope,
                round_idx=current_round,
            )
        elif mode == "uc_vsmc":
            aggregated_expert_state, info, particle_rows = _aggregate_uc_vsmc_expert(
                global_model=global_model,
                global_expert_state=global_group["state_dict"],
                client_expert_states=client_group_states,
                client_metrics=client_metrics,
                expert_weights=expert_weights,
                state=state,
                config=config,
                group_key=group_key,
                expert_scope=expert_scope,
                probe_batch=probe_batch,
                probe_hidden_cache=probe_hidden_cache,
                round_idx=current_round,
            )
        elif mode == "multi_signal_weighted":
            aggregated_expert_state, info, particle_rows = _aggregate_multi_signal_weighted_expert(
                global_model=global_model,
                global_expert_state=global_group["state_dict"],
                client_expert_states=client_group_states,
                client_metrics=client_metrics,
                expert_weights=expert_weights,
                state=state,
                config=config,
                group_key=group_key,
                expert_scope=expert_scope,
                probe_batch=probe_batch,
                probe_hidden_cache=probe_hidden_cache,
                round_idx=current_round,
            )
        elif mode == "uc_weighted":
            aggregated_expert_state, info, particle_rows = _aggregate_uc_weighted_expert(
                global_model=global_model,
                global_expert_state=global_group["state_dict"],
                client_expert_states=client_group_states,
                client_metrics=client_metrics,
                expert_weights=expert_weights,
                state=state,
                config=config,
                group_key=group_key,
                expert_scope=expert_scope,
                probe_batch=probe_batch,
                probe_hidden_cache=probe_hidden_cache,
                round_idx=current_round,
            )
        else:
            raise ValueError(f"Unsupported stateful ablation mode: {mode}")

        merged_expert_groups[expert_scope] = {
            "expert_id": int(expert_id),
            "expert_scope": expert_scope,
            "state_dict": aggregated_expert_state,
        }
        expert_summaries.append(info)
        particle_summaries.extend(particle_rows)

    new_global_state = _merge_with_reference_state(
        fedavg_state,
        ha._merge_expert_groups(merged_expert_groups),
        device=None,
    )
    state.round_idx = current_round
    round_logs = {
        "round": int(current_round),
        "anchor_mode": "fedavg" if mode in {"freq_vsmc", "uc_vsmc"} else "none",
        "expert_summaries": expert_summaries,
        "particle_summaries": particle_summaries,
        "mean_ess_ratio": ha._safe_mean([info.get("ESS_ratio") for info in expert_summaries]),
        "fallback_count": int(sum(1 for info in expert_summaries if bool(info.get("fallback_used", False)))),
    }
    return new_global_state, state, round_logs


def _aggregate_freq_weighted_model_state(
    global_state,
    client_updates,
    num_experts: int,
    device: torch.device,
    round_idx: int,
):
    del num_experts
    shared_weights = ha._client_weights_from_mode(client_updates, "uniform")
    expert_weights = ha._client_weights_from_mode(client_updates, "uniform")
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
    merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    expert_summaries: List[Dict[str, Any]] = []

    max_block = 0
    max_expert = 0
    for _, _, metrics in client_updates:
        router_usage = metrics.get("router_usage")
        if torch.is_tensor(router_usage) and router_usage.dim() == 2:
            max_block = max(max_block, int(router_usage.shape[0]))
            max_expert = max(max_expert, int(router_usage.shape[1]))

    for expert_scope, global_group in global_expert_groups.items():
        block_expert = ha._parse_block_expert_from_scope(expert_scope)
        if block_expert is None:
            continue
        block_id, expert_id = block_expert
        client_group_states = []
        client_metrics = []
        for (_, _, metrics), client_group_map in zip(client_updates, split_client_groups):
            group = client_group_map.get(expert_scope)
            if group is None:
                client_group_states.append(clone_detached_state_dict(global_group["state_dict"]))
            else:
                client_group_states.append(group["state_dict"])
            client_metrics.append(metrics or {})
        aggregated_expert_state, info, _ = _aggregate_freq_weighted_expert(
            global_expert_state=global_group["state_dict"],
            client_expert_states=client_group_states,
            client_metrics=client_metrics,
            expert_weights=expert_weights,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            round_idx=round_idx,
            num_blocks=max(max_block, block_id + 1),
            num_experts=max(max_expert, expert_id + 1),
            device=device,
        )
        merged_expert_groups[expert_scope] = {
            "expert_id": int(expert_id),
            "expert_scope": expert_scope,
            "state_dict": aggregated_expert_state,
        }
        expert_summaries.append(info)

    new_global_state = _merge_with_reference_state(
        fedavg_state,
        ha._merge_expert_groups(merged_expert_groups),
        device=None,
    )
    round_logs = {
        "round": int(round_idx),
        "anchor_mode": "none",
        "expert_summaries": expert_summaries,
        "particle_summaries": [],
        "mean_ess_ratio": 0.0,
        "fallback_count": int(sum(1 for info in expert_summaries if bool(info.get("fallback_used", False)))),
    }
    return new_global_state, round_logs


class ReliabilityAblationAggregator:
    def __init__(self, config: ha.HAVSMCETFConfig, mode: str, logger=None):
        self.config = config
        self.mode = mode
        self.logger = logger
        self.state: Optional[ha.HAVSMCETFState] = None

    def aggregate_model_state(
        self,
        global_model,
        global_state,
        client_updates,
        device: torch.device,
        round_idx: int,
        probe_batch=None,
    ):
        if self.mode == "freq_weighted":
            return _aggregate_freq_weighted_model_state(
                global_state=global_state,
                client_updates=client_updates,
                num_experts=int(getattr(global_model, "switch_num_experts", 0) or 0),
                device=device,
                round_idx=round_idx,
            )

        if (
            self.state is None
            or int(self.state.num_clients) != len(client_updates)
            or int(self.state.num_particles) != int(self.config.num_particles)
        ):
            self.state = ha.build_ha_vsmc_etf_state(
                global_model=global_model,
                global_state=global_state,
                num_clients=len(client_updates),
                config=self.config,
                device=device,
            )
        self.config.current_round_idx = int(round_idx)
        aggregated_state, updated_state, round_logs = _aggregate_mode_with_state(
            mode=self.mode,
            global_model=global_model,
            client_updates=client_updates,
            state=self.state,
            config=self.config,
            probe_batch=probe_batch,
        )
        self.state = updated_state
        return aggregated_state, round_logs


def _install_monkey_patch(mode: str) -> None:
    class PatchedReliabilityAggregator(ReliabilityAblationAggregator):
        def __init__(self, config: ha.HAVSMCETFConfig, logger=None):
            super().__init__(config=config, mode=mode, logger=logger)

    server_module.HAVSMCETFExpertAggregator = PatchedReliabilityAggregator
    ha.HAVSMCETFExpertAggregator = PatchedReliabilityAggregator


def _build_logger(args_namespace) -> logging.Logger:
    data = args_namespace.data_name
    num_client = args_namespace.num_clients
    alpha = args_namespace.alpha
    backbone = args_namespace.backbone_name
    logger_name = args_namespace.save_result + "/logs/" + f"data_{data} client_{num_client} alpha_{alpha} backbone_{backbone}"
    os.makedirs(args_namespace.model_save_path, exist_ok=True)
    os.makedirs(args_namespace.data_save_path, exist_ok=True)
    os.makedirs(args_namespace.save_result + "/logs", exist_ok=True)
    os.makedirs(args_namespace.save_result + "/detail", exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    file_handler = logging.FileHandler(f"{logger_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    custom_args, remaining = _extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    warnings.filterwarnings("ignore")
    _install_monkey_patch(custom_args.reliability_ablation_method)
    args_namespace = parse.parse_args()
    logger = _build_logger(args_namespace)
    server_module.Server(parse=parse, logger=logger).train()


if __name__ == "__main__":
    main()
