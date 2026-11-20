#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import csv
import json
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


_EPS = 1e-12
_LAST_ROUND_CSV = "freq_vsmc_synced_last_round_metrics.csv"
_SUMMARY_JSON = "freq_vsmc_synced_summary.json"


def _build_custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freq_vsmc_synced_sigma", type=float, default=0.0)
    parser.add_argument("--freq_vsmc_synced_rho", type=float, default=0.0)
    parser.add_argument("--freq_vsmc_synced_tau", type=float, default=1.0)
    parser.add_argument("--freq_vsmc_synced_eps", type=float, default=1e-12)
    return parser


def _extract_custom_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    custom_parser = _build_custom_parser()
    custom_args, remaining = custom_parser.parse_known_args(list(argv))
    return custom_args, remaining


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


def _normalize_freq_distribution(
    usage_fractions: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    num_clients = int(usage_fractions.numel())
    nonnegative = usage_fractions.to(dtype=torch.float64).clamp_min(0.0)
    total = float(nonnegative.sum().item())
    if total <= float(eps):
        raw_freq = torch.full(
            (num_clients,),
            1.0 / float(max(num_clients, 1)),
            dtype=torch.float64,
            device=usage_fractions.device,
        )
        centered = raw_freq.clone()
        return raw_freq, centered, True

    raw_freq = nonnegative / nonnegative.sum().clamp_min(float(eps))
    centered = raw_freq.clamp_min(float(eps))
    centered = centered / centered.sum().clamp_min(float(eps))
    return raw_freq, centered, False


def _pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float) -> float:
    a = a.to(dtype=torch.float64)
    b = b.to(dtype=torch.float64)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(a_centered.norm().item() * b_centered.norm().item())
    if denom <= float(eps):
        return 1.0 if torch.allclose(a, b, atol=1e-10, rtol=1e-10) else 0.0
    return float((a_centered * b_centered).sum().item() / denom)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float) -> float:
    p = p.to(dtype=torch.float64)
    q = q.to(dtype=torch.float64)
    p_safe = p.clamp_min(float(eps))
    q_safe = q.clamp_min(float(eps))
    return float(torch.sum(p_safe * (torch.log(p_safe) - torch.log(q_safe))).item())


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


def _deterministic_noise(
    shape: torch.Size,
    sigma: float,
    seed: int,
    offset: int,
    device: torch.device,
    deterministic: bool,
) -> torch.Tensor:
    if float(sigma) <= 0.0:
        return torch.zeros(shape, dtype=torch.float64, device=device)
    generator = None
    if bool(deterministic):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(offset))
    return torch.randn(shape, generator=generator, dtype=torch.float64, device=device) * float(sigma)


def _aggregate_freq_vsmc_synced_expert(
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
    del expert_weights
    block_id, expert_id = group_key
    device = state.trust_logits.device
    particle_count = int(config.num_particles)
    num_clients = len(client_expert_states)
    theta_fedavg = weighted_average_state_dicts(
        list(client_expert_states),
        ha._client_weights_from_mode(
            [(collections.OrderedDict(), 1, {}) for _ in range(num_clients)],
            "uniform",
        ),
        fallback_state=global_expert_state,
    )

    usage_fractions, raw_usage_counts = _collect_usage_fractions(
        client_metrics=client_metrics,
        block_id=block_id,
        expert_id=expert_id,
        state=state,
        device=device,
    )
    freq_weights, centered_freq_weights, all_zero_usage = _normalize_freq_distribution(
        usage_fractions=usage_fractions,
        eps=float(getattr(config, "synced_eps", config.eps)),
    )

    rho = float(max(0.0, min(1.0, getattr(config, "synced_rho", 0.0))))
    sigma = float(max(0.0, getattr(config, "synced_sigma", 0.0)))
    tau = float(max(getattr(config, "synced_tau", 1.0), getattr(config, "synced_eps", config.eps)))
    eps = float(max(getattr(config, "synced_eps", config.eps), _EPS))

    prev_prob = state.synced_prob_ema[block_id, expert_id].to(dtype=torch.float64, device=device)
    prob_center = rho * prev_prob + (1.0 - rho) * freq_weights
    prob_center = prob_center.clamp_min(eps)
    prob_center = prob_center / prob_center.sum().clamp_min(eps)
    state.synced_prob_ema[block_id, expert_id] = prob_center

    base_logits = torch.log(prob_center.clamp_min(eps)).unsqueeze(0).repeat(particle_count, 1)
    noise = _deterministic_noise(
        shape=base_logits.shape,
        sigma=sigma,
        seed=int(config.seed),
        offset=block_id * 1000 + expert_id * 50 + int(round_idx),
        device=device,
        deterministic=bool(config.deterministic),
    )
    trust_logits = base_logits + noise
    client_weight_rows = torch.softmax(trust_logits / tau, dim=-1)
    client_weight_rows = client_weight_rows / client_weight_rows.sum(dim=-1, keepdim=True).clamp_min(eps)

    observation_scores = freq_weights
    particle_scores = torch.sum(
        client_weight_rows * observation_scores.unsqueeze(0),
        dim=-1,
    )
    particle_log_weights = particle_scores - torch.logsumexp(particle_scores, dim=0)
    particle_weights = torch.softmax(particle_log_weights, dim=0)
    particle_weights = particle_weights / particle_weights.sum().clamp_min(eps)
    client_weight_final = torch.sum(
        particle_weights.unsqueeze(-1) * client_weight_rows,
        dim=0,
    )
    client_weight_final = client_weight_final / client_weight_final.sum().clamp_min(eps)

    theta_new = weighted_average_state_dicts(
        list(client_expert_states),
        client_weight_final,
        fallback_state=theta_fedavg,
    )
    _finalize_state_update(
        state=state,
        group_key=group_key,
        global_expert_state=global_expert_state,
        theta_new=theta_new,
        config=config,
    )
    state.trust_logits[block_id, expert_id] = trust_logits
    state.particle_log_weights[block_id, expert_id] = torch.log(particle_weights.clamp_min(eps))
    state.client_reliability[block_id, expert_id] = freq_weights
    state.usage_ema[block_id, expert_id] = freq_weights

    ess = 1.0 / particle_weights.pow(2).sum().clamp_min(eps)
    ess_ratio = float((ess / float(max(particle_count, 1))).item())
    vsmc_fedavg_l2_distance = ha._state_l2_distance(theta_new, theta_fedavg)
    pearson_corr = _pearson_corr(freq_weights, client_weight_final, eps=eps)
    kl_value = _kl_divergence(client_weight_final, freq_weights, eps=eps)
    freq_entropy = float(ha._probability_entropy(freq_weights.unsqueeze(0))[0].item())
    vsmc_entropy = float(ha._probability_entropy(client_weight_final.unsqueeze(0))[0].item())
    freq_top = int(torch.argmax(freq_weights).item()) if num_clients > 0 else -1
    vsmc_top = int(torch.argmax(client_weight_final).item()) if num_clients > 0 else -1
    top_match = bool(freq_top == vsmc_top)

    info = ha._build_base_info(
        round_idx=round_idx,
        block_id=block_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        num_particles=particle_count,
    )
    info.update(
        {
            "lambda_expert": 1.0,
            "anchor_mode": "none",
            "temperature": float(tau),
            "ESS": float(ess.item()),
            "ESS_ratio": float(ess_ratio),
            "resampled": False,
            "particle_weight_entropy": float(ha._probability_entropy(particle_weights.unsqueeze(0))[0].item()),
            "particle_max_weight": float(particle_weights.max().item()),
            "client_weight_entropy": float(ha._probability_entropy(client_weight_final.unsqueeze(0))[0].item()),
            "client_max_weight": float(client_weight_final.max().item()) if client_weight_final.numel() > 0 else 0.0,
            "client_min_weight": float(client_weight_final.min().item()) if client_weight_final.numel() > 0 else 0.0,
            "usage_score_mean": float(observation_scores.mean().item()) if observation_scores.numel() > 0 else 0.0,
            "usage_score_std": float(observation_scores.std(unbiased=False).item()) if observation_scores.numel() > 0 else 0.0,
            "history_cos_mean": 0.0,
            "history_cos_std": 0.0,
            "consensus_cos_mean": 0.0,
            "consensus_cos_std": 0.0,
            "normalized_drift_mean": 0.0,
            "normalized_drift_std": 0.0,
            "function_drift_mean": 0.0,
            "function_drift_std": 0.0,
            "observation_score_mean": float(observation_scores.mean().item()) if observation_scores.numel() > 0 else 0.0,
            "observation_score_std": float(observation_scores.std(unbiased=False).item()) if observation_scores.numel() > 0 else 0.0,
            "particle_score_mean": float(particle_scores.mean().item()) if particle_scores.numel() > 0 else 0.0,
            "particle_score_std": float(particle_scores.std(unbiased=False).item()) if particle_scores.numel() > 0 else 0.0,
            "vsmc_fedavg_l2_distance": float(vsmc_fedavg_l2_distance),
            "final_fedavg_l2_distance": float(vsmc_fedavg_l2_distance),
            "final_residual_ratio": 1.0 if vsmc_fedavg_l2_distance > 0.0 else 0.0,
            "velocity_norm": float(ha._state_rms(state.expert_velocity_ema[group_key])),
            "second_moment_norm": float(ha._state_rms(state.expert_second_moment[group_key])),
            "function_drift_available": False,
            "fallback_used": False,
            "fallback_reason": "",
            "confidence": float(particle_scores.mean().item() - particle_scores.std(unbiased=False).item()) if particle_scores.numel() > 0 else 0.0,
            "weights": client_weight_final.detach().cpu().tolist(),
            "particle_weights": particle_weights.detach().cpu().tolist(),
            "usage_score_values": observation_scores.detach().cpu().tolist(),
            "history_cos_values": [],
            "consensus_cos_values": [],
            "normalized_drift_values": [],
            "function_drift_values": [],
            "observation_score_values": observation_scores.detach().cpu().tolist(),
            "particle_score_values": particle_scores.detach().cpu().tolist(),
            "synced_freq_weights": freq_weights.detach().cpu().tolist(),
            "synced_center_weights": centered_freq_weights.detach().cpu().tolist(),
            "synced_prob_center": prob_center.detach().cpu().tolist(),
            "synced_vsmc_weights": client_weight_final.detach().cpu().tolist(),
            "synced_pearson_corr": float(pearson_corr),
            "synced_kl_vsmc_to_freq": float(kl_value),
            "synced_freq_entropy": float(freq_entropy),
            "synced_vsmc_entropy": float(vsmc_entropy),
            "synced_freq_max_weight": float(freq_weights.max().item()) if freq_weights.numel() > 0 else 0.0,
            "synced_vsmc_max_weight": float(client_weight_final.max().item()) if client_weight_final.numel() > 0 else 0.0,
            "synced_freq_top_client": int(freq_top),
            "synced_vsmc_top_client": int(vsmc_top),
            "synced_top_client_match": bool(top_match),
            "synced_all_zero_usage": bool(all_zero_usage),
            "synced_raw_usage_counts": raw_usage_counts.detach().cpu().tolist(),
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
                "resampled": False,
            }
        )
    return theta_new, info, particle_rows


def _aggregate_freq_vsmc_synced_model_state(
    global_model,
    global_state,
    client_updates,
    state: ha.HAVSMCETFState,
    config: ha.HAVSMCETFConfig,
):
    current_round = int(config.current_round_idx) if int(config.current_round_idx) > 0 else int(state.round_idx) + 1
    num_clients = len(client_updates)
    if num_clients != int(state.num_clients):
        raise ValueError(
            f"HA-VSMC-ETF state expects {state.num_clients} clients, but received {num_clients}"
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

        aggregated_expert_state, info, particle_rows = _aggregate_freq_vsmc_synced_expert(
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
        "anchor_mode": "none",
        "expert_summaries": expert_summaries,
        "particle_summaries": particle_summaries,
        "mean_ess_ratio": ha._safe_mean([info.get("ESS_ratio") for info in expert_summaries]),
        "fallback_count": int(sum(1 for info in expert_summaries if bool(info.get("fallback_used", False)))),
    }
    return new_global_state, state, round_logs


class FreqVSMCSyncedAggregator:
    def __init__(self, config: ha.HAVSMCETFConfig, logger=None):
        self.config = config
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
        del probe_batch
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
            self.state.synced_prob_ema = torch.full(
                (
                    int(self.state.num_blocks),
                    int(self.state.num_experts_per_block),
                    int(self.state.num_clients),
                ),
                1.0 / float(max(int(self.state.num_clients), 1)),
                dtype=torch.float64,
                device=device,
            )
        self.config.current_round_idx = int(round_idx)
        aggregated_state, updated_state, round_logs = _aggregate_freq_vsmc_synced_model_state(
            global_model=global_model,
            global_state=global_state,
            client_updates=client_updates,
            state=self.state,
            config=self.config,
        )
        self.state = updated_state
        return aggregated_state, round_logs


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _write_last_round_analysis(server_instance, round_idx: int, server_test_acc: float) -> None:
    round_data = server_instance.latest_ha_vsmc_etf_round_data or {}
    infos = round_data.get("expert_summaries", []) or []
    logs_dir = Path(server_instance.args.save_result) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for info in infos:
        rows.append(
            {
                "round": int(round_idx),
                "block_id": int(info.get("block_id", -1)),
                "expert_id": int(info.get("expert_id", -1)),
                "expert_scope": str(info.get("expert_scope", "")),
                "a_freq": _json_dumps(info.get("synced_freq_weights", [])),
                "a_vsmc_final": _json_dumps(info.get("synced_vsmc_weights", [])),
                "pearson_corr": float(info.get("synced_pearson_corr", 0.0)),
                "kl_vsmc_to_freq": float(info.get("synced_kl_vsmc_to_freq", 0.0)),
                "entropy_freq": float(info.get("synced_freq_entropy", 0.0)),
                "entropy_vsmc": float(info.get("synced_vsmc_entropy", 0.0)),
                "max_weight_freq": float(info.get("synced_freq_max_weight", 0.0)),
                "max_weight_vsmc": float(info.get("synced_vsmc_max_weight", 0.0)),
                "top_client_freq": int(info.get("synced_freq_top_client", -1)),
                "top_client_vsmc": int(info.get("synced_vsmc_top_client", -1)),
                "top_client_match": bool(info.get("synced_top_client_match", False)),
                "all_zero_usage": bool(info.get("synced_all_zero_usage", False)),
                "raw_usage_counts": _json_dumps(info.get("synced_raw_usage_counts", [])),
                "prob_center": _json_dumps(info.get("synced_prob_center", [])),
            }
        )

    csv_path = logs_dir / _LAST_ROUND_CSV
    fieldnames = [
        "round",
        "block_id",
        "expert_id",
        "expert_scope",
        "a_freq",
        "a_vsmc_final",
        "pearson_corr",
        "kl_vsmc_to_freq",
        "entropy_freq",
        "entropy_vsmc",
        "max_weight_freq",
        "max_weight_vsmc",
        "top_client_freq",
        "top_client_vsmc",
        "top_client_match",
        "all_zero_usage",
        "raw_usage_counts",
        "prob_center",
    ]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_corr = sum(float(row["pearson_corr"]) for row in rows) / float(max(len(rows), 1))
    mean_kl = sum(float(row["kl_vsmc_to_freq"]) for row in rows) / float(max(len(rows), 1))
    top_match = sum(1 for row in rows if bool(row["top_client_match"]))
    summary = {
        "final_acc": float(server_test_acc),
        "best_acc": float(server_instance.best_server_test_acc),
        "mean_corr": float(mean_corr),
        "mean_kl": float(mean_kl),
        "top_client_match": int(top_match),
        "num_expert_groups": int(len(rows)),
        "csv_path": str(csv_path),
    }
    with open(logs_dir / _SUMMARY_JSON, "w") as jsonfile:
        json.dump(summary, jsonfile, ensure_ascii=True, indent=2)


def _install_monkey_patches(custom_args: argparse.Namespace) -> None:
    server_module.HAVSMCETFExpertAggregator = FreqVSMCSyncedAggregator
    ha.HAVSMCETFExpertAggregator = FreqVSMCSyncedAggregator

    original_build_config = server_module.Server.build_ha_vsmc_etf_config
    original_record_metrics = server_module.Server.record_ha_vsmc_etf_round_metrics

    def patched_build_config(self):
        config = original_build_config(self)
        config.anchor_mode = "none"
        config.alpha_history_direction = 0.0
        config.alpha_consensus = 0.0
        config.alpha_drift = 0.0
        config.alpha_function_drift = 0.0
        config.use_function_drift = False
        config.prediction_noise_std = 0.0
        config.trust_update_lr = 0.0
        config.reliability_momentum_lr = 0.0
        config.temperature_init = float(custom_args.freq_vsmc_synced_tau)
        config.temperature_min = float(custom_args.freq_vsmc_synced_tau)
        config.temperature_max = float(custom_args.freq_vsmc_synced_tau)
        config.synced_sigma = float(custom_args.freq_vsmc_synced_sigma)
        config.synced_rho = float(custom_args.freq_vsmc_synced_rho)
        config.synced_tau = float(custom_args.freq_vsmc_synced_tau)
        config.synced_eps = float(custom_args.freq_vsmc_synced_eps)
        return config

    def patched_record_metrics(
        self,
        round_idx,
        avg_client_train_loss,
        avg_client_train_acc,
        server_test_loss,
        server_test_acc,
    ):
        original_record_metrics(
            self,
            round_idx,
            avg_client_train_loss,
            avg_client_train_acc,
            server_test_loss,
            server_test_acc,
        )
        if int(round_idx) == int(self.server_epochs):
            _write_last_round_analysis(
                server_instance=self,
                round_idx=int(round_idx),
                server_test_acc=float(server_test_acc),
            )

    server_module.Server.build_ha_vsmc_etf_config = patched_build_config
    server_module.Server.record_ha_vsmc_etf_round_metrics = patched_record_metrics


def _print_final_summary(save_result: str) -> None:
    summary_path = Path(save_result) / "logs" / _SUMMARY_JSON
    if not summary_path.exists():
        return
    with open(summary_path, "r") as jsonfile:
        summary = json.load(jsonfile)
    total_groups = int(summary.get("num_expert_groups", 0))
    print(f"Freq-VSMC-Synced Final ACC={float(summary.get('final_acc', 0.0)):.4f}")
    print(f"Freq-VSMC-Synced Best ACC={float(summary.get('best_acc', 0.0)):.4f}")
    print(f"Mean corr(a_freq, a_vsmc)={float(summary.get('mean_corr', 0.0)):.6f}")
    print(f"Mean KL(a_vsmc || a_freq)={float(summary.get('mean_kl', 0.0)):.6f}")
    print(f"Top-client match experts={int(summary.get('top_client_match', 0))}/{total_groups}")


def main() -> None:
    custom_args, remaining = _extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    warnings.filterwarnings("ignore")
    _install_monkey_patches(custom_args)
    args_namespace = parse.parse_args()
    logger = _build_logger(args_namespace)
    server_module.Server(parse=parse, logger=logger).train()
    _print_final_summary(args_namespace.save_result)


if __name__ == "__main__":
    main()
