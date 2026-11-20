from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from args import str2bool


@dataclass(frozen=True)
class FedMAConfig:
    fedma_scope: str
    matching_method: str
    expert_signature: str
    similarity: str
    non_expert_agg: str
    router_agg: str
    classifier_agg: str
    shared_weight_mode: str
    expert_weight_mode: str
    weight_by_num_samples: bool
    allow_expert_permutation: bool
    keep_num_experts_fixed: bool
    log_matching: bool
    save_match_matrix: bool
    save_expert_similarity: bool
    output_dir: str
    model_save_path: str
    total_rounds: int
    local_epochs: int
    clients_per_round: int
    num_clients: int
    batch_size: int
    learning_rate: float
    device: str
    seed: int
    data_name: str
    data_save_path: str
    backbone_name: str
    switch_layers: int
    switch_num_experts: int


def _has_option(parser, option_string: str) -> bool:
    return option_string in getattr(parser, "_option_string_actions", {})


def ensure_fedma_args(parser) -> None:
    if not _has_option(parser, "--fedma_scope"):
        parser.add_argument(
            "--fedma_scope",
            type=str,
            default="experts_only",
            choices=["experts_only", "all_ffn", "router_signature_only"],
        )
    if not _has_option(parser, "--matching_method"):
        parser.add_argument(
            "--matching_method",
            type=str,
            default="hungarian",
            choices=["hungarian", "greedy", "identity"],
        )
    if not _has_option(parser, "--expert_signature"):
        parser.add_argument(
            "--expert_signature",
            type=str,
            default="weight_flatten",
            choices=["weight_flatten", "weight_mean", "activation_mean", "router_usage"],
        )
    if not _has_option(parser, "--similarity"):
        parser.add_argument(
            "--similarity",
            type=str,
            default="cosine",
            choices=["cosine", "l2"],
        )
    if not _has_option(parser, "--fedma_allow_expert_permutation"):
        parser.add_argument("--fedma_allow_expert_permutation", type=str2bool, default=True)
    if not _has_option(parser, "--fedma_keep_num_experts_fixed"):
        parser.add_argument("--fedma_keep_num_experts_fixed", type=str2bool, default=True)
    if not _has_option(parser, "--fedma_log_matching"):
        parser.add_argument("--fedma_log_matching", type=str2bool, default=True)
    if not _has_option(parser, "--fedma_save_match_matrix"):
        parser.add_argument("--fedma_save_match_matrix", type=str2bool, default=True)
    if not _has_option(parser, "--fedma_save_expert_similarity"):
        parser.add_argument("--fedma_save_expert_similarity", type=str2bool, default=True)
    if not _has_option(parser, "--output_dir"):
        parser.add_argument("--output_dir", type=str, default="")


def resolve_fedma_config(args) -> FedMAConfig:
    num_clients = max(int(getattr(args, "num_clients", 1)), 1)
    default_run_name = (
        f"fedma_{str(getattr(args, 'fedma_scope', 'experts_only')).lower()}_"
        f"{str(getattr(args, 'expert_signature', 'weight_flatten')).lower()}_"
        f"seed{int(getattr(args, 'seed', 0))}"
    )
    base_output_root = os.path.join(str(getattr(args, "save_result", "./save/result")), "baselines_fedma")
    output_dir = str(getattr(args, "output_dir", "")).strip() or os.path.join(base_output_root, default_run_name)
    output_dir = os.path.abspath(output_dir)
    model_save_path = os.path.join(output_dir, "models")

    shared_weight_mode = str(getattr(args, "fedavg_weight_mode", "uniform")).lower()
    expert_weight_mode = str(getattr(args, "expert_weight_mode", "uniform")).lower()

    return FedMAConfig(
        fedma_scope=str(getattr(args, "fedma_scope", "experts_only")).lower(),
        matching_method=str(getattr(args, "matching_method", "hungarian")).lower(),
        expert_signature=str(getattr(args, "expert_signature", "weight_flatten")).lower(),
        similarity=str(getattr(args, "similarity", "cosine")).lower(),
        non_expert_agg="fedavg",
        router_agg="fedavg",
        classifier_agg="fedavg",
        shared_weight_mode=shared_weight_mode,
        expert_weight_mode=expert_weight_mode,
        weight_by_num_samples=expert_weight_mode == "sample_count",
        allow_expert_permutation=bool(getattr(args, "fedma_allow_expert_permutation", True)),
        keep_num_experts_fixed=bool(getattr(args, "fedma_keep_num_experts_fixed", True)),
        log_matching=bool(getattr(args, "fedma_log_matching", True)),
        save_match_matrix=bool(getattr(args, "fedma_save_match_matrix", True)),
        save_expert_similarity=bool(getattr(args, "fedma_save_expert_similarity", True)),
        output_dir=output_dir,
        model_save_path=model_save_path,
        total_rounds=max(int(getattr(args, "server_epochs", 1)), 1),
        local_epochs=max(int(getattr(args, "client_epochs", 1)), 1),
        clients_per_round=num_clients,
        num_clients=num_clients,
        batch_size=max(int(getattr(args, "batch_size", 1)), 1),
        learning_rate=float(getattr(args, "learning_rate", 0.0)),
        device=str(getattr(args, "device", "cpu")),
        seed=int(getattr(args, "seed", 0)),
        data_name=str(getattr(args, "data_name", "")),
        data_save_path=str(getattr(args, "data_save_path", "")),
        backbone_name=str(getattr(args, "backbone_name", "")),
        switch_layers=max(int(getattr(args, "switch_layers", 1)), 1),
        switch_num_experts=max(int(getattr(args, "switch_num_experts", 1)), 1),
    )
