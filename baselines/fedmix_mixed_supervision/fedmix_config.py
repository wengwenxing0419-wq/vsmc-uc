from __future__ import annotations

import os
from dataclasses import dataclass

from args import str2bool


@dataclass(frozen=True)
class FedMixConfig:
    beta: float
    lam: float
    pseudo_thresh: float
    pseudo_loss_weight: float
    use_dynamic_sample_selection: bool
    use_adaptive_aggregation: bool
    label_type: str
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
    shared_weight_mode: str
    expert_weight_mode: str


def _has_option(parser, option_string: str) -> bool:
    return option_string in getattr(parser, "_option_string_actions", {})


def ensure_fedmix_args(parser) -> None:
    if not _has_option(parser, "--fedmix_beta"):
        parser.add_argument("--fedmix_beta", type=float, default=1.5)
    if not _has_option(parser, "--fedmix_lambda"):
        parser.add_argument("--fedmix_lambda", type=float, default=10.0)
    if not _has_option(parser, "--fedmix_pseudo_thresh"):
        parser.add_argument("--fedmix_pseudo_thresh", type=float, default=0.9)
    if not _has_option(parser, "--fedmix_pseudo_loss_weight"):
        parser.add_argument("--fedmix_pseudo_loss_weight", type=float, default=0.5)
    if not _has_option(parser, "--fedmix_use_dynamic_sample_selection"):
        parser.add_argument("--fedmix_use_dynamic_sample_selection", type=str2bool, default=True)
    if not _has_option(parser, "--fedmix_use_adaptive_aggregation"):
        parser.add_argument("--fedmix_use_adaptive_aggregation", type=str2bool, default=True)
    if not _has_option(parser, "--fedmix_label_type"):
        parser.add_argument("--fedmix_label_type", type=str, default="class", choices=["class", "image", "unlabeled"])
    if not _has_option(parser, "--output_dir"):
        parser.add_argument("--output_dir", type=str, default="")


def resolve_fedmix_config(args) -> FedMixConfig:
    num_clients = max(int(getattr(args, "num_clients", 1)), 1)
    default_run_name = (
        f"fedmix_beta{float(getattr(args, 'fedmix_beta', 1.5)):.2f}_"
        f"lambda{float(getattr(args, 'fedmix_lambda', 10.0)):.2f}_"
        f"seed{int(getattr(args, 'seed', 0))}"
    )
    base_output_root = os.path.join(str(getattr(args, "save_result", "./save/result")), "baselines_fedmix")
    output_dir = str(getattr(args, "output_dir", "")).strip() or os.path.join(base_output_root, default_run_name)
    output_dir = os.path.abspath(output_dir)
    model_save_path = os.path.join(output_dir, "models")

    return FedMixConfig(
        beta=float(getattr(args, "fedmix_beta", 1.5)),
        lam=float(getattr(args, "fedmix_lambda", 10.0)),
        pseudo_thresh=float(getattr(args, "fedmix_pseudo_thresh", 0.9)),
        pseudo_loss_weight=float(getattr(args, "fedmix_pseudo_loss_weight", 0.5)),
        use_dynamic_sample_selection=bool(getattr(args, "fedmix_use_dynamic_sample_selection", True)),
        use_adaptive_aggregation=bool(getattr(args, "fedmix_use_adaptive_aggregation", True)),
        label_type=str(getattr(args, "fedmix_label_type", "class")).lower(),
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
        shared_weight_mode=str(getattr(args, "fedavg_weight_mode", "uniform")).lower(),
        expert_weight_mode=str(getattr(args, "expert_weight_mode", "uniform")).lower(),
    )
