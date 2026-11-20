from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


VALID_BACKBONES = (
    "patch_embed",
    "cnn_small",
    "resnet8",
    "resnet10",
    "mobilenet_tiny",
    "resnet18",
    "resnet34",
    "resnet50",
)
VALID_IMPORTANCE_METRICS = ("val_loss",)
VALID_AGGREGATION_STRATEGIES = ("sparse_weighted", "sample_count", "uniform")
VALID_COMBINE_METHODS = ("gated", "weighted_sum", "topk_average", "dynamic_routing")
BACKBONE_CAPACITY_SCALE = {
    "patch_embed": 1.0,
    "cnn_small": 1.0,
    "resnet8": 1.0,
    "resnet10": 1.15,
    "mobilenet_tiny": 1.0,
    "resnet18": 1.5,
    "resnet34": 1.75,
    "resnet50": 2.0,
}


@dataclass(frozen=True)
class HFedMoEConfig:
    enabled: bool
    num_experts: int
    top_k: int
    shared_backbone_name: str
    shared_switch_layers: int
    train_shared_backbone: bool
    expert_backbone_name: str
    expert_hidden_dim: int
    gating_hidden_dim: int
    gate_entropy_lambda: float
    importance_metric: str
    importance_max_batches: int
    importance_eps: float
    resource_budget_mode: bool
    client_memory_budget_mb: float
    client_flops_budget: float
    client_budget_scales: Tuple[float, ...]
    min_active_experts: int
    client_selection_ratio: float
    aggregation_strategy: str
    combine_method: str
    batch_size: int
    client_epochs: int
    learning_rate: float
    client_optimizer: str
    client_weight_decay: float
    client_momentum: float
    client_nesterov: bool
    label_smoothing: float
    grad_clip_norm: float
    use_client_aug: bool
    aug_random_crop_pad: int
    aug_horizontal_flip: float
    aug_random_erasing: float
    mixup_alpha: float
    fedavg_weight_mode: str
    expert_weight_mode: str
    device: str
    seed: int
    num_clients: int
    server_epochs: int
    data_name: str
    data_save_path: str
    model_save_path: str
    save_result: str
    image_size: int
    patch_size: int
    hidden_dim: int
    switch_heads: int
    switch_ffn_dim: int
    switch_num_experts: int
    dropout: float
    num_classes: int
    main_eval_set: str
    main_metric: str


def _parse_budget_scales(raw_value: str) -> Tuple[float, ...]:
    values = []
    for token in str(raw_value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(max(float(token), 0.0))
        except ValueError as exc:
            raise ValueError(f"Invalid hfedmoe_client_budget_scales value: {token}") from exc
    return tuple(values) if values else (1.0,)


def resolve_hfedmoe_config(args) -> HFedMoEConfig:
    enabled = bool(getattr(args, "hfedmoe_enabled", False)) or str(
        getattr(args, "aggregation_method", "")
    ).lower() == "hfedmoe" or str(getattr(args, "method", "")).lower() == "hfedmoe"

    num_clients = max(int(getattr(args, "num_clients", 1)), 1)
    num_experts = max(int(getattr(args, "hfedmoe_num_experts", 8)), 1)
    top_k = min(max(int(getattr(args, "hfedmoe_top_k", 2)), 1), num_experts)
    shared_backbone_name = str(getattr(args, "hfedmoe_shared_backbone_name", "cnn_small"))
    expert_backbone_name = str(getattr(args, "hfedmoe_expert_backbone_name", getattr(args, "backbone_name", "resnet18")))
    if shared_backbone_name not in VALID_BACKBONES:
        raise ValueError(f"Unsupported HFedMoE shared backbone: {shared_backbone_name}")
    if expert_backbone_name not in VALID_BACKBONES:
        raise ValueError(f"Unsupported HFedMoE expert backbone: {expert_backbone_name}")

    importance_metric = str(getattr(args, "hfedmoe_importance_metric", "val_loss")).lower()
    if importance_metric not in VALID_IMPORTANCE_METRICS:
        raise ValueError(f"Unsupported HFedMoE importance metric: {importance_metric}")

    aggregation_strategy = str(getattr(args, "hfedmoe_aggregation_strategy", "sparse_weighted")).lower()
    if aggregation_strategy not in VALID_AGGREGATION_STRATEGIES:
        raise ValueError(f"Unsupported HFedMoE aggregation strategy: {aggregation_strategy}")

    combine_method = str(getattr(args, "hfedmoe_combine_method", "gated")).lower()
    if combine_method not in VALID_COMBINE_METHODS:
        raise ValueError(f"Unsupported HFedMoE combine method: {combine_method}")

    hidden_dim = max(int(getattr(args, "hidden_dim", 128)), 1)
    requested_expert_hidden_dim = int(getattr(args, "hfedmoe_expert_hidden_dim", 0))
    if requested_expert_hidden_dim > 0:
        expert_hidden_dim = requested_expert_hidden_dim
    else:
        scale = BACKBONE_CAPACITY_SCALE.get(expert_backbone_name, 1.0)
        expert_hidden_dim = max(int(round(hidden_dim * scale)), hidden_dim)

    num_classes_by_dataset = {
        "cifar10": 10,
        "cifar100": 100,
        "cinic10": 10,
        "tinyimagenet": 200,
    }
    client_selection_ratio = min(max(float(getattr(args, "hfedmoe_client_selection_ratio", 1.0)), 0.0), 1.0)

    return HFedMoEConfig(
        enabled=enabled,
        num_experts=num_experts,
        top_k=top_k,
        shared_backbone_name=shared_backbone_name,
        shared_switch_layers=max(int(getattr(args, "hfedmoe_shared_switch_layers", 1)), 1),
        train_shared_backbone=bool(getattr(args, "hfedmoe_train_shared_backbone", True)),
        expert_backbone_name=expert_backbone_name,
        expert_hidden_dim=max(int(expert_hidden_dim), 1),
        gating_hidden_dim=max(int(getattr(args, "hfedmoe_gating_hidden_dim", 256)), 1),
        gate_entropy_lambda=max(float(getattr(args, "hfedmoe_gate_entropy_lambda", 0.01)), 0.0),
        importance_metric=importance_metric,
        importance_max_batches=max(int(getattr(args, "hfedmoe_importance_max_batches", 2)), 1),
        importance_eps=max(float(getattr(args, "hfedmoe_importance_eps", 1e-6)), 1e-12),
        resource_budget_mode=bool(getattr(args, "hfedmoe_resource_budget_mode", True)),
        client_memory_budget_mb=max(float(getattr(args, "hfedmoe_client_memory_budget_mb", 2048.0)), 0.0),
        client_flops_budget=max(float(getattr(args, "hfedmoe_client_flops_budget", 2e11)), 0.0),
        client_budget_scales=_parse_budget_scales(getattr(args, "hfedmoe_client_budget_scales", "")),
        min_active_experts=max(int(getattr(args, "hfedmoe_min_active_experts", 1)), 1),
        client_selection_ratio=client_selection_ratio,
        aggregation_strategy=aggregation_strategy,
        combine_method=combine_method,
        batch_size=max(int(getattr(args, "batch_size", 1)), 1),
        client_epochs=max(int(getattr(args, "client_epochs", 1)), 1),
        learning_rate=float(getattr(args, "learning_rate", 5e-4)),
        client_optimizer=str(getattr(args, "client_optimizer", "adam")).lower(),
        client_weight_decay=float(getattr(args, "client_weight_decay", 0.0)),
        client_momentum=float(getattr(args, "client_momentum", 0.0)),
        client_nesterov=bool(getattr(args, "client_nesterov", False)),
        label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
        grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
        use_client_aug=bool(getattr(args, "use_client_aug", False)),
        aug_random_crop_pad=int(getattr(args, "aug_random_crop_pad", 4)),
        aug_horizontal_flip=float(getattr(args, "aug_horizontal_flip", 0.5)),
        aug_random_erasing=float(getattr(args, "aug_random_erasing", 0.0)),
        mixup_alpha=float(getattr(args, "mixup_alpha", 0.0)),
        fedavg_weight_mode=str(getattr(args, "fedavg_weight_mode", "sample_count")).lower(),
        expert_weight_mode=str(getattr(args, "expert_weight_mode", "uniform")).lower(),
        device=str(getattr(args, "device", "cpu")),
        seed=int(getattr(args, "seed", 0)),
        num_clients=num_clients,
        server_epochs=max(int(getattr(args, "server_epochs", 1)), 1),
        data_name=str(getattr(args, "data_name", "cifar10")),
        data_save_path=str(getattr(args, "data_save_path", "")),
        model_save_path=str(getattr(args, "model_save_path", "")),
        save_result=str(getattr(args, "save_result", "")),
        image_size=max(int(getattr(args, "image_size", 32)), 1),
        patch_size=max(int(getattr(args, "patch_size", 4)), 1),
        hidden_dim=hidden_dim,
        switch_heads=max(int(getattr(args, "switch_heads", 1)), 1),
        switch_ffn_dim=max(int(getattr(args, "switch_ffn_dim", 128)), 1),
        switch_num_experts=max(int(getattr(args, "switch_num_experts", 1)), 1),
        dropout=float(getattr(args, "dropout", 0.0)),
        num_classes=num_classes_by_dataset[str(getattr(args, "data_name", "cifar10"))],
        main_eval_set="server_test.pth",
        main_metric="server_test_acc",
    )
