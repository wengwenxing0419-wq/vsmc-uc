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
VALID_SERVER_TEST_MODES = (
    "ensemble_clients",
    "global_only",
    "mean_gate",
    "best_global_mix",
)
VALID_WARMUP_MODES = ("fixed_uniform", "learned")
VALID_BEST_METRICS = ("val_acc", "server_test_acc")


@dataclass(frozen=True)
class PFedMoEConfig:
    enabled: bool
    gate_hidden_dim: int
    client_selection_ratio: float
    aggregation_method: str
    global_backbone_name: str
    global_switch_layers: int
    local_backbones: Tuple[str, ...]
    eval_fallback_to_server_test: bool
    server_test_mode: str
    global_aux_loss: bool
    global_aux_lambda: float
    aggregate_global_aux_head: bool
    alpha_global_target: float
    alpha_global_reg_lambda: float
    gate_warmup_rounds: int
    gate_warmup_mode: str
    freeze_gate_during_warmup: bool
    gate_entropy_lambda: float
    gate_entropy_eps: float
    best_global_mix_alpha: float
    use_val_for_checkpoint: bool
    best_metric: str
    log_personalized_client_test: bool
    log_global_only_acc: bool
    main_eval_set: str
    main_metric: str
    fair_comparison: bool
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
    device: str
    seed: int
    num_clients: int
    server_epochs: int
    data_name: str
    data_save_path: str
    model_save_path: str
    save_result: str
    backbone_name: str
    image_size: int
    patch_size: int
    hidden_dim: int
    switch_layers: int
    switch_heads: int
    switch_ffn_dim: int
    switch_num_experts: int
    dropout: float
    num_classes: int


def _parse_local_backbones(raw_value: str, default_backbone: str) -> Tuple[str, ...]:
    tokens = tuple(
        token.strip()
        for token in str(raw_value or "").split(",")
        if token.strip()
    )
    if not tokens:
        return (str(default_backbone),)
    invalid = [token for token in tokens if token not in VALID_BACKBONES]
    if invalid:
        raise ValueError(f"Unsupported pFedMoE local backbones: {invalid}")
    return tokens


def resolve_pfedmoe_config(args) -> PFedMoEConfig:
    enabled = bool(getattr(args, "pfedmoe_enabled", False)) or str(
        getattr(args, "aggregation_method", "")
    ).lower() == "pfedmoe" or str(getattr(args, "method", "")).lower() == "pfedmoe"
    data_name = str(getattr(args, "data_name", "cifar10"))
    num_classes_by_dataset = {
        "cifar10": 10,
        "cifar100": 100,
        "cinic10": 10,
        "tinyimagenet": 200,
    }
    global_backbone_name = str(getattr(args, "pfedmoe_global_backbone_name", "cnn_small"))
    if global_backbone_name not in VALID_BACKBONES:
        raise ValueError(f"Unsupported pFedMoE global backbone: {global_backbone_name}")

    ratio = float(getattr(args, "pfedmoe_client_selection_ratio", 1.0))
    ratio = min(max(ratio, 0.0), 1.0)

    server_test_mode = str(getattr(args, "pfedmoe_server_test_mode", "ensemble_clients")).lower()
    if server_test_mode not in VALID_SERVER_TEST_MODES:
        raise ValueError(f"Unsupported pFedMoE server test mode: {server_test_mode}")

    warmup_mode = str(getattr(args, "pfedmoe_gate_warmup_mode", "fixed_uniform")).lower()
    if warmup_mode not in VALID_WARMUP_MODES:
        raise ValueError(f"Unsupported pFedMoE gate warmup mode: {warmup_mode}")

    best_metric = str(getattr(args, "pfedmoe_best_metric", "val_acc")).lower()
    if best_metric not in VALID_BEST_METRICS:
        raise ValueError(f"Unsupported pFedMoE best metric: {best_metric}")

    label_smoothing = float(
        getattr(args, "pfedmoe_label_smoothing", getattr(args, "label_smoothing", 0.1))
    )
    weight_decay = float(
        getattr(args, "pfedmoe_weight_decay", getattr(args, "client_weight_decay", 5e-4))
    )
    best_global_mix_alpha = float(getattr(args, "pfedmoe_best_global_mix_alpha", 0.5))
    best_global_mix_alpha = min(max(best_global_mix_alpha, 0.0), 1.0)

    return PFedMoEConfig(
        enabled=enabled,
        gate_hidden_dim=max(int(getattr(args, "pfedmoe_gate_hidden_dim", 512)), 1),
        client_selection_ratio=ratio,
        aggregation_method=str(getattr(args, "pfedmoe_aggregation_method", "weighted_avg")),
        global_backbone_name=global_backbone_name,
        global_switch_layers=max(int(getattr(args, "pfedmoe_global_switch_layers", 1)), 1),
        local_backbones=_parse_local_backbones(
            getattr(args, "pfedmoe_local_backbones", ""),
            getattr(args, "backbone_name", "resnet18"),
        ),
        eval_fallback_to_server_test=bool(getattr(args, "pfedmoe_eval_fallback_to_server_test", True)),
        server_test_mode=server_test_mode,
        global_aux_loss=bool(getattr(args, "pfedmoe_global_aux_loss", True)),
        global_aux_lambda=max(float(getattr(args, "pfedmoe_global_aux_lambda", 0.5)), 0.0),
        aggregate_global_aux_head=bool(getattr(args, "pfedmoe_aggregate_global_aux_head", True)),
        alpha_global_target=min(max(float(getattr(args, "pfedmoe_alpha_global_target", 0.5)), 0.0), 1.0),
        alpha_global_reg_lambda=max(float(getattr(args, "pfedmoe_alpha_global_reg_lambda", 0.1)), 0.0),
        gate_warmup_rounds=max(int(getattr(args, "pfedmoe_gate_warmup_rounds", 5)), 0),
        gate_warmup_mode=warmup_mode,
        freeze_gate_during_warmup=bool(getattr(args, "pfedmoe_freeze_gate_during_warmup", True)),
        gate_entropy_lambda=max(float(getattr(args, "pfedmoe_gate_entropy_lambda", 0.01)), 0.0),
        gate_entropy_eps=max(float(getattr(args, "pfedmoe_gate_entropy_eps", 1e-8)), 1e-12),
        best_global_mix_alpha=best_global_mix_alpha,
        use_val_for_checkpoint=bool(getattr(args, "pfedmoe_use_val_for_checkpoint", True)),
        best_metric=best_metric,
        log_personalized_client_test=bool(getattr(args, "pfedmoe_log_personalized_client_test", True)),
        log_global_only_acc=bool(getattr(args, "pfedmoe_log_global_only_acc", True)),
        main_eval_set="server_test.pth",
        main_metric="server_test_acc",
        fair_comparison=True,
        batch_size=max(int(getattr(args, "batch_size", 1)), 1),
        client_epochs=max(int(getattr(args, "client_epochs", 1)), 1),
        learning_rate=float(getattr(args, "learning_rate", 5e-4)),
        client_optimizer=str(getattr(args, "client_optimizer", "adam")).lower(),
        client_weight_decay=weight_decay,
        client_momentum=float(getattr(args, "client_momentum", 0.0)),
        client_nesterov=bool(getattr(args, "client_nesterov", False)),
        label_smoothing=label_smoothing,
        grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
        use_client_aug=bool(getattr(args, "use_client_aug", False)),
        aug_random_crop_pad=int(getattr(args, "aug_random_crop_pad", 4)),
        aug_horizontal_flip=float(getattr(args, "aug_horizontal_flip", 0.5)),
        aug_random_erasing=float(getattr(args, "aug_random_erasing", 0.0)),
        mixup_alpha=float(getattr(args, "mixup_alpha", 0.0)),
        fedavg_weight_mode=str(getattr(args, "fedavg_weight_mode", "sample_count")).lower(),
        device=str(getattr(args, "device", "cpu")),
        seed=int(getattr(args, "seed", 0)),
        num_clients=max(int(getattr(args, "num_clients", 1)), 1),
        server_epochs=max(int(getattr(args, "server_epochs", 1)), 1),
        data_name=data_name,
        data_save_path=str(getattr(args, "data_save_path", "")),
        model_save_path=str(getattr(args, "model_save_path", "")),
        save_result=str(getattr(args, "save_result", "")),
        backbone_name=str(getattr(args, "backbone_name", "resnet18")),
        image_size=max(int(getattr(args, "image_size", 32)), 1),
        patch_size=max(int(getattr(args, "patch_size", 4)), 1),
        hidden_dim=max(int(getattr(args, "hidden_dim", 128)), 1),
        switch_layers=max(int(getattr(args, "switch_layers", 1)), 1),
        switch_heads=max(int(getattr(args, "switch_heads", 1)), 1),
        switch_ffn_dim=max(int(getattr(args, "switch_ffn_dim", 128)), 1),
        switch_num_experts=max(int(getattr(args, "switch_num_experts", 1)), 1),
        dropout=float(getattr(args, "dropout", 0.0)),
        num_classes=num_classes_by_dataset[data_name],
    )
