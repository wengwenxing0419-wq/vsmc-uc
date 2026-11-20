from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FedJETSConfig:
    enabled: bool
    num_experts: int
    top_k: int
    gate_hidden_dim: int
    num_anchor_clients: int
    anchor_clients_per_round: int
    normal_clients_per_round: int
    common_expert_ckpt: str
    cache_embeddings: bool
    gate_lr: float
    expert_lr: float
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
    backbone_name: str
    image_size: int
    patch_size: int
    hidden_dim: int
    switch_layers: int
    switch_heads: int
    switch_ffn_dim: int
    switch_num_experts: int
    dropout: float
    embedding_dim: int = 0


def resolve_fedjets_config(args) -> FedJETSConfig:
    num_clients = max(int(getattr(args, "num_clients", 1)), 1)
    num_experts = max(int(getattr(args, "fedjets_num_experts", 5)), 1)
    requested_anchor_clients = max(int(getattr(args, "fedjets_num_anchor_clients", num_experts)), 0)
    num_anchor_clients = min(num_clients, max(requested_anchor_clients, min(num_experts, num_clients)))
    non_anchor_clients = max(num_clients - num_anchor_clients, 0)

    top_k = max(1, min(int(getattr(args, "fedjets_top_k", 2)), num_experts))
    anchor_clients_per_round = int(getattr(args, "fedjets_anchor_clients_per_round", num_anchor_clients))
    normal_clients_per_round = int(getattr(args, "fedjets_normal_clients_per_round", non_anchor_clients))
    anchor_clients_per_round = min(max(anchor_clients_per_round, 0), num_anchor_clients)
    normal_clients_per_round = min(max(normal_clients_per_round, 0), non_anchor_clients)
    if num_anchor_clients > 0 and anchor_clients_per_round == 0:
        anchor_clients_per_round = num_anchor_clients
    if non_anchor_clients > 0 and normal_clients_per_round == 0:
        normal_clients_per_round = non_anchor_clients

    base_lr = float(getattr(args, "learning_rate", 5e-4))
    gate_lr = float(getattr(args, "fedjets_gate_lr", -1.0))
    expert_lr = float(getattr(args, "fedjets_expert_lr", -1.0))
    if gate_lr <= 0.0:
        gate_lr = base_lr
    if expert_lr <= 0.0:
        expert_lr = base_lr

    enabled = bool(getattr(args, "fedjets_enabled", False)) or str(
        getattr(args, "aggregation_method", "")
    ).lower() == "fedjets" or str(getattr(args, "method", "")).lower() == "fedjets"

    return FedJETSConfig(
        enabled=enabled,
        num_experts=num_experts,
        top_k=top_k,
        gate_hidden_dim=max(int(getattr(args, "fedjets_gate_hidden_dim", 512)), 1),
        num_anchor_clients=num_anchor_clients,
        anchor_clients_per_round=anchor_clients_per_round,
        normal_clients_per_round=normal_clients_per_round,
        common_expert_ckpt=str(getattr(args, "fedjets_common_expert_ckpt", "") or ""),
        cache_embeddings=bool(getattr(args, "fedjets_cache_embeddings", True)),
        gate_lr=gate_lr,
        expert_lr=expert_lr,
        batch_size=max(int(getattr(args, "batch_size", 1)), 1),
        client_epochs=max(int(getattr(args, "client_epochs", 1)), 1),
        learning_rate=base_lr,
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
        expert_weight_mode=str(getattr(args, "expert_weight_mode", "sample_count")).lower(),
        device=str(getattr(args, "device", "cpu")),
        seed=int(getattr(args, "seed", 0)),
        num_clients=num_clients,
        server_epochs=max(int(getattr(args, "server_epochs", 1)), 1),
        data_name=str(getattr(args, "data_name", "")),
        data_save_path=str(getattr(args, "data_save_path", "")),
        model_save_path=str(getattr(args, "model_save_path", "")),
        save_result=str(getattr(args, "save_result", "")),
        backbone_name=str(getattr(args, "backbone_name", "")),
        image_size=max(int(getattr(args, "image_size", 32)), 1),
        patch_size=max(int(getattr(args, "patch_size", 4)), 1),
        hidden_dim=max(int(getattr(args, "hidden_dim", 128)), 1),
        switch_layers=max(int(getattr(args, "switch_layers", 1)), 1),
        switch_heads=max(int(getattr(args, "switch_heads", 1)), 1),
        switch_ffn_dim=max(int(getattr(args, "switch_ffn_dim", 128)), 1),
        switch_num_experts=max(int(getattr(args, "switch_num_experts", 1)), 1),
        dropout=float(getattr(args, "dropout", 0.0)),
    )
