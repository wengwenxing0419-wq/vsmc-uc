from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FedMoEDAConfig:
    top_p: int
    temperature: float
    personalized_mix: float
    similarity_metric: str
    min_similarity: float
    shared_weight_mode: str
    expert_weight_mode: str
    num_clients: int
    server_epochs: int
    client_epochs: int
    batch_size: int
    learning_rate: float
    device: str
    seed: int
    data_name: str
    data_save_path: str
    save_result: str
    model_save_path: str
    backbone_name: str


def resolve_fedmoe_da_config(args) -> FedMoEDAConfig:
    num_clients = max(int(getattr(args, "num_clients", 1)), 1)
    top_p = max(1, min(int(getattr(args, "fedmoe_da_top_p", 3)), num_clients))
    personalized_mix = float(getattr(args, "fedmoe_da_personalized_mix", 1.0))
    personalized_mix = min(max(personalized_mix, 0.0), 1.0)
    return FedMoEDAConfig(
        top_p=top_p,
        temperature=max(float(getattr(args, "fedmoe_da_temperature", 1.0)), 1e-6),
        personalized_mix=personalized_mix,
        similarity_metric=str(getattr(args, "fedmoe_da_similarity_metric", "router_row_cosine")),
        min_similarity=float(getattr(args, "fedmoe_da_min_similarity", 0.0)),
        shared_weight_mode=str(getattr(args, "fedavg_weight_mode", "uniform")).lower(),
        expert_weight_mode=str(getattr(args, "expert_weight_mode", "uniform")).lower(),
        num_clients=num_clients,
        server_epochs=max(int(getattr(args, "server_epochs", 1)), 1),
        client_epochs=max(int(getattr(args, "client_epochs", 1)), 1),
        batch_size=max(int(getattr(args, "batch_size", 1)), 1),
        learning_rate=float(getattr(args, "learning_rate", 0.0)),
        device=str(getattr(args, "device", "cpu")),
        seed=int(getattr(args, "seed", 0)),
        data_name=str(getattr(args, "data_name", "")),
        data_save_path=str(getattr(args, "data_save_path", "")),
        save_result=str(getattr(args, "save_result", "")),
        model_save_path=str(getattr(args, "model_save_path", "")),
        backbone_name=str(getattr(args, "backbone_name", "")),
    )
