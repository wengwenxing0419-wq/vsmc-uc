import logging
import json
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from args import parse
from fedjets.client import FedJETSClientTrainer
from fedjets.common_expert import CommonExpertWrapper
from fedjets.config import resolve_fedjets_config
from fedjets.expert_pool import ExpertPool
from fedjets.gate import FedJETSGate
from fedjets.utils import clone_state_dict, select_topk_experts
from server import Server


def make_small_model():
    from Models.SwitchTransformer import SwitchTransformerClassifier

    return SwitchTransformerClassifier(
        num_classes=10,
        image_size=8,
        patch_size=4,
        backbone_name="patch_embed",
        hidden_dim=16,
        num_layers=1,
        num_heads=1,
        ffn_dim=32,
        num_experts=2,
        dropout=0.0,
    )


def make_config(tmp_path):
    args = SimpleNamespace(
        fedjets_enabled=True,
        aggregation_method="fedjets",
        method="fedjets",
        fedjets_num_experts=3,
        fedjets_top_k=2,
        fedjets_gate_hidden_dim=32,
        fedjets_num_anchor_clients=1,
        fedjets_anchor_clients_per_round=1,
        fedjets_normal_clients_per_round=1,
        fedjets_common_expert_ckpt="",
        fedjets_cache_embeddings=False,
        fedjets_gate_lr=1e-3,
        fedjets_expert_lr=1e-3,
        batch_size=2,
        client_epochs=1,
        learning_rate=1e-3,
        client_optimizer="adam",
        client_weight_decay=0.0,
        client_momentum=0.0,
        client_nesterov=False,
        label_smoothing=0.0,
        grad_clip_norm=0.0,
        use_client_aug=False,
        aug_random_crop_pad=0,
        aug_horizontal_flip=0.0,
        aug_random_erasing=0.0,
        mixup_alpha=0.0,
        fedavg_weight_mode="sample_count",
        expert_weight_mode="sample_count",
        device="cpu",
        seed=1,
        num_clients=2,
        server_epochs=1,
        data_name="cifar10",
        data_save_path=str(tmp_path / "data"),
        model_save_path=str(tmp_path / "model"),
        save_result=str(tmp_path / "result"),
        backbone_name="patch_embed",
        image_size=8,
        patch_size=4,
        hidden_dim=16,
        switch_layers=1,
        switch_heads=1,
        switch_ffn_dim=32,
        switch_num_experts=2,
        dropout=0.0,
    )
    os.makedirs(args.data_save_path, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.save_result, exist_ok=True)
    return resolve_fedjets_config(args)


def save_client_dataset(root, client_id, split, num_samples=4):
    inputs = torch.randn(num_samples, 3, 8, 8)
    labels = torch.randint(0, 10, (num_samples,))
    torch.save(TensorDataset(inputs, labels), root / f"{client_id}_{split}.pth")


def any_tensor_changed(before_state, after_state):
    for name, before_tensor in before_state.items():
        if torch.is_tensor(before_tensor) and not torch.equal(before_tensor, after_state[name]):
            return True
    return False


def test_fedjets_gate_shape():
    gate = FedJETSGate(embedding_dim=8, num_experts=5, hidden_dim=16)
    probs = gate(torch.randn(4, 8))
    assert probs.shape == (4, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_fedjets_topk_selection_returns_valid_ids():
    gate = FedJETSGate(embedding_dim=4, num_experts=3, hidden_dim=8)
    with torch.no_grad():
        gate.mlp[0].weight.zero_()
        gate.mlp[0].bias.zero_()
        gate.mlp[2].weight.zero_()
        gate.mlp[2].bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
    ids, _ = select_topk_experts(gate, torch.randn(6, 4), k=2, device=torch.device("cpu"))
    assert ids == [2, 1]


def test_common_expert_is_frozen():
    wrapper = CommonExpertWrapper(make_small_model(), device=torch.device("cpu"), cache_embeddings=False)
    assert wrapper.embedding_dim == 16
    assert all(not param.requires_grad for param in wrapper.model.parameters())


def test_normal_client_only_updates_selected_experts(tmp_path):
    config = make_config(tmp_path)
    data_root = tmp_path / "data"
    save_client_dataset(data_root, 1, "train")
    common_expert = CommonExpertWrapper(make_small_model(), device=torch.device("cpu"), cache_embeddings=False)
    gate = FedJETSGate(config.embedding_dim or 16, config.num_experts, config.gate_hidden_dim)
    initial_pool = ExpertPool(config.num_experts, make_small_model, make_small_model().state_dict())
    selected_ids = [0, 2]
    selected_states = initial_pool.state_dict_for(selected_ids)
    before_state = clone_state_dict(selected_states[0])
    trainer = FedJETSClientTrainer(
        config=type(config)(**{**config.__dict__, "embedding_dim": 16}),
        client_id=1,
        round_idx=1,
        role="normal",
        model_factory=make_small_model,
        common_expert=common_expert,
        gate_state=gate.state_dict(),
        expert_states=selected_states,
        selected_expert_ids=selected_ids,
        assigned_expert_id=None,
    )
    update = trainer.train()
    assert sorted(update["expert_states"].keys()) == selected_ids
    assert 1 not in update["expert_states"]
    assert any_tensor_changed(before_state, update["expert_states"][0])


def test_anchor_client_only_updates_assigned_expert(tmp_path):
    config = make_config(tmp_path)
    data_root = tmp_path / "data"
    save_client_dataset(data_root, 1, "train")
    common_expert = CommonExpertWrapper(make_small_model(), device=torch.device("cpu"), cache_embeddings=False)
    gate = FedJETSGate(16, config.num_experts, config.gate_hidden_dim)
    initial_pool = ExpertPool(config.num_experts, make_small_model, make_small_model().state_dict())
    trainer = FedJETSClientTrainer(
        config=type(config)(**{**config.__dict__, "embedding_dim": 16}),
        client_id=1,
        round_idx=1,
        role="anchor",
        model_factory=make_small_model,
        common_expert=common_expert,
        gate_state=gate.state_dict(),
        expert_states=initial_pool.state_dict_for([1]),
        selected_expert_ids=[1],
        assigned_expert_id=1,
    )
    update = trainer.train()
    assert list(update["expert_states"].keys()) == [1]
    assert update["assigned_expert_id"] == 1


def test_fedavg_expert_aggregation_keeps_unselected_experts_unchanged():
    initial_state = make_small_model().state_dict()
    pool = ExpertPool(num_experts=3, model_factory=make_small_model, initial_state=initial_state)
    before = pool.state_dict_for([0, 1, 2])
    updated_state = clone_state_dict(before[1])
    first_key = next(iter(updated_state))
    updated_state[first_key] = updated_state[first_key] + 1.0
    pool.aggregate_expert_updates([{1: updated_state}], [1.0])
    after = pool.state_dict_for([0, 1, 2])
    assert torch.equal(before[0][first_key], after[0][first_key])
    assert torch.equal(before[2][first_key], after[2][first_key])
    assert not torch.equal(before[1][first_key], after[1][first_key])


def test_method_fedjets_smoke(monkeypatch, tmp_path):
    torch.manual_seed(1)
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    result_root = tmp_path / "result"
    data_root.mkdir()
    model_root.mkdir()
    result_root.mkdir()

    for client_id in [1, 2]:
        save_client_dataset(data_root, client_id, "train")
        save_client_dataset(data_root, client_id, "val", num_samples=2)
        save_client_dataset(data_root, client_id, "test", num_samples=2)

    torch.save(TensorDataset(torch.randn(2, 3, 8, 8), torch.randint(0, 10, (2,))), data_root / "server_test.pth")
    torch.save(TensorDataset(torch.randn(2, 3, 8, 8), torch.randint(0, 10, (2,))), data_root / "server_val.pth")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "--data_name",
            "cifar10",
            "--num_clients",
            "2",
            "--device",
            "cpu",
            "--server_epochs",
            "1",
            "--client_epochs",
            "1",
            "--batch_size",
            "2",
            "--learning_rate",
            "0.001",
            "--hidden_dim",
            "16",
            "--switch_layers",
            "1",
            "--switch_heads",
            "1",
            "--switch_ffn_dim",
            "32",
            "--switch_num_experts",
            "2",
            "--backbone_name",
            "patch_embed",
            "--patch_size",
            "4",
            "--image_size",
            "8",
            "--aggregation_method",
            "fedjets",
            "--fedjets_enabled",
            "true",
            "--fedjets_num_experts",
            "2",
            "--fedjets_top_k",
            "1",
            "--fedjets_num_anchor_clients",
            "1",
            "--fedjets_anchor_clients_per_round",
            "1",
            "--fedjets_normal_clients_per_round",
            "1",
            "--fedavg_weight_mode",
            "sample_count",
            "--expert_weight_mode",
            "sample_count",
            "--data_save_path",
            str(data_root),
            "--model_save_path",
            str(model_root),
            "--save_result",
            str(result_root),
        ],
    )

    logger = logging.getLogger("fedjets_smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)

    Server(parse=parse, logger=logger).train()

    assert (model_root / "server.pth").exists()
    assert (result_root / "logs" / "fedjets_round_metrics.csv").exists()
    checkpoint = torch.load(model_root / "server.pth", map_location="cpu")
    assert checkpoint["aggregation_method"] == "fedjets"



def test_method_fedjets_server_test_fallback(monkeypatch, tmp_path):
    torch.manual_seed(1)
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    result_root = tmp_path / "result"
    data_root.mkdir()
    model_root.mkdir()
    result_root.mkdir()

    for client_id in [1, 2]:
        save_client_dataset(data_root, client_id, "train")
        save_client_dataset(data_root, client_id, "val", num_samples=2)

    torch.save(TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))), data_root / "server_test.pth")
    torch.save(TensorDataset(torch.randn(2, 3, 8, 8), torch.randint(0, 10, (2,))), data_root / "server_val.pth")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "--data_name",
            "cifar10",
            "--num_clients",
            "2",
            "--device",
            "cpu",
            "--server_epochs",
            "1",
            "--client_epochs",
            "1",
            "--batch_size",
            "2",
            "--learning_rate",
            "0.001",
            "--hidden_dim",
            "16",
            "--switch_layers",
            "1",
            "--switch_heads",
            "1",
            "--switch_ffn_dim",
            "32",
            "--switch_num_experts",
            "2",
            "--backbone_name",
            "patch_embed",
            "--patch_size",
            "4",
            "--image_size",
            "8",
            "--aggregation_method",
            "fedjets",
            "--fedjets_enabled",
            "true",
            "--fedjets_num_experts",
            "2",
            "--fedjets_top_k",
            "1",
            "--fedjets_num_anchor_clients",
            "1",
            "--fedjets_anchor_clients_per_round",
            "1",
            "--fedjets_normal_clients_per_round",
            "1",
            "--fedavg_weight_mode",
            "sample_count",
            "--expert_weight_mode",
            "sample_count",
            "--data_save_path",
            str(data_root),
            "--model_save_path",
            str(model_root),
            "--save_result",
            str(result_root),
        ],
    )

    logger = logging.getLogger("fedjets_fallback_smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)

    Server(parse=parse, logger=logger).train()

    with open(result_root / "logs" / "run_summary.txt") as summary_file:
        summary = json.load(summary_file)
    assert summary["final_loss"] > 0.0
    assert summary["selected_expert_histogram"]
