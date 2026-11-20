import json
import logging
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from args import parse
from Models.hfedmoe import ExpertPool, HFedMoEGate, HFedMoEModel, build_shared_backbone, select_topk_experts
from hfedmoe.config import resolve_hfedmoe_config
from hfedmoe.runtime import HFedMoERuntime
from server import Server


def make_config(tmp_path):
    args = SimpleNamespace(
        hfedmoe_enabled=True,
        aggregation_method="hfedmoe",
        method="hfedmoe",
        hfedmoe_num_experts=4,
        hfedmoe_top_k=2,
        hfedmoe_shared_backbone_name="patch_embed",
        hfedmoe_shared_switch_layers=1,
        hfedmoe_train_shared_backbone=True,
        hfedmoe_expert_backbone_name="resnet18",
        hfedmoe_expert_hidden_dim=12,
        hfedmoe_gating_hidden_dim=10,
        hfedmoe_gate_entropy_lambda=0.01,
        hfedmoe_importance_metric="val_loss",
        hfedmoe_importance_max_batches=1,
        hfedmoe_importance_eps=1e-6,
        hfedmoe_resource_budget_mode=True,
        hfedmoe_client_memory_budget_mb=4096.0,
        hfedmoe_client_flops_budget=1e12,
        hfedmoe_client_budget_scales="1.0,0.5",
        hfedmoe_min_active_experts=1,
        hfedmoe_client_selection_ratio=1.0,
        hfedmoe_aggregation_strategy="sparse_weighted",
        hfedmoe_combine_method="gated",
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
        expert_weight_mode="uniform",
        device="cpu",
        seed=1,
        num_clients=2,
        server_epochs=1,
        data_name="cifar10",
        data_save_path=str(tmp_path / "data"),
        model_save_path=str(tmp_path / "model"),
        save_result=str(tmp_path / "result"),
        image_size=8,
        patch_size=4,
        hidden_dim=8,
        switch_heads=1,
        switch_ffn_dim=16,
        switch_num_experts=2,
        dropout=0.0,
    )
    return resolve_hfedmoe_config(args)


def save_client_dataset(root, client_id, split, num_samples=4):
    inputs = torch.randn(num_samples, 3, 8, 8)
    labels = torch.randint(0, 10, (num_samples,))
    torch.save(TensorDataset(inputs, labels), root / f"{client_id}_{split}.pth")


def test_hfedmoe_gate_outputs_softmax_probabilities():
    gate = HFedMoEGate(input_dim=8, hidden_dim=10, num_experts=4)
    weights = gate(torch.randn(5, 8))
    assert weights.shape == (5, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_hfedmoe_select_topk_experts_returns_valid_ids():
    gate_weights = torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.3, 0.1, 0.4, 0.2]])
    selected_ids, scores = select_topk_experts(gate_weights, k=2)
    assert len(selected_ids) == 2
    assert all(0 <= expert_id < 4 for expert_id in selected_ids)
    assert scores.numel() == 2


def test_hfedmoe_client_updates_only_selected_experts_and_gate(tmp_path):
    from hfedmoe.client import HFedMoEClientTrainer

    config = make_config(tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    save_client_dataset(data_root, 1, "train")

    shared_backbone = build_shared_backbone(config)
    gate = HFedMoEGate(input_dim=config.hidden_dim, hidden_dim=config.gating_hidden_dim, num_experts=config.num_experts)
    expert_pool = ExpertPool(
        num_experts=config.num_experts,
        input_dim=config.hidden_dim,
        hidden_dim=config.expert_hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    original_shared_backbone_state = {name: tensor.detach().clone() for name, tensor in shared_backbone.state_dict().items()}
    original_gate_state = {name: tensor.detach().clone() for name, tensor in gate.state_dict().items()}
    original_expert_states = expert_pool.state_dict_for(range(config.num_experts))

    trainer = HFedMoEClientTrainer(
        config=config,
        client_id=1,
        round_idx=1,
        logger=logging.getLogger("hfedmoe_client_test"),
        shared_backbone_state=shared_backbone.state_dict(),
        gate_state=gate.state_dict(),
        expert_states=original_expert_states,
    )
    update = trainer.train()

    assert len(update["selected_expert_ids"]) >= 1
    inactive_ids = [expert_id for expert_id in range(config.num_experts) if expert_id not in update["selected_expert_ids"]]
    for expert_id in inactive_ids:
        current_state = trainer.expert_pool.state_dict_for([expert_id])[expert_id]
        baseline_state = original_expert_states[expert_id]
        assert all(torch.allclose(current_state[name], baseline_state[name]) for name in baseline_state)

    assert any(not torch.allclose(update["gate_state"][name], original_gate_state[name]) for name in original_gate_state)
    assert any(
        not torch.allclose(update["shared_backbone_state"][name], original_shared_backbone_state[name])
        for name in original_shared_backbone_state
    )


def test_hfedmoe_server_aggregation_only_updates_active_experts(tmp_path):
    config = make_config(tmp_path)
    runtime = HFedMoERuntime(config, logging.getLogger("hfedmoe_runtime_test"), torch.device("cpu"))
    before_state = runtime.expert_pool.state_dict_for(range(config.num_experts))
    before_shared_backbone_state = {name: tensor.detach().clone() for name, tensor in runtime.shared_backbone_state.items()}

    updates = [
        {
            "num_samples": 4,
            "mean_selected_expert_score": 1.0,
            "selected_expert_scores": {0: 1.0},
            "shared_backbone_state": {name: tensor.detach().clone() + 1.0 for name, tensor in runtime.shared_backbone_state.items()},
            "gate_state": runtime.gate_state,
            "expert_states": {0: before_state[0]},
        },
        {
            "num_samples": 4,
            "mean_selected_expert_score": 1.0,
            "selected_expert_scores": {1: 1.0},
            "shared_backbone_state": {name: tensor.detach().clone() + 3.0 for name, tensor in runtime.shared_backbone_state.items()},
            "gate_state": runtime.gate_state,
            "expert_states": {1: before_state[1]},
        },
    ]
    aggregation_info = runtime.aggregate_updates(updates)
    after_state = runtime.expert_pool.state_dict_for(range(config.num_experts))

    assert sorted(aggregation_info["updated_expert_ids"]) == [0, 1]
    assert all(torch.allclose(after_state[2][name], before_state[2][name]) for name in before_state[2])
    assert aggregation_info["server_agg_expert_params_norm"] >= 0.0
    assert any(not torch.allclose(runtime.shared_backbone_state[name], before_shared_backbone_state[name]) for name in before_shared_backbone_state)


def test_hfedmoe_method_smoke(monkeypatch, tmp_path):
    torch.manual_seed(1)
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    result_root = tmp_path / "result"
    data_root.mkdir()
    model_root.mkdir()
    result_root.mkdir()

    for client_id in [1, 2]:
        save_client_dataset(data_root, client_id, "train")

    torch.save(TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))), data_root / "server_test.pth")

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
            "8",
            "--switch_heads",
            "1",
            "--switch_ffn_dim",
            "16",
            "--switch_num_experts",
            "2",
            "--backbone_name",
            "patch_embed",
            "--patch_size",
            "4",
            "--image_size",
            "8",
            "--aggregation_method",
            "hfedmoe",
            "--hfedmoe_enabled",
            "true",
            "--hfedmoe_num_experts",
            "4",
            "--hfedmoe_top_k",
            "2",
            "--hfedmoe_shared_backbone_name",
            "patch_embed",
            "--hfedmoe_expert_hidden_dim",
            "12",
            "--hfedmoe_gating_hidden_dim",
            "10",
            "--fedavg_weight_mode",
            "sample_count",
            "--data_save_path",
            str(data_root),
            "--model_save_path",
            str(model_root),
            "--save_result",
            str(result_root),
        ],
    )

    logger = logging.getLogger("hfedmoe_smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)

    Server(parse=parse, logger=logger).train()

    with open(result_root / "logs" / "run_summary.txt") as summary_file:
        summary = json.load(summary_file)
    assert summary["algorithm"] == "hfedmoe"
    assert summary["main_eval_set"] == "server_test.pth"
    assert summary["main_metric"] == "server_test_acc"
    assert (result_root / "logs" / "hfedmoe_round_metrics.csv").exists()
