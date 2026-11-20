import json
import logging
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from args import parse
from pfedmoe.config import resolve_pfedmoe_config
from pfedmoe.modeling import (
    PFedMoEGate,
    PFedMoEPersonalModel,
    SwitchTransformerFeatureExtractor,
)
from server import Server


def make_config(tmp_path):
    args = SimpleNamespace(
        pfedmoe_enabled=True,
        aggregation_method="pfedmoe",
        method="pfedmoe",
        pfedmoe_gate_hidden_dim=16,
        pfedmoe_client_selection_ratio=1.0,
        pfedmoe_aggregation_method="weighted_avg",
        pfedmoe_global_backbone_name="cnn_small",
        pfedmoe_global_switch_layers=1,
        pfedmoe_local_backbones="patch_embed,resnet8",
        pfedmoe_eval_fallback_to_server_test=True,
        pfedmoe_server_test_mode="ensemble_clients",
        pfedmoe_global_aux_loss=True,
        pfedmoe_global_aux_lambda=0.5,
        pfedmoe_aggregate_global_aux_head=True,
        pfedmoe_alpha_global_target=0.5,
        pfedmoe_alpha_global_reg_lambda=0.1,
        pfedmoe_gate_warmup_rounds=2,
        pfedmoe_gate_warmup_mode="fixed_uniform",
        pfedmoe_freeze_gate_during_warmup=True,
        pfedmoe_gate_entropy_lambda=0.01,
        pfedmoe_gate_entropy_eps=1e-8,
        pfedmoe_label_smoothing=0.1,
        pfedmoe_weight_decay=5e-4,
        pfedmoe_use_val_for_checkpoint=True,
        pfedmoe_best_metric="val_acc",
        pfedmoe_log_personalized_client_test=True,
        pfedmoe_log_global_only_acc=True,
        pfedmoe_best_global_mix_alpha=0.5,
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
    return resolve_pfedmoe_config(args)


def save_client_dataset(root, client_id, split, num_samples=4):
    inputs = torch.randn(num_samples, 3, 8, 8)
    labels = torch.randint(0, 10, (num_samples,))
    torch.save(TensorDataset(inputs, labels), root / f"{client_id}_{split}.pth")


def make_feature_extractor(backbone_name="patch_embed"):
    return SwitchTransformerFeatureExtractor(
        image_size=8,
        patch_size=4,
        backbone_name=backbone_name,
        hidden_dim=16,
        num_layers=1,
        num_heads=1,
        ffn_dim=32,
        num_experts=2,
        dropout=0.0,
    )


def test_pfedmoe_gate_weights_sum_to_one():
    gate = PFedMoEGate(input_dim=32, hidden_dim=16)
    weights = gate(torch.randn(4, 32))
    assert weights.shape == (4, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_pfedmoe_personal_model_output_shape_matches_head():
    model = PFedMoEPersonalModel(
        global_extractor=make_feature_extractor("patch_embed"),
        local_extractor=make_feature_extractor("patch_embed"),
        gate=PFedMoEGate(input_dim=32, hidden_dim=16),
        prediction_head=torch.nn.Linear(16, 10),
        global_aux_head=torch.nn.Linear(16, 10),
    )
    outputs = model(torch.randn(3, 3, 8, 8))
    assert outputs["mixed_features"].shape == (3, 16)
    assert outputs["logits"].shape == (3, 10)
    assert outputs["global_logits"].shape == (3, 10)


def test_pfedmoe_config_defaults_to_fair_server_test_setup(tmp_path):
    args = SimpleNamespace(
        pfedmoe_enabled=True,
        aggregation_method="pfedmoe",
        method="pfedmoe",
        pfedmoe_gate_hidden_dim=16,
        pfedmoe_client_selection_ratio=1.0,
        pfedmoe_aggregation_method="weighted_avg",
        pfedmoe_global_switch_layers=1,
        pfedmoe_local_backbones="",
        pfedmoe_eval_fallback_to_server_test=True,
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
    config = resolve_pfedmoe_config(args)
    assert config.global_backbone_name == "cnn_small"
    assert config.server_test_mode == "ensemble_clients"
    assert config.main_eval_set == "server_test.pth"
    assert config.main_metric == "server_test_acc"
    assert config.fair_comparison is True
    assert config.global_aux_loss is True


def test_pfedmoe_server_side_shared_extractor_template_is_frozen(tmp_path):
    from pfedmoe.runtime import PFedMoERuntime

    config_args = SimpleNamespace(
        pfedmoe_enabled=True,
        aggregation_method="pfedmoe",
        method="pfedmoe",
        pfedmoe_gate_hidden_dim=16,
        pfedmoe_client_selection_ratio=1.0,
        pfedmoe_aggregation_method="weighted_avg",
        pfedmoe_global_backbone_name="patch_embed",
        pfedmoe_global_switch_layers=1,
        pfedmoe_local_backbones="patch_embed",
        pfedmoe_eval_fallback_to_server_test=True,
        pfedmoe_server_test_mode="ensemble_clients",
        pfedmoe_global_aux_loss=True,
        pfedmoe_global_aux_lambda=0.5,
        pfedmoe_aggregate_global_aux_head=True,
        pfedmoe_alpha_global_target=0.5,
        pfedmoe_alpha_global_reg_lambda=0.1,
        pfedmoe_gate_warmup_rounds=1,
        pfedmoe_gate_warmup_mode="fixed_uniform",
        pfedmoe_freeze_gate_during_warmup=True,
        pfedmoe_gate_entropy_lambda=0.01,
        pfedmoe_gate_entropy_eps=1e-8,
        pfedmoe_label_smoothing=0.1,
        pfedmoe_weight_decay=5e-4,
        pfedmoe_use_val_for_checkpoint=False,
        pfedmoe_best_metric="server_test_acc",
        pfedmoe_log_personalized_client_test=False,
        pfedmoe_log_global_only_acc=True,
        pfedmoe_best_global_mix_alpha=0.5,
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
    runtime = PFedMoERuntime(config_args, logging.getLogger("pfedmoe_frozen"), torch.device("cpu"))
    assert all(not param.requires_grad for param in runtime.frozen_shared_extractor_template.parameters())
    assert runtime.global_num_params > 0


def test_pfedmoe_method_smoke(monkeypatch, tmp_path):
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
            "pfedmoe",
            "--pfedmoe_enabled",
            "true",
            "--pfedmoe_global_backbone_name",
            "cnn_small",
            "--pfedmoe_global_switch_layers",
            "1",
            "--pfedmoe_local_backbones",
            "patch_embed",
            "--pfedmoe_client_selection_ratio",
            "1.0",
            "--pfedmoe_server_test_mode",
            "ensemble_clients",
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

    logger = logging.getLogger("pfedmoe_smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)

    Server(parse=parse, logger=logger).train()

    with open(result_root / "logs" / "run_summary.txt") as summary_file:
        summary = json.load(summary_file)
    assert summary["final_loss"] > 0.0
    assert summary["eval_mode"] == "server_test"
    assert summary["main_eval_set"] == "server_test.pth"
    assert summary["fair_comparison"] is True
    assert summary["server_test_mode"] == "ensemble_clients"
    checkpoint = torch.load(model_root / "server.pth", weights_only=False)
    assert "global_aux_head_state" in checkpoint
