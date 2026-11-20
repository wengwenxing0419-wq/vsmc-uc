from types import SimpleNamespace

import torch

from Models.SwitchTransformer import SwitchTransformerClassifier
from server import (
    AdaptMetrics,
    SelectiveAdaptState,
    Server,
    TeacherRouterMetrics,
    selective_adapt_accept_rule,
    strict_adapt_accept_rule,
    teacher_router_accept_rule,
)


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _config(**overrides):
    base = {
        "server_adapt_late_start_frac": 0.8,
        "server_adapt_min_delta": 3e-4,
        "server_adapt_mid_start_frac": 0.7,
        "server_adapt_mid_min_delta": 7e-4,
        "server_adapt_mid_margin_delta": 8e-4,
        "server_adapt_late_min_delta": 1e-3,
        "server_adapt_margin_delta": 5e-4,
        "server_adapt_late_margin_delta": 1e-3,
        "server_adapt_weak_loss_delta": 1e-4,
        "server_adapt_margin_weight": 0.2,
        "server_adapt_evidence_gate": True,
        "server_adapt_evidence_required_mid": 2,
        "server_adapt_evidence_required_late": 3,
        "server_adapt_router_entropy_tolerance": 0.03,
        "server_adapt_late_cooldown": True,
        "server_adapt_cooldown_start_frac": 0.75,
        "server_adapt_cooldown_after_accept": 1,
        "server_adapt_rescue_if_loss_worse": True,
        "server_adapt_rescue_loss_threshold": 0.01,
        "server_adapt_near_best_protection": True,
        "server_adapt_near_best_tolerance": 0.002,
        "server_adapt_near_best_require_late_rule": True,
        "server_adapt_update_head": True,
        "server_adapt_update_final_norm": True,
        "server_adapt_update_moe_layernorm": True,
        "server_adapt_update_router_temperature": True,
        "server_adapt_update_router_bias": False,
        "server_adapt_update_full_router": False,
        "server_adapt_update_full_dense": False,
        "server_adapt_update_experts": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_strict_accept_rejects_tiny_loss_only_gain():
    before = AdaptMetrics(loss=1.0, acc=0.80, margin=0.10, router_load_entropy=0.7)
    after = AdaptMetrics(loss=0.99999, acc=0.80, margin=0.10, router_load_entropy=0.7)
    accept, reason, debug = strict_adapt_accept_rule(before, after, progress=0.6, config=_config())
    assert accept is False
    assert reason == "strict_reject"
    assert debug["strong_accept"] is False


def test_late_phase_requires_non_decreasing_accuracy():
    before = AdaptMetrics(loss=1.0, acc=0.82, margin=0.10, router_load_entropy=0.7)
    after = AdaptMetrics(loss=0.995, acc=0.81, margin=0.102, router_load_entropy=0.7)
    accept, reason, debug = strict_adapt_accept_rule(before, after, progress=0.9, config=_config())
    assert accept is False
    assert reason in {"late_reject", "late_insufficient_improvement"}
    assert debug["late_phase"] is True


def test_default_server_adapt_selection_excludes_experts_and_router_weights():
    model = SwitchTransformerClassifier(
        num_classes=10,
        backbone_name="patch_embed",
        image_size=8,
        patch_size=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        num_experts=3,
        dropout=0.0,
    )
    server = Server.__new__(Server)
    server.logger = _DummyLogger()
    params, names = server.select_server_adapt_params(model, _config())
    assert params
    assert any(name.startswith("classifier") for name in names)
    assert any(name == "norm.weight" for name in names)
    assert any(name.endswith(".server_temperature") for name in names)
    assert all(".switch_ffn.experts." not in name for name in names)
    assert all(not name.endswith(".router.weight") for name in names)
    assert all(not name.endswith(".router.bias") for name in names)


def test_selective_accept_near_best_can_reject_despite_evidence():
    before = AdaptMetrics(loss=1.0, acc=0.900, margin=0.10, router_load_entropy=0.95)
    after = AdaptMetrics(loss=0.9985, acc=0.900, margin=0.1010, router_load_entropy=0.94)
    state = SelectiveAdaptState(
        best_proxy=0.9010,
        best_accept_loss=0.97,
        cooldown_remaining=0,
        accept_threshold_multiplier=1.0,
    )
    accept, reason, debug = selective_adapt_accept_rule(
        before,
        after,
        progress=0.92,
        config=_config(server_adapt_late_min_delta=1.5e-3, server_adapt_late_margin_delta=1.5e-3),
        state=state,
    )
    assert accept is False
    assert reason == "near_best_reject"
    assert debug["near_best"] is True
    assert debug["near_best_reject"] is True


def test_selective_accept_cooldown_skip_blocks_accept():
    before = AdaptMetrics(loss=1.0, acc=0.85, margin=0.10, router_load_entropy=0.96)
    after = AdaptMetrics(loss=0.990, acc=0.86, margin=0.12, router_load_entropy=0.96)
    state = SelectiveAdaptState(
        best_proxy=0.88,
        best_accept_loss=0.995,
        cooldown_remaining=1,
        accept_threshold_multiplier=1.0,
    )
    accept, reason, debug = selective_adapt_accept_rule(before, after, progress=0.80, config=_config(), state=state)
    assert accept is False
    assert reason == "cooldown_skip"
    assert debug["cooldown_active"] is True
    assert debug["cooldown_skip"] is True


def test_selective_accept_mid_phase_requires_evidence_and_accepts():
    before = AdaptMetrics(loss=1.0, acc=0.80, margin=0.10, router_load_entropy=0.95)
    after = AdaptMetrics(loss=0.9989, acc=0.81, margin=0.1020, router_load_entropy=0.94)
    state = SelectiveAdaptState(
        best_proxy=0.85,
        best_accept_loss=0.99,
        cooldown_remaining=0,
        accept_threshold_multiplier=1.0,
    )
    accept, reason, debug = selective_adapt_accept_rule(before, after, progress=0.75, config=_config(), state=state)
    assert accept is True
    assert reason in {"mid_evidence_accept", "near_best_strict_accept"}
    assert debug["adapt_phase"] == "mid"
    assert debug["evidence_count"] >= debug["evidence_required"]


def test_teacher_router_accept_rule_rejects_when_teacher_loss_worsens():
    before = AdaptMetrics(loss=1.0, acc=0.82, margin=0.10, router_load_entropy=0.95)
    after = AdaptMetrics(loss=0.999, acc=0.82, margin=0.101, router_load_entropy=0.95)
    before_teacher = TeacherRouterMetrics(
        teacher_router_loss=0.30,
        teacher_label_entropy_mean=0.8,
        teacher_conf_mean=0.1,
        teacher_conf_std=0.02,
        teacher_valid_ratio=1.0,
        router_teacher_acc=0.5,
        teacher_proxy_mode=False,
        counterfactual_success=True,
        counterfactual_fallback_reason="",
    )
    after_teacher = TeacherRouterMetrics(
        teacher_router_loss=0.35,
        teacher_label_entropy_mean=0.8,
        teacher_conf_mean=0.1,
        teacher_conf_std=0.02,
        teacher_valid_ratio=1.0,
        router_teacher_acc=0.5,
        teacher_proxy_mode=False,
        counterfactual_success=True,
        counterfactual_fallback_reason="",
    )
    accept, reason, debug = teacher_router_accept_rule(
        before,
        after,
        before_teacher,
        after_teacher,
        progress=0.7,
        config=_config(),
    )
    assert accept is False
    assert reason == "teacher_router_reject"
    assert debug["router_evidence"] is False


def test_v12_teacher_router_selection_includes_router_bias():
    model = SwitchTransformerClassifier(
        num_classes=10,
        backbone_name="patch_embed",
        image_size=8,
        patch_size=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        num_experts=3,
        dropout=0.0,
    )
    server = Server.__new__(Server)
    server.logger = _DummyLogger()
    cfg = _config(
        server_adapt_version="v12_teacher_router",
        server_adapt_teacher_router=True,
        server_adapt_teacher_update_router_bias=True,
        server_adapt_teacher_update_router_temperature=True,
        server_adapt_teacher_update_full_router=False,
    )
    params, names = server.select_server_adapt_params(model, cfg)
    assert any(name.endswith(".server_bias") for name in names)
    assert any(name.endswith(".server_temperature") for name in names)


def test_fedavg_expert_weight_mode_only_changes_expert_aggregation():
    server = Server.__new__(Server)
    server.device = torch.device("cpu")
    server.args = SimpleNamespace(
        fedavg_weight_mode="sample_count",
        expert_weight_mode="uniform",
    )
    client_updates = [
        (
            {
                "classifier.weight": torch.tensor([1.0]),
                "blocks.0.switch_ffn.experts.0.0.weight": torch.tensor([10.0]),
            },
            9,
            {},
        ),
        (
            {
                "classifier.weight": torch.tensor([3.0]),
                "blocks.0.switch_ffn.experts.0.0.weight": torch.tensor([20.0]),
            },
            1,
            {},
        ),
    ]
    aggregated = server._compute_fedavg_state(client_updates)
    assert torch.allclose(aggregated["classifier.weight"], torch.tensor([1.2]))
    assert torch.allclose(aggregated["blocks.0.switch_ffn.experts.0.0.weight"], torch.tensor([15.0]))


def test_fedavg_uniform_weight_mode_changes_all_aggregation():
    server = Server.__new__(Server)
    server.device = torch.device("cpu")
    server.args = SimpleNamespace(
        fedavg_weight_mode="uniform",
        expert_weight_mode="uniform",
    )
    client_updates = [
        (
            {
                "classifier.weight": torch.tensor([1.0]),
                "blocks.0.switch_ffn.experts.0.0.weight": torch.tensor([10.0]),
            },
            9,
            {},
        ),
        (
            {
                "classifier.weight": torch.tensor([3.0]),
                "blocks.0.switch_ffn.experts.0.0.weight": torch.tensor([20.0]),
            },
            1,
            {},
        ),
    ]
    aggregated = server._compute_fedavg_state(client_updates)
    assert torch.allclose(aggregated["classifier.weight"], torch.tensor([2.0]))
    assert torch.allclose(aggregated["blocks.0.switch_ffn.experts.0.0.weight"], torch.tensor([15.0]))
