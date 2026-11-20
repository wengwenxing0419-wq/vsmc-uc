import collections

import torch

from Models.SwitchTransformer import SwitchTransformerClassifier
from aggregation.history_aware_vsmc_expert_aggregator import (
    HAVSMCETFConfig,
    aggregate_with_ha_vsmc_etf,
    build_ha_vsmc_etf_state,
    get_expert_param_groups,
    get_non_expert_param_names,
)


def _build_model():
    model = SwitchTransformerClassifier(
        num_classes=2,
        image_size=4,
        patch_size=4,
        backbone_name="patch_embed",
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        num_experts=2,
        dropout=0.0,
    )
    model.eval()
    return model


def _clone_state_dict(model):
    return collections.OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
    )


def _shift_expert(state_dict, expert_id: int, delta: float):
    updated = collections.OrderedDict()
    prefix = f"blocks.0.switch_ffn.experts.{expert_id}."
    for name, tensor in state_dict.items():
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and name.startswith(prefix):
            updated[name] = tensor.detach().clone() + float(delta)
        else:
            updated[name] = tensor.detach().clone() if torch.is_tensor(tensor) else tensor
    return updated


def test_expert_group_helpers_identify_switch_experts():
    model = _build_model()
    groups = get_expert_param_groups(model)
    non_expert_names = get_non_expert_param_names(model)

    assert (0, 0) in groups
    assert (0, 1) in groups
    assert groups[(0, 0)]
    assert all(".switch_ffn.experts." not in name for name in non_expert_names)


def test_ha_vsmc_prefers_more_reliable_client_without_loss_signal():
    model = _build_model()
    global_state = _clone_state_dict(model)
    client_a = _shift_expert(global_state, expert_id=0, delta=0.05)
    client_b = _shift_expert(global_state, expert_id=0, delta=1.0)
    config = HAVSMCETFConfig(
        num_particles=4,
        lambda_max=1.0,
        lambda_warmup_rounds=0,
        alpha_usage=1.0,
        alpha_history_direction=0.0,
        alpha_consensus=0.0,
        alpha_drift=0.5,
        alpha_function_drift=0.0,
        collapse_penalty=0.0,
        use_function_drift=False,
        deterministic=True,
    )
    state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=config,
        device=torch.device("cpu"),
    )

    aggregated_state, updated_state, round_logs = aggregate_with_ha_vsmc_etf(
        global_model=model,
        client_updates=[
            (client_a, 10, {"router_usage": torch.tensor([[100.0, 1.0]])}),
            (client_b, 10, {"router_usage": torch.tensor([[1.0, 1.0]])}),
        ],
        state=state,
        config=config,
        probe_batch=None,
    )

    del updated_state
    expert_info = next(info for info in round_logs["expert_summaries"] if info["expert_id"] == 0)
    weight_name = "blocks.0.switch_ffn.experts.0.0.weight"
    dist_to_a = torch.norm(aggregated_state[weight_name] - client_a[weight_name]).item()
    dist_to_b = torch.norm(aggregated_state[weight_name] - client_b[weight_name]).item()

    assert expert_info["fallback_used"] is False
    assert expert_info["lambda_expert"] > 0.0
    assert dist_to_a < dist_to_b


def test_ha_vsmc_falls_back_to_fedavg_when_router_usage_is_zero():
    model = _build_model()
    global_state = _clone_state_dict(model)
    client_a = _shift_expert(global_state, expert_id=0, delta=0.2)
    client_b = _shift_expert(global_state, expert_id=0, delta=0.6)
    config = HAVSMCETFConfig(
        num_particles=4,
        lambda_max=1.0,
        lambda_warmup_rounds=0,
        use_function_drift=False,
        deterministic=True,
    )
    state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=config,
        device=torch.device("cpu"),
    )

    aggregated_state, _, round_logs = aggregate_with_ha_vsmc_etf(
        global_model=model,
        client_updates=[
            (client_a, 10, {"router_usage": torch.zeros(1, 2)}),
            (client_b, 10, {"router_usage": torch.zeros(1, 2)}),
        ],
        state=state,
        config=config,
        probe_batch=None,
    )

    expert_info = next(info for info in round_logs["expert_summaries"] if info["expert_id"] == 0)
    weight_name = "blocks.0.switch_ffn.experts.0.0.weight"
    expected = 0.5 * (client_a[weight_name] + client_b[weight_name])

    assert expert_info["fallback_used"] is True
    assert expert_info["fallback_reason"] == "all_router_usage_zero"
    assert torch.allclose(aggregated_state[weight_name], expected, atol=1e-6)


def test_ha_vsmc_no_anchor_uses_full_vsmc_update():
    model = _build_model()
    global_state = _clone_state_dict(model)
    client_a = _shift_expert(global_state, expert_id=0, delta=0.05)
    client_b = _shift_expert(global_state, expert_id=0, delta=1.0)
    base_kwargs = dict(
        num_particles=4,
        lambda_max=0.2,
        lambda_warmup_rounds=0,
        alpha_usage=1.0,
        alpha_history_direction=0.0,
        alpha_consensus=0.0,
        alpha_drift=0.5,
        alpha_function_drift=0.0,
        collapse_penalty=0.0,
        use_function_drift=False,
        deterministic=True,
    )
    anchor_config = HAVSMCETFConfig(**base_kwargs, anchor_mode="fedavg")
    no_anchor_config = HAVSMCETFConfig(**base_kwargs, anchor_mode="none")

    anchor_state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=anchor_config,
        device=torch.device("cpu"),
    )
    no_anchor_state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=no_anchor_config,
        device=torch.device("cpu"),
    )

    client_updates = [
        (client_a, 10, {"router_usage": torch.tensor([[100.0, 1.0]])}),
        (client_b, 10, {"router_usage": torch.tensor([[1.0, 1.0]])}),
    ]
    anchored_state, _, anchored_logs = aggregate_with_ha_vsmc_etf(
        global_model=model,
        client_updates=client_updates,
        state=anchor_state,
        config=anchor_config,
        probe_batch=None,
    )
    no_anchor_state_dict, _, no_anchor_logs = aggregate_with_ha_vsmc_etf(
        global_model=model,
        client_updates=client_updates,
        state=no_anchor_state,
        config=no_anchor_config,
        probe_batch=None,
    )

    weight_name = "blocks.0.switch_ffn.experts.0.0.weight"
    anchored_info = next(info for info in anchored_logs["expert_summaries"] if info["expert_id"] == 0)
    no_anchor_info = next(info for info in no_anchor_logs["expert_summaries"] if info["expert_id"] == 0)

    anchored_dist = torch.norm(anchored_state[weight_name] - client_a[weight_name]).item()
    no_anchor_dist = torch.norm(no_anchor_state_dict[weight_name] - client_a[weight_name]).item()

    assert anchored_info["anchor_mode"] == "fedavg"
    assert no_anchor_info["anchor_mode"] == "none"
    assert no_anchor_info["lambda_expert"] == 1.0
    assert no_anchor_info["final_residual_ratio"] > 0.999
    assert no_anchor_dist < anchored_dist
    assert not torch.allclose(no_anchor_state_dict[weight_name], anchored_state[weight_name])


def test_ha_vsmc_pure_vsmc_logs_and_forbids_fedavg_blend():
    model = _build_model()
    global_state = _clone_state_dict(model)
    client_a = _shift_expert(global_state, expert_id=0, delta=0.05)
    client_b = _shift_expert(global_state, expert_id=0, delta=1.0)
    config = HAVSMCETFConfig(
        num_particles=4,
        anchor_mode="none",
        pure_vsmc=True,
        lambda_max=0.2,
        lambda_warmup_rounds=0,
        alpha_usage=1.0,
        alpha_history_direction=0.0,
        alpha_consensus=0.0,
        alpha_drift=0.5,
        alpha_function_drift=0.0,
        collapse_penalty=0.0,
        use_function_drift=False,
        deterministic=True,
    )
    state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=config,
        device=torch.device("cpu"),
    )

    aggregated_state, _, round_logs = aggregate_with_ha_vsmc_etf(
        global_model=model,
        client_updates=[
            (client_a, 10, {"router_usage": torch.tensor([[100.0, 1.0]])}),
            (client_b, 10, {"router_usage": torch.tensor([[1.0, 1.0]])}),
        ],
        state=state,
        config=config,
        probe_batch=None,
    )

    weight_name = "blocks.0.switch_ffn.experts.0.0.weight"
    expert_info = next(info for info in round_logs["expert_summaries"] if info["expert_id"] == 0)
    assert round_logs["pure_vsmc"] is True
    assert round_logs["fedavg_blend_enabled"] is False
    assert round_logs["expert_output"] == "theta_vsmc"
    assert expert_info["pure_vsmc"] is True
    assert expert_info["fedavg_blend_enabled"] is False
    assert expert_info["expert_output"] == "theta_vsmc"
    assert expert_info["anchor_mode"] == "none"
    assert expert_info["lambda_expert"] == 1.0
    assert expert_info["final_residual_ratio"] > 0.999
    assert torch.isfinite(aggregated_state[weight_name]).all()


def test_ha_vsmc_pure_vsmc_rejects_fedavg_fallback():
    model = _build_model()
    global_state = _clone_state_dict(model)
    config = HAVSMCETFConfig(
        num_particles=4,
        anchor_mode="none",
        pure_vsmc=True,
        lambda_warmup_rounds=0,
        use_function_drift=False,
        deterministic=True,
    )
    state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=config,
        device=torch.device("cpu"),
    )

    try:
        aggregate_with_ha_vsmc_etf(
            global_model=model,
            client_updates=[
                (_shift_expert(global_state, expert_id=0, delta=0.2), 10, {"router_usage": torch.zeros(1, 2)}),
                (_shift_expert(global_state, expert_id=0, delta=0.6), 10, {"router_usage": torch.zeros(1, 2)}),
            ],
            state=state,
            config=config,
            probe_batch=None,
        )
    except RuntimeError as exc:
        assert "forbidden fallback" in str(exc)
    else:
        raise AssertionError("Pure VSMC mode should reject FedAvg fallback paths")



def test_ha_vsmc_non_pure_keeps_anchor_blend_when_postprocess_flags_are_enabled():
    model = _build_model()
    global_state = _clone_state_dict(model)
    client_a = _shift_expert(global_state, expert_id=0, delta=0.05)
    client_b = _shift_expert(global_state, expert_id=0, delta=1.0)
    config = HAVSMCETFConfig(
        num_particles=4,
        anchor_mode="fedavg",
        pure_vsmc=False,
        lambda_max=0.2,
        lambda_warmup_rounds=0,
        alpha_usage=1.0,
        alpha_history_direction=0.0,
        alpha_consensus=0.0,
        alpha_drift=0.5,
        alpha_function_drift=0.0,
        collapse_penalty=0.0,
        use_function_drift=False,
        prob_ema_rho=0.5,
        reliability_floor_eps=0.05,
        reliability_floor_source="usage",
        top_k_clients=1,
        entropy_target="moderate",
        adaptive_temperature=True,
        deterministic=True,
    )
    state = build_ha_vsmc_etf_state(
        global_model=model,
        global_state=global_state,
        num_clients=2,
        config=config,
        device=torch.device("cpu"),
    )

    aggregated_state, _, round_logs = aggregate_with_ha_vsmc_etf(
        global_model=model,
        client_updates=[
            (client_a, 10, {"router_usage": torch.tensor([[100.0, 1.0]])}),
            (client_b, 10, {"router_usage": torch.tensor([[1.0, 1.0]])}),
        ],
        state=state,
        config=config,
        probe_batch=None,
    )

    weight_name = "blocks.0.switch_ffn.experts.0.0.weight"
    expert_info = next(info for info in round_logs["expert_summaries"] if info["expert_id"] == 0)
    fedavg_weight = 0.5 * (client_a[weight_name] + client_b[weight_name])

    assert expert_info["pure_vsmc"] is False
    assert expert_info["fedavg_blend_enabled"] is True
    assert expert_info["expert_output"] == "blend(theta_fedavg,theta_vsmc)"
    assert expert_info["lambda_expert"] < 1.0
    assert torch.norm(aggregated_state[weight_name] - fedavg_weight).item() > 0.0
    assert torch.norm(aggregated_state[weight_name] - client_a[weight_name]).item() > 0.0
