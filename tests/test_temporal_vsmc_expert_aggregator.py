import collections

import torch

from temporal_vsmc_expert_aggregator import (
    TemporalExpertMemory,
    TemporalVSMCAggregatorConfig,
    TemporalVSMCExpertAggregator,
    aggregate_expert_particles_temporal,
    compute_temporal_features,
)


def _state(value):
    return collections.OrderedDict(
        {
            "blocks.0.switch_ffn.experts.0.0.weight": torch.full((2, 2), float(value)),
            "blocks.0.switch_ffn.experts.0.0.bias": torch.full((2,), float(value)),
        }
    )


def test_temporal_alignment_increases_weight_for_consistent_update():
    global_state = _state(0.0)
    aligned_client = _state(1.0)
    misaligned_client = _state(-1.0)
    memory = TemporalExpertMemory(
        prev_global_state=global_state,
        momentum_delta_state=_state(1.0),
        last_round=2,
    )
    config = TemporalVSMCAggregatorConfig(
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        sample_beta=0.0,
        route_beta=0.0,
        align_beta=1.0,
        jump_beta=0.0,
        warmup_rounds=0,
    )

    _, info = aggregate_expert_particles_temporal(
        global_expert_state=global_state,
        client_expert_states=[aligned_client, misaligned_client],
        client_sample_counts=[10, 10],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        num_experts=2,
        config=config,
        round_idx=3,
        temporal_memory=memory,
    )

    assert info["temporal_enabled"] is True
    assert info["weights"][0] > info["weights"][1]


def test_temporal_jump_penalizes_large_deviation_from_history():
    global_state = _state(0.0)
    close_client = _state(1.0)
    jumpy_client = _state(3.0)
    memory = TemporalExpertMemory(
        prev_global_state=global_state,
        momentum_delta_state=_state(1.0),
        last_round=2,
    )
    config = TemporalVSMCAggregatorConfig(
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        sample_beta=0.0,
        route_beta=0.0,
        align_beta=0.0,
        jump_beta=1.0,
        warmup_rounds=0,
    )

    _, info = aggregate_expert_particles_temporal(
        global_expert_state=global_state,
        client_expert_states=[close_client, jumpy_client],
        client_sample_counts=[10, 10],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        num_experts=2,
        config=config,
        round_idx=3,
        temporal_memory=memory,
    )

    assert info["weights"][0] > info["weights"][1]


def test_temporal_warmup_disables_temporal_terms():
    global_state = _state(0.0)
    client_a = _state(1.0)
    client_b = _state(-1.0)
    memory = TemporalExpertMemory(
        prev_global_state=global_state,
        momentum_delta_state=_state(1.0),
        last_round=1,
    )
    config = TemporalVSMCAggregatorConfig(
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        sample_beta=0.0,
        route_beta=0.0,
        align_beta=1.0,
        jump_beta=1.0,
        warmup_rounds=5,
    )

    _, info = aggregate_expert_particles_temporal(
        global_expert_state=global_state,
        client_expert_states=[client_a, client_b],
        client_sample_counts=[10, 10],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        num_experts=2,
        config=config,
        round_idx=2,
        temporal_memory=memory,
    )

    assert info["temporal_enabled"] is False
    assert info["warmup_used"] is True
    assert abs(info["weights"][0] - info["weights"][1]) < 1e-6


def test_align_only_mode_zeroes_jump_term():
    features = compute_temporal_features(
        client_expert_state=_state(2.0),
        prev_global_state=_state(0.0),
        temporal_reference_state=_state(1.0),
        score_mode="align_only",
    )

    assert abs(features["alignment"] - 1.0) < 1e-6
    assert features["jump"] == 0.0


def test_history_size_uses_average_delta_reference():
    global_state = _state(0.0)
    client_a = _state(1.0)
    client_b = _state(-1.0)
    memory = TemporalExpertMemory(
        prev_global_state=global_state,
        momentum_delta_state=_state(1.0),
        delta_history_states=[_state(1.0), _state(-1.0), _state(1.0)],
        last_round=3,
    )
    config = TemporalVSMCAggregatorConfig(
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        sample_beta=0.0,
        route_beta=0.0,
        align_beta=1.0,
        jump_beta=0.0,
        warmup_rounds=0,
        history_size=3,
        score_mode="align_only",
    )

    _, info = aggregate_expert_particles_temporal(
        global_expert_state=global_state,
        client_expert_states=[client_a, client_b],
        client_sample_counts=[10, 10],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        num_experts=2,
        config=config,
        round_idx=4,
        temporal_memory=memory,
    )

    assert info["temporal_enabled"] is True
    assert info["weights"][0] > info["weights"][1]


def test_temporal_memory_updates_after_aggregation():
    global_state = collections.OrderedDict(_state(0.0))
    client_updates = [
        (_state(1.0), 1, {"route_counts": {0: 1}}),
        (_state(3.0), 3, {"route_counts": {0: 1}}),
    ]
    aggregator = TemporalVSMCExpertAggregator(
        TemporalVSMCAggregatorConfig(
            loss_beta=0.0,
            usage_beta=0.0,
            drift_beta=0.0,
            sample_beta=1.0,
            route_beta=0.0,
            align_beta=0.0,
            jump_beta=0.0,
            warmup_rounds=0,
        )
    )

    aggregated_state, round_infos, updated_memory = aggregator.aggregate_model_state(
        global_state=global_state,
        client_updates=client_updates,
        num_experts=1,
        round_idx=1,
    )

    scope = "blocks.0.switch_ffn.experts.0"
    assert scope in updated_memory
    memory = updated_memory[scope]
    assert memory.last_round == 1
    assert round_infos[0]["expert_scope"] == scope
    assert torch.allclose(
        memory.prev_global_state["blocks.0.switch_ffn.experts.0.0.weight"],
        aggregated_state["blocks.0.switch_ffn.experts.0.0.weight"],
    )
    assert memory.delta_history_states is not None
    assert len(memory.delta_history_states) == 1
