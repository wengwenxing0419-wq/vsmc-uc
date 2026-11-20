import collections

import torch

from temporal_latent_expert_aggregator import (
    LatentExpertMemory,
    TemporalLatentAggregatorConfig,
    TemporalLatentExpertAggregator,
    _build_projection_matrix,
    _project_delta,
    aggregate_expert_particles_temporal_latent,
    flatten_state_dict_to_vector,
)


def _state(value):
    return collections.OrderedDict(
        {
            "blocks.0.switch_ffn.experts.0.0.weight": torch.full((2, 2), float(value)),
            "blocks.0.switch_ffn.experts.0.0.bias": torch.full((2,), float(value)),
        }
    )


def test_temporal_latent_prefers_closer_observation_to_prior():
    global_state = _state(0.0)
    memory = LatentExpertMemory(
        projection_matrix=_build_projection_matrix(4, 6, 0, "blocks.0.switch_ffn.experts.0"),
        z_mean=torch.zeros(4, dtype=torch.float64),
        z_cov_diag=torch.full((4,), 1e-6, dtype=torch.float64),
        flat_keys=list(global_state.keys()),
        last_round=2,
    )
    config = TemporalLatentAggregatorConfig(
        latent_dim=4,
        transition_rho=1.0,
        process_noise=1e-6,
        obs_noise_base=1.0,
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        route_beta=0.0,
        warmup_rounds=0,
        projection_seed=0,
    )

    _, info, _ = aggregate_expert_particles_temporal_latent(
        global_expert_state=global_state,
        client_expert_states=[_state(0.1), _state(3.0)],
        client_sample_counts=[10, 10],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        expert_scope="blocks.0.switch_ffn.experts.0",
        num_experts=2,
        config=config,
        round_idx=3,
        latent_memory=memory,
    )

    assert info["latent_enabled"] is True
    assert info["weights"][0] > info["weights"][1]


def test_temporal_latent_lower_noise_gets_higher_weight_when_equally_close():
    global_state = _state(0.0)
    client_state = _state(1.0)
    config = TemporalLatentAggregatorConfig(
        latent_dim=4,
        transition_rho=0.0,
        process_noise=1e-6,
        obs_noise_base=1.0,
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        route_beta=0.0,
        loss_noise_beta=2.0,
        route_noise_beta=0.0,
        sample_noise_beta=0.0,
        warmup_rounds=0,
        projection_seed=0,
    )

    _, info, _ = aggregate_expert_particles_temporal_latent(
        global_expert_state=global_state,
        client_expert_states=[client_state, client_state],
        client_sample_counts=[10, 10],
        client_metrics=[
            {"expert_losses": {0: 0.1}, "route_counts": {0: 1}},
            {"expert_losses": {0: 2.0}, "route_counts": {0: 1}},
        ],
        expert_id=0,
        expert_scope="blocks.0.switch_ffn.experts.0",
        num_experts=2,
        config=config,
        round_idx=1,
        latent_memory=None,
    )

    assert info["weights"][0] > info["weights"][1]
    assert info["obs_noise_values"][0] < info["obs_noise_values"][1]


def test_temporal_latent_memory_updates_after_aggregation():
    global_state = collections.OrderedDict(_state(0.0))
    aggregator = TemporalLatentExpertAggregator(
        TemporalLatentAggregatorConfig(
            latent_dim=4,
            transition_rho=0.9,
            process_noise=0.1,
            warmup_rounds=0,
            projection_seed=7,
            loss_beta=0.0,
            usage_beta=0.0,
            drift_beta=0.0,
            route_beta=0.0,
        )
    )
    client_updates = [
        (_state(1.0), 1, {"route_counts": {0: 1}}),
        (_state(3.0), 3, {"route_counts": {0: 1}}),
    ]

    aggregated_state, round_infos, memory = aggregator.aggregate_model_state(
        global_state=global_state,
        client_updates=client_updates,
        num_experts=1,
        round_idx=1,
    )

    scope = "blocks.0.switch_ffn.experts.0"
    assert scope in memory
    assert memory[scope].last_round == 1
    assert memory[scope].z_mean.numel() == 4
    assert round_infos[0]["expert_scope"] == scope
    assert aggregated_state["blocks.0.switch_ffn.experts.0.0.weight"].shape == torch.Size([2, 2])


def test_projection_is_deterministic_for_scope_and_seed():
    state = _state(0.0)
    flat = flatten_state_dict_to_vector(state, list(state.keys()))
    proj_a = _build_projection_matrix(4, flat.numel(), 11, "blocks.0.switch_ffn.experts.0")
    proj_b = _build_projection_matrix(4, flat.numel(), 11, "blocks.0.switch_ffn.experts.0")
    delta = flatten_state_dict_to_vector(_state(1.0), list(state.keys())) - flat

    assert torch.allclose(proj_a, proj_b)
    assert torch.allclose(_project_delta(delta, proj_a), _project_delta(delta, proj_b))


def test_latent_fedavg_mixing_increases_ess_and_preserves_probability_simplex():
    global_state = _state(0.0)
    base_config = TemporalLatentAggregatorConfig(
        latent_dim=4,
        transition_rho=1.0,
        process_noise=1e-6,
        obs_noise_base=1.0,
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        route_beta=0.0,
        warmup_rounds=0,
        projection_seed=0,
    )
    memory = LatentExpertMemory(
        projection_matrix=_build_projection_matrix(4, 6, 0, "blocks.0.switch_ffn.experts.0"),
        z_mean=torch.zeros(4, dtype=torch.float64),
        z_cov_diag=torch.full((4,), 1e-6, dtype=torch.float64),
        flat_keys=list(global_state.keys()),
        last_round=2,
    )

    _, info_pure, _ = aggregate_expert_particles_temporal_latent(
        global_expert_state=global_state,
        client_expert_states=[_state(0.1), _state(3.0)],
        client_sample_counts=[1, 9],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        expert_scope="blocks.0.switch_ffn.experts.0",
        num_experts=2,
        config=base_config,
        round_idx=3,
        latent_memory=memory,
    )

    mixed_config = TemporalLatentAggregatorConfig(**{**base_config.__dict__, "fedavg_mix": 0.5})
    _, info_mixed, _ = aggregate_expert_particles_temporal_latent(
        global_expert_state=global_state,
        client_expert_states=[_state(0.1), _state(3.0)],
        client_sample_counts=[1, 9],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        expert_scope="blocks.0.switch_ffn.experts.0",
        num_experts=2,
        config=mixed_config,
        round_idx=3,
        latent_memory=memory,
    )

    assert info_mixed["ess"] >= info_pure["ess"]
    assert abs(sum(info_mixed["weights"]) - 1.0) < 1e-6


def test_latent_min_weight_applies_probability_floor():
    global_state = _state(0.0)
    config = TemporalLatentAggregatorConfig(
        latent_dim=4,
        transition_rho=1.0,
        process_noise=1e-6,
        obs_noise_base=1.0,
        loss_beta=0.0,
        usage_beta=0.0,
        drift_beta=0.0,
        route_beta=0.0,
        warmup_rounds=0,
        projection_seed=0,
        min_weight=0.2,
    )
    memory = LatentExpertMemory(
        projection_matrix=_build_projection_matrix(4, 6, 0, "blocks.0.switch_ffn.experts.0"),
        z_mean=torch.zeros(4, dtype=torch.float64),
        z_cov_diag=torch.full((4,), 1e-6, dtype=torch.float64),
        flat_keys=list(global_state.keys()),
        last_round=2,
    )

    _, info, _ = aggregate_expert_particles_temporal_latent(
        global_expert_state=global_state,
        client_expert_states=[_state(0.1), _state(3.0)],
        client_sample_counts=[1, 9],
        client_metrics=[{"route_counts": {0: 1}}, {"route_counts": {0: 1}}],
        expert_id=0,
        expert_scope="blocks.0.switch_ffn.experts.0",
        num_experts=2,
        config=config,
        round_idx=3,
        latent_memory=memory,
    )

    assert min(info["weights"]) >= 0.2 - 1e-6
    assert abs(sum(info["weights"]) - 1.0) < 1e-6
