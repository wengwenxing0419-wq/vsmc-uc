import collections

import torch

from aggregation.vsmc_expert_aggregator import (
    VSMCExpertAggregationConfig,
    aggregate_one_expert_vsmc,
    aggregate_expert_particles,
    compute_log_f,
    compute_log_r,
    gaussian_log_prob_diag,
    safe_logsumexp_normalize,
    stratified_resampling,
    weighted_average_state_dicts,
)


def _state(value):
    return collections.OrderedDict(
        {
            "blocks.0.switch_ffn.experts.0.0.weight": torch.full((2, 2), float(value)),
            "blocks.0.switch_ffn.experts.0.0.bias": torch.full((2,), float(value)),
        }
    )


def test_safe_logsumexp_normalize_uniform():
    weights = safe_logsumexp_normalize(torch.tensor([0.0, 0.0, 0.0]), temperature=1.0, eps=1e-12)
    expected = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    assert torch.allclose(weights, expected, atol=1e-6)


def test_gaussian_log_prob_diag_prefers_matching_mean():
    x = torch.tensor([0.0, 0.0])
    mean = torch.tensor([0.0, 0.0])
    far = torch.tensor([10.0, 10.0])
    assert gaussian_log_prob_diag(x, mean, sigma=1.0).item() > gaussian_log_prob_diag(far, mean, sigma=1.0).item()


def test_log_weight_formula_identity():
    log_f = torch.tensor([-1.0], dtype=torch.float64)
    log_g = torch.tensor([-2.0], dtype=torch.float64)
    log_r = torch.tensor([-3.0], dtype=torch.float64)
    log_w = log_f + log_g - log_r
    assert torch.allclose(log_w, torch.tensor([0.0], dtype=torch.float64))


def test_low_loss_client_gets_higher_weight():
    config = VSMCExpertAggregationConfig(
        sigma_f=1.0,
        sigma_r=1.0,
        loss_beta=1.0,
        route_beta=0.0,
        usage_beta=0.0,
        sample_beta=0.0,
    )
    _, info = aggregate_one_expert_vsmc(
        previous_global_expert_state=_state(0.0),
        current_client_expert_states=[_state(1.0), _state(1.0)],
        client_metrics=[
            {"expert_losses": {0: 0.1}, "expert_usage": {0: 0.5}, "route_counts": {0: 10}},
            {"expert_losses": {0: 2.0}, "expert_usage": {0: 0.5}, "route_counts": {0: 10}},
        ],
        sample_counts=[10, 10],
        expert_id=0,
        num_experts=2,
        config=config,
    )
    assert info["weights"][0] > info["weights"][1]


def test_log_f_prefers_state_closer_to_ancestor():
    config = VSMCExpertAggregationConfig(loss_beta=0.0, route_beta=0.0, usage_beta=0.0, sample_beta=0.0)
    ancestor = _state(0.0)
    near = _state(0.1)
    far = _state(5.0)
    assert compute_log_f(near, ancestor, config).item() > compute_log_f(far, ancestor, config).item()


def test_log_r_changes_with_proposal_distance():
    config = VSMCExpertAggregationConfig(loss_beta=0.0, route_beta=0.0, usage_beta=0.0, sample_beta=0.0)
    proposal = _state(3.0)
    a = _state(0.0)
    b = _state(2.9)
    assert compute_log_r(b, proposal, config).item() > compute_log_r(a, proposal, config).item()


def test_weighted_average_state_dicts_matches_expected_value():
    aggregated = weighted_average_state_dicts([_state(1.0), _state(3.0)], torch.tensor([0.25, 0.75]))
    assert torch.allclose(aggregated["blocks.0.switch_ffn.experts.0.0.weight"], torch.full((2, 2), 2.5), atol=1e-6)


def test_non_floating_tensors_do_not_break_aggregation():
    client_a = _state(1.0)
    client_b = _state(3.0)
    client_a["blocks.0.switch_ffn.experts.0.num_batches_tracked"] = torch.tensor(4, dtype=torch.long)
    client_b["blocks.0.switch_ffn.experts.0.num_batches_tracked"] = torch.tensor(9, dtype=torch.long)

    aggregated = weighted_average_state_dicts(
        [client_a, client_b],
        torch.tensor([0.5, 0.5]),
    )
    assert aggregated["blocks.0.switch_ffn.experts.0.num_batches_tracked"].item() == 4


def test_nan_log_weight_falls_back_to_sample_count_fedavg():
    config = VSMCExpertAggregationConfig(
        fallback_to_fedavg_on_nan=True,
        loss_beta=1.0,
        route_beta=0.0,
        usage_beta=0.0,
        sample_beta=0.0,
    )
    aggregated, info = aggregate_expert_particles(
        global_expert_state=_state(0.0),
        client_expert_states=[_state(1.0), _state(5.0)],
        client_sample_counts=[1, 3],
        client_metrics=[
            {"expert_losses": {0: float("nan")}, "route_counts": {0: 1}},
            {"expert_losses": {0: 1.0}, "route_counts": {0: 1}},
        ],
        expert_id=0,
        num_experts=2,
        config=config,
    )
    assert info["fallback_used"] is True
    assert torch.allclose(aggregated["blocks.0.switch_ffn.experts.0.0.weight"], torch.full((2, 2), 4.0), atol=1e-6)


def test_stratified_resampling_prefers_high_weight_index():
    weights = torch.tensor([0.9, 0.1], dtype=torch.float64)
    indices = stratified_resampling(weights, generator=torch.Generator().manual_seed(0))
    assert indices.shape == torch.Size([2])
    assert (indices == 0).sum().item() >= 1
