import math

import torch

from aggregation.beta_precision_fedtra_aggregator import (
    BetaPrecisionFedTRAConfig,
    BetaPrecisionFedTRAAggregator,
)


def _expert_state(value: float):
    return {
        "blocks.0.switch_ffn.experts.0.0.weight": torch.full((2, 2), value),
    }


def _metrics(client_id: int, usage: float, route_count: float):
    return {
        "client_id": client_id,
        "train_expert_usage": {0: usage},
        "train_route_counts": {0: route_count},
        "train_num_tokens": route_count,
        "train_layer_route_counts": {0: {0: route_count}},
    }


def _aggregator(tmp_path):
    return BetaPrecisionFedTRAAggregator(
        BetaPrecisionFedTRAConfig(
            num_particles=16,
            kappa_u=10.0,
            kappa_c=10.0,
            omd_lr=0.05,
            gamma_min=1e-3,
            seed=7,
            log_dir=str(tmp_path),
        )
    )


def test_initial_precision_shares_recover_base_concentrations(tmp_path):
    aggregator = _aggregator(tmp_path)
    assert torch.allclose(
        aggregator.gamma,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    assert math.isclose(
        aggregator.gamma[0].item() * aggregator.total_concentration, 10.0
    )
    assert math.isclose(
        aggregator.gamma[1].item() * aggregator.total_concentration, 10.0
    )


def test_beta_log_probability_matches_torch_distribution(tmp_path):
    aggregator = _aggregator(tmp_path)
    x = 0.37
    mean = torch.tensor([0.25, 0.6], dtype=torch.float64)
    concentration = 8.0
    actual = aggregator._beta_log_prob(x, concentration, mean)
    expected = torch.distributions.Beta(
        concentration * mean,
        concentration * (1.0 - mean),
    ).log_prob(torch.full_like(mean, x))
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_omd_preserves_simplex_and_lower_bound(tmp_path):
    aggregator = _aggregator(tmp_path)
    updated = aggregator._update_gamma(torch.tensor([2.0, -1.0], dtype=torch.float64))
    assert torch.isfinite(updated).all()
    assert math.isclose(updated.sum().item(), 1.0, abs_tol=1e-12)
    assert updated.min().item() >= aggregator.config.gamma_min


def test_aggregate_model_state_updates_weights_and_logs(tmp_path):
    aggregator = _aggregator(tmp_path)
    global_state = _expert_state(0.0)
    client_updates = [
        (
            _expert_state(1.0),
            10,
            _metrics(1, usage=0.8, route_count=8.0),
        ),
        (
            _expert_state(-1.0),
            10,
            _metrics(2, usage=0.2, route_count=2.0),
        ),
    ]
    aggregated, info = aggregator.aggregate_model_state(
        global_state=global_state,
        client_updates=client_updates,
        num_experts=1,
    )
    assert torch.isfinite(aggregated["blocks.0.switch_ffn.experts.0.0.weight"]).all()
    assert info["num_pairs"] == 2
    assert math.isclose(aggregator.gamma.sum().item(), 1.0, abs_tol=1e-12)
    assert aggregator.gamma.min().item() >= aggregator.config.gamma_min
    assert math.isfinite(info["predictive_nll"])
    assert math.isfinite(info["surrogate_nll"])
    assert (tmp_path / "beta_precision_fedtra_round_metrics.csv").exists()
    assert (tmp_path / "beta_precision_fedtra_pair_metrics.csv").exists()


def test_single_client_consistency_is_neutral(tmp_path):
    aggregator = _aggregator(tmp_path)
    scores = aggregator._leave_one_out_consistency(
        [_expert_state(1.0)],
        _expert_state(0.0),
    )
    assert scores == [0.5]
