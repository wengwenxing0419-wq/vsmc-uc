from args import parse
from fedmoe_da.config import resolve_fedmoe_da_config


def test_fedmoe_da_reuses_shared_args_from_parser():
    args = parse.parse_args(
        [
            "--aggregation_method",
            "fedmoe_da",
            "--num_clients",
            "7",
            "--server_epochs",
            "11",
            "--client_epochs",
            "2",
            "--batch_size",
            "64",
            "--learning_rate",
            "0.01",
            "--fedavg_weight_mode",
            "sample_count",
            "--expert_weight_mode",
            "uniform",
            "--fedmoe_da_top_p",
            "5",
            "--seed",
            "13",
        ]
    )

    config = resolve_fedmoe_da_config(args)

    assert args.aggregation_method == "fedmoe_da"
    assert config.num_clients == 7
    assert config.server_epochs == 11
    assert config.client_epochs == 2
    assert config.batch_size == 64
    assert config.learning_rate == 0.01
    assert config.shared_weight_mode == "sample_count"
    assert config.expert_weight_mode == "uniform"
    assert config.top_p == 5
    assert config.seed == 13


def test_fedmoe_da_config_clamps_method_specific_values():
    args = parse.parse_args(
        [
            "--aggregation_method",
            "fedmoe_da",
            "--num_clients",
            "4",
            "--fedmoe_da_top_p",
            "99",
            "--fedmoe_da_personalized_mix",
            "1.5",
        ]
    )

    config = resolve_fedmoe_da_config(args)

    assert config.top_p == 4
    assert config.personalized_mix == 1.0
