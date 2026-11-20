import collections

import torch

from fedmoe_da.aggregator import FedMoEDAExpertAggregator
from fedmoe_da.config import FedMoEDAConfig


EXPERT_NAME = "blocks.0.switch_ffn.experts.0.weight"
ROUTER_NAME = "blocks.0.switch_ffn.router.weight"
ROUTER_BIAS_NAME = "blocks.0.switch_ffn.router.bias"
SHARED_NAME = "shared.weight"


def make_state(shared_value, expert_value, router_vector):
    return collections.OrderedDict(
        {
            SHARED_NAME: torch.tensor([shared_value], dtype=torch.float32),
            ROUTER_NAME: torch.tensor([router_vector], dtype=torch.float32),
            ROUTER_BIAS_NAME: torch.tensor([0.0], dtype=torch.float32),
            EXPERT_NAME: torch.tensor([expert_value], dtype=torch.float32),
        }
    )


def test_fedmoe_da_shared_path_stays_on_shared_args_weighting():
    config = FedMoEDAConfig(
        top_p=2,
        temperature=1.0,
        personalized_mix=0.0,
        similarity_metric="router_row_cosine",
        min_similarity=0.0,
        shared_weight_mode="sample_count",
        expert_weight_mode="sample_count",
        num_clients=2,
        server_epochs=1,
        client_epochs=1,
        batch_size=1,
        learning_rate=0.01,
        device="cpu",
        seed=1,
        data_name="cifar10",
        data_save_path="",
        save_result="",
        model_save_path="",
        backbone_name="resnet18",
    )
    aggregator = FedMoEDAExpertAggregator(config)
    global_state = make_state(0.0, 0.0, [1.0, 0.0])
    client_updates = [
        (make_state(1.0, 1.0, [1.0, 0.0]), 1, {"client_id": 1}),
        (make_state(3.0, 9.0, [0.0, 1.0]), 3, {"client_id": 2}),
    ]

    aggregated_state, _ = aggregator.aggregate_model_state(
        global_state=global_state,
        client_updates=client_updates,
        device=torch.device("cpu"),
        round_idx=1,
    )

    assert torch.isclose(aggregated_state[SHARED_NAME], torch.tensor([2.5])).all()
    assert torch.isclose(aggregated_state[EXPERT_NAME], torch.tensor([7.0])).all()


def test_fedmoe_da_expert_path_moves_toward_router_similar_neighbors():
    config = FedMoEDAConfig(
        top_p=2,
        temperature=1000.0,
        personalized_mix=1.0,
        similarity_metric="router_row_cosine",
        min_similarity=0.0,
        shared_weight_mode="uniform",
        expert_weight_mode="uniform",
        num_clients=3,
        server_epochs=1,
        client_epochs=1,
        batch_size=1,
        learning_rate=0.01,
        device="cpu",
        seed=1,
        data_name="cifar10",
        data_save_path="",
        save_result="",
        model_save_path="",
        backbone_name="resnet18",
    )
    aggregator = FedMoEDAExpertAggregator(config)
    global_state = make_state(0.0, 0.0, [1.0, 0.0])
    client_updates = [
        (make_state(0.0, 0.0, [1.0, 0.0]), 1, {"client_id": 1}),
        (make_state(0.0, 10.0, [0.2, 1.0]), 1, {"client_id": 2}),
        (make_state(0.0, 30.0, [0.0, 1.0]), 1, {"client_id": 3}),
    ]

    aggregated_state, round_data = aggregator.aggregate_model_state(
        global_state=global_state,
        client_updates=client_updates,
        device=torch.device("cpu"),
        round_idx=1,
    )

    assert float(aggregated_state[EXPERT_NAME].item()) > (40.0 / 3.0)
    assert round_data["num_expert_groups"] == 1
    assert round_data["mean_selected_similarity"] >= round_data["mean_pairwise_similarity"]
