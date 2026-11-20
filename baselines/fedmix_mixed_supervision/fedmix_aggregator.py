from __future__ import annotations

import collections
from typing import Any, Dict, MutableMapping, Sequence, Tuple

import torch

from .fedmix_config import FedMixConfig
from .fedmix_utils import (
    TensorState,
    clone_state_dict,
    compute_client_weights,
    merge_state_dicts,
    split_expert_and_non_expert_state,
    weighted_average_state_dicts,
)


class FedMixAggregator:
    def __init__(self, config: FedMixConfig, logger=None):
        self.config = config
        self.logger = logger

    @staticmethod
    def _clone_state_to_cpu(state_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        cloned = collections.OrderedDict()
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().cpu().clone()
            else:
                cloned[key] = value
        return cloned

    def _compute_adaptive_expert_weights(
        self,
        client_losses: Sequence[float],
        client_num_samples: Sequence[int],
    ) -> Sequence[float]:
        sample_weights = compute_client_weights(client_num_samples, mode="sample_count")
        stabilized_losses = [max(float(loss), 1e-8) for loss in client_losses]
        loss_powers = [loss ** float(self.config.beta) for loss in stabilized_losses]
        loss_total = sum(loss_powers)
        if loss_total <= 0.0:
            return compute_client_weights(client_num_samples, mode=self.config.expert_weight_mode)
        adaptive_terms = [value / loss_total for value in loss_powers]
        mixed_weights = [
            float(sample_weight) + float(self.config.lam) * float(adaptive_term)
            for sample_weight, adaptive_term in zip(sample_weights, adaptive_terms)
        ]
        mixed_total = sum(mixed_weights)
        if mixed_total <= 0.0:
            return compute_client_weights(client_num_samples, mode=self.config.expert_weight_mode)
        return [weight / mixed_total for weight in mixed_weights]

    def aggregate(
        self,
        global_state: TensorState,
        client_states: Sequence[TensorState],
        client_losses: Sequence[float],
        client_num_samples: Sequence[int],
    ) -> Tuple[TensorState, Dict[str, Any]]:
        if not client_states:
            raise ValueError("Cannot aggregate an empty list of client states")
        if len(client_states) != len(client_num_samples) or len(client_states) != len(client_losses):
            raise ValueError("client_states, client_losses, and client_num_samples must have the same length")

        global_state_cpu = self._clone_state_to_cpu(global_state)
        client_states_cpu = [self._clone_state_to_cpu(state_dict) for state_dict in client_states]
        _, global_non_expert_state = split_expert_and_non_expert_state(global_state_cpu)
        global_expert_state, _ = split_expert_and_non_expert_state(global_state_cpu)

        client_expert_states = []
        client_non_expert_states = []
        for state_dict in client_states_cpu:
            expert_state, non_expert_state = split_expert_and_non_expert_state(state_dict)
            client_expert_states.append(expert_state)
            client_non_expert_states.append(non_expert_state)

        shared_weights = compute_client_weights(client_num_samples, mode=self.config.shared_weight_mode)
        if self.config.use_adaptive_aggregation:
            expert_weights = self._compute_adaptive_expert_weights(client_losses, client_num_samples)
        else:
            expert_weights = compute_client_weights(client_num_samples, mode=self.config.expert_weight_mode)

        aggregated_non_expert_state = weighted_average_state_dicts(
            client_non_expert_states,
            shared_weights,
            fallback_state=global_non_expert_state,
        )
        aggregated_expert_state = weighted_average_state_dicts(
            client_expert_states,
            expert_weights,
            fallback_state=global_expert_state,
        )
        aggregated_state = merge_state_dicts(aggregated_non_expert_state, aggregated_expert_state)
        aggregated_state = clone_state_dict(aggregated_state)

        metrics = {
            "fedmix_beta": float(self.config.beta),
            "fedmix_lambda": float(self.config.lam),
            "fedmix_pseudo_thresh": float(self.config.pseudo_thresh),
            "fedmix_use_dynamic_sample_selection": bool(self.config.use_dynamic_sample_selection),
            "fedmix_use_adaptive_aggregation": bool(self.config.use_adaptive_aggregation),
            "fedmix_shared_weight_mode": self.config.shared_weight_mode,
            "fedmix_expert_weight_mode": self.config.expert_weight_mode,
            "fedmix_avg_client_loss": float(sum(float(loss) for loss in client_losses) / max(len(client_losses), 1)),
            "fedmix_max_expert_weight": float(max(expert_weights)) if expert_weights else 0.0,
            "fedmix_min_expert_weight": float(min(expert_weights)) if expert_weights else 0.0,
        }
        return aggregated_state, {
            "metrics": metrics,
            "shared_weights": list(shared_weights),
            "expert_weights": list(expert_weights),
        }

    def aggregate_model_state(
        self,
        global_state: TensorState,
        client_updates,
        device: torch.device,
    ) -> Tuple[TensorState, Dict[str, Any]]:
        client_states = [state_dict for state_dict, _, _ in client_updates]
        client_num_samples = [int(num_samples) for _, num_samples, _ in client_updates]
        client_losses = [float(metrics.get("train_loss", 0.0)) for _, _, metrics in client_updates]
        aggregated_state, round_data = self.aggregate(
            global_state=global_state,
            client_states=client_states,
            client_losses=client_losses,
            client_num_samples=client_num_samples,
        )
        aggregated_state = collections.OrderedDict(
            (key, value.to(device) if torch.is_tensor(value) else value)
            for key, value in aggregated_state.items()
        )
        return aggregated_state, round_data
