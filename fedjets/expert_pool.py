from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch

from .utils import average_state_dicts, clone_state_dict


class ExpertPool:
    def __init__(self, num_experts: int, model_factory, initial_state: Mapping[str, torch.Tensor]):
        self.num_experts = max(int(num_experts), 1)
        self.model_factory = model_factory
        base_state = clone_state_dict(initial_state)
        self._expert_states: List[Dict[str, torch.Tensor]] = [clone_state_dict(base_state) for _ in range(self.num_experts)]

    def get_experts(self, indices: Iterable[int], device: torch.device | None = None):
        models = {}
        for index in indices:
            model = self.model_factory()
            model.load_state_dict(self._expert_states[int(index)])
            if device is not None:
                model = model.to(device)
            models[int(index)] = model
        return models

    def state_dict_for(self, indices: Iterable[int]) -> Dict[int, Dict[str, torch.Tensor]]:
        return {
            int(index): clone_state_dict(self._expert_states[int(index)])
            for index in indices
        }

    def load_expert_state(self, index: int, state: Mapping[str, torch.Tensor]):
        self._expert_states[int(index)] = clone_state_dict(state)

    def aggregate_expert_updates(self, updates, weights):
        per_expert_updates: Dict[int, List[Mapping[str, torch.Tensor]]] = {}
        per_expert_weights: Dict[int, List[float]] = {}

        for client_update, client_weight in zip(updates, weights):
            for expert_id, state in client_update.items():
                expert_id = int(expert_id)
                per_expert_updates.setdefault(expert_id, []).append(state)
                per_expert_weights.setdefault(expert_id, []).append(float(client_weight))

        updated_client_counts = {}
        for expert_id, state_dicts in per_expert_updates.items():
            self._expert_states[expert_id] = average_state_dicts(
                state_dicts,
                per_expert_weights[expert_id],
            )
            updated_client_counts[expert_id] = len(state_dicts)
        return updated_client_counts

    def all_states(self) -> List[Dict[str, torch.Tensor]]:
        return [clone_state_dict(state) for state in self._expert_states]
