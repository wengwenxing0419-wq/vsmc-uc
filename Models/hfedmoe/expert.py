from __future__ import annotations

import collections
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import nn


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> collections.OrderedDict:
    return collections.OrderedDict(
        (name, tensor.detach().clone().cpu() if torch.is_tensor(tensor) else tensor)
        for name, tensor in state_dict.items()
    )


class HFedMoEExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertPool(nn.Module):
    def __init__(self, num_experts: int, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            [HFedMoEExpert(input_dim, hidden_dim, num_classes, dropout=dropout) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor, expert_ids: Sequence[int]) -> torch.Tensor:
        selected = [self.experts[int(expert_id)](x) for expert_id in expert_ids]
        if not selected:
            raise ValueError("ExpertPool.forward requires at least one expert id")
        return torch.stack(selected, dim=1)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    def get_params(self):
        return [param for expert in self.experts for param in expert.parameters()]

    def state_dict_for(self, expert_ids: Iterable[int]) -> Dict[int, collections.OrderedDict]:
        return {
            int(expert_id): clone_state_dict(self.experts[int(expert_id)].state_dict())
            for expert_id in expert_ids
        }

    def load_expert_state(self, expert_id: int, state_dict: Mapping[str, torch.Tensor]):
        self.experts[int(expert_id)].load_state_dict(state_dict)

    def load_expert_states(self, state_dicts: Mapping[int, Mapping[str, torch.Tensor]]):
        for expert_id, state_dict in state_dicts.items():
            self.load_expert_state(int(expert_id), state_dict)

    def expert_param_count(self) -> int:
        if self.num_experts <= 0:
            return 0
        return int(sum(param.numel() for param in self.experts[0].parameters()))
