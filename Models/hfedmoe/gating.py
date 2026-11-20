from __future__ import annotations

import torch
from torch import nn


class HFedMoEGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(z), dim=-1)
