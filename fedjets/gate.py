from __future__ import annotations

import torch
from torch import nn


class FedJETSGate(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_experts),
        )

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(z), dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.scores(z)
