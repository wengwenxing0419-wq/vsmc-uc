from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from pfedmoe.modeling import SwitchTransformerFeatureExtractor

from .expert import ExpertPool
from .gating import HFedMoEGate


def build_shared_backbone(config) -> SwitchTransformerFeatureExtractor:
    return SwitchTransformerFeatureExtractor(
        image_size=config.image_size,
        patch_size=config.patch_size,
        backbone_name=config.shared_backbone_name,
        hidden_dim=config.hidden_dim,
        num_layers=config.shared_switch_layers,
        num_heads=config.switch_heads,
        ffn_dim=config.switch_ffn_dim,
        num_experts=config.switch_num_experts,
        dropout=config.dropout,
    )


def select_topk_experts(gate_weights: torch.Tensor, k: int) -> Tuple[list[int], torch.Tensor]:
    if gate_weights.dim() == 1:
        scores = gate_weights
    elif gate_weights.dim() == 2:
        scores = gate_weights.mean(dim=0)
    else:
        raise ValueError("gate_weights must be a [num_experts] or [batch, num_experts] tensor")
    k = min(max(int(k), 1), int(scores.numel()))
    topk = torch.topk(scores, k=k, dim=-1)
    return [int(index) for index in topk.indices.detach().cpu().tolist()], topk.values.detach().cpu()


class HFedMoEModel(nn.Module):
    def __init__(
        self,
        shared_backbone: nn.Module,
        expert_pool: ExpertPool,
        gating_net: HFedMoEGate,
        top_k: int = 2,
        combine_method: str = "gated",
    ):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.expert_pool = expert_pool
        self.gating = gating_net
        self.top_k = max(int(top_k), 1)
        self.combine_method = str(combine_method).lower()

    def _combine_logits(self, expert_logits: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        if self.combine_method == "topk_average":
            return expert_logits.mean(dim=1)
        normalized_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (expert_logits * normalized_weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor, active_expert_ids: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        features = self.shared_backbone(x)
        gate_weights = self.gating(features)
        num_classes = None

        if active_expert_ids is not None:
            active_ids = [int(expert_id) for expert_id in active_expert_ids]
            if not active_ids:
                raise ValueError("active_expert_ids must contain at least one expert id")
            expert_logits = self.expert_pool(features, active_ids)
            subset_gate_weights = gate_weights[:, active_ids]
            if subset_gate_weights.size(1) > self.top_k:
                local_topk_values, local_topk_indices = torch.topk(
                    subset_gate_weights,
                    k=min(self.top_k, subset_gate_weights.size(1)),
                    dim=-1,
                )
                gather_index = local_topk_indices.unsqueeze(-1).expand(-1, -1, expert_logits.size(-1))
                selected_expert_logits = torch.gather(expert_logits, 1, gather_index)
                active_tensor = torch.tensor(active_ids, device=x.device, dtype=torch.long)
                selected_expert_ids = active_tensor[local_topk_indices]
                selected_gate_weights = local_topk_values
            else:
                selected_expert_logits = expert_logits
                selected_gate_weights = subset_gate_weights
                selected_expert_ids = torch.tensor(active_ids, device=x.device, dtype=torch.long).view(1, -1).expand(x.size(0), -1)
        else:
            all_expert_logits = self.expert_pool.forward_all(features)
            topk_values, topk_indices = torch.topk(
                gate_weights,
                k=min(self.top_k, gate_weights.size(-1)),
                dim=-1,
            )
            gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, all_expert_logits.size(-1))
            selected_expert_logits = torch.gather(all_expert_logits, 1, gather_index)
            selected_gate_weights = topk_values
            selected_expert_ids = topk_indices

        logits = self._combine_logits(selected_expert_logits, selected_gate_weights)
        num_classes = selected_expert_logits.size(-1)
        return {
            "logits": logits,
            "shared_features": features,
            "gate_weights": gate_weights,
            "selected_gate_weights": selected_gate_weights / selected_gate_weights.sum(dim=1, keepdim=True).clamp_min(1e-8),
            "selected_expert_ids": selected_expert_ids,
            "selected_expert_logits": selected_expert_logits.view(selected_expert_logits.size(0), -1, num_classes),
        }
