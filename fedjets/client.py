from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from utils import record_result

from .gate import FedJETSGate
from .utils import clone_state_dict


class FedJETSClientTrainer:
    def __init__(
        self,
        config,
        client_id: int,
        round_idx: int,
        role: str,
        model_factory,
        common_expert,
        gate_state: Dict[str, torch.Tensor],
        expert_states: Dict[int, Dict[str, torch.Tensor]],
        selected_expert_ids,
        assigned_expert_id: Optional[int] = None,
    ):
        self.config = config
        self.client_id = int(client_id)
        self.round_idx = int(round_idx)
        self.role = str(role)
        self.model_factory = model_factory
        self.common_expert = common_expert
        self.selected_expert_ids = [int(expert_id) for expert_id in selected_expert_ids]
        self.assigned_expert_id = None if assigned_expert_id is None else int(assigned_expert_id)
        self.device = torch.device(self.config.device)

        self.train_dataset = self._load_client_split("train")
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError(f"Client {self.client_id} has no training samples for FedJETs")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.gate = FedJETSGate(
            embedding_dim=self.config.embedding_dim,
            num_experts=self.config.num_experts,
            hidden_dim=self.config.gate_hidden_dim,
        ).to(self.device)
        self.gate.load_state_dict(gate_state)

        self.experts = {}
        for expert_id, state in expert_states.items():
            model = self.model_factory().to(self.device)
            model.load_state_dict(state)
            self.experts[int(expert_id)] = model

        if self.role == "anchor":
            if self.assigned_expert_id is None:
                raise ValueError("Anchor clients require an assigned expert id")
            if self.assigned_expert_id not in self.experts:
                raise ValueError("Assigned expert id is missing from the local expert payload")
            self.selected_expert_ids = [self.assigned_expert_id]
        elif not self.selected_expert_ids:
            raise ValueError("Normal clients require at least one selected expert")

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.gate_optimizer = self._build_optimizer(self.gate.parameters(), self.config.gate_lr)
        expert_params = []
        for expert_id in self.selected_expert_ids:
            expert_params.extend(self.experts[expert_id].parameters())
        self.expert_optimizer = self._build_optimizer(expert_params, self.config.expert_lr)

    def _load_client_split(self, split: str):
        path = os.path.join(self.config.data_save_path, f"{self.client_id}_{split}.pth")
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=False)

    def _build_optimizer(self, params, learning_rate: float):
        optimizer_name = str(self.config.client_optimizer).lower()
        params = list(params)
        if optimizer_name == "adam":
            return optim.Adam(params, lr=learning_rate, weight_decay=self.config.client_weight_decay)
        if optimizer_name == "adamw":
            return optim.AdamW(params, lr=learning_rate, weight_decay=self.config.client_weight_decay)
        if optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=learning_rate,
                momentum=self.config.client_momentum,
                nesterov=self.config.client_nesterov,
                weight_decay=self.config.client_weight_decay,
            )
        raise ValueError(f"Unsupported client optimizer for FedJETs: {self.config.client_optimizer}")

    def _random_crop_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        pad = int(self.config.aug_random_crop_pad)
        if pad <= 0:
            return inputs
        batch_size, _, height, width = inputs.shape
        padded = F.pad(inputs, (pad, pad, pad, pad), mode="reflect")
        max_offset_h = padded.shape[-2] - height
        max_offset_w = padded.shape[-1] - width
        top_offsets = torch.randint(0, max_offset_h + 1, (batch_size,), device=inputs.device)
        left_offsets = torch.randint(0, max_offset_w + 1, (batch_size,), device=inputs.device)
        crops = []
        for idx in range(batch_size):
            top = int(top_offsets[idx].item())
            left = int(left_offsets[idx].item())
            crops.append(padded[idx:idx + 1, :, top:top + height, left:left + width])
        return torch.cat(crops, dim=0)

    def _horizontal_flip_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        flip_prob = float(self.config.aug_horizontal_flip)
        if flip_prob <= 0.0:
            return inputs
        flip_mask = torch.rand(inputs.size(0), device=inputs.device) < flip_prob
        if flip_mask.any():
            inputs = inputs.clone()
            inputs[flip_mask] = torch.flip(inputs[flip_mask], dims=[3])
        return inputs

    def _random_erasing_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        erase_prob = float(self.config.aug_random_erasing)
        if erase_prob <= 0.0:
            return inputs
        inputs = inputs.clone()
        batch_size, _, height, width = inputs.shape
        for idx in range(batch_size):
            if torch.rand(1, device=inputs.device).item() >= erase_prob:
                continue
            erase_h = max(1, int(height * torch.empty(1).uniform_(0.1, 0.25).item()))
            erase_w = max(1, int(width * torch.empty(1).uniform_(0.1, 0.25).item()))
            top = int(torch.randint(0, max(height - erase_h + 1, 1), (1,), device=inputs.device).item())
            left = int(torch.randint(0, max(width - erase_w + 1, 1), (1,), device=inputs.device).item())
            inputs[idx, :, top:top + erase_h, left:left + erase_w] = 0.0
        return inputs

    def _apply_batch_augmentation(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.config.use_client_aug:
            return inputs
        inputs = self._random_crop_batch(inputs)
        inputs = self._horizontal_flip_batch(inputs)
        inputs = self._random_erasing_batch(inputs)
        return inputs

    def _sample_mixup(self, inputs: torch.Tensor, labels: torch.Tensor):
        alpha = float(self.config.mixup_alpha)
        if alpha <= 0.0 or inputs.size(0) < 2:
            return inputs, labels, labels, 1.0
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = float(beta_dist.sample().item())
        perm = torch.randperm(inputs.size(0), device=inputs.device)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[perm]
        return mixed_inputs, labels, labels[perm], lam

    def _supervised_loss(self, logits, labels, labels_a, labels_b, mixup_lambda):
        if mixup_lambda < 1.0:
            return (
                mixup_lambda
                * F.cross_entropy(logits, labels_a, reduction="mean", label_smoothing=self.config.label_smoothing)
                + (1.0 - mixup_lambda)
                * F.cross_entropy(logits, labels_b, reduction="mean", label_smoothing=self.config.label_smoothing)
            )
        return self.criterion(logits, labels)

    def _forward_selected_experts(self, mixed_inputs: torch.Tensor):
        expert_logits = []
        for expert_id in self.selected_expert_ids:
            logits, _ = self.experts[expert_id](mixed_inputs)
            expert_logits.append(logits)
        return expert_logits

    def _normal_client_step(self, mixed_inputs, labels, labels_a, labels_b, mixup_lambda):
        features = self.common_expert.extract_features(mixed_inputs)
        gate_scores = self.gate(features)
        local_scores = gate_scores[:, self.selected_expert_ids]
        local_scores = local_scores / local_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expert_logits = self._forward_selected_experts(mixed_inputs)
        stacked_logits = torch.stack(expert_logits, dim=1)
        mixed_logits = torch.sum(local_scores.unsqueeze(-1) * stacked_logits, dim=1)
        loss = self._supervised_loss(mixed_logits, labels, labels_a, labels_b, mixup_lambda)
        return loss, mixed_logits

    def _anchor_client_step(self, mixed_inputs, labels, labels_a, labels_b, mixup_lambda):
        features = self.common_expert.extract_features(mixed_inputs)
        gate_logits = self.gate.logits(features)
        gate_targets = torch.full(
            (mixed_inputs.size(0),),
            fill_value=int(self.assigned_expert_id),
            device=self.device,
            dtype=torch.long,
        )
        expert_logits, _ = self.experts[self.assigned_expert_id](mixed_inputs)
        expert_loss = self._supervised_loss(expert_logits, labels, labels_a, labels_b, mixup_lambda)
        gate_loss = F.cross_entropy(gate_logits, gate_targets)
        return expert_loss + gate_loss, expert_logits

    def train(self):
        sample_count = len(self.train_loader.dataset)
        running_loss = 0.0
        running_corrects = 0
        last_loss = 0.0

        for epoch in range(self.config.client_epochs):
            self.gate.train()
            for expert_id in self.selected_expert_ids:
                self.experts[expert_id].train()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in self.train_loader:
                inputs = self._apply_batch_augmentation(inputs.to(self.device))
                labels = labels.to(self.device)
                mixed_inputs, labels_a, labels_b, mixup_lambda = self._sample_mixup(inputs, labels)

                self.gate_optimizer.zero_grad()
                self.expert_optimizer.zero_grad()
                if self.role == "anchor":
                    loss, logits = self._anchor_client_step(mixed_inputs, labels, labels_a, labels_b, mixup_lambda)
                else:
                    loss, logits = self._normal_client_step(mixed_inputs, labels, labels_a, labels_b, mixup_lambda)
                loss.backward()
                if self.config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.gate.parameters(), max_norm=self.config.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(
                        [param for expert_id in self.selected_expert_ids for param in self.experts[expert_id].parameters()],
                        max_norm=self.config.grad_clip_norm,
                    )
                self.gate_optimizer.step()
                self.expert_optimizer.step()

                running_loss += float(loss.item()) * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                running_corrects += int(torch.sum(preds == labels).item())
                last_loss = float(loss.item())

            train_loss = running_loss / float(sample_count)
            train_acc = running_corrects / float(sample_count)
            record_result(
                {
                    "T": self.round_idx,
                    "client_epoch": epoch + 1,
                    "client_id": self.client_id,
                    "client_train_loss": train_loss,
                    "client_train_acc": train_acc,
                }
            )

        return {
            "client_id": self.client_id,
            "role": self.role,
            "num_samples": sample_count,
            "selected_expert_ids": list(self.selected_expert_ids),
            "assigned_expert_id": self.assigned_expert_id,
            "gate_state": clone_state_dict(self.gate.state_dict()),
            "expert_states": {
                int(expert_id): clone_state_dict(self.experts[expert_id].state_dict())
                for expert_id in self.selected_expert_ids
            },
            "train_loss": float(running_loss / float(sample_count)),
            "train_acc": float(running_corrects / float(sample_count)),
            "last_batch_loss": float(last_loss),
        }
