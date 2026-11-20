from __future__ import annotations

import os
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Models.hfedmoe import ExpertPool, HFedMoEGate, HFedMoEModel, build_shared_backbone
from utils import record_result

from .utils import clone_state_dict


class HFedMoEClientTrainer:
    def __init__(
        self,
        config,
        client_id: int,
        round_idx: int,
        logger,
        shared_backbone_state,
        gate_state,
        expert_states,
    ):
        self.config = config
        self.client_id = int(client_id)
        self.round_idx = int(round_idx)
        self.logger = logger
        self.device = torch.device(self.config.device)
        self.train_dataset = self._load_client_split("train")
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError(f"Client {self.client_id} has no training data for HFedMoE")
        self.val_dataset = self._load_client_split("val")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.shared_backbone = build_shared_backbone(self.config).to(self.device)
        self.shared_backbone.load_state_dict(shared_backbone_state)
        for param in self.shared_backbone.parameters():
            param.requires_grad = bool(self.config.train_shared_backbone)
        if self.config.train_shared_backbone:
            self.shared_backbone.train()
        else:
            self.shared_backbone.eval()

        self.gate = HFedMoEGate(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.gating_hidden_dim,
            num_experts=self.config.num_experts,
        ).to(self.device)
        self.gate.load_state_dict(gate_state)

        self.expert_pool = ExpertPool(
            num_experts=self.config.num_experts,
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.expert_hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        ).to(self.device)
        self.expert_pool.load_expert_states(expert_states)

        self.model = HFedMoEModel(
            shared_backbone=self.shared_backbone,
            expert_pool=self.expert_pool,
            gating_net=self.gate,
            top_k=self.config.top_k,
            combine_method=self.config.combine_method,
        ).to(self.device)

        self.importance_scores = self._compute_expert_importance()
        self.selected_expert_ids, self.resource_profile = self._select_experts_for_client()
        self._freeze_inactive_experts()
        self.optimizer = self._build_optimizer()

    def _load_client_split(self, split: str):
        path = os.path.join(self.config.data_save_path, f"{self.client_id}_{split}.pth")
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=False)

    def _importance_loader(self):
        dataset = self.val_dataset if self.val_dataset is not None and len(self.val_dataset) > 0 else self.train_dataset
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

    def _compute_client_budget_scale(self) -> float:
        scales = self.config.client_budget_scales
        return float(scales[(self.client_id - 1) % len(scales)])

    def _expert_memory_mb(self) -> float:
        return max(float(self.expert_pool.expert_param_count()) * 4.0 / (1024.0 ** 2), 1e-6)

    def _expert_flops_proxy(self) -> float:
        return max(float(self.expert_pool.expert_param_count()) * 2.0, 1.0)

    def _select_experts_for_client(self):
        sorted_experts = sorted(self.importance_scores.items(), key=lambda item: item[1], reverse=True)
        budget_scale = self._compute_client_budget_scale()
        memory_budget = float(self.config.client_memory_budget_mb) * budget_scale
        flops_budget = float(self.config.client_flops_budget) * budget_scale
        expert_memory_mb = self._expert_memory_mb()
        expert_flops = self._expert_flops_proxy()

        allowed = self.config.top_k
        if self.config.resource_budget_mode:
            candidate_limits = [self.config.top_k, self.config.num_experts]
            if memory_budget > 0.0:
                candidate_limits.append(max(int(memory_budget / expert_memory_mb), self.config.min_active_experts))
            if flops_budget > 0.0:
                candidate_limits.append(max(int(flops_budget / expert_flops), self.config.min_active_experts))
            allowed = min(candidate_limits)
        allowed = max(self.config.min_active_experts, min(int(allowed), self.config.num_experts))
        selected_ids = [int(expert_id) for expert_id, _ in sorted_experts[:allowed]]
        if not selected_ids:
            selected_ids = [int(sorted_experts[0][0])] if sorted_experts else [0]

        used_memory = float(len(selected_ids)) * expert_memory_mb
        used_flops = float(len(selected_ids)) * expert_flops
        utilization_terms = []
        if memory_budget > 0.0:
            utilization_terms.append(used_memory / memory_budget)
        if flops_budget > 0.0:
            utilization_terms.append(used_flops / flops_budget)
        resource_utilization = float(sum(utilization_terms) / len(utilization_terms)) if utilization_terms else 0.0

        return selected_ids, {
            "budget_scale": budget_scale,
            "allowed_expert_count": int(allowed),
            "memory_budget_mb": memory_budget,
            "flops_budget": flops_budget,
            "expert_memory_mb": expert_memory_mb,
            "expert_flops_proxy": expert_flops,
            "resource_utilization": resource_utilization,
        }

    def _compute_expert_importance(self) -> Dict[int, float]:
        loader = self._importance_loader()
        was_training = self.model.training
        self.model.eval()
        scores: Dict[int, float] = {}
        with torch.no_grad():
            for expert_id in range(self.config.num_experts):
                loss_sum = 0.0
                sample_count = 0
                for batch_idx, (inputs, labels) in enumerate(loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs, active_expert_ids=[expert_id])
                    loss = F.cross_entropy(
                        outputs["logits"],
                        labels,
                        reduction="sum",
                        label_smoothing=self.config.label_smoothing,
                    )
                    loss_sum += float(loss.item())
                    sample_count += int(labels.numel())
                    if batch_idx + 1 >= self.config.importance_max_batches:
                        break
                avg_loss = loss_sum / float(max(sample_count, 1))
                scores[int(expert_id)] = 1.0 / float(avg_loss + self.config.importance_eps)
        self.model.train(was_training)
        return scores

    def _freeze_inactive_experts(self):
        selected = set(self.selected_expert_ids)
        for expert_id, expert in enumerate(self.expert_pool.experts):
            requires_grad = expert_id in selected
            for param in expert.parameters():
                param.requires_grad = requires_grad
        for param in self.gate.parameters():
            param.requires_grad = True

    def _optimizer_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        if self.config.train_shared_backbone:
            params.extend([param for param in self.shared_backbone.parameters() if param.requires_grad])
        params.extend([param for param in self.gate.parameters() if param.requires_grad])
        for expert_id in self.selected_expert_ids:
            params.extend([param for param in self.expert_pool.experts[int(expert_id)].parameters() if param.requires_grad])
        return params

    def _build_optimizer(self):
        params = self._optimizer_parameters()
        optimizer_name = str(self.config.client_optimizer).lower()
        if optimizer_name == "adam":
            return optim.Adam(params, lr=self.config.learning_rate, weight_decay=self.config.client_weight_decay)
        if optimizer_name == "adamw":
            return optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.client_weight_decay)
        if optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.client_momentum,
                nesterov=self.config.client_nesterov,
                weight_decay=self.config.client_weight_decay,
            )
        raise ValueError(f"Unsupported HFedMoE client optimizer: {self.config.client_optimizer}")

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

    def _mixup_cross_entropy(self, logits, labels, labels_a, labels_b, mixup_lambda):
        if mixup_lambda < 1.0:
            return (
                mixup_lambda * F.cross_entropy(logits, labels_a, reduction="mean", label_smoothing=self.config.label_smoothing)
                + (1.0 - mixup_lambda) * F.cross_entropy(logits, labels_b, reduction="mean", label_smoothing=self.config.label_smoothing)
            )
        return F.cross_entropy(logits, labels, reduction="mean", label_smoothing=self.config.label_smoothing)

    def train(self):
        start_time = time.perf_counter()
        sample_count = len(self.train_loader.dataset)
        train_loss = 0.0
        train_acc = 0.0
        gate_entropy = 0.0
        usage_histogram = {int(expert_id): 0 for expert_id in self.selected_expert_ids}

        for epoch in range(self.config.client_epochs):
            self.model.train()
            if self.config.train_shared_backbone:
                self.shared_backbone.train()
            else:
                self.shared_backbone.eval()
            running_loss = 0.0
            running_corrects = 0
            entropy_sum = 0.0
            entropy_weight = 0
            usage_histogram = {int(expert_id): 0 for expert_id in self.selected_expert_ids}

            for inputs, labels in self.train_loader:
                inputs = self._apply_batch_augmentation(inputs.to(self.device))
                labels = labels.to(self.device)
                mixed_inputs, labels_a, labels_b, mixup_lambda = self._sample_mixup(inputs, labels)

                self.optimizer.zero_grad()
                outputs = self.model(mixed_inputs, active_expert_ids=self.selected_expert_ids)
                logits = outputs["logits"]
                task_loss = self._mixup_cross_entropy(logits, labels, labels_a, labels_b, mixup_lambda)
                gate_probs = outputs["gate_weights"]
                gate_entropy = -(gate_probs * torch.log(gate_probs.clamp_min(self.config.importance_eps))).sum(dim=1).mean()
                loss = task_loss - float(self.config.gate_entropy_lambda) * gate_entropy
                loss.backward()
                if self.config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self._optimizer_parameters(), max_norm=self.config.grad_clip_norm)
                self.optimizer.step()

                running_loss += float(task_loss.item()) * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                running_corrects += int(torch.sum(preds == labels).item())
                gate_probs = outputs["gate_weights"].detach()
                entropy = -(gate_probs * torch.log(gate_probs.clamp_min(self.config.importance_eps))).sum(dim=1).mean()
                entropy_sum += float(entropy.item()) * inputs.size(0)
                entropy_weight += int(inputs.size(0))
                top_expert_ids = outputs["selected_expert_ids"][..., 0].detach().cpu().tolist()
                for expert_id in top_expert_ids:
                    usage_histogram[int(expert_id)] = usage_histogram.get(int(expert_id), 0) + 1

            train_loss = running_loss / float(sample_count)
            train_acc = running_corrects / float(sample_count)
            gate_entropy = entropy_sum / float(max(entropy_weight, 1))
            record_result(
                {
                    "T": self.round_idx,
                    "client_epoch": epoch + 1,
                    "client_id": self.client_id,
                    "client_train_loss": train_loss,
                    "client_train_acc": train_acc,
                }
            )

        train_seconds = time.perf_counter() - start_time
        selected_score_values = [float(self.importance_scores[int(expert_id)]) for expert_id in self.selected_expert_ids]
        mean_selected_score = sum(selected_score_values) / float(max(len(selected_score_values), 1))
        efficiency_metric = train_seconds / float(max(self.resource_profile["resource_utilization"], 1e-8))

        return {
            "client_id": self.client_id,
            "num_samples": sample_count,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "gate_entropy": float(gate_entropy),
            "train_seconds": float(train_seconds),
            "resource_utilization": float(self.resource_profile["resource_utilization"]),
            "efficiency_metric": float(efficiency_metric),
            "selected_expert_ids": [int(expert_id) for expert_id in self.selected_expert_ids],
            "selected_expert_scores": {int(expert_id): float(self.importance_scores[int(expert_id)]) for expert_id in self.selected_expert_ids},
            "mean_selected_expert_score": float(mean_selected_score),
            "per_expert_usage_frequency": usage_histogram,
            "shared_backbone_state": clone_state_dict(self.shared_backbone.state_dict()),
            "gate_state": clone_state_dict(self.gate.state_dict()),
            "expert_states": self.expert_pool.state_dict_for(self.selected_expert_ids),
        }
