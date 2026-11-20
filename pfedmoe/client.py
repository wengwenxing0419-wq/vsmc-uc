from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import record_result

from .modeling import PFedMoEPersonalModel


class PFedMoEClientTrainer:
    def __init__(
        self,
        config,
        client_id: int,
        round_idx: int,
        logger,
        global_extractor,
        local_extractor,
        gate,
        prediction_head,
        global_aux_head,
    ):
        self.config = config
        self.client_id = int(client_id)
        self.round_idx = int(round_idx)
        self.logger = logger
        self.device = torch.device(self.config.device)
        self.train_dataset = self._load_client_split("train")
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError(f"Client {self.client_id} has no training samples for pFedMoE")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.global_extractor = global_extractor.to(self.device)
        self.local_extractor = local_extractor.to(self.device)
        self.gate = gate.to(self.device)
        self.prediction_head = prediction_head.to(self.device)
        self.global_aux_head = global_aux_head.to(self.device)
        self.model = PFedMoEPersonalModel(
            global_extractor=self.global_extractor,
            local_extractor=self.local_extractor,
            gate=self.gate,
            prediction_head=self.prediction_head,
            global_aux_head=self.global_aux_head,
        ).to(self.device)

        self.in_gate_warmup = self.round_idx <= int(self.config.gate_warmup_rounds)
        self.freeze_gate = self.in_gate_warmup and bool(self.config.freeze_gate_during_warmup)
        self.optimizer = self._build_optimizer()

    def _load_client_split(self, split: str):
        path = os.path.join(self.config.data_save_path, f"{self.client_id}_{split}.pth")
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=False)

    def _optimizer_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        params.extend(list(self.global_extractor.parameters()))
        params.extend(list(self.local_extractor.parameters()))
        params.extend(list(self.prediction_head.parameters()))
        params.extend(list(self.global_aux_head.parameters()))
        if not self.freeze_gate:
            params.extend(list(self.gate.parameters()))
        return params

    def _build_optimizer(self):
        params = self._optimizer_params()
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
        raise ValueError(f"Unsupported client optimizer for pFedMoE: {self.config.client_optimizer}")

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

    def _compute_ce_loss(self, logits, labels, labels_a, labels_b, mixup_lambda):
        if mixup_lambda < 1.0:
            return (
                mixup_lambda
                * F.cross_entropy(logits, labels_a, reduction="mean", label_smoothing=self.config.label_smoothing)
                + (1.0 - mixup_lambda)
                * F.cross_entropy(logits, labels_b, reduction="mean", label_smoothing=self.config.label_smoothing)
            )
        return F.cross_entropy(logits, labels, reduction="mean", label_smoothing=self.config.label_smoothing)

    def _gate_override(self, batch_size: int):
        if not self.in_gate_warmup or self.config.gate_warmup_mode != "fixed_uniform":
            return None
        return torch.full((batch_size, 2), 0.5, device=self.device, dtype=torch.float32)

    def train(self):
        sample_count = len(self.train_loader.dataset)
        train_loss = 0.0
        train_acc = 0.0
        mean_alpha_global = 0.0
        mean_alpha_local = 0.0
        mean_gate_entropy = 0.0
        mean_loss_mix = 0.0
        mean_loss_global = 0.0
        mean_alpha_reg_loss = 0.0
        global_aux_acc = 0.0

        for epoch in range(self.config.client_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            alpha_global_sum = 0.0
            alpha_local_sum = 0.0
            alpha_weight = 0
            loss_mix_sum = 0.0
            loss_global_sum = 0.0
            alpha_reg_sum = 0.0
            entropy_sum = 0.0
            global_aux_corrects = 0

            for inputs, labels in self.train_loader:
                inputs = self._apply_batch_augmentation(inputs.to(self.device))
                labels = labels.to(self.device)
                mixed_inputs, labels_a, labels_b, mixup_lambda = self._sample_mixup(inputs, labels)
                gate_override = self._gate_override(mixed_inputs.size(0))

                self.optimizer.zero_grad()
                outputs = self.model(mixed_inputs, gate_override=gate_override)
                logits_mix = outputs["logits"]
                global_logits = outputs["global_logits"]
                gate_weights = outputs["gate_weights"]

                loss_mix = self._compute_ce_loss(logits_mix, labels, labels_a, labels_b, mixup_lambda)
                if self.config.global_aux_loss and global_logits is not None:
                    loss_global = self._compute_ce_loss(global_logits, labels, labels_a, labels_b, mixup_lambda)
                else:
                    loss_global = torch.zeros((), device=self.device)

                mean_alpha_global_batch = gate_weights[:, 0].mean()
                alpha_reg_loss = F.relu(
                    torch.tensor(self.config.alpha_global_target, device=self.device) - mean_alpha_global_batch
                ).pow(2)
                entropy = -(gate_weights * torch.log(gate_weights.clamp_min(self.config.gate_entropy_eps))).sum(dim=1).mean()
                total_loss = (
                    loss_mix
                    + self.config.global_aux_lambda * loss_global
                    + self.config.alpha_global_reg_lambda * alpha_reg_loss
                    - self.config.gate_entropy_lambda * entropy
                )
                total_loss.backward()
                if self.config.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.optimizer.step()

                running_loss += float(total_loss.item()) * inputs.size(0)
                preds = torch.argmax(logits_mix, dim=1)
                running_corrects += int(torch.sum(preds == labels).item())
                alpha_global_sum += float(gate_weights[:, 0].detach().sum().item())
                alpha_local_sum += float(gate_weights[:, 1].detach().sum().item())
                alpha_weight += int(gate_weights.size(0))
                loss_mix_sum += float(loss_mix.item()) * inputs.size(0)
                loss_global_sum += float(loss_global.item()) * inputs.size(0)
                alpha_reg_sum += float(alpha_reg_loss.item()) * inputs.size(0)
                entropy_sum += float(entropy.item()) * inputs.size(0)
                if global_logits is not None:
                    global_aux_corrects += int(torch.sum(torch.argmax(global_logits, dim=1) == labels).item())

            train_loss = running_loss / float(sample_count)
            train_acc = running_corrects / float(sample_count)
            mean_alpha_global = alpha_global_sum / float(max(alpha_weight, 1))
            mean_alpha_local = alpha_local_sum / float(max(alpha_weight, 1))
            mean_gate_entropy = entropy_sum / float(sample_count)
            mean_loss_mix = loss_mix_sum / float(sample_count)
            mean_loss_global = loss_global_sum / float(sample_count)
            mean_alpha_reg_loss = alpha_reg_sum / float(sample_count)
            global_aux_acc = global_aux_corrects / float(sample_count)
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
            "num_samples": sample_count,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "loss_mix": float(mean_loss_mix),
            "loss_global": float(mean_loss_global),
            "global_aux_acc": float(global_aux_acc),
            "alpha_global_reg_loss": float(mean_alpha_reg_loss),
            "mean_gate_entropy": float(mean_gate_entropy),
            "mean_alpha_global": float(mean_alpha_global),
            "mean_alpha_local": float(mean_alpha_local),
            "global_extractor_state": {
                name: tensor.detach().cpu()
                for name, tensor in self.global_extractor.state_dict().items()
            },
            "global_aux_head_state": {
                name: tensor.detach().cpu()
                for name, tensor in self.global_aux_head.state_dict().items()
            },
            "local_state": {
                "local_extractor_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.local_extractor.state_dict().items()
                },
                "gate_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.gate.state_dict().items()
                },
                "prediction_head_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.prediction_head.state_dict().items()
                },
            },
        }
