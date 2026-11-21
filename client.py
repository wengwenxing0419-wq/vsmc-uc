import argparse
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from args import apply_plain_fedavg_baseline_overrides
from utils import record_result


class Client:
    def __init__(
        self,
        parse: argparse.ArgumentParser,
        client_id: int,
        logger,
        c_T: int,
        model_factory=None,
        global_state=None,
    ):
        self.args = apply_plain_fedavg_baseline_overrides(parse.parse_args())
        self.client_id = client_id
        if model_factory is None:
            raise ValueError("Client requires a model_factory for standard FL training")
        self.model = model_factory()
        self.device = torch.device(self.args.device)
        self.global_state = global_state
        if global_state is not None:
            self.model.load_state_dict(global_state)
        self.reference_state = {
            name: tensor.detach().clone().to(self.device)
            for name, tensor in self.model.state_dict().items()
            if torch.is_tensor(tensor)
        }
        self.c_T = c_T
        self.client_epochs = self.args.client_epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        self.optimizer = self.build_optimizer()
        self.batch_size = self.args.batch_size
        self.train_loader = None
        self.val_loader = None
        self.logger = logger
        self.get_dataloader()
        self.model_path = os.path.join(self.args.model_save_path, f"{self.client_id}.pth")

    def _compute_client_lr(self) -> float:
        base_lr = float(self.args.learning_rate)
        schedule = str(self.args.client_lr_schedule).lower()
        if schedule == "none":
            lr = base_lr
        elif schedule == "cosine":
            total_rounds = max(int(self.args.server_epochs), 1)
            progress = 0.0 if total_rounds <= 1 else float(self.c_T) / float(total_rounds - 1)
            min_lr = base_lr * float(self.args.client_min_lr_ratio)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            lr = min_lr + (base_lr - min_lr) * cosine
        else:
            raise ValueError(f"Unsupported client lr schedule: {self.args.client_lr_schedule}")

        warmup_rounds = max(int(getattr(self.args, "client_lr_warmup_rounds", 0)), 0)
        if warmup_rounds > 0:
            warmup_scale = min(1.0, float(self.c_T + 1) / float(warmup_rounds))
            lr = lr * warmup_scale
        return lr

    def _group_model_parameters(self):
        groups = {"shared": [], "router": [], "expert": [], "classifier": []}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if ".switch_ffn.experts." in name:
                groups["expert"].append(param)
            elif ".switch_ffn.router." in name:
                groups["router"].append(param)
            elif name.startswith("classifier"):
                groups["classifier"].append(param)
            else:
                groups["shared"].append(param)
        return groups

    def build_optimizer(self):
        optimizer_name = str(self.args.client_optimizer).lower()
        current_lr = self._compute_client_lr()
        if not self.args.use_client_specialization:
            if optimizer_name == "adam":
                return optim.Adam(
                    self.model.parameters(),
                    lr=current_lr,
                    weight_decay=self.args.client_weight_decay,
                )
            if optimizer_name == "adamw":
                return optim.AdamW(
                    self.model.parameters(),
                    lr=current_lr,
                    weight_decay=self.args.client_weight_decay,
                )
            if optimizer_name == "sgd":
                return optim.SGD(
                    self.model.parameters(),
                    lr=current_lr,
                    momentum=self.args.client_momentum,
                    nesterov=self.args.client_nesterov,
                    weight_decay=self.args.client_weight_decay,
                )
            raise ValueError(f"Unsupported client optimizer: {self.args.client_optimizer}")

        groups = self._group_model_parameters()
        group_specs = [
            ("shared", self.args.client_shared_lr_scale, self.args.client_shared_weight_decay),
            ("router", self.args.client_router_lr_scale, self.args.client_router_weight_decay),
            ("expert", self.args.client_expert_lr_scale, self.args.client_expert_weight_decay),
            ("classifier", self.args.client_classifier_lr_scale, self.args.client_classifier_weight_decay),
        ]
        param_groups = []
        for name, lr_scale, wd_override in group_specs:
            params = groups[name]
            if not params:
                continue
            weight_decay = self.args.client_weight_decay if wd_override is None else wd_override
            param_groups.append({
                "params": params,
                "lr": current_lr * float(lr_scale),
                "weight_decay": weight_decay,
                "group_name": name,
            })
        if optimizer_name == "adam":
            return optim.Adam(param_groups, lr=current_lr, weight_decay=self.args.client_weight_decay)
        if optimizer_name == "adamw":
            return optim.AdamW(param_groups, lr=current_lr, weight_decay=self.args.client_weight_decay)
        if optimizer_name == "sgd":
            return optim.SGD(
                param_groups,
                lr=current_lr,
                momentum=self.args.client_momentum,
                nesterov=self.args.client_nesterov,
                weight_decay=self.args.client_weight_decay,
            )
        raise ValueError(f"Unsupported client optimizer: {self.args.client_optimizer}")

    def _random_crop_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        pad = int(self.args.aug_random_crop_pad)
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
        flip_prob = float(self.args.aug_horizontal_flip)
        if flip_prob <= 0.0:
            return inputs
        flip_mask = torch.rand(inputs.size(0), device=inputs.device) < flip_prob
        if flip_mask.any():
            inputs = inputs.clone()
            inputs[flip_mask] = torch.flip(inputs[flip_mask], dims=[3])
        return inputs

    def _random_erasing_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        erase_prob = float(self.args.aug_random_erasing)
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
        if not self.args.use_client_aug:
            return inputs
        inputs = self._random_crop_batch(inputs)
        inputs = self._horizontal_flip_batch(inputs)
        inputs = self._random_erasing_batch(inputs)
        return inputs

    def _sample_mixup(self, inputs: torch.Tensor, labels: torch.Tensor):
        alpha = float(self.args.mixup_alpha)
        if alpha <= 0.0 or inputs.size(0) < 2:
            return inputs, labels, labels, 1.0
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = float(beta_dist.sample().item())
        perm = torch.randperm(inputs.size(0), device=inputs.device)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[perm]
        return mixed_inputs, labels, labels[perm], lam

    def save_client_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_client_split(self, split: str):
        data_path = os.path.join(self.args.data_save_path, f"{self.client_id}_{split}.pth")
        if not os.path.exists(data_path):
            return None
        return torch.load(data_path, weights_only=False)

    def get_dataloader(self):
        train_dataset = self.load_client_split("train")
        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError(
                f"Client {self.client_id} has no training samples. "
                "Please regenerate data with a larger alpha or fewer clients."
            )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = self.load_client_split("val")
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def _init_expert_metric_buffers(self):
        num_experts = int(self.args.switch_num_experts)
        return {
            "route_counts": torch.zeros(num_experts, dtype=torch.float64),
            "loss_sums": torch.zeros(num_experts, dtype=torch.float64),
            "loss_weights": torch.zeros(num_experts, dtype=torch.float64),
            "router_entropy_sum": 0.0,
            "router_entropy_weight": 0.0,
            "router_aux_loss_sum": 0.0,
            "router_aux_loss_weight": 0.0,
            "load_balance_gap_sum": 0.0,
            "load_balance_gap_weight": 0.0,
            "layer_route_counts": None,
            "num_tokens": 0.0,
        }

    def _accumulate_expert_metrics(self, buffers: Dict[str, torch.Tensor], router_stats, sample_losses):
        sample_expert_counts = router_stats.get("sample_expert_counts_by_layer")
        if sample_expert_counts is None:
            return

        sample_expert_counts = sample_expert_counts.detach().to(dtype=torch.float64)
        if sample_expert_counts.dim() == 2:
            per_sample_counts = sample_expert_counts
            per_layer_counts = sample_expert_counts.unsqueeze(0)
        else:
            per_sample_counts = sample_expert_counts.sum(dim=0)
            per_layer_counts = sample_expert_counts

        route_counts = per_sample_counts.sum(dim=0)
        per_sample_total = per_sample_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized_counts = per_sample_counts / per_sample_total
        sample_losses = sample_losses.detach().to(dtype=torch.float64)

        buffers["route_counts"] += route_counts.cpu()
        buffers["loss_sums"] += (normalized_counts * sample_losses.unsqueeze(-1)).sum(dim=0).cpu()
        buffers["loss_weights"] += normalized_counts.sum(dim=0).cpu()
        buffers["num_tokens"] += float(route_counts.sum().item())
        per_layer_route_counts = per_layer_counts.sum(dim=1).cpu()
        if buffers["layer_route_counts"] is None:
            buffers["layer_route_counts"] = torch.zeros_like(per_layer_route_counts)
        buffers["layer_route_counts"] += per_layer_route_counts

        router_probs = router_stats.get("router_probs")
        if router_probs is not None:
            router_probs = router_probs.detach().to(dtype=torch.float64)
            entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-12))).sum(dim=-1).mean()
            buffers["router_entropy_sum"] += float(entropy.item()) * router_probs.shape[0]
            buffers["router_entropy_weight"] += float(router_probs.shape[0])

        aux_loss = router_stats.get("aux_loss")
        if aux_loss is not None:
            aux_loss = aux_loss.detach().to(dtype=torch.float64)
            sample_weight = float(sample_losses.numel())
            buffers["router_aux_loss_sum"] += float(aux_loss.item()) * sample_weight
            buffers["router_aux_loss_weight"] += sample_weight

        density = route_counts / route_counts.sum().clamp_min(1.0)
        uniform_density = torch.full_like(density, 1.0 / max(int(density.numel()), 1))
        load_balance_gap = torch.abs(density - uniform_density).mean()
        buffers["load_balance_gap_sum"] += float(load_balance_gap.item()) * float(route_counts.sum().item())
        buffers["load_balance_gap_weight"] += float(route_counts.sum().item())

    def _finalize_expert_metrics(self, buffers: Dict[str, torch.Tensor], default_loss: float) -> Dict[str, Dict[int, float]]:
        route_counts = buffers["route_counts"]
        loss_sums = buffers["loss_sums"]
        loss_weights = buffers["loss_weights"]
        num_tokens = max(float(buffers["num_tokens"]), 0.0)

        expert_losses = {}
        expert_usage = {}
        route_count_dict = {}
        for expert_id in range(route_counts.numel()):
            route_count = float(route_counts[expert_id].item())
            loss_weight = float(loss_weights[expert_id].item())
            route_count_dict[expert_id] = route_count
            expert_usage[expert_id] = route_count / num_tokens if num_tokens > 0.0 else 0.0
            if loss_weight > 0.0:
                expert_losses[expert_id] = float(loss_sums[expert_id].item() / loss_weight)
            else:
                expert_losses[expert_id] = float(default_loss)

        router_usage = (
            buffers["layer_route_counts"].detach().clone().to(dtype=torch.float32)
            if buffers["layer_route_counts"] is not None
            else torch.zeros(
                int(getattr(self.args, "switch_layers", 0)),
                int(getattr(self.args, "switch_num_experts", 0)),
                dtype=torch.float32,
            )
        )

        return {
            "expert_losses": expert_losses,
            "expert_usage": expert_usage,
            "route_counts": route_count_dict,
            "num_tokens": num_tokens,
            "router_entropy": (
                float(buffers["router_entropy_sum"]) / max(float(buffers["router_entropy_weight"]), 1.0)
            ),
            "router_aux_loss": (
                float(buffers["router_aux_loss_sum"]) / max(float(buffers["router_aux_loss_weight"]), 1.0)
            ),
            "load_balance_gap": (
                float(buffers["load_balance_gap_sum"]) / max(float(buffers["load_balance_gap_weight"]), 1.0)
            ),
            "layer_route_counts": (
                {
                    int(layer_idx): {
                        int(expert_id): float(value.item())
                        for expert_id, value in enumerate(layer_counts)
                    }
                    for layer_idx, layer_counts in enumerate(buffers["layer_route_counts"])
                }
                if buffers["layer_route_counts"] is not None
                else {}
            ),
            "router_usage": router_usage,
        }

    def _proximal_penalty(self) -> torch.Tensor:
        if self.global_state is None:
            return torch.zeros((), device=self.device)
        beta_by_group = {
            "shared": float(self.args.client_shared_prox_beta),
            "router": float(self.args.client_router_prox_beta),
            "expert": float(self.args.client_expert_prox_beta),
            "classifier": float(self.args.client_classifier_prox_beta),
        }
        penalty = torch.zeros((), device=self.device)
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.reference_state:
                continue
            if ".switch_ffn.experts." in name:
                beta = beta_by_group["expert"]
            elif ".switch_ffn.router." in name:
                beta = beta_by_group["router"]
            elif name.startswith("classifier"):
                beta = beta_by_group["classifier"]
            else:
                beta = beta_by_group["shared"]
            if beta <= 0.0:
                continue
            ref = self.reference_state[name].to(device=param.device, dtype=param.dtype)
            penalty = penalty + 0.5 * beta * (param - ref).pow(2).mean()
        return penalty

    def _expert_diversity_penalty(self) -> torch.Tensor:
        if float(self.args.client_expert_diversity_beta) <= 0.0:
            return torch.zeros((), device=self.device)
        penalties = []
        for block in getattr(self.model, 'blocks', []):
            experts = getattr(block.switch_ffn, 'experts', None)
            if experts is None or len(experts) < 2:
                continue
            expert_vectors: List[torch.Tensor] = []
            for expert in experts:
                flat_parts = [param.reshape(-1) for param in expert.parameters() if param.requires_grad]
                if flat_parts:
                    expert_vectors.append(torch.cat(flat_parts, dim=0))
            for i in range(len(expert_vectors)):
                for j in range(i + 1, len(expert_vectors)):
                    cosine = F.cosine_similarity(expert_vectors[i], expert_vectors[j], dim=0)
                    penalties.append(cosine.pow(2))
        if not penalties:
            return torch.zeros((), device=self.device)
        return torch.stack(penalties).mean()

    def _router_balance_penalty(self, router_stats) -> torch.Tensor:
        if float(self.args.client_router_balance_beta) <= 0.0:
            return torch.zeros((), device=self.device)
        sample_counts = router_stats.get("sample_expert_counts_by_layer")
        if sample_counts is None:
            return torch.zeros((), device=self.device)
        sample_counts = sample_counts.float()
        if sample_counts.dim() == 2:
            sample_counts = sample_counts.unsqueeze(0)
        density = sample_counts.sum(dim=1)
        density = density / density.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        uniform = torch.full_like(density, 1.0 / max(density.shape[-1], 1))
        return torch.abs(density - uniform).mean()

    def _router_entropy_penalty(self, router_stats) -> torch.Tensor:
        if float(self.args.client_router_entropy_beta) <= 0.0:
            return torch.zeros((), device=self.device)
        router_probs = router_stats.get("router_probs_by_layer")
        if router_probs is None:
            return torch.zeros((), device=self.device)
        entropy = -(router_probs.clamp_min(1e-12) * torch.log(router_probs.clamp_min(1e-12))).sum(dim=-1).mean()
        return -entropy

    def _router_consistency_penalty(self, inputs: torch.Tensor, router_stats) -> torch.Tensor:
        if float(self.args.client_router_consistency_beta) <= 0.0:
            return torch.zeros((), device=self.device)
        primary_probs = router_stats.get("router_probs_by_layer")
        if primary_probs is None:
            return torch.zeros((), device=self.device)
        _, second_router_stats = self.model(inputs)
        second_probs = second_router_stats.get("router_probs_by_layer")
        if second_probs is None:
            return torch.zeros((), device=self.device)
        temperature = max(float(self.args.client_consistency_temperature), 1e-6)
        p = (primary_probs / temperature).clamp_min(1e-12)
        q = (second_probs / temperature).clamp_min(1e-12)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        kl_pq = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
        kl_qp = (q * (torch.log(q) - torch.log(p))).sum(dim=-1)
        return 0.5 * (kl_pq.mean() + kl_qp.mean())

    def _compose_training_loss(self, mixed_inputs, labels, labels_a, labels_b, mixup_lambda, outputs, router_stats):
        if mixup_lambda < 1.0:
            sample_losses = (
                mixup_lambda * F.cross_entropy(outputs, labels_a, reduction="none", label_smoothing=self.args.label_smoothing)
                + (1.0 - mixup_lambda) * F.cross_entropy(outputs, labels_b, reduction="none", label_smoothing=self.args.label_smoothing)
            )
        else:
            sample_losses = F.cross_entropy(outputs, labels, reduction="none", label_smoothing=self.args.label_smoothing)
        ce_loss = sample_losses.mean()
        aux_loss = self.args.aux_loss_weight * router_stats["aux_loss"]
        prox_penalty = self._proximal_penalty()
        diversity_penalty = float(self.args.client_expert_diversity_beta) * self._expert_diversity_penalty()
        router_balance_penalty = float(self.args.client_router_balance_beta) * self._router_balance_penalty(router_stats)
        router_entropy_penalty = float(self.args.client_router_entropy_beta) * self._router_entropy_penalty(router_stats)
        router_consistency_penalty = float(self.args.client_router_consistency_beta) * self._router_consistency_penalty(mixed_inputs, router_stats)
        total_loss = ce_loss + aux_loss + prox_penalty + diversity_penalty + router_balance_penalty + router_entropy_penalty + router_consistency_penalty
        loss_terms = {
            "ce_loss": float(ce_loss.detach().item()),
            "aux_loss": float(aux_loss.detach().item()),
            "prox_penalty": float(prox_penalty.detach().item()),
            "diversity_penalty": float(diversity_penalty.detach().item()),
            "router_balance_penalty": float(router_balance_penalty.detach().item()),
            "router_entropy_penalty": float(router_entropy_penalty.detach().item()),
            "router_consistency_penalty": float(router_consistency_penalty.detach().item()),
        }
        return total_loss, sample_losses, loss_terms

    def _evaluate_loader(self, loader: Optional[DataLoader]):
        if loader is None or len(loader.dataset) == 0:
            return None

        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        expert_buffers = self._init_expert_metric_buffers()

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, router_stats = self.model(inputs)
                sample_losses = F.cross_entropy(outputs, labels, reduction="none")
                loss = sample_losses.mean() + self.args.aux_loss_weight * router_stats["aux_loss"]

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                self._accumulate_expert_metrics(expert_buffers, router_stats, sample_losses)

        mean_loss = running_loss / len(loader.dataset)
        mean_acc = (running_corrects.double() / len(loader.dataset)).item()
        metrics = self._finalize_expert_metrics(expert_buffers, default_loss=mean_loss)
        metrics["loss"] = float(mean_loss)
        metrics["acc"] = float(mean_acc)
        return metrics

    def train(self):
        self.model.to(self.device)
        train_metric_buffers = self._init_expert_metric_buffers()

        for epoch in range(self.client_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self._apply_batch_augmentation(inputs)
                mixed_inputs, labels_a, labels_b, mixup_lambda = self._sample_mixup(inputs, labels)
                self.optimizer.zero_grad()
                outputs, router_stats = self.model(mixed_inputs)
                loss, sample_losses, loss_terms = self._compose_training_loss(mixed_inputs, labels, labels_a, labels_b, mixup_lambda, outputs, router_stats)
                loss.backward()
                if self.args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip_norm)
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                self._accumulate_expert_metrics(train_metric_buffers, router_stats, sample_losses)

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects.double() / len(self.train_loader.dataset)
            train_acc_value = train_acc.item()
            record_result(
                record_dic={
                    "T": self.c_T,
                    "client_epoch": epoch + 1,
                    "client_id": self.client_id,
                    "client_train_loss": train_loss,
                    "client_train_acc": train_acc_value,
                }
            )

        train_expert_metrics = self._finalize_expert_metrics(train_metric_buffers, default_loss=train_loss)
        val_metrics = self._evaluate_loader(self.val_loader)
        expert_metric_source = val_metrics if val_metrics is not None else train_expert_metrics

        self.save_client_model()
        client_state = {name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()}
        client_metrics = {
            "client_id": int(self.client_id),
            "num_samples": len(self.train_loader.dataset),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc_value),
            "train_size": len(self.train_loader.dataset),
            "val_loss": float(val_metrics["loss"]) if val_metrics is not None else float(train_loss),
            "val_acc": float(val_metrics["acc"]) if val_metrics is not None else float(train_acc_value),
            "expert_losses": dict(expert_metric_source["expert_losses"]),
            "expert_usage": dict(expert_metric_source["expert_usage"]),
            "route_counts": dict(expert_metric_source["route_counts"]),
            # Explicit training-time routing observations for FedTRA-style u.
            "train_expert_usage": dict(train_expert_metrics["expert_usage"]),
            "train_route_counts": dict(train_expert_metrics["route_counts"]),
            "train_num_tokens": float(train_expert_metrics["num_tokens"]),
            "train_layer_route_counts": dict(train_expert_metrics["layer_route_counts"]),
            "num_tokens": float(expert_metric_source["num_tokens"]),
            "router_entropy": float(expert_metric_source["router_entropy"]),
            "router_aux_loss": float(expert_metric_source["router_aux_loss"]),
            "load_balance_gap": float(expert_metric_source["load_balance_gap"]),
            "layer_route_counts": dict(expert_metric_source["layer_route_counts"]),
            "router_usage": train_expert_metrics["router_usage"].detach().cpu(),
            "use_client_specialization": bool(self.args.use_client_specialization),
        }
        return client_state, len(self.train_loader.dataset), client_metrics
