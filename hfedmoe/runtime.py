from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Models.hfedmoe import ExpertPool, HFedMoEGate, HFedMoEModel, build_shared_backbone

from .client import HFedMoEClientTrainer
from .config import HFedMoEConfig, resolve_hfedmoe_config
from .utils import average_state_dicts, clone_state_dict, histogram_to_json, merge_histograms, state_dict_l2_norm


ROUND_FIELDNAMES = [
    "round",
    "active_clients",
    "avg_client_train_loss",
    "avg_client_train_acc",
    "mean_client_expert_scores",
    "mean_gate_entropy",
    "efficiency_metric",
    "avg_resource_utilization",
    "avg_selected_expert_count",
    "server_test_loss",
    "server_test_acc",
    "best_acc_so_far",
    "best_acc_round",
    "per_expert_usage_frequency",
    "server_agg_expert_params_norm",
    "resource_budget_mode",
    "aggregation_strategy",
    "combine_method",
    "main_eval_set",
    "main_metric",
]


class HFedMoERuntime:
    def __init__(self, args, logger, device: torch.device):
        self.args = args
        self.logger = logger
        self.device = device
        self.config = args if isinstance(args, HFedMoEConfig) else resolve_hfedmoe_config(args)
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_result, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_result, "detail"), exist_ok=True)

        self.client_ids = list(range(1, self.config.num_clients + 1))
        self.round_metrics_path = os.path.join(self.config.save_result, "logs", "hfedmoe_round_metrics.csv")
        self.run_summary_path = os.path.join(self.config.save_result, "logs", "run_summary.txt")
        self.checkpoint_path = os.path.join(self.config.model_save_path, "server.pth")
        self.best_server_test_acc = float("-inf")
        self.best_server_test_round = 0

        shared_backbone = build_shared_backbone(self.config)
        self.shared_backbone_state = clone_state_dict(shared_backbone.state_dict())
        self.gate_state = clone_state_dict(self._build_gate().state_dict())
        self.expert_pool = self._build_expert_pool()

        self._init_round_metrics_csv()
        self.save_checkpoint()

    def _build_gate(self) -> HFedMoEGate:
        return HFedMoEGate(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.gating_hidden_dim,
            num_experts=self.config.num_experts,
        )

    def _build_expert_pool(self) -> ExpertPool:
        return ExpertPool(
            num_experts=self.config.num_experts,
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.expert_hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )

    def _build_model(self) -> HFedMoEModel:
        shared_backbone = build_shared_backbone(self.config).to(self.device)
        shared_backbone.load_state_dict(self.shared_backbone_state)
        shared_backbone.eval()
        gate = self._build_gate().to(self.device)
        gate.load_state_dict(self.gate_state)
        expert_pool = self._build_expert_pool().to(self.device)
        expert_pool.load_expert_states(self.expert_pool.state_dict_for(range(self.config.num_experts)))
        model = HFedMoEModel(
            shared_backbone=shared_backbone,
            expert_pool=expert_pool,
            gating_net=gate,
            top_k=self.config.top_k,
            combine_method=self.config.combine_method,
        ).to(self.device)
        model.eval()
        return model

    def _init_round_metrics_csv(self):
        with open(self.round_metrics_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ROUND_FIELDNAMES)
            writer.writeheader()

    def _record_round_metrics(self, record: Dict[str, object]):
        with open(self.round_metrics_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ROUND_FIELDNAMES)
            writer.writerow(record)

    def _load_named_dataset(self, filename: str):
        path = os.path.join(self.config.data_save_path, filename)
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=False)

    def _sample_active_clients(self, round_idx: int):
        num_active = max(1, math.ceil(self.config.client_selection_ratio * len(self.client_ids)))
        if num_active >= len(self.client_ids):
            return list(self.client_ids)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + round_idx * 997)
        perm = torch.randperm(len(self.client_ids), generator=generator)[:num_active].tolist()
        return sorted(self.client_ids[index] for index in perm)

    def _shared_backbone_weight(self, update: Mapping[str, object]) -> float:
        return float(update["num_samples"])

    def _global_gate_weight(self, update: Mapping[str, object]) -> float:
        if self.config.aggregation_strategy == "uniform":
            return 1.0
        if self.config.aggregation_strategy == "sample_count":
            return float(update["num_samples"])
        return float(update["num_samples"]) * float(update["mean_selected_expert_score"])

    def _expert_weight(self, update: Mapping[str, object], expert_id: int) -> float:
        score = float(update["selected_expert_scores"][int(expert_id)])
        if self.config.aggregation_strategy == "uniform":
            return score if score > 0.0 else 1.0
        if self.config.aggregation_strategy == "sample_count":
            return float(update["num_samples"])
        return float(update["num_samples"]) * score

    def aggregate_updates(self, updates: Sequence[Mapping[str, object]]):
        if not updates:
            return {
                "updated_expert_ids": [],
                "server_agg_expert_params_norm": 0.0,
            }

        if self.config.train_shared_backbone:
            shared_weights = [self._shared_backbone_weight(update) for update in updates]
            self.shared_backbone_state = average_state_dicts([update["shared_backbone_state"] for update in updates], shared_weights)

        gate_weights = [self._global_gate_weight(update) for update in updates]
        self.gate_state = average_state_dicts([update["gate_state"] for update in updates], gate_weights)

        updated_expert_ids: List[int] = []
        updated_expert_norms: List[float] = []
        for expert_id in range(self.config.num_experts):
            participating_states = []
            participating_weights = []
            for update in updates:
                expert_states = update["expert_states"]
                if int(expert_id) not in expert_states:
                    continue
                participating_states.append(expert_states[int(expert_id)])
                participating_weights.append(self._expert_weight(update, expert_id))
            if not participating_states:
                continue
            aggregated_state = average_state_dicts(participating_states, participating_weights)
            self.expert_pool.load_expert_state(expert_id, aggregated_state)
            updated_expert_ids.append(int(expert_id))
            updated_expert_norms.append(state_dict_l2_norm(aggregated_state))

        return {
            "updated_expert_ids": updated_expert_ids,
            "server_agg_expert_params_norm": float(sum(updated_expert_norms) / max(len(updated_expert_norms), 1)),
        }

    def evaluate_server_test(self):
        dataset = self._load_named_dataset(self.config.main_eval_set)
        if dataset is None or len(dataset) == 0:
            return {
                "server_test_loss": 0.0,
                "server_test_acc": 0.0,
                "mean_gate_entropy": 0.0,
                "per_expert_usage_frequency": {},
                "main_eval_set": self.config.main_eval_set,
                "main_metric": self.config.main_metric,
            }

        model = self._build_model()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        entropy_sum = 0.0
        entropy_weight = 0
        usage_histogram: Dict[int, int] = {}

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, labels, reduction="sum")
                preds = torch.argmax(logits, dim=1)
                total_loss += float(loss.item())
                total_correct += int(torch.sum(preds == labels).item())
                total_samples += int(labels.numel())
                gate_weights = outputs["gate_weights"]
                entropy = -(gate_weights * torch.log(gate_weights.clamp_min(1e-8))).sum(dim=1).mean()
                entropy_sum += float(entropy.item()) * inputs.size(0)
                entropy_weight += int(inputs.size(0))
                topk_ids = outputs["selected_expert_ids"].detach().cpu().tolist()
                for sample_ids in topk_ids:
                    for expert_id in sample_ids:
                        usage_histogram[int(expert_id)] = usage_histogram.get(int(expert_id), 0) + 1

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "server_test_loss": total_loss / float(max(total_samples, 1)),
            "server_test_acc": total_correct / float(max(total_samples, 1)),
            "mean_gate_entropy": entropy_sum / float(max(entropy_weight, 1)),
            "per_expert_usage_frequency": usage_histogram,
            "main_eval_set": self.config.main_eval_set,
            "main_metric": self.config.main_metric,
        }

    def save_checkpoint(self):
        checkpoint = {
            "aggregation_method": "hfedmoe",
            "shared_backbone_state": clone_state_dict(self.shared_backbone_state),
            "gate_state": clone_state_dict(self.gate_state),
            "expert_states": self.expert_pool.state_dict_for(range(self.config.num_experts)),
            "config": dict(self.config.__dict__),
        }
        torch.save(checkpoint, self.checkpoint_path)

    def train(self):
        for round_idx in range(1, self.config.server_epochs + 1):
            self.logger.info(
                f"============================== HFedMoE Round {round_idx}/{self.config.server_epochs} start =============================="
            )
            active_clients = self._sample_active_clients(round_idx)
            gate_state_snapshot = clone_state_dict(self.gate_state)
            expert_state_snapshot = self.expert_pool.state_dict_for(range(self.config.num_experts))
            updates = []
            for client_id in active_clients:
                trainer = HFedMoEClientTrainer(
                    config=self.config,
                    client_id=client_id,
                    round_idx=round_idx,
                    logger=self.logger,
                    shared_backbone_state=self.shared_backbone_state,
                    gate_state=gate_state_snapshot,
                    expert_states=expert_state_snapshot,
                )
                updates.append(trainer.train())

            aggregation_info = self.aggregate_updates(updates)
            evaluation_metrics = self.evaluate_server_test()
            self.save_checkpoint()

            if evaluation_metrics["server_test_acc"] > self.best_server_test_acc:
                self.best_server_test_acc = float(evaluation_metrics["server_test_acc"])
                self.best_server_test_round = int(round_idx)

            avg_train_loss = sum(float(update["train_loss"]) for update in updates) / float(max(len(updates), 1))
            avg_train_acc = sum(float(update["train_acc"]) for update in updates) / float(max(len(updates), 1))
            mean_client_expert_scores = sum(float(update["mean_selected_expert_score"]) for update in updates) / float(max(len(updates), 1))
            mean_gate_entropy = sum(float(update["gate_entropy"]) for update in updates) / float(max(len(updates), 1))
            efficiency_metric = sum(float(update["efficiency_metric"]) for update in updates) / float(max(len(updates), 1))
            avg_resource_utilization = sum(float(update["resource_utilization"]) for update in updates) / float(max(len(updates), 1))
            avg_selected_expert_count = sum(len(update["selected_expert_ids"]) for update in updates) / float(max(len(updates), 1))
            per_expert_usage_frequency = merge_histograms(update["per_expert_usage_frequency"] for update in updates)

            self._record_round_metrics(
                {
                    "round": round_idx,
                    "active_clients": len(active_clients),
                    "avg_client_train_loss": avg_train_loss,
                    "avg_client_train_acc": avg_train_acc,
                    "mean_client_expert_scores": mean_client_expert_scores,
                    "mean_gate_entropy": mean_gate_entropy,
                    "efficiency_metric": efficiency_metric,
                    "avg_resource_utilization": avg_resource_utilization,
                    "avg_selected_expert_count": avg_selected_expert_count,
                    "server_test_loss": evaluation_metrics["server_test_loss"],
                    "server_test_acc": evaluation_metrics["server_test_acc"],
                    "best_acc_so_far": self.best_server_test_acc,
                    "best_acc_round": self.best_server_test_round,
                    "per_expert_usage_frequency": histogram_to_json(per_expert_usage_frequency),
                    "server_agg_expert_params_norm": aggregation_info["server_agg_expert_params_norm"],
                    "resource_budget_mode": bool(self.config.resource_budget_mode),
                    "aggregation_strategy": self.config.aggregation_strategy,
                    "combine_method": self.config.combine_method,
                    "main_eval_set": self.config.main_eval_set,
                    "main_metric": self.config.main_metric,
                }
            )

            with open(self.run_summary_path, "w") as summary_file:
                json.dump(
                    {
                        "algorithm": "hfedmoe",
                        "rounds": round_idx,
                        "final_acc": evaluation_metrics["server_test_acc"],
                        "final_loss": evaluation_metrics["server_test_loss"],
                        "best_acc": self.best_server_test_acc,
                        "best_round": self.best_server_test_round,
                        "num_experts": self.config.num_experts,
                        "top_k": self.config.top_k,
                        "resource_budget_mode": self.config.resource_budget_mode,
                        "agg_strategy": self.config.aggregation_strategy,
                        "combine_method": self.config.combine_method,
                        "main_eval_set": self.config.main_eval_set,
                        "main_metric": self.config.main_metric,
                        "mean_client_expert_scores": mean_client_expert_scores,
                        "mean_gate_entropy": mean_gate_entropy,
                        "efficiency_metric": efficiency_metric,
                        "per_expert_usage_frequency": per_expert_usage_frequency,
                        "server_agg_expert_params_norm": aggregation_info["server_agg_expert_params_norm"],
                        "shared_backbone_name": self.config.shared_backbone_name,
                        "train_shared_backbone": self.config.train_shared_backbone,
                        "gate_entropy_lambda": self.config.gate_entropy_lambda,
                        "expert_backbone_name": self.config.expert_backbone_name,
                        "avg_selected_expert_count": avg_selected_expert_count,
                    },
                    summary_file,
                    indent=2,
                    sort_keys=True,
                )

            self.logger.info(
                f"--hfedmoe_round:{round_idx} --avg_client_train_loss:{avg_train_loss:.4f} "
                f"--avg_client_train_acc:{avg_train_acc:.4f} "
                f"--server_test_loss:{evaluation_metrics['server_test_loss']:.4f} "
                f"--server_test_acc:{evaluation_metrics['server_test_acc']:.4f} "
                f"--mean_client_expert_scores:{mean_client_expert_scores:.4f}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
