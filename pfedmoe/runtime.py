from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .client import PFedMoEClientTrainer
from .config import resolve_pfedmoe_config
from .modeling import (
    PFedMoEGate,
    PFedMoEPersonalModel,
    SwitchTransformerFeatureExtractor,
    clone_state_dict,
    count_parameters,
    freeze_module,
)


ROUND_FIELDNAMES = [
    "round",
    "active_clients",
    "avg_client_train_loss",
    "avg_client_train_acc",
    "avg_loss_mix",
    "avg_loss_global",
    "avg_global_aux_acc",
    "avg_alpha_global_reg_loss",
    "avg_mean_gate_entropy",
    "mean_alpha_global",
    "mean_alpha_local",
    "server_test_loss",
    "server_test_acc",
    "global_only_acc",
    "ensemble_clients_acc",
    "personalized_client_test_acc",
    "train_test_gap",
    "best_acc_so_far",
    "best_acc_round",
    "server_test_mode",
    "main_eval_set",
    "aggregation_weight_mode",
    "checkpoint_metric_name",
    "checkpoint_metric_value",
]


class PFedMoERuntime:
    def __init__(self, args, logger, device: torch.device):
        self.args = args
        self.logger = logger
        self.device = device
        self.config = resolve_pfedmoe_config(args)
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_result, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_result, "detail"), exist_ok=True)

        self.client_ids = list(range(1, self.config.num_clients + 1))
        self.round_metrics_path = os.path.join(self.config.save_result, "logs", "pfedmoe_round_metrics.csv")
        self.run_summary_path = os.path.join(self.config.save_result, "logs", "run_summary.txt")
        self.checkpoint_path = os.path.join(self.config.model_save_path, "server.pth")
        self.best_checkpoint_path = os.path.join(self.config.model_save_path, "best_server.pth")
        self.best_server_test_acc = float("-inf")
        self.best_server_test_round = 0
        self.best_checkpoint_metric_value = float("-inf")
        self.best_checkpoint_metric_name = self.config.best_metric
        self.best_checkpoint_round = 0

        shared_template = self._build_global_extractor()
        freeze_module(shared_template)
        self.frozen_shared_extractor_template = shared_template
        self.global_extractor_state = clone_state_dict(shared_template.state_dict())
        self.global_aux_head_state = clone_state_dict(self._build_global_aux_head().state_dict())
        self.client_personal_states = self._initialize_client_personal_states()
        self.global_num_params = count_parameters(self._build_global_extractor()) + count_parameters(self._build_global_aux_head())
        self.local_backbone_param_counts = {
            backbone_name: count_parameters(self._build_local_extractor(backbone_name))
            for backbone_name in sorted(set(self.config.local_backbones))
        }
        local_counts = [
            self.local_backbone_param_counts[self._local_backbone_for_client(client_id)]
            for client_id in self.client_ids
        ]
        self.avg_local_num_params = float(sum(local_counts)) / float(max(len(local_counts), 1))
        self.global_to_local_param_ratio = (
            float(self.global_num_params) / float(self.avg_local_num_params)
            if self.avg_local_num_params > 0.0
            else 0.0
        )
        self._init_round_metrics_csv()
        self.save_checkpoint()

    def _build_feature_extractor(self, backbone_name: str, num_layers: int) -> SwitchTransformerFeatureExtractor:
        return SwitchTransformerFeatureExtractor(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            backbone_name=backbone_name,
            hidden_dim=self.config.hidden_dim,
            num_layers=num_layers,
            num_heads=self.config.switch_heads,
            ffn_dim=self.config.switch_ffn_dim,
            num_experts=self.config.switch_num_experts,
            dropout=self.config.dropout,
        )

    def _build_global_extractor(self):
        return self._build_feature_extractor(
            backbone_name=self.config.global_backbone_name,
            num_layers=self.config.global_switch_layers,
        )

    def _build_local_extractor(self, backbone_name: str):
        return self._build_feature_extractor(
            backbone_name=backbone_name,
            num_layers=self.config.switch_layers,
        )

    def _build_gate(self):
        return PFedMoEGate(
            input_dim=2 * self.config.hidden_dim,
            hidden_dim=self.config.gate_hidden_dim,
        )

    def _build_prediction_head(self):
        return nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def _build_global_aux_head(self):
        return nn.Linear(self.config.hidden_dim, self.config.num_classes)

    def _local_backbone_for_client(self, client_id: int) -> str:
        backbones = self.config.local_backbones
        return backbones[(int(client_id) - 1) % len(backbones)]

    def _initialize_client_personal_states(self):
        states = {}
        for client_id in self.client_ids:
            backbone_name = self._local_backbone_for_client(client_id)
            local_extractor = self._build_local_extractor(backbone_name)
            gate = self._build_gate()
            head = self._build_prediction_head()
            states[int(client_id)] = {
                "backbone_name": backbone_name,
                "local_extractor_state": clone_state_dict(local_extractor.state_dict()),
                "gate_state": clone_state_dict(gate.state_dict()),
                "prediction_head_state": clone_state_dict(head.state_dict()),
            }
        return states

    def _init_round_metrics_csv(self):
        with open(self.round_metrics_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ROUND_FIELDNAMES)
            writer.writeheader()

    def _record_round_metrics(self, record: Dict[str, object]):
        with open(self.round_metrics_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ROUND_FIELDNAMES)
            writer.writerow(record)

    def _load_client_split(self, client_id: int, split: str):
        path = os.path.join(self.config.data_save_path, f"{int(client_id)}_{split}.pth")
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=False)

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

    def _compute_client_weights(self, updates: Sequence[Mapping[str, object]]):
        if not updates:
            return []
        mode = str(self.config.fedavg_weight_mode).lower()
        if mode == "uniform":
            return [1.0 / float(len(updates))] * len(updates)
        sample_counts = [max(float(update["num_samples"]), 0.0) for update in updates]
        total = sum(sample_counts)
        if total <= 0.0:
            return [1.0 / float(len(updates))] * len(updates)
        return [count / total for count in sample_counts]

    def _aggregate_state_key(self, updates: Sequence[Mapping[str, object]], key: str):
        weights = self._compute_client_weights(updates)
        aggregated = {}
        for update, weight in zip(updates, weights):
            state = update[key]
            for name, tensor in state.items():
                tensor = tensor.to(dtype=torch.float32)
                if name not in aggregated:
                    aggregated[name] = tensor * float(weight)
                else:
                    aggregated[name] += tensor * float(weight)
        return clone_state_dict(aggregated)

    def _aggregate_global_branch(self, updates: Sequence[Mapping[str, object]]):
        self.global_extractor_state = self._aggregate_state_key(updates, "global_extractor_state")
        if self.config.aggregate_global_aux_head:
            self.global_aux_head_state = self._aggregate_state_key(updates, "global_aux_head_state")

    def _build_personal_model_for_client(self, client_id: int):
        personal_state = self.client_personal_states[int(client_id)]
        global_extractor = self._build_global_extractor().to(self.device)
        global_extractor.load_state_dict(self.global_extractor_state)
        local_extractor = self._build_local_extractor(personal_state["backbone_name"]).to(self.device)
        local_extractor.load_state_dict(personal_state["local_extractor_state"])
        gate = self._build_gate().to(self.device)
        gate.load_state_dict(personal_state["gate_state"])
        prediction_head = self._build_prediction_head().to(self.device)
        prediction_head.load_state_dict(personal_state["prediction_head_state"])
        global_aux_head = self._build_global_aux_head().to(self.device)
        global_aux_head.load_state_dict(self.global_aux_head_state)
        model = PFedMoEPersonalModel(
            global_extractor=global_extractor,
            local_extractor=local_extractor,
            gate=gate,
            prediction_head=prediction_head,
            global_aux_head=global_aux_head,
        ).to(self.device)
        model.eval()
        return model

    def _fixed_gate_override(self, batch_size: int, alpha_global: float):
        alpha_global = float(min(max(alpha_global, 0.0), 1.0))
        alpha_local = 1.0 - alpha_global
        return torch.tensor([[alpha_global, alpha_local]], device=self.device, dtype=torch.float32).expand(batch_size, -1)

    def _evaluate_dataset(self, model, dataset, gate_override: Optional[torch.Tensor] = None):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_global = 0.0
        alpha_local = 0.0
        alpha_weight = 0
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch_override = None
                if gate_override is not None:
                    batch_override = gate_override
                    if batch_override.size(0) != inputs.size(0):
                        batch_override = gate_override[: inputs.size(0)]
                outputs = model(inputs, gate_override=batch_override)
                logits = outputs["logits"]
                gate_weights = outputs["gate_weights"]
                loss = F.cross_entropy(logits, labels, reduction="sum")
                preds = torch.argmax(logits, dim=1)
                total_loss += float(loss.item())
                total_correct += int(torch.sum(preds == labels).item())
                total_samples += int(labels.numel())
                alpha_global += float(gate_weights[:, 0].sum().item())
                alpha_local += float(gate_weights[:, 1].sum().item())
                alpha_weight += int(gate_weights.size(0))
        return {
            "loss_sum": total_loss,
            "correct": total_correct,
            "num_samples": total_samples,
            "mean_alpha_global": alpha_global / float(max(alpha_weight, 1)),
            "mean_alpha_local": alpha_local / float(max(alpha_weight, 1)),
        }

    def _evaluate_global_only_dataset(self, dataset):
        global_extractor = self._build_global_extractor().to(self.device)
        global_extractor.load_state_dict(self.global_extractor_state)
        global_aux_head = self._build_global_aux_head().to(self.device)
        global_aux_head.load_state_dict(self.global_aux_head_state)
        global_extractor.eval()
        global_aux_head.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                features = global_extractor(inputs)
                logits = global_aux_head(features)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                preds = torch.argmax(logits, dim=1)
                total_loss += float(loss.item())
                total_correct += int(torch.sum(preds == labels).item())
                total_samples += int(labels.numel())
        return {
            "loss_sum": total_loss,
            "correct": total_correct,
            "num_samples": total_samples,
            "mean_alpha_global": 1.0,
            "mean_alpha_local": 0.0,
        }

    def _evaluate_server_dataset(self, dataset, mode: str):
        mode = str(mode).lower()
        if mode == "global_only":
            metrics = self._evaluate_global_only_dataset(dataset)
            metrics["mode"] = mode
            return metrics

        models = [self._build_personal_model_for_client(client_id) for client_id in self.client_ids]
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_global = 0.0
        alpha_local = 0.0
        alpha_weight = 0
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                per_model_outputs = []
                if mode == "best_global_mix":
                    gate_override = self._fixed_gate_override(inputs.size(0), self.config.best_global_mix_alpha)
                else:
                    gate_override = None

                for model in models:
                    outputs = model(inputs, gate_override=gate_override)
                    per_model_outputs.append((model, outputs))

                if mode == "mean_gate":
                    alpha_bar = torch.stack([outputs["gate_weights"] for _, outputs in per_model_outputs], dim=0).mean(dim=0)
                    logits = torch.stack(
                        [
                            model.prediction_head(
                                alpha_bar[:, 0:1] * outputs["global_features"]
                                + alpha_bar[:, 1:2] * outputs["local_features"]
                            )
                            for model, outputs in per_model_outputs
                        ],
                        dim=0,
                    ).mean(dim=0)
                    gate_weights = alpha_bar
                else:
                    logits = torch.stack([outputs["logits"] for _, outputs in per_model_outputs], dim=0).mean(dim=0)
                    gate_weights = torch.stack([outputs["gate_weights"] for _, outputs in per_model_outputs], dim=0).mean(dim=0)

                loss = F.cross_entropy(logits, labels, reduction="sum")
                preds = torch.argmax(logits, dim=1)
                total_loss += float(loss.item())
                total_correct += int(torch.sum(preds == labels).item())
                total_samples += int(labels.numel())
                alpha_global += float(gate_weights[:, 0].sum().item())
                alpha_local += float(gate_weights[:, 1].sum().item())
                alpha_weight += int(gate_weights.size(0))

        for model in models:
            del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "loss_sum": total_loss,
            "correct": total_correct,
            "num_samples": total_samples,
            "mean_alpha_global": alpha_global / float(max(alpha_weight, 1)),
            "mean_alpha_local": alpha_local / float(max(alpha_weight, 1)),
            "mode": mode,
        }

    def _finalize_eval_metrics(self, metrics: Dict[str, float]):
        num_samples = int(metrics.get("num_samples", 0))
        if num_samples <= 0:
            return {
                "loss": 0.0,
                "acc": 0.0,
                "mean_alpha_global": 0.0,
                "mean_alpha_local": 0.0,
            }
        return {
            "loss": float(metrics["loss_sum"]) / float(num_samples),
            "acc": float(metrics["correct"]) / float(num_samples),
            "mean_alpha_global": float(metrics["mean_alpha_global"]),
            "mean_alpha_local": float(metrics["mean_alpha_local"]),
        }

    def _evaluate_personalized_client_tests(self):
        per_client_accuracy = {}
        losses = []
        accs = []
        alpha_globals = []
        alpha_locals = []
        sample_counts = []
        for client_id in self.client_ids:
            dataset = self._load_client_split(client_id, "test")
            if dataset is None or len(dataset) == 0:
                continue
            model = self._build_personal_model_for_client(client_id)
            metrics = self._evaluate_dataset(model, dataset)
            per_client_accuracy[int(client_id)] = metrics["correct"] / float(max(metrics["num_samples"], 1))
            losses.append(metrics["loss_sum"] / float(max(metrics["num_samples"], 1)))
            accs.append(per_client_accuracy[int(client_id)])
            alpha_globals.append(metrics["mean_alpha_global"])
            alpha_locals.append(metrics["mean_alpha_local"])
            sample_counts.append(metrics["num_samples"])
            del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if not sample_counts:
            return None
        total_weight = float(sum(sample_counts))
        return {
            "loss": sum(loss * count for loss, count in zip(losses, sample_counts)) / total_weight,
            "acc": sum(acc * count for acc, count in zip(accs, sample_counts)) / total_weight,
            "mean_alpha_global": sum(alpha * count for alpha, count in zip(alpha_globals, sample_counts)) / total_weight,
            "mean_alpha_local": sum(alpha * count for alpha, count in zip(alpha_locals, sample_counts)) / total_weight,
            "per_client_accuracy": per_client_accuracy,
        }

    def evaluate_server_test(self):
        server_test_dataset = self._load_named_dataset(self.config.main_eval_set)
        if server_test_dataset is None or len(server_test_dataset) == 0:
            self.logger.warning("pFedMoE evaluation could not find server_test.pth; returning zero metrics.")
            return {
                "server_test_loss": 0.0,
                "server_test_acc": 0.0,
                "mean_alpha_global": 0.0,
                "mean_alpha_local": 0.0,
                "eval_mode": "missing_server_test",
                "server_test_mode": self.config.server_test_mode,
                "main_eval_set": self.config.main_eval_set,
                "main_metric": self.config.main_metric,
                "fair_comparison": self.config.fair_comparison,
                "global_only_acc": None,
                "global_only_loss": None,
                "ensemble_clients_acc": None,
                "ensemble_clients_loss": None,
                "personalized_client_test_acc": None,
                "personalized_client_test_loss": None,
                "per_client_accuracy": {},
            }

        main_metrics = self._evaluate_server_dataset(server_test_dataset, self.config.server_test_mode)
        main_summary = self._finalize_eval_metrics(main_metrics)

        global_only_summary = None
        if self.config.log_global_only_acc or self.config.server_test_mode == "global_only":
            global_only_summary = self._finalize_eval_metrics(
                self._evaluate_server_dataset(server_test_dataset, "global_only")
            )

        if self.config.server_test_mode == "ensemble_clients":
            ensemble_summary = dict(main_summary)
        else:
            ensemble_summary = self._finalize_eval_metrics(
                self._evaluate_server_dataset(server_test_dataset, "ensemble_clients")
            )

        personalized_summary = None
        if self.config.log_personalized_client_test:
            personalized_summary = self._evaluate_personalized_client_tests()

        return {
            "server_test_loss": main_summary["loss"],
            "server_test_acc": main_summary["acc"],
            "mean_alpha_global": main_summary["mean_alpha_global"],
            "mean_alpha_local": main_summary["mean_alpha_local"],
            "eval_mode": "server_test",
            "server_test_mode": self.config.server_test_mode,
            "main_eval_set": self.config.main_eval_set,
            "main_metric": self.config.main_metric,
            "fair_comparison": self.config.fair_comparison,
            "global_only_acc": None if global_only_summary is None else global_only_summary["acc"],
            "global_only_loss": None if global_only_summary is None else global_only_summary["loss"],
            "ensemble_clients_acc": ensemble_summary["acc"],
            "ensemble_clients_loss": ensemble_summary["loss"],
            "personalized_client_test_acc": None if personalized_summary is None else personalized_summary["acc"],
            "personalized_client_test_loss": None if personalized_summary is None else personalized_summary["loss"],
            "personalized_client_test_mean_alpha_global": None if personalized_summary is None else personalized_summary["mean_alpha_global"],
            "personalized_client_test_mean_alpha_local": None if personalized_summary is None else personalized_summary["mean_alpha_local"],
            "per_client_accuracy": {} if personalized_summary is None else personalized_summary["per_client_accuracy"],
        }

    def evaluate_validation(self):
        if not self.config.use_val_for_checkpoint:
            return None
        server_val_dataset = self._load_named_dataset("server_val.pth")
        if server_val_dataset is None or len(server_val_dataset) == 0:
            return None
        metrics = self._evaluate_server_dataset(server_val_dataset, self.config.server_test_mode)
        summary = self._finalize_eval_metrics(metrics)
        return {
            "val_loss": summary["loss"],
            "val_acc": summary["acc"],
        }

    def save_checkpoint(self, path: Optional[str] = None):
        checkpoint = {
            "aggregation_method": "pfedmoe",
            "global_extractor_state": clone_state_dict(self.global_extractor_state),
            "global_aux_head_state": clone_state_dict(self.global_aux_head_state),
            "client_personal_states": self.client_personal_states,
            "config": dict(self.config.__dict__),
        }
        torch.save(checkpoint, path or self.checkpoint_path)

    def _checkpoint_metric(self, evaluation_metrics: Dict[str, object], validation_metrics: Optional[Dict[str, float]]):
        if self.config.best_metric == "val_acc" and validation_metrics is not None:
            return "val_acc", float(validation_metrics["val_acc"])
        return "server_test_acc", float(evaluation_metrics["server_test_acc"])

    def _write_run_summary(
        self,
        round_idx: int,
        evaluation_metrics: Dict[str, object],
        validation_metrics: Optional[Dict[str, float]],
        avg_metrics: Dict[str, float],
    ):
        summary = {
            "rounds": round_idx,
            "main_eval_set": self.config.main_eval_set,
            "main_metric": self.config.main_metric,
            "fair_comparison": self.config.fair_comparison,
            "server_test_mode": self.config.server_test_mode,
            "final_acc": float(evaluation_metrics["server_test_acc"]),
            "final_loss": float(evaluation_metrics["server_test_loss"]),
            "best_acc": float(self.best_server_test_acc),
            "best_round": int(self.best_server_test_round),
            "eval_mode": str(evaluation_metrics["eval_mode"]),
            "mean_alpha_global": float(evaluation_metrics["mean_alpha_global"]),
            "mean_alpha_local": float(evaluation_metrics["mean_alpha_local"]),
            "pfedmoe_global_backbone_name": self.config.global_backbone_name,
            "global_num_params": int(self.global_num_params),
            "local_backbone_names": list(sorted(set(self.config.local_backbones))),
            "avg_local_num_params": float(self.avg_local_num_params),
            "global_to_local_param_ratio": float(self.global_to_local_param_ratio),
            "global_only_acc": evaluation_metrics["global_only_acc"],
            "ensemble_clients_acc": evaluation_metrics["ensemble_clients_acc"],
            "personalized_client_test_acc": evaluation_metrics["personalized_client_test_acc"],
            "client_backbones": {
                str(client_id): state["backbone_name"]
                for client_id, state in self.client_personal_states.items()
            },
            "avg_client_train_loss": float(avg_metrics["avg_client_train_loss"]),
            "avg_client_train_acc": float(avg_metrics["avg_client_train_acc"]),
            "avg_loss_mix": float(avg_metrics["avg_loss_mix"]),
            "avg_loss_global": float(avg_metrics["avg_loss_global"]),
            "avg_global_aux_acc": float(avg_metrics["avg_global_aux_acc"]),
            "avg_alpha_global_reg_loss": float(avg_metrics["avg_alpha_global_reg_loss"]),
            "avg_mean_gate_entropy": float(avg_metrics["avg_mean_gate_entropy"]),
            "best_checkpoint_metric_name": self.best_checkpoint_metric_name,
            "best_checkpoint_metric_value": float(self.best_checkpoint_metric_value),
            "best_checkpoint_round": int(self.best_checkpoint_round),
        }
        if validation_metrics is not None:
            summary.update(validation_metrics)
        with open(self.run_summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)

    def train(self):
        for round_idx in range(1, self.config.server_epochs + 1):
            self.logger.info(
                f"============================== pFedMoE Round {round_idx}/{self.config.server_epochs} start =============================="
            )
            active_clients = self._sample_active_clients(round_idx)
            updates = []
            for client_id in active_clients:
                personal_state = self.client_personal_states[int(client_id)]
                global_extractor = self._build_global_extractor()
                global_extractor.load_state_dict(self.global_extractor_state)
                for param in global_extractor.parameters():
                    param.requires_grad = True
                local_extractor = self._build_local_extractor(personal_state["backbone_name"])
                local_extractor.load_state_dict(personal_state["local_extractor_state"])
                gate = self._build_gate()
                gate.load_state_dict(personal_state["gate_state"])
                prediction_head = self._build_prediction_head()
                prediction_head.load_state_dict(personal_state["prediction_head_state"])
                global_aux_head = self._build_global_aux_head()
                global_aux_head.load_state_dict(self.global_aux_head_state)

                trainer = PFedMoEClientTrainer(
                    config=self.config,
                    client_id=client_id,
                    round_idx=round_idx,
                    logger=self.logger,
                    global_extractor=global_extractor,
                    local_extractor=local_extractor,
                    gate=gate,
                    prediction_head=prediction_head,
                    global_aux_head=global_aux_head,
                )
                update = trainer.train()
                updates.append(update)
                self.client_personal_states[int(client_id)].update(update["local_state"])

            self._aggregate_global_branch(updates)
            evaluation_metrics = self.evaluate_server_test()
            validation_metrics = self.evaluate_validation()
            self.save_checkpoint()

            if evaluation_metrics["server_test_acc"] > self.best_server_test_acc:
                self.best_server_test_acc = float(evaluation_metrics["server_test_acc"])
                self.best_server_test_round = int(round_idx)

            checkpoint_metric_name, checkpoint_metric_value = self._checkpoint_metric(evaluation_metrics, validation_metrics)
            if checkpoint_metric_value > self.best_checkpoint_metric_value:
                self.best_checkpoint_metric_name = checkpoint_metric_name
                self.best_checkpoint_metric_value = checkpoint_metric_value
                self.best_checkpoint_round = int(round_idx)
                self.save_checkpoint(self.best_checkpoint_path)

            avg_metrics = {
                "avg_client_train_loss": sum(float(update["train_loss"]) for update in updates) / float(max(len(updates), 1)),
                "avg_client_train_acc": sum(float(update["train_acc"]) for update in updates) / float(max(len(updates), 1)),
                "avg_loss_mix": sum(float(update["loss_mix"]) for update in updates) / float(max(len(updates), 1)),
                "avg_loss_global": sum(float(update["loss_global"]) for update in updates) / float(max(len(updates), 1)),
                "avg_global_aux_acc": sum(float(update["global_aux_acc"]) for update in updates) / float(max(len(updates), 1)),
                "avg_alpha_global_reg_loss": sum(float(update["alpha_global_reg_loss"]) for update in updates) / float(max(len(updates), 1)),
                "avg_mean_gate_entropy": sum(float(update["mean_gate_entropy"]) for update in updates) / float(max(len(updates), 1)),
                "mean_alpha_global": sum(float(update["mean_alpha_global"]) for update in updates) / float(max(len(updates), 1)),
                "mean_alpha_local": sum(float(update["mean_alpha_local"]) for update in updates) / float(max(len(updates), 1)),
            }
            train_test_gap = avg_metrics["avg_client_train_acc"] - float(evaluation_metrics["server_test_acc"])

            self._record_round_metrics(
                {
                    "round": round_idx,
                    "active_clients": len(active_clients),
                    "avg_client_train_loss": avg_metrics["avg_client_train_loss"],
                    "avg_client_train_acc": avg_metrics["avg_client_train_acc"],
                    "avg_loss_mix": avg_metrics["avg_loss_mix"],
                    "avg_loss_global": avg_metrics["avg_loss_global"],
                    "avg_global_aux_acc": avg_metrics["avg_global_aux_acc"],
                    "avg_alpha_global_reg_loss": avg_metrics["avg_alpha_global_reg_loss"],
                    "avg_mean_gate_entropy": avg_metrics["avg_mean_gate_entropy"],
                    "mean_alpha_global": avg_metrics["mean_alpha_global"],
                    "mean_alpha_local": avg_metrics["mean_alpha_local"],
                    "server_test_loss": evaluation_metrics["server_test_loss"],
                    "server_test_acc": evaluation_metrics["server_test_acc"],
                    "global_only_acc": evaluation_metrics["global_only_acc"],
                    "ensemble_clients_acc": evaluation_metrics["ensemble_clients_acc"],
                    "personalized_client_test_acc": evaluation_metrics["personalized_client_test_acc"],
                    "train_test_gap": train_test_gap,
                    "best_acc_so_far": self.best_server_test_acc,
                    "best_acc_round": self.best_server_test_round,
                    "server_test_mode": self.config.server_test_mode,
                    "main_eval_set": self.config.main_eval_set,
                    "aggregation_weight_mode": self.config.fedavg_weight_mode,
                    "checkpoint_metric_name": checkpoint_metric_name,
                    "checkpoint_metric_value": checkpoint_metric_value,
                }
            )
            self._write_run_summary(round_idx, evaluation_metrics, validation_metrics, avg_metrics)
            self.logger.info(
                f"--pfedmoe_round:{round_idx} --avg_client_train_loss:{avg_metrics['avg_client_train_loss']:.4f} "
                f"--avg_client_train_acc:{avg_metrics['avg_client_train_acc']:.4f} "
                f"--mean_alpha_global:{avg_metrics['mean_alpha_global']:.4f} "
                f"--mean_alpha_local:{avg_metrics['mean_alpha_local']:.4f} "
                f"--server_test_loss:{evaluation_metrics['server_test_loss']:.4f} "
                f"--server_test_acc:{evaluation_metrics['server_test_acc']:.4f} "
                f"--server_test_mode:{self.config.server_test_mode}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
