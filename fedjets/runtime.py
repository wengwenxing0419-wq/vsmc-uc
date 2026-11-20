from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .client import FedJETSClientTrainer
from .common_expert import CommonExpertWrapper
from .config import resolve_fedjets_config
from .expert_pool import ExpertPool
from .gate import FedJETSGate
from .utils import (
    average_state_dicts,
    clone_state_dict,
    compute_client_weights,
    merge_histograms,
    sample_client_subset,
    select_anchor_clients,
    select_topk_experts,
)


ROUND_FIELDNAMES = [
    "round",
    "avg_client_train_loss",
    "avg_client_train_acc",
    "server_test_loss",
    "server_test_acc",
    "best_acc_so_far",
    "best_acc_round",
    "active_anchor_clients",
    "active_normal_clients",
    "updated_expert_count",
    "gate_weight_mode",
    "expert_weight_mode",
    "communication_ratio",
]


class FedJETSRuntime:
    def __init__(self, args, logger, model_factory, device: torch.device):
        self.args = args
        self.logger = logger
        self.model_factory = model_factory
        self.device = device
        self.config = resolve_fedjets_config(args)

        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_result, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.config.save_result, "detail"), exist_ok=True)

        base_model = self.model_factory()
        base_state = clone_state_dict(base_model.state_dict())

        common_model = self.model_factory()
        if self.config.common_expert_ckpt:
            common_model.load_state_dict(self._load_model_state(self.config.common_expert_ckpt))
        else:
            common_model.load_state_dict(base_state)
        self.common_expert = CommonExpertWrapper(
            common_model,
            device=self.device,
            cache_embeddings=self.config.cache_embeddings,
        )
        self.config = type(self.config)(**{**self.config.__dict__, "embedding_dim": self.common_expert.embedding_dim})

        self.gate = FedJETSGate(
            embedding_dim=self.config.embedding_dim,
            num_experts=self.config.num_experts,
            hidden_dim=self.config.gate_hidden_dim,
        ).to(self.device)
        self.expert_pool = ExpertPool(
            num_experts=self.config.num_experts,
            model_factory=self.model_factory,
            initial_state=base_state,
        )

        self.client_ids = list(range(1, self.config.num_clients + 1))
        self.anchor_client_ids, self.anchor_assignments = select_anchor_clients(
            self.client_ids,
            self.config.num_anchor_clients,
            self.config.num_experts,
        )
        self.normal_client_ids = [client_id for client_id in self.client_ids if client_id not in self.anchor_assignments]

        self.round_metrics_path = os.path.join(self.config.save_result, "logs", "fedjets_round_metrics.csv")
        self.run_summary_path = os.path.join(self.config.save_result, "logs", "run_summary.txt")
        self.checkpoint_path = os.path.join(self.config.model_save_path, "server.pth")
        self.best_server_test_acc = float("-inf")
        self.best_server_test_round = 0
        self._init_round_metrics_csv()
        self.save_checkpoint()

    def _load_model_state(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, Mapping):
            if "model_state" in checkpoint:
                return checkpoint["model_state"]
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
        return checkpoint

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
        active_anchor_clients = sample_client_subset(
            self.anchor_client_ids,
            self.config.anchor_clients_per_round,
            self.config.seed + round_idx * 193,
        )
        active_normal_clients = sample_client_subset(
            self.normal_client_ids,
            self.config.normal_clients_per_round,
            self.config.seed + round_idx * 389,
        )
        return active_anchor_clients, active_normal_clients

    def _select_normal_client_experts(self, client_id: int) -> List[int]:
        dataset = self._load_client_split(client_id, "train")
        if dataset is None or len(dataset) == 0:
            raise ValueError(f"Client {client_id} is missing a train split for FedJETs expert selection")
        embeddings = self.common_expert.get_cached_client_embeddings(
            client_id=client_id,
            split="train",
            dataset=dataset,
            batch_size=self.config.batch_size,
        )
        selected_ids, _ = select_topk_experts(
            self.gate,
            embeddings,
            k=self.config.top_k,
            device=self.device,
        )
        return selected_ids

    def run_stage_a_collect_updates(self, round_idx: int):
        active_anchor_clients, active_normal_clients = self._sample_active_clients(round_idx)
        updates = []
        client_train_losses = []
        client_train_accs = []

        gate_state = clone_state_dict(self.gate.state_dict())
        for client_id in active_anchor_clients:
            assigned_expert_id = self.anchor_assignments[client_id]
            trainer = FedJETSClientTrainer(
                config=self.config,
                client_id=client_id,
                round_idx=round_idx,
                role="anchor",
                model_factory=self.model_factory,
                common_expert=self.common_expert,
                gate_state=gate_state,
                expert_states=self.expert_pool.state_dict_for([assigned_expert_id]),
                selected_expert_ids=[assigned_expert_id],
                assigned_expert_id=assigned_expert_id,
            )
            update = trainer.train()
            updates.append(update)
            client_train_losses.append(update["train_loss"])
            client_train_accs.append(update["train_acc"])

        for client_id in active_normal_clients:
            selected_expert_ids = self._select_normal_client_experts(client_id)
            trainer = FedJETSClientTrainer(
                config=self.config,
                client_id=client_id,
                round_idx=round_idx,
                role="normal",
                model_factory=self.model_factory,
                common_expert=self.common_expert,
                gate_state=gate_state,
                expert_states=self.expert_pool.state_dict_for(selected_expert_ids),
                selected_expert_ids=selected_expert_ids,
                assigned_expert_id=None,
            )
            update = trainer.train()
            updates.append(update)
            client_train_losses.append(update["train_loss"])
            client_train_accs.append(update["train_acc"])

        return {
            "updates": updates,
            "active_anchor_clients": active_anchor_clients,
            "active_normal_clients": active_normal_clients,
            "avg_client_train_loss": sum(client_train_losses) / max(len(client_train_losses), 1),
            "avg_client_train_acc": sum(client_train_accs) / max(len(client_train_accs), 1),
        }

    def run_stage_b_aggregate(self, stage_a_data):
        updates = stage_a_data["updates"]
        gate_weights = compute_client_weights(
            [int(update["num_samples"]) for update in updates],
            self.config.fedavg_weight_mode,
        )
        aggregated_gate_state = average_state_dicts(
            [update["gate_state"] for update in updates],
            gate_weights,
        )
        self.gate.load_state_dict(aggregated_gate_state)

        expert_weights = compute_client_weights(
            [int(update["num_samples"]) for update in updates],
            self.config.expert_weight_mode,
        )
        updated_expert_counts = self.expert_pool.aggregate_expert_updates(
            [update["expert_states"] for update in updates],
            expert_weights,
        )
        return {
            "updated_expert_counts": updated_expert_counts,
        }

    def run_stage_c_finalize(self, round_idx: int, stage_a_data, stage_b_data, evaluation_metrics):
        self.save_checkpoint()
        if evaluation_metrics["server_test_acc"] > self.best_server_test_acc:
            self.best_server_test_acc = float(evaluation_metrics["server_test_acc"])
            self.best_server_test_round = int(round_idx)

        communication_ratio = self._estimate_round_communication_ratio(
            num_anchor=len(stage_a_data["active_anchor_clients"]),
            num_normal=len(stage_a_data["active_normal_clients"]),
        )
        self._record_round_metrics(
            {
                "round": round_idx,
                "avg_client_train_loss": stage_a_data["avg_client_train_loss"],
                "avg_client_train_acc": stage_a_data["avg_client_train_acc"],
                "server_test_loss": evaluation_metrics["server_test_loss"],
                "server_test_acc": evaluation_metrics["server_test_acc"],
                "best_acc_so_far": self.best_server_test_acc,
                "best_acc_round": self.best_server_test_round,
                "active_anchor_clients": len(stage_a_data["active_anchor_clients"]),
                "active_normal_clients": len(stage_a_data["active_normal_clients"]),
                "updated_expert_count": len(stage_b_data["updated_expert_counts"]),
                "gate_weight_mode": self.config.fedavg_weight_mode,
                "expert_weight_mode": self.config.expert_weight_mode,
                "communication_ratio": communication_ratio,
            }
        )

        summary = {
            "rounds": round_idx,
            "final_acc": evaluation_metrics["server_test_acc"],
            "final_loss": evaluation_metrics["server_test_loss"],
            "best_acc": self.best_server_test_acc,
            "best_round": self.best_server_test_round,
            "anchor_clients": self.anchor_client_ids,
            "anchor_assignments": self.anchor_assignments,
            "communication_ratio": communication_ratio,
            "selected_expert_histogram": evaluation_metrics["selected_expert_histogram"],
        }
        with open(self.run_summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)

    def _estimate_round_communication_ratio(self, num_anchor: int, num_normal: int) -> float:
        total_active = num_anchor + num_normal
        if total_active <= 0:
            return 0.0
        mean_experts_sent = (
            float(num_anchor) * 1.0 + float(num_normal) * float(self.config.top_k)
        ) / float(total_active)
        return mean_experts_sent / float(self.config.num_experts)

    def save_checkpoint(self):
        checkpoint = {
            "aggregation_method": "fedjets",
            "gate_state": clone_state_dict(self.gate.state_dict()),
            "expert_states": self.expert_pool.all_states(),
            "anchor_assignments": dict(self.anchor_assignments),
            "common_expert_source": self.config.common_expert_ckpt or "server_init_model",
            "config": dict(self.config.__dict__),
        }
        torch.save(checkpoint, self.checkpoint_path)


    def _evaluate_dataset_with_selected_experts(
        self,
        gate_model,
        dataset,
        selected_expert_ids: Sequence[int],
    ):
        selected_experts = self.expert_pool.get_experts(selected_expert_ids, device=self.device)
        for expert in selected_experts.values():
            expert.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        expert_histogram: Dict[int, int] = {}
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                features = self.common_expert.extract_features(inputs)
                gate_scores = gate_model(features)[:, selected_expert_ids]
                route_indices = torch.argmax(gate_scores, dim=-1)

                logits_per_expert = {}
                for local_index, expert_id in enumerate(selected_expert_ids):
                    logits_per_expert[local_index], _ = selected_experts[expert_id](inputs)
                chosen_logits = torch.zeros(
                    inputs.size(0),
                    logits_per_expert[0].size(-1),
                    device=self.device,
                )
                for local_index, expert_id in enumerate(selected_expert_ids):
                    mask = route_indices == local_index
                    if mask.any():
                        chosen_logits[mask] = logits_per_expert[local_index][mask]
                        expert_histogram[int(expert_id)] = expert_histogram.get(int(expert_id), 0) + int(mask.sum().item())

                loss = F.cross_entropy(chosen_logits, labels, reduction="sum")
                preds = torch.argmax(chosen_logits, dim=1)
                total_loss += float(loss.item())
                total_correct += int(torch.sum(preds == labels).item())
                total_samples += int(labels.numel())
        return total_loss, total_correct, total_samples, expert_histogram

    def _evaluate_dataset_as_unseen_client(
        self,
        gate_model,
        dataset,
        cache_client_id: int,
        cache_split: str,
    ):
        client_embeddings = self.common_expert.get_cached_client_embeddings(
            client_id=cache_client_id,
            split=cache_split,
            dataset=dataset,
            batch_size=self.config.batch_size,
        )
        selected_expert_ids, _ = select_topk_experts(
            gate_model,
            client_embeddings,
            k=self.config.top_k,
            device=self.device,
        )
        return self._evaluate_dataset_with_selected_experts(
            gate_model=gate_model,
            dataset=dataset,
            selected_expert_ids=selected_expert_ids,
        )

    def evaluate_server_test(self):
        gate_state = clone_state_dict(self.gate.state_dict())
        gate_model = FedJETSGate(
            embedding_dim=self.config.embedding_dim,
            num_experts=self.config.num_experts,
            hidden_dim=self.config.gate_hidden_dim,
        ).to(self.device)
        gate_model.load_state_dict(gate_state)
        gate_model.eval()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        per_client_accuracy = {}
        histogram_parts = []
        evaluated_clients = 0

        for client_id in self.client_ids:
            dataset = self._load_client_split(client_id, "test")
            if dataset is None or len(dataset) == 0:
                continue

            client_embeddings = self.common_expert.get_cached_client_embeddings(
                client_id=client_id,
                split="test",
                dataset=dataset,
                batch_size=self.config.batch_size,
            )
            selected_expert_ids, _ = select_topk_experts(
                gate_model,
                client_embeddings,
                k=self.config.top_k,
                device=self.device,
            )
            client_loss, client_correct, client_total, client_histogram = self._evaluate_dataset_with_selected_experts(
                gate_model=gate_model,
                dataset=dataset,
                selected_expert_ids=selected_expert_ids,
            )

            total_loss += client_loss
            total_correct += client_correct
            total_samples += client_total
            per_client_accuracy[int(client_id)] = client_correct / float(max(client_total, 1))
            histogram_parts.append(client_histogram)
            evaluated_clients += 1

        if evaluated_clients == 0:
            server_test_dataset = self._load_named_dataset("server_test.pth")
            if server_test_dataset is not None and len(server_test_dataset) > 0:
                self.logger.warning(
                    "FedJETs evaluation fallback activated: client-specific *_test.pth splits were not found; "
                    "using server_test.pth as a single unseen-client evaluation set."
                )
                fallback_loss, fallback_correct, fallback_total, fallback_histogram = self._evaluate_dataset_as_unseen_client(
                    gate_model=gate_model,
                    dataset=server_test_dataset,
                    cache_client_id=0,
                    cache_split="server_test",
                )
                total_loss += fallback_loss
                total_correct += fallback_correct
                total_samples += fallback_total
                per_client_accuracy["server_test"] = fallback_correct / float(max(fallback_total, 1))
                histogram_parts.append(fallback_histogram)
            else:
                self.logger.warning(
                    "FedJETs evaluation found no client-specific *_test.pth splits and no server_test.pth fallback; "
                    "returning zero-accuracy placeholder metrics."
                )

        mean_loss = total_loss / float(max(total_samples, 1))
        mean_acc = total_correct / float(max(total_samples, 1))
        return {
            "server_test_loss": mean_loss,
            "server_test_acc": mean_acc,
            "per_client_accuracy": per_client_accuracy,
            "selected_expert_histogram": merge_histograms(histogram_parts),
            "communication_ratio": self._estimate_round_communication_ratio(
                num_anchor=self.config.anchor_clients_per_round,
                num_normal=self.config.normal_clients_per_round,
            ),
        }

    def train(self):
        for round_idx in range(1, self.config.server_epochs + 1):
            self.logger.info(
                f"============================== FedJETs Round {round_idx}/{self.config.server_epochs} start =============================="
            )
            stage_a_data = self.run_stage_a_collect_updates(round_idx)
            stage_b_data = self.run_stage_b_aggregate(stage_a_data)
            evaluation_metrics = self.evaluate_server_test()
            self.run_stage_c_finalize(round_idx, stage_a_data, stage_b_data, evaluation_metrics)
            self.logger.info(
                f"--fedjets_round:{round_idx} --avg_client_train_loss:{stage_a_data['avg_client_train_loss']:.4f} "
                f"--avg_client_train_acc:{stage_a_data['avg_client_train_acc']:.4f} "
                f"--server_test_loss:{evaluation_metrics['server_test_loss']:.4f} "
                f"--server_test_acc:{evaluation_metrics['server_test_acc']:.4f} "
                f"--updated_experts:{len(stage_b_data['updated_expert_counts'])}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
