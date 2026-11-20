from __future__ import annotations

import copy
import csv
import logging
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from args import parse
from client import Client
from server import Server
from utils import record_result

from .fedmix_aggregator import FedMixAggregator
from .fedmix_config import ensure_fedmix_args, resolve_fedmix_config
from .fedmix_pseudo import generate_pseudo_labels


ROUND_METRIC_FIELDNAMES = [
    "round",
    "train_loss",
    "train_acc",
    "eval_loss",
    "eval_acc",
    "best_acc",
    "fedmix_beta",
    "fedmix_lambda",
    "fedmix_pseudo_thresh",
    "fedmix_use_dynamic_sample_selection",
    "fedmix_use_adaptive_aggregation",
    "fedmix_shared_weight_mode",
    "fedmix_expert_weight_mode",
    "fedmix_avg_client_loss",
    "fedmix_avg_selected_ratio",
    "fedmix_avg_pseudo_confidence",
    "fedmix_avg_agreement_ratio",
    "fedmix_avg_pseudo_loss",
    "fedmix_max_expert_weight",
    "fedmix_min_expert_weight",
    "num_selected_clients",
    "total_client_samples",
]


def _freeze_parser_with_namespace(parser, namespace) -> None:
    frozen_namespace = copy.deepcopy(namespace)

    def _parse_args(args=None, namespace_override=None):
        del args, namespace_override
        return copy.deepcopy(frozen_namespace)

    def _parse_known_args(args=None, namespace_override=None):
        del args, namespace_override
        return copy.deepcopy(frozen_namespace), []

    parser.parse_args = _parse_args
    parser.parse_known_args = _parse_known_args


def _build_logger(output_dir: str) -> logging.Logger:
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(f"fedmix::{output_dir}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    file_handler = logging.FileHandler(os.path.join(logs_dir, "fedmix.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class FedMixClient(Client):
    def __init__(
        self,
        parse,
        client_id: int,
        logger,
        c_T: int,
        model_factory,
        global_state,
        fedmix_config,
    ):
        self.fedmix_config = fedmix_config
        self._teacher_factory = model_factory
        super().__init__(
            parse=parse,
            client_id=client_id,
            logger=logger,
            c_T=c_T,
            model_factory=model_factory,
            global_state=global_state,
        )
        self.teacher_model = self._teacher_factory().to(self.device)
        self.teacher_model.load_state_dict(global_state, strict=True)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    def train(self):
        self.model.to(self.device)
        self.teacher_model.to(self.device)
        train_metric_buffers = self._init_expert_metric_buffers()
        selected_count = 0.0
        agreement_count = 0.0
        total_seen = 0.0
        pseudo_conf_sum = 0.0
        pseudo_conf_weight = 0.0
        pseudo_loss_sum = 0.0
        pseudo_loss_weight = 0.0

        for epoch in range(self.client_epochs):
            self.model.train()
            self.teacher_model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self._apply_batch_augmentation(inputs)
                mixed_inputs, labels_a, labels_b, mixup_lambda = self._sample_mixup(inputs, labels)

                self.optimizer.zero_grad()
                outputs, router_stats = self.model(mixed_inputs)
                supervised_loss, sample_losses, _ = self._compose_training_loss(
                    mixed_inputs,
                    labels,
                    labels_a,
                    labels_b,
                    mixup_lambda,
                    outputs,
                    router_stats,
                )

                with torch.no_grad():
                    teacher_outputs, _ = self.teacher_model(mixed_inputs)
                    pseudo_data = generate_pseudo_labels(
                        student_logits=outputs.detach(),
                        teacher_logits=teacher_outputs.detach(),
                        threshold=self.fedmix_config.pseudo_thresh,
                        label_type=self.fedmix_config.label_type,
                    )

                if self.fedmix_config.use_dynamic_sample_selection:
                    selection_mask = pseudo_data["mask"]
                else:
                    selection_mask = torch.ones_like(pseudo_data["mask"], dtype=torch.bool)

                pseudo_loss = outputs.new_zeros(())
                if selection_mask.any():
                    pseudo_loss = F.cross_entropy(outputs[selection_mask], pseudo_data["pseudo_labels"][selection_mask])
                    pseudo_conf_sum += float(pseudo_data["confidence"][selection_mask].sum().item())
                    pseudo_conf_weight += float(selection_mask.sum().item())
                total_loss = supervised_loss + float(self.fedmix_config.pseudo_loss_weight) * pseudo_loss
                total_loss.backward()
                if self.args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip_norm)
                self.optimizer.step()

                batch_size = float(inputs.size(0))
                running_loss += float(total_loss.item()) * batch_size
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                self._accumulate_expert_metrics(train_metric_buffers, router_stats, sample_losses)

                total_seen += batch_size
                selected_count += float(selection_mask.sum().item())
                agreement_count += float(pseudo_data["agreement"].sum().item())
                pseudo_loss_sum += float(pseudo_loss.detach().item()) * batch_size
                pseudo_loss_weight += batch_size

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
            "num_tokens": float(expert_metric_source["num_tokens"]),
            "router_entropy": float(expert_metric_source["router_entropy"]),
            "router_aux_loss": float(expert_metric_source["router_aux_loss"]),
            "load_balance_gap": float(expert_metric_source["load_balance_gap"]),
            "layer_route_counts": dict(expert_metric_source["layer_route_counts"]),
            "router_usage": train_expert_metrics["router_usage"].detach().cpu(),
            "use_client_specialization": bool(self.args.use_client_specialization),
            "fedmix_selected_ratio": float(selected_count / max(total_seen, 1.0)),
            "fedmix_pseudo_confidence": float(pseudo_conf_sum / max(pseudo_conf_weight, 1.0)),
            "fedmix_agreement_ratio": float(agreement_count / max(total_seen, 1.0)),
            "fedmix_pseudo_loss": float(pseudo_loss_sum / max(pseudo_loss_weight, 1.0)),
        }
        return client_state, len(self.train_loader.dataset), client_metrics


class FedMixServer(Server):
    def __init__(self, parse, logger):
        super().__init__(parse=parse, logger=logger)
        self.fedmix_config = resolve_fedmix_config(self.args)
        self.fedmix_aggregator = FedMixAggregator(config=self.fedmix_config, logger=self.logger)
        self.round_metrics_path = os.path.join(self.fedmix_config.output_dir, "logs", "round_metrics.csv")
        self.run_summary_path = os.path.join(self.fedmix_config.output_dir, "logs", "run_summary.txt")
        self.min_eval_loss = float("inf")
        self.final_eval_loss = float("inf")
        self.final_eval_acc = 0.0
        self.final_round = 0
        self._init_round_metrics_csv()
        self.logger.info(
            "FedMix mixed-supervision baseline selected: shared/common experiment settings are inherited "
            "from args.py; client training adds teacher-student pseudo supervision and expert aggregation "
            "uses FedMix-style adaptive weights."
        )

    def _init_round_metrics_csv(self) -> None:
        os.makedirs(os.path.dirname(self.round_metrics_path), exist_ok=True)
        with open(self.round_metrics_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ROUND_METRIC_FIELDNAMES)
            writer.writeheader()

    def _append_round_metrics(self, row: Dict[str, object]) -> None:
        with open(self.round_metrics_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ROUND_METRIC_FIELDNAMES)
            writer.writerow(row)

    def _write_run_summary(self) -> None:
        summary_text = (
            "method: FedMix mixed supervision\n"
            f"beta: {float(self.fedmix_config.beta):.4f}\n"
            f"lambda: {float(self.fedmix_config.lam):.4f}\n"
            f"pseudo_thresh: {float(self.fedmix_config.pseudo_thresh):.4f}\n"
            f"dynamic_sample_selection: {bool(self.fedmix_config.use_dynamic_sample_selection)}\n"
            f"adaptive_aggregation: {bool(self.fedmix_config.use_adaptive_aggregation)}\n"
            f"seed: {int(self.args.seed)}\n"
            f"final_acc: {float(self.final_eval_acc):.4f}\n"
            f"best_acc: {float(self.best_server_test_acc):.4f}\n"
            f"best_round: {int(self.best_server_test_round)}\n"
            f"final_loss: {float(self.final_eval_loss):.4f}\n"
            f"min_eval_loss: {float(self.min_eval_loss):.4f}\n"
            f"total_rounds: {int(self.final_round)}\n"
        )
        with open(self.run_summary_path, "w", encoding="utf-8") as handle:
            handle.write(summary_text)

    def train(self):
        total_rounds = int(self.fedmix_config.total_rounds)
        for round_idx in range(total_rounds):
            self.logger.info(
                f"============================== FedMix Round {round_idx + 1}/{total_rounds} start =============================="
            )
            global_state = self.load_server_state()
            client_updates = []
            client_train_loss_list = []
            client_train_acc_list = []
            selected_ratio_list = []
            pseudo_confidence_list = []
            agreement_ratio_list = []
            pseudo_loss_list = []

            for client_id in self.clients_id_list:
                client = FedMixClient(
                    parse=parse,
                    client_id=client_id,
                    logger=self.logger,
                    c_T=round_idx,
                    model_factory=self.build_model,
                    global_state=global_state,
                    fedmix_config=self.fedmix_config,
                )
                client_state, num_samples, metrics = client.train()
                client_updates.append((client_state, num_samples, metrics))
                client_train_loss_list.append(float(metrics["train_loss"]))
                client_train_acc_list.append(float(metrics["train_acc"]))
                selected_ratio_list.append(float(metrics.get("fedmix_selected_ratio", 0.0)))
                pseudo_confidence_list.append(float(metrics.get("fedmix_pseudo_confidence", 0.0)))
                agreement_ratio_list.append(float(metrics.get("fedmix_agreement_ratio", 0.0)))
                pseudo_loss_list.append(float(metrics.get("fedmix_pseudo_loss", 0.0)))

            aggregated_state, round_data = self.fedmix_aggregator.aggregate_model_state(
                global_state=global_state,
                client_updates=client_updates,
                device=self.device,
            )
            self.model.load_state_dict(aggregated_state, strict=True)
            self.save_server_model()

            server_test_loss, server_test_acc = self.evaluate_server_test()
            self.final_eval_loss = float(server_test_loss)
            self.final_eval_acc = float(server_test_acc)
            self.final_round = int(round_idx + 1)
            self.min_eval_loss = min(float(self.min_eval_loss), float(server_test_loss))
            if float(server_test_acc) > float(self.best_server_test_acc):
                self.best_server_test_acc = float(server_test_acc)
                self.best_server_test_round = int(round_idx + 1)

            avg_client_train_loss = sum(client_train_loss_list) / max(len(client_train_loss_list), 1)
            avg_client_train_acc = sum(client_train_acc_list) / max(len(client_train_acc_list), 1)
            total_client_samples = sum(int(num_samples) for _, num_samples, _ in client_updates)
            round_metrics = dict(round_data["metrics"])
            log_row = {
                "round": int(round_idx + 1),
                "train_loss": float(avg_client_train_loss),
                "train_acc": float(avg_client_train_acc),
                "eval_loss": float(server_test_loss),
                "eval_acc": float(server_test_acc),
                "best_acc": float(self.best_server_test_acc),
                "fedmix_avg_selected_ratio": float(sum(selected_ratio_list) / max(len(selected_ratio_list), 1)),
                "fedmix_avg_pseudo_confidence": float(sum(pseudo_confidence_list) / max(len(pseudo_confidence_list), 1)),
                "fedmix_avg_agreement_ratio": float(sum(agreement_ratio_list) / max(len(agreement_ratio_list), 1)),
                "fedmix_avg_pseudo_loss": float(sum(pseudo_loss_list) / max(len(pseudo_loss_list), 1)),
                "num_selected_clients": int(len(self.clients_id_list)),
                "total_client_samples": int(total_client_samples),
                **round_metrics,
            }
            self._append_round_metrics(log_row)

            self.logger.info(
                f"--fedmix --round:{round_idx + 1} --beta:{float(self.fedmix_config.beta):.3f} "
                f"--lambda:{float(self.fedmix_config.lam):.3f} "
                f"--pseudo_thresh:{float(self.fedmix_config.pseudo_thresh):.3f}"
            )
            self.logger.info(
                f"--round:{round_idx + 1} --avg_client_train_loss:{avg_client_train_loss:.4f} "
                f"--avg_client_train_acc:{avg_client_train_acc:.4f} "
                f"--server_eval_loss:{server_test_loss:.4f} --server_eval_acc:{server_test_acc:.4f} "
                f"--selected_ratio:{log_row['fedmix_avg_selected_ratio']:.4f}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._write_run_summary()


def main():
    ensure_fedmix_args(parse)
    args = parse.parse_args()
    fedmix_config = resolve_fedmix_config(args)

    args.output_dir = fedmix_config.output_dir
    args.save_result = fedmix_config.output_dir
    args.model_save_path = fedmix_config.model_save_path
    args.aggregation_method = "fedavg"
    args.method = ""
    args.server_adapt_enabled = False

    os.makedirs(fedmix_config.output_dir, exist_ok=True)
    os.makedirs(fedmix_config.model_save_path, exist_ok=True)
    logger = _build_logger(fedmix_config.output_dir)
    _freeze_parser_with_namespace(parse, args)
    FedMixServer(parse=parse, logger=logger).train()
