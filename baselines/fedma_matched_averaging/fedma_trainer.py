from __future__ import annotations

import copy
import csv
import json
import logging
import os
from typing import Dict, List

import torch

from args import parse
from client import Client
from server import Server

from .fedma_aggregator import FedMAAggregator
from .fedma_config import ensure_fedma_args, resolve_fedma_config
from .fedma_matcher import FedMAMatcher


ROUND_METRIC_FIELDNAMES = [
    "round",
    "train_loss",
    "train_acc",
    "eval_loss",
    "eval_acc",
    "best_acc",
    "fedma_scope",
    "expert_signature",
    "matching_method",
    "similarity",
    "matching_solver",
    "fedma_avg_match_similarity",
    "fedma_min_match_similarity",
    "fedma_max_match_similarity",
    "fedma_matching_changed_ratio",
    "fedma_num_experts",
    "num_selected_clients",
    "total_client_samples",
    "num_moe_layers",
    "num_experts_per_layer",
    "matching_time_sec",
    "aggregation_time_sec",
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
    logger = logging.getLogger(f"fedma::{output_dir}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    file_handler = logging.FileHandler(os.path.join(logs_dir, "fedma.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class FedMAServer(Server):
    def __init__(self, parse, logger):
        super().__init__(parse=parse, logger=logger)
        self.fedma_config = resolve_fedma_config(self.args)
        self.matcher = FedMAMatcher(
            signature_type=self.fedma_config.expert_signature,
            similarity=self.fedma_config.similarity,
            matching_method=self.fedma_config.matching_method,
            keep_num_experts_fixed=self.fedma_config.keep_num_experts_fixed,
            device=self.fedma_config.device,
        )
        self.fedma_aggregator = FedMAAggregator(
            config=self.fedma_config,
            matcher=self.matcher,
            logger=self.logger,
        )
        self.round_metrics_path = os.path.join(self.fedma_config.output_dir, "logs", "round_metrics.csv")
        self.matching_dir = os.path.join(self.fedma_config.output_dir, "logs")
        self.similarity_dir = os.path.join(self.fedma_config.output_dir, "logs")
        self.run_summary_path = os.path.join(self.fedma_config.output_dir, "logs", "run_summary.txt")
        self.min_eval_loss = float("inf")
        self.final_eval_loss = float("inf")
        self.final_eval_acc = 0.0
        self.final_round = 0
        self._init_round_metrics_csv()
        self.logger.info(
            "FedMA-style matched averaging baseline selected: shared/common experiment settings are inherited "
            "from args.py; non-expert parameters stay on the original FedAvg path while MoE experts are "
            "matched layer-wise before averaging."
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

    def _select_client_ids(self, round_idx: int) -> List[int]:
        if self.fedma_config.clients_per_round >= len(self.clients_id_list):
            return list(self.clients_id_list)
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed) + int(round_idx) + 1)
        permutation = torch.randperm(len(self.clients_id_list), generator=generator).tolist()
        selected = [self.clients_id_list[index] for index in permutation[: self.fedma_config.clients_per_round]]
        return sorted(selected)

    def _save_matching_artifacts(self, round_idx: int, round_data: Dict[str, object]) -> None:
        if self.fedma_config.save_match_matrix:
            matching_path = os.path.join(self.matching_dir, f"matching_round_{int(round_idx)}.json")
            with open(matching_path, "w", encoding="utf-8") as handle:
                json.dump(round_data["matching_json"], handle, indent=2, ensure_ascii=False)
        if self.fedma_config.save_expert_similarity:
            similarity_path = os.path.join(self.similarity_dir, f"similarity_round_{int(round_idx)}.pt")
            torch.save(round_data["similarity_tensors"], similarity_path)

    def _write_run_summary(self) -> None:
        summary_text = (
            "method: FedMA-style matched expert averaging\n"
            f"scope: {self.fedma_config.fedma_scope}\n"
            f"signature: {self.fedma_config.expert_signature}\n"
            f"matching: {self.fedma_config.matching_method}\n"
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
        total_rounds = int(self.fedma_config.total_rounds)
        for round_idx in range(total_rounds):
            self.logger.info(
                f"============================== FedMA Round {round_idx + 1}/{total_rounds} start =============================="
            )
            selected_client_ids = self._select_client_ids(round_idx)
            global_state = self.load_server_state()
            client_updates = []
            client_train_loss_list = []
            client_train_acc_list = []

            for client_id in selected_client_ids:
                client = Client(
                    parse=parse,
                    client_id=client_id,
                    logger=self.logger,
                    c_T=round_idx,
                    model_factory=self.build_model,
                    global_state=global_state,
                )
                client_state, num_samples, metrics = client.train()
                client_updates.append((client_state, num_samples, metrics))
                client_train_loss_list.append(float(metrics["train_loss"]))
                client_train_acc_list.append(float(metrics["train_acc"]))

            aggregated_state, round_data = self.fedma_aggregator.aggregate_model_state(
                global_state=global_state,
                client_updates=client_updates,
                device=self.device,
                round_idx=round_idx + 1,
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
                "num_selected_clients": int(len(selected_client_ids)),
                "total_client_samples": int(total_client_samples),
                **round_metrics,
            }
            self._append_round_metrics(log_row)
            self._save_matching_artifacts(round_idx + 1, round_data)

            self.logger.info(
                f"--fedma --round:{round_idx + 1} --scope:{self.fedma_config.fedma_scope} "
                f"--signature:{self.fedma_config.expert_signature} --matching:{self.fedma_config.matching_method} "
                f"--matching_solver:{round_metrics.get('matching_solver', self.fedma_config.matching_method)}"
            )
            self.logger.info(
                f"--round:{round_idx + 1} --avg_client_train_loss:{avg_client_train_loss:.4f} "
                f"--avg_client_train_acc:{avg_client_train_acc:.4f} "
                f"--server_eval_loss:{server_test_loss:.4f} --server_eval_acc:{server_test_acc:.4f} "
                f"--match_avg:{float(round_metrics.get('fedma_avg_match_similarity', 0.0)):.4f}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._write_run_summary()


def main():
    ensure_fedma_args(parse)
    args = parse.parse_args()
    fedma_config = resolve_fedma_config(args)

    args.output_dir = fedma_config.output_dir
    args.save_result = fedma_config.output_dir
    args.model_save_path = fedma_config.model_save_path
    args.aggregation_method = "fedavg"
    args.method = ""
    args.server_adapt_enabled = False

    os.makedirs(fedma_config.output_dir, exist_ok=True)
    os.makedirs(fedma_config.model_save_path, exist_ok=True)
    logger = _build_logger(fedma_config.output_dir)
    _freeze_parser_with_namespace(parse, args)
    FedMAServer(parse=parse, logger=logger).train()
