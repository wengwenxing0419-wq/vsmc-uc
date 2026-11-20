#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from args import parse
import server as server_module
from aggregation import history_aware_vsmc_expert_aggregator as ha
from aggregation.vsmc_expert_aggregator import (
    _merge_with_reference_state,
    clone_detached_state_dict,
    split_state_dict_by_expert,
    weighted_average_state_dicts,
)


_ABALATION_METHODS = (
    "freq_weighted",
    "freq_vsmc",
    "multi_signal_weighted",
    "uc_weighted",
    "uc_vsmc",
)


def _load_base_module():
    module_path = REPO_DIR / "scripts" / "reliability_ablation_train.py"
    spec = importlib.util.spec_from_file_location("reliability_ablation_train_base_sc", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base ablation script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def _build_custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reliability_ablation_method",
        type=str,
        required=True,
        choices=_ABALATION_METHODS,
    )
    return parser


def _extract_custom_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    custom_parser = _build_custom_parser()
    custom_args, remaining = custom_parser.parse_known_args(list(argv))
    return custom_args, remaining


def _aggregate_freq_weighted_model_state_sc(
    global_state,
    client_updates,
    num_experts: int,
    device: torch.device,
    round_idx: int,
):
    del num_experts
    shared_weights = ha._client_weights_from_mode(client_updates, "sample_count")
    expert_weights = ha._client_weights_from_mode(client_updates, "uniform")
    fedavg_state = weighted_average_state_dicts(
        [state_dict for state_dict, _, _ in client_updates],
        shared_weights,
        fallback_state=global_state,
    )
    _, global_expert_groups = split_state_dict_by_expert(global_state, num_experts=None)
    split_client_groups = [
        split_state_dict_by_expert(state_dict, num_experts=None)[1]
        for state_dict, _, _ in client_updates
    ]
    merged_expert_groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    expert_summaries: List[Dict[str, Any]] = []

    max_block = 0
    max_expert = 0
    for _, _, metrics in client_updates:
        router_usage = metrics.get("router_usage")
        if torch.is_tensor(router_usage) and router_usage.dim() == 2:
            max_block = max(max_block, int(router_usage.shape[0]))
            max_expert = max(max_expert, int(router_usage.shape[1]))

    for expert_scope, global_group in global_expert_groups.items():
        block_expert = ha._parse_block_expert_from_scope(expert_scope)
        if block_expert is None:
            continue
        block_id, expert_id = block_expert
        client_group_states = []
        client_metrics = []
        for (_, _, metrics), client_group_map in zip(client_updates, split_client_groups):
            group = client_group_map.get(expert_scope)
            if group is None:
                client_group_states.append(clone_detached_state_dict(global_group["state_dict"]))
            else:
                client_group_states.append(group["state_dict"])
            client_metrics.append(metrics or {})
        aggregated_expert_state, info, _ = BASE._aggregate_freq_weighted_expert(
            global_expert_state=global_group["state_dict"],
            client_expert_states=client_group_states,
            client_metrics=client_metrics,
            expert_weights=expert_weights,
            block_id=block_id,
            expert_id=expert_id,
            expert_scope=expert_scope,
            round_idx=round_idx,
            num_blocks=max(max_block, block_id + 1),
            num_experts=max(max_expert, expert_id + 1),
            device=device,
        )
        merged_expert_groups[expert_scope] = {
            "expert_id": int(expert_id),
            "expert_scope": expert_scope,
            "state_dict": aggregated_expert_state,
        }
        expert_summaries.append(info)

    new_global_state = _merge_with_reference_state(
        fedavg_state,
        ha._merge_expert_groups(merged_expert_groups),
        device=None,
    )
    round_logs = {
        "round": int(round_idx),
        "anchor_mode": "none",
        "expert_summaries": expert_summaries,
        "particle_summaries": [],
        "mean_ess_ratio": 0.0,
        "fallback_count": int(sum(1 for info in expert_summaries if bool(info.get("fallback_used", False)))),
    }
    return new_global_state, round_logs


class ReliabilityAblationSCNonExpertAggregator:
    def __init__(self, config: ha.HAVSMCETFConfig, mode: str, logger=None):
        self.config = config
        self.mode = mode
        self.logger = logger
        self.state: Optional[ha.HAVSMCETFState] = None

    def aggregate_model_state(
        self,
        global_model,
        global_state,
        client_updates,
        device: torch.device,
        round_idx: int,
        probe_batch=None,
    ):
        if self.mode == "freq_weighted":
            return _aggregate_freq_weighted_model_state_sc(
                global_state=global_state,
                client_updates=client_updates,
                num_experts=int(getattr(global_model, "switch_num_experts", 0) or 0),
                device=device,
                round_idx=round_idx,
            )

        if (
            self.state is None
            or int(self.state.num_clients) != len(client_updates)
            or int(self.state.num_particles) != int(self.config.num_particles)
        ):
            self.state = ha.build_ha_vsmc_etf_state(
                global_model=global_model,
                global_state=global_state,
                num_clients=len(client_updates),
                config=self.config,
                device=device,
            )
        self.config.current_round_idx = int(round_idx)
        aggregated_state, updated_state, round_logs = BASE._aggregate_mode_with_state(
            mode=self.mode,
            global_model=global_model,
            client_updates=client_updates,
            state=self.state,
            config=self.config,
            probe_batch=probe_batch,
        )
        self.state = updated_state
        return aggregated_state, round_logs


def _install_monkey_patch(mode: str) -> None:
    class PatchedReliabilityAggregator(ReliabilityAblationSCNonExpertAggregator):
        def __init__(self, config: ha.HAVSMCETFConfig, logger=None):
            super().__init__(config=config, mode=mode, logger=logger)

    server_module.HAVSMCETFExpertAggregator = PatchedReliabilityAggregator
    ha.HAVSMCETFExpertAggregator = PatchedReliabilityAggregator


def _build_logger(args_namespace) -> logging.Logger:
    data = args_namespace.data_name
    num_client = args_namespace.num_clients
    alpha = args_namespace.alpha
    backbone = args_namespace.backbone_name
    logger_name = args_namespace.save_result + "/logs/" + f"data_{data} client_{num_client} alpha_{alpha} backbone_{backbone}"
    os.makedirs(args_namespace.model_save_path, exist_ok=True)
    os.makedirs(args_namespace.data_save_path, exist_ok=True)
    os.makedirs(args_namespace.save_result + "/logs", exist_ok=True)
    os.makedirs(args_namespace.save_result + "/detail", exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    file_handler = logging.FileHandler(f"{logger_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    custom_args, remaining = _extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    warnings.filterwarnings("ignore")
    _install_monkey_patch(custom_args.reliability_ablation_method)
    original_build_config = server_module.Server.build_ha_vsmc_etf_config

    def patched_build_config(self):
        config = original_build_config(self)
        config.shared_weight_mode = "sample_count"
        return config

    server_module.Server.build_ha_vsmc_etf_config = patched_build_config
    args_namespace = parse.parse_args()
    logger = _build_logger(args_namespace)
    server_module.Server(parse=parse, logger=logger).train()


if __name__ == "__main__":
    main()
