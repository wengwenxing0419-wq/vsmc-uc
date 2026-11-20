#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from args import parse
import server as server_module
from aggregation import history_aware_vsmc_expert_aggregator as ha


def _load_base_module():
    module_path = REPO_DIR / "scripts" / "freq_vsmc_synced_train.py"
    spec = importlib.util.spec_from_file_location("freq_vsmc_synced_train_sc_base", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base synced script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def _install_monkey_patches(custom_args) -> None:
    server_module.HAVSMCETFExpertAggregator = BASE.FreqVSMCSyncedAggregator
    ha.HAVSMCETFExpertAggregator = BASE.FreqVSMCSyncedAggregator

    original_build_config = server_module.Server.build_ha_vsmc_etf_config
    original_record_metrics = server_module.Server.record_ha_vsmc_etf_round_metrics

    def patched_build_config(self):
        config = original_build_config(self)
        config.anchor_mode = "none"
        config.shared_weight_mode = "sample_count"
        config.expert_weight_mode = "uniform"
        config.alpha_history_direction = 0.0
        config.alpha_consensus = 0.0
        config.alpha_drift = 0.0
        config.alpha_function_drift = 0.0
        config.use_function_drift = False
        config.prediction_noise_std = 0.0
        config.trust_update_lr = 0.0
        config.reliability_momentum_lr = 0.0
        config.temperature_init = float(custom_args.freq_vsmc_synced_tau)
        config.temperature_min = float(custom_args.freq_vsmc_synced_tau)
        config.temperature_max = float(custom_args.freq_vsmc_synced_tau)
        config.synced_sigma = float(custom_args.freq_vsmc_synced_sigma)
        config.synced_rho = float(custom_args.freq_vsmc_synced_rho)
        config.synced_tau = float(custom_args.freq_vsmc_synced_tau)
        config.synced_eps = float(custom_args.freq_vsmc_synced_eps)
        return config

    def patched_record_metrics(
        self,
        round_idx,
        avg_client_train_loss,
        avg_client_train_acc,
        server_test_loss,
        server_test_acc,
    ):
        original_record_metrics(
            self,
            round_idx,
            avg_client_train_loss,
            avg_client_train_acc,
            server_test_loss,
            server_test_acc,
        )
        if int(round_idx) == int(self.server_epochs):
            BASE._write_last_round_analysis(
                server_instance=self,
                round_idx=int(round_idx),
                server_test_acc=float(server_test_acc),
            )

    server_module.Server.build_ha_vsmc_etf_config = patched_build_config
    server_module.Server.record_ha_vsmc_etf_round_metrics = patched_record_metrics


def main() -> None:
    custom_args, remaining = BASE._extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    warnings.filterwarnings("ignore")
    _install_monkey_patches(custom_args)
    args_namespace = parse.parse_args()
    logger = BASE._build_logger(args_namespace)
    server_module.Server(parse=parse, logger=logger).train()
    summary_path = Path(args_namespace.save_result) / "logs" / BASE._SUMMARY_JSON
    if summary_path.exists():
        BASE._print_final_summary(args_namespace.save_result)


if __name__ == "__main__":
    main()
