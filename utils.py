import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import parse


CLIENT_RESULT_FIELDNAMES = [
    "T",
    "client_epoch",
    "client_id",
    "client_train_loss",
    "client_train_acc",
]

VSMC_GROUP_DETAIL_FIELDNAMES = [
    "round",
    "expert_id",
    "expert_scope",
    "num_particles",
    "log_f_mean",
    "log_f_min",
    "log_f_max",
    "log_g_mean",
    "log_g_min",
    "log_g_max",
    "log_r_mean",
    "log_r_min",
    "log_r_max",
    "log_q_mean",
    "proposal_loss",
    "mean_a",
    "std_a",
    "mean_b",
    "std_b",
    "latent_coeff0_mean",
    "latent_coeff0_std",
    "latent_coeff1_mean",
    "latent_coeff1_std",
    "val_reward_mean",
    "val_reward_max",
    "log_w_min",
    "log_w_max",
    "log_w_mean",
    "min_alpha",
    "max_alpha",
    "mean_alpha",
    "ess",
    "normalized_ess",
    "logZ_increment",
    "logZ_total",
    "loss_mean",
    "loss_min",
    "loss_max",
    "server_val_loss_mean",
    "server_val_base_loss",
    "server_val_replaced_loss_mean",
    "server_val_reward_mean",
    "server_val_reward_std",
    "delta_reward_mean",
    "delta_reward_std",
    "credit_mean",
    "credit_std",
    "novelty_mean",
    "stability_mean",
    "usage_mean",
    "route_count_sum",
    "alpha_g_server_ess",
    "alpha_r_ess",
    "alpha_final_ess",
    "delta_norm_mean",
    "delta_norm_max",
    "fedavg_delta_norm",
    "expert_server_lr",
    "expert_ema_rho",
    "router_usage_entropy",
    "router_usage_min",
    "router_usage_max",
    "fallback_used",
    "fallback_reason",
    "resampled",
    "feature_min",
    "feature_max",
    "feature_mean",
    "feature_has_nan",
    "particle_delta_norm_mean",
    "particle_delta_mse_mean",
    "fedavg_delta_mse",
    "particle_delta_norm_min",
    "particle_delta_norm_max",
    "particle_to_prev_mse_mean",
    "particle_to_fedavg_mse_mean",
    "particle_to_fedavg_mse_min",
    "particle_to_fedavg_mse_max",
    "proposal_grad_norm",
    "proposal_param_delta_norm",
    "logZ_requires_grad",
    "log_f_requires_grad",
    "log_g_requires_grad",
    "log_q_requires_grad",
    "raw_reward_mean",
    "raw_reward_std",
    "raw_reward_min",
    "raw_reward_max",
    "normalized_reward_mean",
    "normalized_reward_std",
    "log_g_std",
    "log_g_min",
    "log_g_max",
    "server_val_particle_loss_mean",
    "server_val_particle_loss_min",
    "server_val_particle_loss_max",
    "particle_requires_grad",
    "differentiable_g",
    "normalized_reward_min",
    "normalized_reward_max",
    "base_correct_confidence",
    "particle_correct_confidence_mean",
    "particle_correct_confidence_min",
    "particle_correct_confidence_max",
    "confidence_reward_mean",
    "confidence_reward_std",
    "reward_zero_std",
    "log_w_std",
    "alpha_entropy",
    "alpha_uniform_l1",
    "alpha_max_before_clip",
    "alpha_max_after_clip",
    "alpha_mixed_with_uniform",
    "bad_alpha_fallback",
    "any_particle_nan",
    "any_particle_inf",
    "any_aggregated_expert_nan",
    "any_global_nan",
    "delta_div_norm",
    "delta_div_enabled",
    "learned_vsmc_logq_coef",
    "log_f_contribution_mean",
    "log_g_contribution_mean",
    "log_q_contribution_mean",
    "log_q_contribution_abs_mean",
    "log_q_dominates",
    "fedavg_particle_alpha",
    "fedavg_particle_val_reward",
    "fedavg_particle_raw_reward",
    "fedavg_particle_normalized_reward",
    "fedavg_particle_log_g",
    "best_particle_index",
    "best_particle_is_fedavg",
]


def _get_csv_path():
    args, _ = parse.parse_known_args()
    data, num_client, alpha, backbone = args.data_name, args.num_clients, args.alpha, args.backbone_name
    return args.save_result + "/detail/" + f"data_{data} client_{num_client} alpha_{alpha} backbone_{backbone}.csv"


def _get_vsmc_round_csv_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "vsmc_aggregation_round_metrics.csv")


def _get_vsmc_round_summary_csv_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "vsmc_round_summary.csv")


def _get_vsmc_group_detail_log_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "vsmc_group_details.log")


def _get_vsmc_run_summary_txt_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "run_summary.txt")


def _get_server_adapt_metrics_csv_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "server_adapt_metrics.csv")


def _get_ha_vsmc_etf_round_summary_csv_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "ha_vsmc_etf_round_summary.csv")


def _get_ha_vsmc_etf_expert_summary_csv_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "ha_vsmc_etf_expert_summary.csv")


def _get_ha_vsmc_etf_particle_summary_csv_path():
    args, _ = parse.parse_known_args()
    return os.path.join(args.save_result, "logs", "ha_vsmc_etf_particle_summary.csv")


VSMC_ROUND_SUMMARY_FIELDNAMES = [
    "round",
    "method",
    "mode",
    "num_groups",
    "group_level",
    "avg_client_train_loss",
    "avg_client_train_acc",
    "server_test_loss",
    "server_test_acc",
    "best_acc_so_far",
    "best_acc_round",
    "mean_normalized_ess",
    "differentiable_g_group_ratio",
    "reward_zero_std_count",
    "reward_std_mean",
    "log_g_std_mean",
    "log_g_effective_group_ratio",
    "log_q_dominates_ratio",
    "best_particle_is_fedavg_rate",
    "fedavg_particle_alpha_mean",
    "proposal_grad_norm_mean",
    "proposal_param_delta_norm_mean",
    "fallback_count",
    "used_reinforce_count",
    "phase",
]

SERVER_ADAPT_METRICS_FIELDNAMES = [
    "round",
    "server_adapt_version",
    "adapt_attempted",
    "adapt_accepted",
    "adapt_accept_reason",
    "accept_reason",
    "adapt_phase",
    "split_accept",
    "train_batches",
    "accept_batches",
    "strict_accept",
    "selected_param_count",
    "selected_param_names",
    "train_loss_before",
    "train_loss_after",
    "train_acc_before",
    "train_acc_after",
    "train_margin_before",
    "train_margin_after",
    "accept_loss_before",
    "accept_loss_after",
    "accept_acc_before",
    "accept_acc_after",
    "accept_margin_before",
    "accept_margin_after",
    "loss_delta_accept",
    "margin_delta_accept",
    "acc_delta_accept",
    "router_entropy_delta_accept",
    "loss_improved",
    "acc_improved",
    "margin_improved",
    "margin_not_worse",
    "loss_not_worse",
    "loss_evidence",
    "margin_evidence",
    "acc_evidence",
    "router_evidence",
    "evidence_count",
    "evidence_required",
    "strong_accept",
    "acc_accept",
    "late_strict_accept",
    "late_phase",
    "late_insufficient_improvement",
    "effective_min_delta",
    "effective_margin_delta",
    "accept_threshold_multiplier",
    "accept_rate_window",
    "accept_rate_window_size",
    "router_bias_enabled",
    "router_temperature_enabled",
    "full_router_enabled",
    "router_aux_available",
    "router_load_entropy_before",
    "router_load_entropy_after",
    "near_best",
    "near_best_reject",
    "cooldown_active",
    "cooldown_remaining",
    "cooldown_skip",
    "rescue_adapt",
    "adapt_delta_norm_raw",
    "adapt_delta_norm_ema",
    "adapt_param_delta_norm",
    "ema_applied",
    "rollback_reason",
    "adapt_loss_before",
    "adapt_loss_after",
    "adapt_margin_before",
    "adapt_margin_after",
    "teacher_router_enabled",
    "teacher_router_loss_before",
    "teacher_router_loss_after",
    "teacher_router_weight",
    "teacher_label_entropy_mean",
    "teacher_conf_mean",
    "teacher_conf_std",
    "teacher_valid_ratio",
    "router_teacher_acc_before",
    "router_teacher_acc_after",
    "teacher_proxy_mode",
    "counterfactual_success",
    "counterfactual_fallback_reason",
    "progress",
    "error_info",
]

HA_VSMC_ETF_ROUND_SUMMARY_FIELDNAMES = [
    "round",
    "anchor_mode",
    "pure_vsmc",
    "fedavg_blend_enabled",
    "expert_output",
    "function_drift_enabled",
    "train_loss",
    "eval_loss",
    "eval_acc",
    "best_acc",
    "final_acc",
    "num_experts_logged",
    "fallback_count",
    "mean_lambda_expert",
    "mean_ESS_ratio",
    "mean_particle_weight_entropy",
    "mean_client_weight_entropy",
    "mean_expert_weight_entropy",
    "mean_vsmc_fedavg_l2_distance",
    "mean_final_residual_ratio",
    "mean_max_particle_weight",
    "mean_max_client_weight",
    "mean_min_client_weight",
    "mean_active_client_count_per_expert",
]

HA_VSMC_ETF_EXPERT_SUMMARY_FIELDNAMES = [
    "round",
    "block_id",
    "expert_id",
    "expert_scope",
    "anchor_mode",
    "pure_vsmc",
    "fedavg_blend_enabled",
    "expert_output",
    "function_drift_enabled",
    "lambda_expert",
    "temperature",
    "ESS",
    "ESS_ratio",
    "resampled",
    "particle_weight_entropy",
    "particle_max_weight",
    "max_particle_weight",
    "client_weight_entropy",
    "expert_weight_entropy",
    "client_max_weight",
    "client_min_weight",
    "max_client_weight_mean",
    "min_client_weight_mean",
    "active_client_count_per_expert",
    "usage_score_mean",
    "usage_score_std",
    "history_cos_mean",
    "history_cos_std",
    "consensus_cos_mean",
    "consensus_cos_std",
    "normalized_drift_mean",
    "normalized_drift_std",
    "function_drift_mean",
    "function_drift_std",
    "usage_signal_mean",
    "usage_signal_std",
    "history_signal_mean",
    "history_signal_std",
    "consensus_signal_mean",
    "consensus_signal_std",
    "drift_signal_mean",
    "drift_signal_std",
    "function_signal_mean",
    "function_signal_std",
    "observation_score_mean",
    "observation_score_std",
    "reliability_score_mean",
    "reliability_score_std",
    "particle_score_mean",
    "particle_score_std",
    "vsmc_fedavg_l2_distance",
    "final_fedavg_l2_distance",
    "final_residual_ratio",
    "velocity_norm",
    "second_moment_norm",
    "function_drift_available",
    "fallback_used",
    "fallback_reason",
    "fallback_type",
    "confidence",
    "score_transform",
]

HA_VSMC_ETF_PARTICLE_SUMMARY_FIELDNAMES = [
    "round",
    "block_id",
    "expert_id",
    "particle_id",
    "particle_weight",
    "particle_score",
    "particle_entropy",
    "particle_max_client_weight",
    "particle_min_client_weight",
    "dominant_client_idx",
    "resampled",
]


class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction="batchmean"):
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, logits1, logits2):
        log_probs1 = F.log_softmax(logits1 / self.temperature, dim=1)
        probs2 = F.softmax(logits2 / self.temperature, dim=1)
        kl_div = F.kl_div(log_probs1, probs2 + 1e-7, reduction=self.reduction)
        kl_div *= self.temperature ** 2
        return kl_div


def init_result_csv():
    csv_path = _get_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CLIENT_RESULT_FIELDNAMES)
        writer.writeheader()


def record_result(record_dic: dict):
    csv_path = _get_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CLIENT_RESULT_FIELDNAMES)
        writer.writerow(record_dic)


def init_vsmc_round_metrics_csv():
    detail_csv_path = _get_vsmc_round_csv_path()
    summary_csv_path = _get_vsmc_round_summary_csv_path()
    group_log_path = _get_vsmc_group_detail_log_path()
    for csv_path, fieldnames in [
        (detail_csv_path, VSMC_GROUP_DETAIL_FIELDNAMES),
        (summary_csv_path, VSMC_ROUND_SUMMARY_FIELDNAMES),
    ]:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    os.makedirs(os.path.dirname(group_log_path), exist_ok=True)
    with open(group_log_path, "w") as logfile:
        logfile.write("")


def record_vsmc_round_metric(record_dic: dict):
    csv_path = _get_vsmc_round_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=VSMC_GROUP_DETAIL_FIELDNAMES)
        writer.writerow(record_dic)


def record_vsmc_round_summary(record_dic: dict):
    csv_path = _get_vsmc_round_summary_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=VSMC_ROUND_SUMMARY_FIELDNAMES)
        writer.writerow(record_dic)


def append_vsmc_group_detail_log(line: str):
    log_path = _get_vsmc_group_detail_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as logfile:
        logfile.write(line.rstrip("\n") + "\n")


def write_vsmc_run_summary(text: str):
    txt_path = _get_vsmc_run_summary_txt_path()
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as txtfile:
        txtfile.write(text)


def init_server_adapt_metrics_csv():
    csv_path = _get_server_adapt_metrics_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SERVER_ADAPT_METRICS_FIELDNAMES)
        writer.writeheader()


def record_server_adapt_metric(record_dic: dict):
    csv_path = _get_server_adapt_metrics_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SERVER_ADAPT_METRICS_FIELDNAMES)
        writer.writerow(record_dic)


def init_ha_vsmc_etf_metrics_csv():
    for csv_path, fieldnames in [
        (_get_ha_vsmc_etf_round_summary_csv_path(), HA_VSMC_ETF_ROUND_SUMMARY_FIELDNAMES),
        (_get_ha_vsmc_etf_expert_summary_csv_path(), HA_VSMC_ETF_EXPERT_SUMMARY_FIELDNAMES),
        (_get_ha_vsmc_etf_particle_summary_csv_path(), HA_VSMC_ETF_PARTICLE_SUMMARY_FIELDNAMES),
    ]:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


def record_ha_vsmc_etf_round_summary(record_dic: dict):
    csv_path = _get_ha_vsmc_etf_round_summary_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HA_VSMC_ETF_ROUND_SUMMARY_FIELDNAMES)
        writer.writerow(record_dic)


def record_ha_vsmc_etf_expert_summary(record_dic: dict):
    csv_path = _get_ha_vsmc_etf_expert_summary_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HA_VSMC_ETF_EXPERT_SUMMARY_FIELDNAMES)
        writer.writerow(record_dic)


def record_ha_vsmc_etf_particle_summary(record_dic: dict):
    csv_path = _get_ha_vsmc_etf_particle_summary_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HA_VSMC_ETF_PARTICLE_SUMMARY_FIELDNAMES)
        writer.writerow(record_dic)
