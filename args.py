import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def apply_plain_fedavg_baseline_overrides(args):
    if not bool(getattr(args, "use_plain_fedavg_baseline", False)):
        return args

    args.aggregation_method = "fedavg"
    args.fedavg_weight_mode = "uniform"
    args.expert_weight_mode = "uniform"
    args.client_optimizer = "sgd"
    args.client_momentum = 0.0
    args.grad_clip_norm = 0.0
    args.use_client_specialization = False
    args.aux_loss_weight = 0.0
    args.client_shared_prox_beta = 0.0
    args.client_router_prox_beta = 0.0
    args.client_classifier_prox_beta = 0.0
    args.client_expert_prox_beta = 0.0
    args.client_expert_diversity_beta = 0.0
    args.client_router_consistency_beta = 0.0
    args.client_router_entropy_beta = 0.0
    args.client_router_balance_beta = 0.0
    args.server_adapt_enabled = False
    return args


parse = argparse.ArgumentParser(description="system")

###################################### data settings ################################################
parse.add_argument("--data_name",type=str,default="cifar10",choices=['cifar10', 'cifar100', 'cinic10', 'tinyimagenet'])
parse.add_argument("--data_path",type=str,default="./data")
parse.add_argument("--train_ratio",type=float,default=0.75)
parse.add_argument("--val_ratio",type=float,default=0.1)
parse.add_argument("--test_ratio",type=float,default=0.15)
parse.add_argument("--data_save_path",type=str,default="./save/data")
parse.add_argument("--batch_size",type=int,default=32)
parse.add_argument("--min_datasize",type=int,default=32)
parse.add_argument("--alpha",type=float,default=0.1)


###################################### base settings ################################################
parse.add_argument("--num_clients",type=int,default=4)
parse.add_argument("--server_epochs",type=int,default=100)
parse.add_argument("--client_epochs",type=int,default=1)
parse.add_argument("--device",type=str,default="cuda")
parse.add_argument("--save_result",type=str,default="./save/result")
parse.add_argument("--seed",type=int,default=1)


###################################### model settings ################################################
parse.add_argument("--topK",type=int,default=2)
parse.add_argument("--num_experts",type=int,default=3)
parse.add_argument("--dropout",type=float,default=0.2)
parse.add_argument("--learning_rate",type=float,default=5e-4)
parse.add_argument("--client_optimizer",type=str,default="adam",choices=["adam","adamw","sgd"])
parse.add_argument("--client_weight_decay",type=float,default=0.0)
parse.add_argument("--client_momentum",type=float,default=0.0)
parse.add_argument("--client_nesterov",type=str2bool,default=False)
parse.add_argument("--label_smoothing",type=float,default=0.0)
parse.add_argument("--grad_clip_norm",type=float,default=0.0)
parse.add_argument("--use_client_aug",type=str2bool,default=False)
parse.add_argument("--aug_random_crop_pad",type=int,default=4)
parse.add_argument("--aug_horizontal_flip",type=float,default=0.5)
parse.add_argument("--aug_random_erasing",type=float,default=0.0)
parse.add_argument("--mixup_alpha",type=float,default=0.0)
parse.add_argument("--client_lr_schedule",type=str,default="none",choices=["none","cosine"])
parse.add_argument("--client_min_lr_ratio",type=float,default=0.1)
parse.add_argument("--client_lr_warmup_rounds",type=int,default=0)
parse.add_argument("--use_client_specialization",type=str2bool,default=False)
parse.add_argument("--use_plain_fedavg_baseline",type=str2bool,default=False)
parse.add_argument("--client_shared_lr_scale",type=float,default=0.5)
parse.add_argument("--client_router_lr_scale",type=float,default=0.75)
parse.add_argument("--client_expert_lr_scale",type=float,default=1.5)
parse.add_argument("--client_classifier_lr_scale",type=float,default=1.0)
parse.add_argument("--client_shared_weight_decay",type=float,default=None)
parse.add_argument("--client_router_weight_decay",type=float,default=None)
parse.add_argument("--client_expert_weight_decay",type=float,default=None)
parse.add_argument("--client_classifier_weight_decay",type=float,default=None)
parse.add_argument("--client_shared_prox_beta",type=float,default=0.0)
parse.add_argument("--client_router_prox_beta",type=float,default=0.0)
parse.add_argument("--client_classifier_prox_beta",type=float,default=0.0)
parse.add_argument("--client_expert_prox_beta",type=float,default=0.0)
parse.add_argument("--client_expert_diversity_beta",type=float,default=0.0)
parse.add_argument("--client_router_consistency_beta",type=float,default=0.0)
parse.add_argument("--client_router_entropy_beta",type=float,default=0.0)
parse.add_argument("--client_router_balance_beta",type=float,default=0.0)
parse.add_argument("--client_consistency_temperature",type=float,default=1.0)
parse.add_argument("--out_dim",type=int,default=100)
parse.add_argument("--hidden_dim",type=int,default=128)
parse.add_argument("--model_save_path",type=str,default="./save/model")
parse.add_argument("--model_name",type=str,default="switch_transformer",choices=["switch_transformer"])
parse.add_argument(
    "--backbone_name",
    type=str,
    default="resnet18",
    choices=[
        "patch_embed",
        "patch_embed_switch",
        "patch_tokenizer_switch",
        "simplecnn",
        "vit_tiny",
        "vit_small",
        "vit_tiny_real",
        "resnet18",
        "resnet34",
        "resnet50",
    ],
)
parse.add_argument(
    "--resnet_stem_mode",
    type=str,
    default="auto",
    choices=["auto", "cifar", "imagenet"],
)
parse.add_argument("--image_size",type=int,default=32)
parse.add_argument("--patch_size",type=int,default=4)
parse.add_argument("--switch_layers",type=int,default=2)
parse.add_argument("--switch_heads",type=int,default=4)
parse.add_argument("--switch_ffn_dim",type=int,default=256)
parse.add_argument("--switch_num_experts",type=int,default=4)
parse.add_argument("--vit_real_embed_dim",type=int,default=192)
parse.add_argument("--vit_real_depth",type=int,default=6)
parse.add_argument("--vit_real_num_heads",type=int,default=3)
parse.add_argument("--vit_real_mlp_ratio",type=float,default=4.0)
parse.add_argument("--vit_real_dropout",type=float,default=0.1)
parse.add_argument("--vit_real_attention_dropout",type=float,default=0.0)
parse.add_argument("--vit_real_use_cls_token",type=str2bool,default=True)
parse.add_argument("--vit_real_use_learnable_pos_embed",type=str2bool,default=True)
parse.add_argument("--aux_loss_weight",type=float,default=0.0)
parse.add_argument("--method",type=str,default="")
parse.add_argument(
    "--aggregation_method",
    "--aggregation",
    "--algorithm",
    dest="aggregation_method",
    type=str,
    default="fedavg",
    choices=[
        "fedavg",
        "fedavg_expert_only",
        "fedavg_shared_only",
        "vsmc_expert",
        "beta_precision_fedtra",
        "joint_vsmc_expert",
        "temporal_vsmc_expert",
        "temporal_latent_vsmc_expert",
        "learned_vsmc",
        "learned_vsmc_moe_state_policy_k8",
        "learned_vsmc_moe_v11_server_adapt",
        "ha_vsmc_etf",
        "history_aware_vsmc_expert_trust_filter",
        "fedmoe_da",
        "fedjets",
        "pfedmoe",
        "hfedmoe",
    ],
)
parse.add_argument("--vsmc_epa_version",type=str,default="")
parse.add_argument("--learned_vsmc_moe_state_policy",type=str2bool,default=False)
parse.add_argument(
    "--fedavg_weight_mode",
    type=str,
    default="uniform",
    choices=["sample_count", "uniform"],
)
parse.add_argument(
    "--expert_weight_mode",
    type=str,
    default="uniform",
    choices=["sample_count", "uniform"],
)
parse.add_argument("--fedmoe_da_top_p",type=int,default=3)
parse.add_argument("--fedmoe_da_temperature",type=float,default=1.0)
parse.add_argument("--fedmoe_da_personalized_mix",type=float,default=1.0)
parse.add_argument(
    "--fedmoe_da_similarity_metric",
    type=str,
    default="router_row_cosine",
    choices=["router_row_cosine", "expert_cosine"],
)
parse.add_argument("--fedmoe_da_min_similarity",type=float,default=0.0)
parse.add_argument("--fedjets_enabled",type=str2bool,default=False)
parse.add_argument("--fedjets_num_experts",type=int,default=5)
parse.add_argument("--fedjets_top_k",type=int,default=2)
parse.add_argument("--fedjets_gate_hidden_dim",type=int,default=512)
parse.add_argument("--fedjets_num_anchor_clients",type=int,default=5)
parse.add_argument("--fedjets_anchor_clients_per_round",type=int,default=5)
parse.add_argument("--fedjets_normal_clients_per_round",type=int,default=5)
parse.add_argument("--fedjets_common_expert_ckpt",type=str,default="")
parse.add_argument("--fedjets_cache_embeddings",type=str2bool,default=True)
parse.add_argument("--fedjets_gate_lr",type=float,default=-1.0)
parse.add_argument("--fedjets_expert_lr",type=float,default=-1.0)
parse.add_argument("--pfedmoe_enabled",type=str2bool,default=False)
parse.add_argument("--pfedmoe_gate_hidden_dim",type=int,default=512)
parse.add_argument("--pfedmoe_client_selection_ratio",type=float,default=1.0)
parse.add_argument("--pfedmoe_aggregation_method",type=str,default="weighted_avg")
parse.add_argument(
    "--pfedmoe_global_backbone_name",
    type=str,
    default="cnn_small",
    choices=["patch_embed", "cnn_small", "resnet8", "resnet10", "mobilenet_tiny", "resnet18", "resnet34", "resnet50"],
)
parse.add_argument("--pfedmoe_global_switch_layers",type=int,default=1)
parse.add_argument("--pfedmoe_local_backbones",type=str,default="")
parse.add_argument("--pfedmoe_eval_fallback_to_server_test",type=str2bool,default=True)
parse.add_argument(
    "--pfedmoe_server_test_mode",
    type=str,
    default="ensemble_clients",
    choices=["ensemble_clients", "global_only", "mean_gate", "best_global_mix"],
)
parse.add_argument("--pfedmoe_global_aux_loss",type=str2bool,default=True)
parse.add_argument("--pfedmoe_global_aux_lambda",type=float,default=0.5)
parse.add_argument("--pfedmoe_aggregate_global_aux_head",type=str2bool,default=True)
parse.add_argument("--pfedmoe_alpha_global_target",type=float,default=0.5)
parse.add_argument("--pfedmoe_alpha_global_reg_lambda",type=float,default=0.1)
parse.add_argument("--pfedmoe_gate_warmup_rounds",type=int,default=5)
parse.add_argument("--pfedmoe_gate_warmup_mode",type=str,default="fixed_uniform",choices=["fixed_uniform", "learned"])
parse.add_argument("--pfedmoe_freeze_gate_during_warmup",type=str2bool,default=True)
parse.add_argument("--pfedmoe_gate_entropy_lambda",type=float,default=0.01)
parse.add_argument("--pfedmoe_gate_entropy_eps",type=float,default=1e-8)
parse.add_argument("--pfedmoe_label_smoothing",type=float,default=0.1)
parse.add_argument("--pfedmoe_weight_decay",type=float,default=5e-4)
parse.add_argument("--pfedmoe_use_val_for_checkpoint",type=str2bool,default=True)
parse.add_argument("--pfedmoe_best_metric",type=str,default="val_acc",choices=["val_acc", "server_test_acc"])
parse.add_argument("--pfedmoe_log_personalized_client_test",type=str2bool,default=True)
parse.add_argument("--pfedmoe_log_global_only_acc",type=str2bool,default=True)
parse.add_argument("--pfedmoe_best_global_mix_alpha",type=float,default=0.5)
parse.add_argument("--hfedmoe_enabled",type=str2bool,default=False)
parse.add_argument("--hfedmoe_num_experts",type=int,default=8)
parse.add_argument("--hfedmoe_top_k",type=int,default=2)
parse.add_argument(
    "--hfedmoe_shared_backbone_name",
    type=str,
    default="cnn_small",
    choices=["patch_embed", "cnn_small", "resnet8", "resnet10", "mobilenet_tiny", "resnet18", "resnet34", "resnet50"],
)
parse.add_argument("--hfedmoe_shared_switch_layers",type=int,default=1)
parse.add_argument("--hfedmoe_train_shared_backbone",type=str2bool,default=True)
parse.add_argument(
    "--hfedmoe_expert_backbone_name",
    type=str,
    default="resnet18",
    choices=["patch_embed", "cnn_small", "resnet8", "resnet10", "mobilenet_tiny", "resnet18", "resnet34", "resnet50"],
)
parse.add_argument("--hfedmoe_expert_hidden_dim",type=int,default=0)
parse.add_argument("--hfedmoe_gating_hidden_dim",type=int,default=256)
parse.add_argument("--hfedmoe_gate_entropy_lambda",type=float,default=0.01)
parse.add_argument("--hfedmoe_importance_metric",type=str,default="val_loss",choices=["val_loss"])
parse.add_argument("--hfedmoe_importance_max_batches",type=int,default=2)
parse.add_argument("--hfedmoe_importance_eps",type=float,default=1e-6)
parse.add_argument("--hfedmoe_resource_budget_mode",type=str2bool,default=True)
parse.add_argument("--hfedmoe_client_memory_budget_mb",type=float,default=2048.0)
parse.add_argument("--hfedmoe_client_flops_budget",type=float,default=2e11)
parse.add_argument("--hfedmoe_client_budget_scales",type=str,default="")
parse.add_argument("--hfedmoe_min_active_experts",type=int,default=1)
parse.add_argument("--hfedmoe_client_selection_ratio",type=float,default=1.0)
parse.add_argument(
    "--hfedmoe_aggregation_strategy",
    type=str,
    default="sparse_weighted",
    choices=["sparse_weighted", "sample_count", "uniform"],
)
parse.add_argument(
    "--hfedmoe_combine_method",
    type=str,
    default="gated",
    choices=["gated", "weighted_sum", "topk_average", "dynamic_routing"],
)
parse.add_argument("--ha_vsmc_enabled",type=str2bool,default=True)
parse.add_argument("--ha_vsmc_num_particles",type=int,default=8)
parse.add_argument("--ha_vsmc_state_level",type=str,default="block_expert")
parse.add_argument("--ha_vsmc_expert_velocity_beta",type=float,default=0.90)
parse.add_argument("--ha_vsmc_second_moment_beta",type=float,default=0.99)
parse.add_argument("--ha_vsmc_client_reliability_beta",type=float,default=0.90)
parse.add_argument("--ha_vsmc_usage_beta",type=float,default=0.90)
parse.add_argument("--ha_vsmc_temperature_init",type=float,default=1.0)
parse.add_argument("--ha_vsmc_temperature_min",type=float,default=0.5)
parse.add_argument("--ha_vsmc_temperature_max",type=float,default=2.0)
parse.add_argument("--ha_vsmc_lambda_max",type=float,default=0.20)
parse.add_argument("--ha_vsmc_lambda_warmup_rounds",type=int,default=10)
parse.add_argument("--ha_vsmc_anchor_mode",type=str,default="fedavg",choices=["fedavg", "none"])
parse.add_argument("--ha_vsmc_pure_vsmc",type=str2bool,default=False)
parse.add_argument("--ha_vsmc_score_transform",type=str,default="zscore",choices=["zscore", "zscore_clip", "rank_based"])
parse.add_argument("--ha_vsmc_score_clip_low",type=float,default=-2.0)
parse.add_argument("--ha_vsmc_score_clip_high",type=float,default=2.0)
parse.add_argument("--ha_vsmc_temperature_scale",type=float,default=1.0)
parse.add_argument("--ha_vsmc_prob_ema_rho",type=float,default=0.0)
parse.add_argument("--ha_vsmc_reliability_floor_eps",type=float,default=0.0)
parse.add_argument("--ha_vsmc_reliability_floor_source",type=str,default="none",choices=["none", "uniform_active", "usage"])
parse.add_argument("--ha_vsmc_top_k_clients",type=int,default=0)
parse.add_argument("--ha_vsmc_adaptive_temperature",type=str2bool,default=False)
parse.add_argument("--ha_vsmc_entropy_target",type=str,default="none",choices=["none", "moderate"])
parse.add_argument("--ha_vsmc_entropy_target_min_max_weight",type=float,default=0.25)
parse.add_argument("--ha_vsmc_entropy_target_max_max_weight",type=float,default=0.55)
parse.add_argument("--ha_vsmc_ess_threshold_ratio",type=float,default=0.50)
parse.add_argument("--ha_vsmc_trust_update_lr",type=float,default=0.20)
parse.add_argument("--ha_vsmc_reliability_momentum_lr",type=float,default=0.10)
parse.add_argument("--ha_vsmc_prediction_noise_std",type=float,default=0.03)
parse.add_argument("--ha_vsmc_alpha_usage",type=float,default=0.10)
parse.add_argument("--ha_vsmc_alpha_history_direction",type=float,default=0.25)
parse.add_argument("--ha_vsmc_alpha_consensus",type=float,default=0.25)
parse.add_argument("--ha_vsmc_alpha_drift",type=float,default=0.20)
parse.add_argument("--ha_vsmc_alpha_function_drift",type=float,default=0.20)
parse.add_argument("--ha_vsmc_collapse_penalty",type=float,default=0.05)
parse.add_argument("--ha_vsmc_probe_batch_size",type=int,default=128)
parse.add_argument("--ha_vsmc_use_function_drift",type=str2bool,default=True)
parse.add_argument("--ha_vsmc_function_drift_metric",type=str,default="cosine",choices=["cosine"])
parse.add_argument("--ha_vsmc_function_drift_interval",type=int,default=1)
parse.add_argument("--ha_vsmc_require_function_drift",type=str2bool,default=False)
parse.add_argument("--ha_vsmc_use_validation_loss",type=str2bool,default=False)
parse.add_argument("--ha_vsmc_normalize_observation_components",type=str2bool,default=True)
parse.add_argument("--ha_vsmc_min_router_usage",type=float,default=1e-6)
parse.add_argument("--ha_vsmc_eps",type=float,default=1e-8)
parse.add_argument("--ha_vsmc_log_every_round",type=str2bool,default=True)
parse.add_argument("--ha_vsmc_deterministic",type=str2bool,default=True)
parse.add_argument("--ha_vsmc_small_entropy_bonus",type=float,default=0.0)
parse.add_argument("--ha_vsmc_reliability_weighted_update",type=str2bool,default=False)
parse.add_argument("--ha_vsmc_save_client_weights",type=str2bool,default=True)
parse.add_argument("--ha_vsmc_anti_collapse_enabled",type=str2bool,default=False)
parse.add_argument("--ha_vsmc_anti_collapse_max_weight_thresh",type=float,default=0.75)
parse.add_argument("--ha_vsmc_anti_collapse_neff_thresh",type=float,default=1.8)
parse.add_argument("--ha_vsmc_anti_collapse_entropy_thresh",type=float,default=0.5)
parse.add_argument("--ha_vsmc_anti_collapse_eps",type=float,default=0.05)
parse.add_argument("--ha_vsmc_anti_collapse_prior",type=str,default="usage",choices=["usage", "uniform_active", "rank_prior"])
parse.add_argument("--ha_vsmc_coverage_bonus_beta",type=float,default=0.0)
parse.add_argument("--ha_vsmc_proxy_gain_gamma",type=float,default=0.0)
parse.add_argument("--ha_vsmc_proxy_samples_per_class",type=int,default=2)
parse.add_argument("--ha_vsmc_proxy_batch_size",type=int,default=128)
parse.add_argument("--moe_state_num_particles",type=int,default=8)
parse.add_argument("--moe_state_elite_top_k",type=int,default=4)
parse.add_argument("--moe_state_ess_threshold",type=float,default=0.5)
parse.add_argument("--moe_state_jitter_std",type=float,default=0.03)
parse.add_argument("--moe_state_logq_coef",type=float,default=0.1)
parse.add_argument("--moe_state_reward_temperature",type=float,default=10.0)
parse.add_argument("--moe_state_transition_coef",type=float,default=0.05)
parse.add_argument("--moe_state_proposal_lr",type=float,default=1e-4)
parse.add_argument("--moe_state_proposal_hidden_dim",type=int,default=128)
parse.add_argument("--moe_state_residual_latent_dim",type=int,default=8)
parse.add_argument("--moe_state_max_residual_ratio",type=float,default=0.15)
parse.add_argument("--moe_state_aggregation_modes",type=str,default="fedavg,shrink075,shrink05,momentum,clipped_mean,latent_residual")
parse.add_argument("--moe_state_momentum_beta",type=float,default=0.9)
parse.add_argument("--moe_state_clip_norm_quantile",type=float,default=0.8)
parse.add_argument("--moe_state_block_scale_min",type=float,default=0.25)
parse.add_argument("--moe_state_block_scale_max",type=float,default=1.25)
parse.add_argument("--moe_state_val_loss_coef",type=float,default=1.0)
parse.add_argument("--moe_state_val_loss_improve_coef",type=float,default=0.5)
parse.add_argument("--moe_state_val_acc_improve_coef",type=float,default=0.2)
parse.add_argument("--moe_state_router_balance_coef",type=float,default=0.1)
parse.add_argument("--moe_state_expert_conflict_coef",type=float,default=0.05)
parse.add_argument("--moe_state_update_norm_coef",type=float,default=0.02)
parse.add_argument("--moe_state_class_balance_coef",type=float,default=0.05)
parse.add_argument("--moe_state_val_batches",type=int,default=1)
parse.add_argument("--moe_state_full_eval_interval",type=int,default=5)
parse.add_argument("--moe_state_train_proposal",type=str2bool,default=True)
parse.add_argument("--moe_state_detach_resampling",type=str2bool,default=True)
parse.add_argument("--moe_state_log_diagnostics",type=str2bool,default=True)
parse.add_argument("--moe_state_selector_mode",type=str,default="vsmc",choices=["vsmc", "oracle", "reward_softmax", "uniform"])
parse.add_argument("--learned_vsmc_num_particles",type=int,default=4)
parse.add_argument("--learned_vsmc_proposal_lr",type=float,default=1e-4)
parse.add_argument("--learned_vsmc_train_steps",type=int,default=1)
parse.add_argument("--learned_vsmc_sigma_f",type=float,default=1.0)
parse.add_argument("--learned_vsmc_server_g_beta",type=float,default=1.0)
parse.add_argument("--learned_vsmc_use_reward_g",type=str2bool,default=True)
parse.add_argument("--learned_vsmc_val_batches",type=int,default=4)
parse.add_argument("--learned_vsmc_grad_clip",type=float,default=1.0)
parse.add_argument("--learned_vsmc_use_reinforce_if_not_differentiable",type=str2bool,default=True)
parse.add_argument("--learned_vsmc_require_differentiable_g",type=str2bool,default=False)
parse.add_argument("--learned_vsmc_reward_type",type=str,default="loss_plus_confidence",choices=["loss", "loss_zscore", "loss_plus_confidence"])
parse.add_argument("--learned_vsmc_confidence_reward_weight",type=float,default=0.3)
parse.add_argument("--learned_vsmc_loss_reward_weight",type=float,default=0.7)
parse.add_argument("--learned_vsmc_latent_dim",type=int,default=4)
parse.add_argument("--learned_vsmc_latent_prior_sigma",type=float,default=1.0)
parse.add_argument("--learned_vsmc_latent_mean_scale",type=float,default=0.25)
parse.add_argument("--learned_vsmc_latent_residual_max",type=float,default=0.75)
parse.add_argument("--learned_vsmc_include_zero_particle",type=str2bool,default=False)
parse.add_argument("--learned_vsmc_use_block_joint",type=str2bool,default=False)
parse.add_argument("--learned_vsmc_block_latent_dim",type=int,default=8)
parse.add_argument("--learned_vsmc_anneal_start_ratio",type=float,default=0.6)
parse.add_argument("--learned_vsmc_final_residual_max",type=float,default=0.15)
parse.add_argument("--learned_vsmc_final_mix_fedavg_candidate",type=float,default=0.5)
parse.add_argument("--learned_vsmc_mean_a_scale",type=float,default=0.5)
parse.add_argument("--learned_vsmc_mean_b_scale",type=float,default=0.2)
parse.add_argument("--learned_vsmc_init_logvar",type=float,default=-4.0)
parse.add_argument("--learned_vsmc_logvar_scale",type=float,default=2.0)
parse.add_argument("--learned_vsmc_a_min",type=float,default=0.0)
parse.add_argument("--learned_vsmc_a_max",type=float,default=1.5)
parse.add_argument("--learned_vsmc_b_min",type=float,default=-0.3)
parse.add_argument("--learned_vsmc_b_max",type=float,default=0.3)
parse.add_argument("--learned_vsmc_reward_clip",type=float,default=5.0)
parse.add_argument("--learned_vsmc_large_loss",type=float,default=20.0)
parse.add_argument("--learned_vsmc_feature_clip",type=float,default=10.0)
parse.add_argument("--learned_vsmc_logq_coef",type=float,default=0.03)
parse.add_argument("--learned_vsmc_normalize_reward",type=str2bool,default=True)
parse.add_argument("--learned_vsmc_reward_norm_eps",type=float,default=1e-8)
parse.add_argument("--learned_vsmc_include_fedavg_particle",type=str2bool,default=True)
parse.add_argument("--learned_vsmc_mix_fedavg_candidate",type=float,default=0.2)
parse.add_argument("--learned_vsmc_mix_uniform",type=float,default=0.2)
parse.add_argument("--learned_vsmc_alpha_max",type=float,default=0.7)
parse.add_argument("--learned_vsmc_min_normalized_ess",type=float,default=0.25)
parse.add_argument("--learned_vsmc_mode",type=str,default="default",choices=["default", "sanity_fedavg_particle", "server_reward_selector", "g_only_particle_selector"])
parse.add_argument("--vsmc_temperature",type=float,default=1.0)
parse.add_argument("--vsmc_sigma_f",type=float,default=1.0)
parse.add_argument("--vsmc_sigma_r",type=float,default=1.0)
parse.add_argument("--vsmc_obs_temp",type=float,default=1.0)
parse.add_argument("--vsmc_loss_beta",type=float,default=1.0)
parse.add_argument("--vsmc_usage_beta",type=float,default=0.2)
parse.add_argument("--vsmc_drift_beta",type=float,default=0.05)
parse.add_argument("--vsmc_sample_beta",type=float,default=0.1)
parse.add_argument("--vsmc_route_beta",type=float,default=0.2)
parse.add_argument("--vsmc_eps",type=float,default=1e-12)
parse.add_argument(
    "--vsmc_normalize_loss",
    type=str,
    default="zscore",
    choices=["zscore", "minmax", "none"],
)
parse.add_argument("--vsmc_normalize_distance",type=str2bool,default=True)
parse.add_argument("--vsmc_target_usage",type=float,default=None)
parse.add_argument("--vsmc_min_route_count",type=int,default=1)
parse.add_argument("--vsmc_ess_threshold",type=float,default=0.35)
parse.add_argument("--vsmc_use_resampling",type=str2bool,default=True)
parse.add_argument("--vsmc_resampling_mode",type=str,default="stratified",choices=["stratified"])
parse.add_argument("--vsmc_keep_particle_bank",type=str2bool,default=True)
parse.add_argument("--vsmc_missing_client_decay",type=float,default=0.9)
parse.add_argument("--vsmc_resampling_noise_std",type=float,default=0.0)
parse.add_argument("--vsmc_fallback_to_fedavg_on_nan",type=str2bool,default=True)
parse.add_argument("--vsmc_fallback_on_low_ess",type=str2bool,default=False)
parse.add_argument("--vsmc_low_ess_threshold",type=float,default=0.15)
parse.add_argument("--vsmc_fallback_when_all_route_zero",type=str2bool,default=True)
parse.add_argument("--vsmc_clamp_log_weight",type=float,default=30.0)
parse.add_argument("--vsmc_use_server_validation",type=str2bool,default=False)
parse.add_argument("--vsmc_server_val_beta",type=float,default=1.0)
parse.add_argument("--vsmc_server_val_batches",type=int,default=2)
parse.add_argument(
    "--vsmc_mode",
    type=str,
    default="baseline",
    choices=[
        "baseline",
        "pure_f_only",
        "pure_r_only",
        "g_only",
        "g_minus_r",
        "full",
        "server_validated_delta_decoupled",
        "strong_signal_joint",
        "block_particle_gain_filter",
    ],
)
parse.add_argument("--vsmc_aggregate_delta",type=str2bool,default=True)
parse.add_argument("--vsmc_expert_server_lr",type=float,default=0.5)
parse.add_argument("--vsmc_use_server_validation_g",type=str2bool,default=True)
parse.add_argument("--vsmc_server_val_batch_size",type=int,default=128)
parse.add_argument("--vsmc_server_val_every_round",type=int,default=1)
parse.add_argument("--vsmc_server_g_beta",type=float,default=5.0)
parse.add_argument("--vsmc_server_g_clip",type=float,default=5.0)
parse.add_argument("--vsmc_decouple_mu",type=float,default=0.2)
parse.add_argument("--vsmc_base_alpha_source",type=str,default="r",choices=["f", "r", "fr", "fedavg"])
parse.add_argument("--vsmc_lambda_r",type=float,default=0.3)
parse.add_argument("--vsmc_r_bonus_clip",type=float,default=3.0)
parse.add_argument("--vsmc_lambda_g",type=float,default=1.0)
parse.add_argument("--vsmc_use_local_g_if_no_server_val",type=str2bool,default=True)
parse.add_argument("--vsmc_use_ema_f_smoothing",type=str2bool,default=True)
parse.add_argument("--vsmc_expert_ema_rho",type=float,default=0.2)
parse.add_argument("--vsmc_mix_with_fedavg",type=float,default=0.1)
parse.add_argument("--vsmc_alpha_max",type=float,default=0.7)
parse.add_argument("--vsmc_normalized_ess_min",type=float,default=0.2)
parse.add_argument("--vsmc_fallback_to_fedavg_on_bad_alpha",type=str2bool,default=True)
parse.add_argument("--vsmc_use_router_usage_calibration",type=str2bool,default=False)
parse.add_argument("--vsmc_router_bias_lr",type=float,default=0.01)
parse.add_argument("--use_server_router_finetune",type=str2bool,default=False)
parse.add_argument("--router_finetune_steps",type=int,default=1)
parse.add_argument("--router_finetune_lr",type=float,default=1e-4)
parse.add_argument("--router_finetune_batches",type=int,default=1)
parse.add_argument("--use_expert_alignment",type=str2bool,default=False)
parse.add_argument("--alignment_metric",type=str,default="output_cosine",choices=["output_cosine"])
parse.add_argument("--alignment_batch_size",type=int,default=128)
parse.add_argument("--alignment_update_every",type=int,default=1)
parse.add_argument("--alignment_strength",type=float,default=0.1)
parse.add_argument("--vsmc_target_usage_override",type=float,default=None)
parse.add_argument("--vsmc_replace_reward_beta",type=float,default=0.35)
parse.add_argument("--vsmc_delta_reward_beta",type=float,default=0.30)
parse.add_argument("--vsmc_stability_beta",type=float,default=0.10)
parse.add_argument("--vsmc_novelty_beta",type=float,default=0.10)
parse.add_argument("--vsmc_credit_beta",type=float,default=0.15)
parse.add_argument("--vsmc_signal_temperature",type=float,default=1.0)
parse.add_argument("--vsmc_reward_ema_rho",type=float,default=0.7)
parse.add_argument("--vsmc_credit_momentum",type=float,default=0.8)
parse.add_argument("--vsmc_uniform_mix",type=float,default=0.05)
parse.add_argument("--vsmc_novelty_clip",type=float,default=2.0)
parse.add_argument("--vsmc_stability_clip",type=float,default=2.0)
parse.add_argument("--vsmc_signal_reward_clip",type=float,default=5.0)
parse.add_argument(
    "--vsmc_server_val_metric",
    type=str,
    default="loss",
    choices=["loss"],
)
parse.add_argument("--joint_expert_transition_beta",type=float,default=1.0)
parse.add_argument("--joint_local_obs_beta",type=float,default=1.0)
parse.add_argument("--joint_global_obs_beta",type=float,default=1.0)
parse.add_argument("--joint_proposal_beta",type=float,default=1.0)
parse.add_argument("--joint_router_consistency_beta",type=float,default=0.5)
parse.add_argument("--joint_router_validation_beta",type=float,default=0.5)
parse.add_argument("--joint_router_entropy_beta",type=float,default=0.1)
parse.add_argument("--joint_router_aux_beta",type=float,default=0.1)
parse.add_argument("--joint_router_load_balance_beta",type=float,default=0.1)
parse.add_argument("--joint_proposal_ema_decay",type=float,default=0.8)
parse.add_argument("--joint_use_client_specific_proposal",type=str2bool,default=True)
parse.add_argument("--joint_use_router_validation",type=str2bool,default=True)
parse.add_argument("--joint_route_weight_temperature",type=float,default=1.0)
parse.add_argument("--vsmc_history_decay",type=float,default=0.9)
parse.add_argument("--vsmc_missing_decay",type=float,default=0.9)
parse.add_argument("--vsmc_degeneration_ess_threshold",type=float,default=0.2)
parse.add_argument("--vsmc_degeneration_temperature_scale",type=float,default=2.0)
parse.add_argument("--vsmc_fallback_to_fedavg_on_degenerate",type=str2bool,default=False)
parse.add_argument("--vsmc_debug",type=str2bool,default=False)
parse.add_argument("--vsmc_variant",type=str,default="baseline",choices=["baseline", "full_vsmc"])
parse.add_argument("--beta_precision_num_particles", type=int, default=32)
parse.add_argument("--beta_precision_kappa_u", type=float, default=10.0)
parse.add_argument("--beta_precision_kappa_c", type=float, default=10.0)
parse.add_argument("--beta_precision_omd_lr", type=float, default=0.05)
parse.add_argument("--beta_precision_gamma_min", type=float, default=1e-3)
parse.add_argument("--beta_precision_transition_rho", type=float, default=0.9)
parse.add_argument("--beta_precision_transition_std", type=float, default=0.15)
parse.add_argument("--beta_precision_initial_std", type=float, default=1.0)
parse.add_argument("--beta_precision_mean_slope_u", type=float, default=1.0)
parse.add_argument("--beta_precision_mean_slope_c", type=float, default=1.0)
parse.add_argument("--beta_precision_mean_intercept_c", type=float, default=0.0)
parse.add_argument("--beta_precision_eps", type=float, default=1e-6)
parse.add_argument("--temporal_momentum",type=float,default=0.8)
parse.add_argument("--temporal_align_beta",type=float,default=0.5)
parse.add_argument("--temporal_jump_beta",type=float,default=0.2)
parse.add_argument("--temporal_warmup_rounds",type=int,default=2)
parse.add_argument(
    "--temporal_score_mode",
    type=str,
    default="align_jump",
    choices=["align_jump", "align_only", "align_cosine_jump"],
)
parse.add_argument("--temporal_history_size",type=int,default=1)
parse.add_argument("--temporal_debug",type=str2bool,default=False)
parse.add_argument("--latent_vsmc_dim",type=int,default=16)
parse.add_argument("--latent_transition_rho",type=float,default=0.9)
parse.add_argument("--latent_process_noise",type=float,default=0.1)
parse.add_argument("--latent_obs_noise_base",type=float,default=1.0)
parse.add_argument("--latent_loss_noise_beta",type=float,default=1.0)
parse.add_argument("--latent_route_noise_beta",type=float,default=0.2)
parse.add_argument("--latent_sample_noise_beta",type=float,default=0.2)
parse.add_argument(
    "--full_vsmc_proposal_mode",
    type=str,
    default="posterior_diag",
    choices=["prior_only", "posterior_diag", "blend"],
)
parse.add_argument("--full_vsmc_proposal_blend",type=float,default=0.5)
parse.add_argument("--full_vsmc_proposal_var_scale",type=float,default=1.0)
parse.add_argument("--full_vsmc_use_proposal_particles",type=str2bool,default=False)
parse.add_argument("--full_vsmc_num_particles",type=int,default=8)
parse.add_argument("--full_vsmc_responsibility_temperature",type=float,default=1.0)
parse.add_argument("--full_vsmc_particle_temperature",type=float,default=1.0)
parse.add_argument("--full_vsmc_metric_ema_beta",type=float,default=0.8)
parse.add_argument("--full_vsmc_responsibility_floor",type=float,default=0.0)
parse.add_argument("--full_vsmc_prev_weight_decay",type=float,default=1.0)
parse.add_argument("--latent_warmup_rounds",type=int,default=3)
parse.add_argument("--latent_projection_seed",type=int,default=0)
parse.add_argument("--latent_fedavg_mix",type=float,default=0.0)
parse.add_argument("--latent_min_weight",type=float,default=0.0)
parse.add_argument("--latent_debug",type=str2bool,default=False)
parse.add_argument("--server_adapt_enabled",type=str2bool,default=False)
parse.add_argument("--server_adapt_version",type=str,default="v11c_strict_accept")
parse.add_argument("--server_adapt_start_frac",type=float,default=0.50)
parse.add_argument("--server_adapt_steps",type=int,default=1)
parse.add_argument("--server_adapt_lr",type=float,default=1e-4)
parse.add_argument("--server_adapt_weight_decay",type=float,default=0.0)
parse.add_argument("--server_adapt_batches",type=int,default=3)
parse.add_argument("--server_adapt_train_batches",type=int,default=2)
parse.add_argument("--server_adapt_accept_batches",type=int,default=1)
parse.add_argument("--server_adapt_split_accept",type=str2bool,default=True)
parse.add_argument("--server_adapt_update_head",type=str2bool,default=True)
parse.add_argument("--server_adapt_update_final_norm",type=str2bool,default=True)
parse.add_argument("--server_adapt_update_moe_layernorm",type=str2bool,default=True)
parse.add_argument("--server_adapt_update_router_temperature",type=str2bool,default=True)
parse.add_argument("--server_adapt_update_router_bias",type=str2bool,default=False)
parse.add_argument("--server_adapt_update_full_router",type=str2bool,default=False)
parse.add_argument("--server_adapt_update_full_dense",type=str2bool,default=False)
parse.add_argument("--server_adapt_update_experts",type=str2bool,default=False)
parse.add_argument("--server_adapt_rollback",type=str2bool,default=True)
parse.add_argument("--server_adapt_delta_ema",type=str2bool,default=True)
parse.add_argument("--server_adapt_ema_decay",type=float,default=0.8)
parse.add_argument("--server_adapt_margin_weight",type=float,default=0.2)
parse.add_argument("--server_adapt_entropy_weight",type=float,default=0.01)
parse.add_argument("--server_adapt_strict_accept",type=str2bool,default=True)
parse.add_argument("--server_adapt_min_delta",type=float,default=3e-4)
parse.add_argument("--server_adapt_mid_start_frac",type=float,default=0.70)
parse.add_argument("--server_adapt_mid_min_delta",type=float,default=7e-4)
parse.add_argument("--server_adapt_mid_margin_delta",type=float,default=8e-4)
parse.add_argument("--server_adapt_late_min_delta",type=float,default=1e-3)
parse.add_argument("--server_adapt_margin_delta",type=float,default=5e-4)
parse.add_argument("--server_adapt_late_margin_delta",type=float,default=1e-3)
parse.add_argument("--server_adapt_weak_loss_delta",type=float,default=1e-4)
parse.add_argument("--server_adapt_late_start_frac",type=float,default=0.8)
parse.add_argument("--server_adapt_evidence_gate",type=str2bool,default=True)
parse.add_argument("--server_adapt_evidence_required_mid",type=int,default=2)
parse.add_argument("--server_adapt_evidence_required_late",type=int,default=3)
parse.add_argument("--server_adapt_router_entropy_tolerance",type=float,default=0.03)
parse.add_argument("--server_adapt_late_cooldown",type=str2bool,default=True)
parse.add_argument("--server_adapt_cooldown_start_frac",type=float,default=0.75)
parse.add_argument("--server_adapt_cooldown_after_accept",type=int,default=1)
parse.add_argument("--server_adapt_rescue_if_loss_worse",type=str2bool,default=True)
parse.add_argument("--server_adapt_rescue_loss_threshold",type=float,default=0.01)
parse.add_argument("--server_adapt_near_best_protection",type=str2bool,default=True)
parse.add_argument("--server_adapt_near_best_tolerance",type=float,default=0.002)
parse.add_argument("--server_adapt_near_best_require_late_rule",type=str2bool,default=True)
parse.add_argument("--server_adapt_teacher_router",type=str2bool,default=True)
parse.add_argument("--server_adapt_teacher_router_weight",type=float,default=0.15)
parse.add_argument("--server_adapt_teacher_start_frac",type=float,default=0.50)
parse.add_argument("--server_adapt_teacher_batches",type=int,default=2)
parse.add_argument("--server_adapt_teacher_topk_experts",type=str,default="all")
parse.add_argument("--server_adapt_teacher_metric",type=str,default="margin")
parse.add_argument("--server_adapt_teacher_conf_threshold",type=float,default=0.0)
parse.add_argument("--server_adapt_teacher_update_router_bias",type=str2bool,default=True)
parse.add_argument("--server_adapt_teacher_update_router_temperature",type=str2bool,default=True)
parse.add_argument("--server_adapt_teacher_update_full_router",type=str2bool,default=False)
parse.add_argument("--server_adapt_teacher_loss_stopgrad",type=str2bool,default=True)
parse.add_argument("--server_adapt_accept_rate_control",type=str2bool,default=True)
parse.add_argument("--server_adapt_accept_rate_window",type=int,default=10)
parse.add_argument("--server_adapt_accept_rate_high",type=float,default=0.9)
parse.add_argument("--server_adapt_accept_rate_low",type=float,default=0.1)
parse.add_argument("--server_adapt_accept_rate_target_low",type=float,default=0.3)
parse.add_argument("--server_adapt_accept_rate_target_high",type=float,default=0.8)
parse.add_argument("--server_adapt_accept_multiplier_min",type=float,default=0.5)
parse.add_argument("--server_adapt_accept_multiplier_max",type=float,default=4.0)
parse.add_argument("--server_adapt_router_temp_min",type=float,default=0.5)
parse.add_argument("--server_adapt_router_temp_max",type=float,default=2.0)


###################################### other settings ################################################
parse.add_argument("--KLtemperature",type=int,default=2)

parse.add_argument("--max_degree",type=int,default=3)

parse.add_argument("--low_energy_thr",type=float,default=0.5)
parse.add_argument("--tail_energy_thr",type=float,default=0.05)
parse.add_argument("--tail_ratio_thr",type=float,default=0.3)
parse.add_argument("--account",type=float,default=0.85)
