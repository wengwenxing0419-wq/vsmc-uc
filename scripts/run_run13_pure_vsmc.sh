#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/wwx/Vsmc-2"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/experiments/run_13}"
DATA_DIR="$RUN_ROOT/data_cifar100_alpha0.1_10client_seed3"
SEED="${SEED:-3}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPO_DIR"
source /home/wwx/code/vsmc/bin/activate

if [[ ! -f "$DATA_DIR/server_test.pth" ]]; then
  python data.py     --data_name cifar100     --num_clients 10     --alpha 0.1     --seed "$SEED"     --data_save_path "$DATA_DIR"
fi

run_exp() {
  local name="$1"
  shift
  local out="$RUN_ROOT/$name/seed${SEED}"
  mkdir -p "$out/model" "$out/result"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python train.py     --data_name cifar100     --num_clients 10     --alpha 0.1     --seed "$SEED"     --device "$DEVICE"     --server_epochs 100     --client_epochs 1     --batch_size 32     --learning_rate 5e-4     --client_optimizer adam     --backbone_name resnet18     --aggregation_method ha_vsmc_etf     --fedavg_weight_mode sample_count     --expert_weight_mode uniform     --ha_vsmc_num_particles 8     --ha_vsmc_lambda_max 0.20     --ha_vsmc_anchor_mode none     --ha_vsmc_pure_vsmc true     --ha_vsmc_use_validation_loss false     --ha_vsmc_use_function_drift true     --ha_vsmc_log_every_round true     --ha_vsmc_score_transform rank_based     --ha_vsmc_save_client_weights true     --data_save_path "$DATA_DIR"     --model_save_path "$out/model"     --save_result "$out/result"     "$@"
}

run_exp pure_vsmc_rank_entropy_guard_eps0.05   --ha_vsmc_anti_collapse_enabled true   --ha_vsmc_anti_collapse_eps 0.05   --ha_vsmc_anti_collapse_prior usage

run_exp pure_vsmc_rank_entropy_guard_eps0.10   --ha_vsmc_anti_collapse_enabled true   --ha_vsmc_anti_collapse_eps 0.10   --ha_vsmc_anti_collapse_prior usage

run_exp pure_vsmc_coverage_rank_beta0.1   --ha_vsmc_coverage_bonus_beta 0.1

run_exp pure_vsmc_coverage_rank_beta0.2   --ha_vsmc_coverage_bonus_beta 0.2

run_exp pure_vsmc_rank_proxy_gain_gamma0.1   --ha_vsmc_proxy_gain_gamma 0.1   --ha_vsmc_proxy_samples_per_class 2

run_exp pure_vsmc_rank_proxy_gain_gamma0.2   --ha_vsmc_proxy_gain_gamma 0.2   --ha_vsmc_proxy_samples_per_class 2
