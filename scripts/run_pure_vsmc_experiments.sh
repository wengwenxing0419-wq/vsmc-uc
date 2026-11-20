#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/home/wwx/Vsmc-2"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/experiments/run_9_pure_vsmc}"
DATA_NAME="${DATA_NAME:-cifar100}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
ALPHA="${ALPHA:-0.1}"
SEED="${SEED:-3}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SERVER_EPOCHS="${SERVER_EPOCHS:-50}"
CLIENT_EPOCHS="${CLIENT_EPOCHS:-1}"
BACKBONE_NAME="${BACKBONE_NAME:-resnet18}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CLIENT_OPTIMIZER="${CLIENT_OPTIMIZER:-adam}"

DATA_DIR="$RUN_ROOT/data_cifar100_alpha${ALPHA}_10client_seed${SEED}"

run_variant() {
  local label="$1"
  shift
  local run_dir="$RUN_ROOT/${label}/seed${SEED}"
  mkdir -p "$run_dir/model" "$run_dir/result"
  echo
  echo "[run] $label"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$REPO_DIR/train.py"     --data_name "$DATA_NAME"     --num_clients "$NUM_CLIENTS"     --alpha "$ALPHA"     --seed "$SEED"     --device "$DEVICE"     --server_epochs "$SERVER_EPOCHS"     --client_epochs "$CLIENT_EPOCHS"     --batch_size "$BATCH_SIZE"     --learning_rate "$LEARNING_RATE"     --client_optimizer "$CLIENT_OPTIMIZER"     --backbone_name "$BACKBONE_NAME"     --aggregation_method ha_vsmc_etf     --fedavg_weight_mode sample_count     --expert_weight_mode uniform     --ha_vsmc_num_particles 8     --ha_vsmc_lambda_max 0.20     --ha_vsmc_anchor_mode none     --ha_vsmc_pure_vsmc true     --ha_vsmc_use_validation_loss false     --ha_vsmc_use_function_drift true     --ha_vsmc_log_every_round true     --data_save_path "$DATA_DIR"     --model_save_path "$run_dir/model"     --save_result "$run_dir/result"     "$@"
}

run_variant "pure_vsmc_baseline"
run_variant "pure_vsmc_low_temperature"   --ha_vsmc_temperature_scale 0.67
run_variant "pure_vsmc_signal_normalized"   --ha_vsmc_score_transform zscore_clip   --ha_vsmc_score_clip_low -1.5   --ha_vsmc_score_clip_high 1.5
run_variant "pure_vsmc_rank_based_score"   --ha_vsmc_score_transform rank_based
run_variant "pure_vsmc_prob_ema_rho0.5"   --ha_vsmc_prob_ema_rho 0.5
run_variant "pure_vsmc_reliability_floor_eps0.05_freq"   --ha_vsmc_reliability_floor_eps 0.05   --ha_vsmc_reliability_floor_source usage
