#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKBONE_NAME="${BACKBONE_NAME:-resnet18}"
ROOT_DIR="${ROOT_DIR:-$REPO_DIR/runs/client50_vsmc_expert_${BACKBONE_NAME}}"
DATA_NAME="${DATA_NAME:-cifar10}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
ALPHA="${ALPHA:-0.1}"
SEEDS="${SEEDS:-1 2 3}"
DEVICE="${DEVICE:-cuda}"
SERVER_EPOCHS="${SERVER_EPOCHS:-100}"
CLIENT_EPOCHS="${CLIENT_EPOCHS:-1}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

ensure_seed_data() {
  local seed="$1"
  local data_dir="$ROOT_DIR/seed${seed}/data"
  local train_count

  mkdir -p "$data_dir"
  train_count="$(find "$data_dir" -maxdepth 1 -name '*_train.pth' | wc -l | tr -d ' ')"

  if [[ -f "$data_dir/server_test.pth" && "$train_count" == "$NUM_CLIENTS" ]]; then
    echo "[skip] seed${seed} data already exists at $data_dir"
    return
  fi

  echo "[data] generating seed${seed} split into $data_dir"
  run_cmd python "$REPO_DIR/data.py" \
    --data_name "$DATA_NAME" \
    --num_clients "$NUM_CLIENTS" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --data_save_path "$data_dir"
}

run_config() {
  local seed="$1"
  local label="$2"
  shift 2

  local seed_dir="$ROOT_DIR/seed${seed}"
  local data_dir="$seed_dir/data"
  local run_dir="$seed_dir/$label"

  mkdir -p "$run_dir/model" "$run_dir/result"

  run_cmd python "$REPO_DIR/train.py" \
    --data_name "$DATA_NAME" \
    --num_clients "$NUM_CLIENTS" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --device "$DEVICE" \
    --server_epochs "$SERVER_EPOCHS" \
    --client_epochs "$CLIENT_EPOCHS" \
    --backbone_name "$BACKBONE_NAME" \
    --data_save_path "$data_dir" \
    --model_save_path "$run_dir/model" \
    --save_result "$run_dir/result" \
    "$@"
}

echo "[root] $ROOT_DIR"
echo "[config] data=$DATA_NAME clients=$NUM_CLIENTS alpha=$ALPHA device=$DEVICE backbone=$BACKBONE_NAME"
echo "[config] seeds=$SEEDS server_epochs=$SERVER_EPOCHS client_epochs=$CLIENT_EPOCHS"

mkdir -p "$ROOT_DIR"

for seed in $SEEDS; do
  ensure_seed_data "$seed"

  run_config "$seed" "fedavg" \
    --aggregation_method fedavg

  run_config "$seed" "vsmc_expert" \
    --aggregation_method vsmc_expert \
    --vsmc_temperature 1.0 \
    --vsmc_loss_beta 1.0 \
    --vsmc_usage_beta 0.3 \
    --vsmc_drift_beta 0.05 \
    --vsmc_sample_beta 0.2 \
    --vsmc_route_beta 0.2

  run_config "$seed" "vsmc_expert_no_drift" \
    --aggregation_method vsmc_expert \
    --vsmc_temperature 1.0 \
    --vsmc_loss_beta 1.0 \
    --vsmc_usage_beta 0.3 \
    --vsmc_drift_beta 0.0 \
    --vsmc_sample_beta 0.2 \
    --vsmc_route_beta 0.2

  run_config "$seed" "vsmc_expert_no_usage" \
    --aggregation_method vsmc_expert \
    --vsmc_temperature 1.0 \
    --vsmc_loss_beta 1.0 \
    --vsmc_usage_beta 0.0 \
    --vsmc_drift_beta 0.05 \
    --vsmc_sample_beta 0.2 \
    --vsmc_route_beta 0.2

  run_config "$seed" "vsmc_expert_temp05" \
    --aggregation_method vsmc_expert \
    --vsmc_temperature 0.5 \
    --vsmc_loss_beta 1.0 \
    --vsmc_usage_beta 0.3 \
    --vsmc_drift_beta 0.05 \
    --vsmc_sample_beta 0.2 \
    --vsmc_route_beta 0.2

  run_config "$seed" "vsmc_expert_temp20" \
    --aggregation_method vsmc_expert \
    --vsmc_temperature 2.0 \
    --vsmc_loss_beta 1.0 \
    --vsmc_usage_beta 0.3 \
    --vsmc_drift_beta 0.05 \
    --vsmc_sample_beta 0.2 \
    --vsmc_route_beta 0.2
done

echo
echo "[done] FedAvg, VSMC expert aggregation, and requested ablations finished under $ROOT_DIR"
