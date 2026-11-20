#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKBONE_NAME="${BACKBONE_NAME:-resnet18}"
DATA_NAME="${DATA_NAME:-cifar10}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
ALPHA="${ALPHA:-0.1}"
SEEDS="${SEEDS:-1 2 3}"
DEVICE="${DEVICE:-cuda}"
SERVER_EPOCHS="${SERVER_EPOCHS:-100}"
CLIENT_EPOCHS="${CLIENT_EPOCHS:-1}"
ROOT_DIR="${ROOT_DIR:-$REPO_DIR/runs/client50_fedavg_vs_recursive_hd05_${BACKBONE_NAME}}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-$REPO_DIR/run_5}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

resolve_data_dir() {
  local seed="$1"
  local legacy_data_dir="$LEGACY_DATA_ROOT/seed${seed}/data"
  local local_data_dir="$ROOT_DIR/seed${seed}/data"
  local train_count

  if [[ -d "$legacy_data_dir" ]]; then
    train_count="$(find "$legacy_data_dir" -maxdepth 1 -name '*_train.pth' | wc -l | tr -d ' ')"
    if [[ -f "$legacy_data_dir/server_test.pth" && "$train_count" == "$NUM_CLIENTS" ]]; then
      echo "$legacy_data_dir"
      return
    fi
  fi

  mkdir -p "$local_data_dir"
  train_count="$(find "$local_data_dir" -maxdepth 1 -name '*_train.pth' | wc -l | tr -d ' ')"
  if [[ -f "$local_data_dir/server_test.pth" && "$train_count" == "$NUM_CLIENTS" ]]; then
    echo "$local_data_dir"
    return
  fi

  echo "[data] generating seed${seed} split into $local_data_dir" >&2
  run_cmd python "$REPO_DIR/data.py" \
    --data_name "$DATA_NAME" \
    --num_clients "$NUM_CLIENTS" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --data_save_path "$local_data_dir" >&2

  echo "$local_data_dir"
}

run_config() {
  local seed="$1"
  local label="$2"
  local data_dir="$3"
  shift 3

  local seed_dir="$ROOT_DIR/seed${seed}"
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
echo "[legacy_data_root] $LEGACY_DATA_ROOT"
echo "[config] data=$DATA_NAME clients=$NUM_CLIENTS alpha=$ALPHA device=$DEVICE backbone=$BACKBONE_NAME"
echo "[config] seeds=$SEEDS server_epochs=$SERVER_EPOCHS client_epochs=$CLIENT_EPOCHS"
echo "[config] methods=fedavg vsmc_expert_recursive_hd05"

mkdir -p "$ROOT_DIR"

for seed in $SEEDS; do
  data_dir="$(resolve_data_dir "$seed")"
  echo "[seed${seed}] data_dir=$data_dir"

  run_config "$seed" "fedavg" "$data_dir" \
    --aggregation_method fedavg

  run_config "$seed" "vsmc_expert_recursive_hd05" "$data_dir" \
    --aggregation_method vsmc_expert \
    --vsmc_temperature 1.0 \
    --vsmc_loss_beta 1.0 \
    --vsmc_usage_beta 0.3 \
    --vsmc_drift_beta 0.05 \
    --vsmc_sample_beta 0.2 \
    --vsmc_route_beta 0.2 \
    --vsmc_history_decay 0.5
done

echo
echo "[done] reruns finished under $ROOT_DIR"
