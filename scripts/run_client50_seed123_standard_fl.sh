#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKBONE_NAME="${BACKBONE_NAME:-resnet18}"
ROOT_DIR="${ROOT_DIR:-$REPO_DIR/runs/client50_standard_fl_${BACKBONE_NAME}}"
DATA_NAME="${DATA_NAME:-cifar10}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
ALPHA="${ALPHA:-0.1}"
SEEDS="${SEEDS:-1 2 3}"
METHODS="${METHODS:-fedavg fedavg_expert_only vsmc_expert}"
DEVICE="${DEVICE:-cuda}"
SERVER_EPOCHS="${SERVER_EPOCHS:-100}"
CLIENT_EPOCHS="${CLIENT_EPOCHS:-1}"
VSMC_EXTRA_ARGS="${VSMC_EXTRA_ARGS:---vsmc_temperature 1.0 --vsmc_loss_beta 1.0 --vsmc_usage_beta 0.3 --vsmc_drift_beta 0.05 --vsmc_sample_beta 0.2 --vsmc_route_beta 0.2}"

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

run_experiment() {
  local seed="$1"
  local method="$2"
  local seed_dir="$ROOT_DIR/seed${seed}"
  local data_dir="$seed_dir/data"
  local run_dir="$seed_dir/$method"
  local cmd=()

  mkdir -p "$run_dir/model" "$run_dir/result"

  echo "[run] seed${seed} / ${method}"
  cmd=(
    python "$REPO_DIR/train.py"
    --data_name "$DATA_NAME" \
    --num_clients "$NUM_CLIENTS" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --device "$DEVICE" \
    --server_epochs "$SERVER_EPOCHS" \
    --client_epochs "$CLIENT_EPOCHS" \
    --backbone_name "$BACKBONE_NAME" \
    --aggregation_method "$method" \
    --data_save_path "$data_dir" \
    --model_save_path "$run_dir/model" \
    --save_result "$run_dir/result"
  )

  if [[ "$method" == "vsmc_expert" ]]; then
    read -r -a extra_args <<< "$VSMC_EXTRA_ARGS"
    cmd+=("${extra_args[@]}")
  fi

  run_cmd "${cmd[@]}"
}

echo "[root] $ROOT_DIR"
echo "[config] data=$DATA_NAME clients=$NUM_CLIENTS alpha=$ALPHA device=$DEVICE backbone=$BACKBONE_NAME"
echo "[config] seeds=$SEEDS methods=$METHODS server_epochs=$SERVER_EPOCHS client_epochs=$CLIENT_EPOCHS"

mkdir -p "$ROOT_DIR"

for seed in $SEEDS; do
  ensure_seed_data "$seed"
  for method in $METHODS; do
    run_experiment "$seed" "$method"
  done
done

echo
echo "[done] all requested runs finished under $ROOT_DIR"
run_cmd python "$REPO_DIR/scripts/summarize_method_comparison.py" \
  --root "$ROOT_DIR" \
  --seeds $SEEDS \
  --methods $METHODS
