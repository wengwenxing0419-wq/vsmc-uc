#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
BASE_DIR="${2:-/home/wwx/code/run_g_fix}"
SEED="${3:-3}"
NUM_ROUNDS="${4:-50}"

DATA_DIR="${BASE_DIR}/shared_data_seed${SEED}"
mkdir -p "${BASE_DIR}"

python /home/wwx/code/data.py \
  --data_name cifar10 \
  --num_clients 10 \
  --alpha 0.5 \
  --seed "${SEED}" \
  --data_save_path "${DATA_DIR}"

run_train() {
  local exp_name="$1"
  shift
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python /home/wwx/code/train.py \
    --data_name cifar10 \
    --num_clients 10 \
    --alpha 0.5 \
    --seed "${SEED}" \
    --device cuda \
    --server_epochs "${NUM_ROUNDS}" \
    --client_epochs 1 \
    --backbone_name resnet18 \
    --data_save_path "${DATA_DIR}" \
    --model_save_path "${BASE_DIR}/${exp_name}/model" \
    --save_result "${BASE_DIR}/${exp_name}/result" \
    "$@"
}

run_train fedavg_seed3 \
  --aggregation_method fedavg

run_train default_old_seed3 \
  --aggregation_method learned_vsmc \
  --learned_vsmc_mode default \
  --learned_vsmc_logq_coef 0.1 \
  --learned_vsmc_val_batches 1 \
  --learned_vsmc_reward_type loss

run_train default_gfix_seed3 \
  --aggregation_method learned_vsmc \
  --learned_vsmc_mode default \
  --learned_vsmc_logq_coef 0.03 \
  --learned_vsmc_val_batches 4 \
  --learned_vsmc_reward_type loss_plus_confidence

run_train g_only_selector_seed3 \
  --aggregation_method learned_vsmc \
  --learned_vsmc_mode g_only_particle_selector \
  --learned_vsmc_logq_coef 0.0 \
  --learned_vsmc_val_batches 4 \
  --learned_vsmc_reward_type loss_plus_confidence

run_train loss_zscore_seed3 \
  --aggregation_method learned_vsmc \
  --learned_vsmc_mode default \
  --learned_vsmc_logq_coef 0.03 \
  --learned_vsmc_val_batches 4 \
  --learned_vsmc_reward_type loss_zscore

python /home/wwx/code/scripts/summarize_learned_vsmc_g_fix.py \
  --base_dir "${BASE_DIR}"
