#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/wwx/code/run_boost_vsmc}"
DATA_NAME="${DATA_NAME:-cifar10}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
ALPHA="${ALPHA:-0.5}"
BACKBONE="${BACKBONE:-resnet18}"
SERVER_EPOCHS="${SERVER_EPOCHS:-50}"
CLIENT_EPOCHS="${CLIENT_EPOCHS:-1}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(1 2 3)
fi

mkdir -p "${ROOT_DIR}"

run_train() {
  local run_name="$1"
  local seed="$2"
  shift 2
  local data_dir="${ROOT_DIR}/data_seed${seed}"
  local model_dir="${ROOT_DIR}/${run_name}/seed${seed}/model"
  local result_dir="${ROOT_DIR}/${run_name}/seed${seed}/result"

  "${PYTHON_BIN}" /home/wwx/code/data.py \
    --data_name "${DATA_NAME}" \
    --num_clients "${NUM_CLIENTS}" \
    --alpha "${ALPHA}" \
    --seed "${seed}" \
    --data_save_path "${data_dir}"

  "${PYTHON_BIN}" /home/wwx/code/train.py \
    --data_name "${DATA_NAME}" \
    --num_clients "${NUM_CLIENTS}" \
    --alpha "${ALPHA}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --server_epochs "${SERVER_EPOCHS}" \
    --client_epochs "${CLIENT_EPOCHS}" \
    --backbone_name "${BACKBONE}" \
    --data_save_path "${data_dir}" \
    --model_save_path "${model_dir}" \
    --save_result "${result_dir}" \
    "$@"
}

for seed in "${SEEDS[@]}"; do
  run_train "fedavg" "${seed}" \
    --aggregation_method fedavg

  run_train "pure_f_only" "${seed}" \
    --aggregation_method vsmc_expert \
    --vsmc_mode pure_f_only \
    --vsmc_loss_beta 0.0 \
    --vsmc_route_beta 0.0 \
    --vsmc_usage_beta 0.0 \
    --vsmc_sample_beta 0.0 \
    --vsmc_sigma_f 0.03 \
    --vsmc_sigma_r 1000000.0

  run_train "g_minus_r" "${seed}" \
    --aggregation_method vsmc_expert \
    --vsmc_mode g_minus_r \
    --vsmc_sigma_f 1000000.0 \
    --vsmc_sigma_r 0.03

  run_train "server_validated_delta_decoupled" "${seed}" \
    --aggregation_method vsmc_expert \
    --vsmc_mode server_validated_delta_decoupled \
    --vsmc_expert_server_lr 0.5 \
    --vsmc_decouple_mu 0.2 \
    --vsmc_base_alpha_source r \
    --vsmc_lambda_r 0.3 \
    --vsmc_server_g_beta 5.0 \
    --vsmc_expert_ema_rho 0.2 \
    --vsmc_mix_with_fedavg 0.1 \
    --vsmc_use_server_validation_g true \
    --vsmc_server_val_batches 1 \
    --vsmc_aggregate_delta true

  run_train "delta_r_only" "${seed}" \
    --aggregation_method vsmc_expert \
    --vsmc_mode server_validated_delta_decoupled \
    --vsmc_use_server_validation_g false \
    --vsmc_base_alpha_source r \
    --vsmc_expert_server_lr 0.5 \
    --vsmc_decouple_mu 0.0 \
    --vsmc_aggregate_delta true

  run_train "delta_r_router_calib" "${seed}" \
    --aggregation_method vsmc_expert \
    --vsmc_mode server_validated_delta_decoupled \
    --vsmc_use_router_usage_calibration true \
    --vsmc_router_bias_lr 0.01 \
    --vsmc_expert_server_lr 0.5 \
    --vsmc_decouple_mu 0.2 \
    --vsmc_base_alpha_source r \
    --vsmc_lambda_r 0.3 \
    --vsmc_server_g_beta 5.0 \
    --vsmc_expert_ema_rho 0.2 \
    --vsmc_mix_with_fedavg 0.1
done
