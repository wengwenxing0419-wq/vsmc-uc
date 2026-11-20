#!/usr/bin/env bash
set -e

python /home/wwx/code/scripts/auto_stage_vsmc_experiments.py \
  --base_output_dir "${1:-run_auto_stage_seed3}" \
  --seeds "${2:-3}" \
  --num_rounds "${3:-50}" \
  --gpu_ids "${4:-0}" \
  --include_alignment "${5:-false}" \
  --run_all_stages "${6:-false}"
