#!/usr/bin/env bash
# Same samples/update as eight GPUs, with the same per-GPU environment load.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
export SPATIAL_WOVR_GPUS="${SPATIAL_WOVR_GPUS:-0,1,2,3}"
export SPATIAL_WOVR_PLACEMENT=0-3
export SPATIAL_WOVR_RUNTIME="${SPATIAL_WOVR_RUNTIME:-$project_root/runtime_artifacts/spatial_wovr_4gpu}"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-/data12/qianb2608/grpo_runs/spatial_wovr_pi05_12000_4gpu/$stamp}"
exec bash "$script_dir/run_spatial_wovr_8gpu.sh" \
  env.train.total_num_envs=64 env.train.rollout_epoch=16 \
  env.eval.total_num_envs=32 env.eval.rollout_epoch=16 \
  runner.logger.experiment_name=spatial_wovr_wrist_v3_pi05_12000_4gpu_kir_first_close_pre16 \
  "$@"
