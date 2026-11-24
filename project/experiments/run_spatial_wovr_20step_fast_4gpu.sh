#!/usr/bin/env bash
# Opt-in serial dual-model optimizations; leaves running/baseline jobs untouched.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-$project_root/grpo_runs/spatial_wovr20_fast_pi05_12000_4gpu/$stamp}"
export SPATIAL_WOVR_RUNTIME="${SPATIAL_WOVR_RUNTIME:-$project_root/runtime_artifacts/spatial_wovr_fast_4gpu}"
export SPATIAL_WOVR_TMP_ROOT="${SPATIAL_WOVR_TMP_ROOT:-$project_root/../r}"
export SPATIAL_WOVR_FAST_TRANSFER="${SPATIAL_WOVR_FAST_TRANSFER:-1}"
exec bash "$script_dir/run_spatial_wovr_20step_4gpu.sh" \
  "env.train.cosmos_batch_size=${SPATIAL_WOVR_FAST_COSMOS_BATCH:-4}" \
  env.train.cosmos_wrist_parallel_generation=false \
  env.train.fast_frame_transfer=true \
  env.train.skip_successful_env_generation=true \
  "env.train.video_cfg.record_every_n_episodes=${SPATIAL_WOVR_FAST_VIDEO_STRIDE:-4}" \
  runner.logger.experiment_name=spatial_wovr20_fast_wrist_v3_pi05_12000_4gpu \
  "$@"
