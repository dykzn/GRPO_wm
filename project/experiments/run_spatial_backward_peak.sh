#!/usr/bin/env bash
# One actual native update (8 micro-batch backwards per GPU); no eval/checkpoint.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-/data12/qianb2608/grpo_runs/spatial_backward_peak/$stamp}"
exec bash "$script_dir/run_spatial_wovr_8gpu.sh" \
  runner.max_epochs=1 runner.max_steps=1 runner.save_interval=-1 runner.val_check_interval=-1 \
  runner.logger.experiment_name=spatial_backward_peak \
  env.train.rollout_epoch=1 env.train.max_steps_per_rollout_epoch=128 \
  actor.gpu_memory_profile.enabled=true actor.gpu_memory_profile.expected_optimizer_updates=1 \
  algorithm.update_epoch=1 \
  "$@"
