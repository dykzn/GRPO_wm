#!/usr/bin/env bash
# Two real GRPO updates test Wan -> backward -> Wan -> backward cache reuse.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-$project_root/grpo_runs/spatial_shared_pool_probe_8gpu/$stamp}"
exec bash "$script_dir/run_spatial_wovr_shared_pool_8gpu.sh" \
  runner.max_epochs=2 runner.max_steps=2 \
  runner.save_interval=-1 runner.val_check_interval=-1 runner.resume_dir=null \
  runner.logger.experiment_name=spatial_shared_pool_probe_8gpu \
  env.train.rollout_epoch=1 env.train.max_steps_per_rollout_epoch=128 \
  actor.gpu_memory_profile.enabled=true \
  actor.gpu_memory_profile.expected_optimizer_updates=1 \
  algorithm.update_epoch=1 \
  "$@"
