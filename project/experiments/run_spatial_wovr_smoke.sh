#!/usr/bin/env bash
# One rollout/GRPO cycle: 128 envs * (128 actions / 8) = 2048 samples.
# Keeps the production 240-action episode limit, micro=32 and accumulation=8.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_spatial_wovr_8gpu.sh" \
  runner.max_epochs=1 runner.max_steps=1 runner.save_interval=1 runner.val_check_interval=-1 \
  runner.logger.experiment_name=spatial_wovr_8gpu_smoke \
  env.train.rollout_epoch=1 env.train.max_steps_per_rollout_epoch=128 \
  "$@"
