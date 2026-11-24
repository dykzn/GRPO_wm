#!/usr/bin/env bash
# Twenty full GRPO cycles; save/evaluate every five, resume state at 10 and 20.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_8gpu/$stamp}"
exec bash "$script_dir/run_spatial_wovr_8gpu.sh" \
  runner.max_epochs=20 runner.max_steps=20 \
  runner.save_interval=5 runner.val_check_interval=5 \
  runner.logger.experiment_name=spatial_wovr20_wrist_v3_pi05_12000_8gpu_kir_first_close_pre16 \
  'actor.fsdp_config.save_training_state_at_steps=[10,20]' \
  "$@"
