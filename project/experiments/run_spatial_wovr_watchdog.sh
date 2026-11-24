#!/usr/bin/env bash
# Independent watcher; attach in the GPU terminal without restarting training.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_root="${SPATIAL_WATCH_LOG_ROOT:-$script_dir/../runtime_artifacts/training_watchdog}"
mkdir -p "$log_root"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
exec > >(tee "$log_root/launcher_$stamp.log") 2>&1
if [[ -n "${SPATIAL_WATCH_RUN:-}" ]]; then
  run_args=(--run-dir "$SPATIAL_WATCH_RUN")
else
  run_args=(--latest-run-root "${SPATIAL_WATCH_RUN_ROOT:-/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_8gpu}")
fi
exec python3 "$script_dir/watch_training_email.py" "${run_args[@]}" "$@"
