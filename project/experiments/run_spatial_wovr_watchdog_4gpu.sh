#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPATIAL_WATCH_RUN_ROOT="${SPATIAL_WATCH_RUN_ROOT:-/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_4gpu}"
exec bash "$script_dir/run_spatial_wovr_watchdog.sh" "$@"
