#!/usr/bin/env bash
# Eight collectors, then concurrent four-rank Wan and four-rank Cosmos updates.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WM_UPDATE_GPUS="${WM_UPDATE_GPUS:-0,1,2,3,4,5,6,7}"
export WM_UPDATE_ACCUMULATION="${WM_UPDATE_ACCUMULATION:-4}"
exec bash "$script_dir/run_spatial_wm_update_4gpu.sh" "$@"
