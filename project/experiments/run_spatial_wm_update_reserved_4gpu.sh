#!/usr/bin/env bash
# Two Wan ranks and two Cosmos ranks; training allocations reuse the cache.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WM_UPDATE_RESERVE_MEMORY=1
export WM_UPDATE_GPUS="${WM_UPDATE_GPUS:-0,1,2,3}"
export WM_UPDATE_MEMORY_HEADROOM_GIB="${WM_UPDATE_MEMORY_HEADROOM_GIB:-3.5}"
exec bash "$script_dir/run_spatial_wm_update_4gpu.sh" "$@"
