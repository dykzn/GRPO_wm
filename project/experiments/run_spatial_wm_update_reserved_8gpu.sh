#!/usr/bin/env bash
# Four Wan ranks and four Cosmos ranks, each retaining its own reusable cache.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WM_UPDATE_RESERVE_MEMORY=1
export WM_UPDATE_MEMORY_HEADROOM_GIB="${WM_UPDATE_MEMORY_HEADROOM_GIB:-3.5}"
exec bash "$script_dir/run_spatial_wm_update_8gpu.sh" "$@"
