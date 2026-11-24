#!/usr/bin/env bash
# Stage-specific launcher. No GPU work happens for check/prepare/inspect/dry-run.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "$script_dir/.." && pwd)"
export RLINF_ENV="${RLINF_ENV:-/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi}"
if [[ "${1:-}" == "--check" ]]; then
  shift
  set -- check "$@"
fi
exec "$RLINF_ENV/bin/python" "$script_dir/wm_update.py" "$@"
