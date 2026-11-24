#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${WM_EVAL_PYTHON:-/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
exec "$python_bin" "$script_dir/wm_eval.py" "$@"
