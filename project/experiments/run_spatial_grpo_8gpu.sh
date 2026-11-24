#!/usr/bin/env bash
# Spatial + Pi05 step 12000 + LIBERO; micro=32, accumulation=8, retained cache.
# --check verifies the local assets and composes the config without using GPUs.
# Optional Hydra overrides apply to both configuration checks and training.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
check_only=false
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
    shift
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash $0 [--check] [Hydra overrides...]"
    echo "Example: bash $0 actor.micro_batch_size=32"
    exit 0
fi

export RLINF_ROOT="${RLINF_ROOT:-/data41/qianb2608/wan/RLinf_opd}"
runtime_venv="${RLINF_ENV:-/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi}"
python_bin="$runtime_venv/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
export PI05_CHECKPOINT_ROOT="${PI05_CHECKPOINT_ROOT:-$project_root/checkpoints/pi05/Pi05_few_shot_libero/global_step_12000}"
export SPATIAL_GRPO_RUNTIME="${SPATIAL_GRPO_RUNTIME:-$project_root/runtime_artifacts/spatial_grpo_8gpu}"
run_stamp="$(date -u +%Y%m%d_%H%M%S)"
export SPATIAL_GRPO_OUTPUT="${SPATIAL_GRPO_OUTPUT:-/data73/qianb2608/grpo_runs/spatial_pi05_12000_grpo_8gpu/$run_stamp}"
export LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-$RLINF_ROOT/.venv/libero/libero/libero}"
export LIBERO_CONFIG_PATH="$SPATIAL_GRPO_RUNTIME/libero_config"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$RLINF_ROOT/.assets/.cache/openpi}"
export OPENPI_DATA_HOME="$TOKENIZER_DIR"
export PATH="$runtime_venv/bin:$PATH"
export VIRTUAL_ENV="$runtime_venv"
# The installed editable package and activate script contain obsolete /home
# paths. Select the real source tree and interpreter explicitly.
export PYTHONPATH="$script_dir:$RLINF_ROOT:$RLINF_ROOT/.venv/libero:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export RLINF_EGL_PHYSICAL_DEVICE_ID="${RLINF_EGL_PHYSICAL_DEVICE_ID:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${SPATIAL_GRPO_GPUS:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export USE_TF=0 USE_FLAX=0 JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# Keep the allocator's default garbage collection behavior. This PyTorch
# version rejects an explicit garbage_collection_threshold of 1.0.
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
unset PYTORCH_NO_CUDA_MEMORY_CACHING
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export RLINF_NODE_RANK=0
# Ray treats 'local' as a new private cluster, even if other heads exist.
# The driver owns this cluster and shuts it down when it exits.
export RAY_ADDRESS=local RAY_USAGE_STATS_ENABLED=0
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export REPO_PATH="$RLINF_ROOT"
export MPLCONFIGDIR="$SPATIAL_GRPO_RUNTIME/matplotlib"
mkdir -p "$SPATIAL_GRPO_RUNTIME" "$MPLCONFIGDIR" "$LIBERO_CONFIG_PATH"

"$python_bin" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import yaml

root = Path(os.environ["PI05_CHECKPOINT_ROOT"])
weight = root / "actor/model_state_dict/full_weights.pt"
stats = root / "physical-intelligence/libero/norm_stats.json"
manifest_path = root.parent / "model_manifest.json"
manifest = json.loads(manifest_path.read_text())
record = next(item for item in manifest["checkpoints"] if item["step"] == 12000)
if not weight.is_file() or weight.stat().st_size != record["size_bytes"]:
    raise SystemExit(f"Incomplete 12k checkpoint: {weight}; expected {record['size_bytes']} bytes")
json.loads(stats.read_text())
runtime = Path(os.environ["SPATIAL_GRPO_RUNTIME"])
cache = runtime / "verified_checkpoint.json"
identity = {
    "path": str(weight.resolve()),
    "size": weight.stat().st_size,
    "mtime_ns": weight.stat().st_mtime_ns,
    "sha256": record["sha256"],
}
verified = json.loads(cache.read_text()) if cache.is_file() else None
if verified != identity:
    print(f"Checking 12k checkpoint SHA-256: {weight}", flush=True)
    digest = hashlib.sha256()
    with weight.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != identity["sha256"]:
        raise SystemExit("12k checkpoint SHA-256 does not match its release manifest")
    cache.write_text(json.dumps(identity, indent=2) + "\n")
print("12k checkpoint: SHA-256 verified", flush=True)
benchmark = Path(os.environ["LIBERO_BENCHMARK_ROOT"])
for relative in ["assets", "bddl_files/libero_spatial", "init_files/libero_spatial"]:
    if not (benchmark / relative).is_dir():
        raise SystemExit(f"Missing LIBERO resource: {benchmark / relative}")
tokenizer = Path(os.environ["TOKENIZER_DIR"]) / "paligemma_tokenizer.model"
if not tokenizer.is_file():
    raise SystemExit(f"Missing tokenizer: {tokenizer}")
paths = {
    "benchmark_root": str(benchmark),
    "bddl_files": str(benchmark / "bddl_files"),
    "init_states": str(benchmark / "init_files"),
    "assets": str(benchmark / "assets"),
    "datasets": str(benchmark.parent / "datasets"),
}
(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").write_text(yaml.safe_dump(paths))
print("LIBERO Spatial assets and matching policy statistics: present", flush=True)
PY

command=(
    "$python_bin" "$script_dir/train_spatial_grpo.py"
    --config-path "$script_dir/config"
    --config-name spatial_grpo_pi05_8gpu
    "$@"
)

if [[ "$check_only" == true ]]; then
    "${command[@]}" --cfg job --resolve > "$SPATIAL_GRPO_RUNTIME/resolved_config.yaml"
    echo "Config composed: $SPATIAL_GRPO_RUNTIME/resolved_config.yaml"
    exit 0
fi

"$python_bin" - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"This job requires 8 visible CUDA GPUs; this process sees {count}. Run it in the GPU terminal.")
for index in range(count):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
PY

export TMPDIR="$(mktemp -d /tmp/spatial-grpo8-XXXXXX)"
export RAY_TMPDIR="$TMPDIR"
mkdir -p "$SPATIAL_GRPO_OUTPUT"
cp "$script_dir/config/spatial_grpo_pi05_8gpu.yaml" "$SPATIAL_GRPO_OUTPUT/launch_config.yaml"
cp "$script_dir/"spatial_grpo_{memory,actor}.py "$script_dir/train_spatial_grpo.py" "$SPATIAL_GRPO_OUTPUT/"
"${command[@]}" --cfg job --resolve > "$SPATIAL_GRPO_OUTPUT/resolved_config.yaml"
printf 'Output: %s\n' "$SPATIAL_GRPO_OUTPUT"
if command -v nvidia-smi >/dev/null 2>&1; then
    # Keep the host GPU IDs and competing processes next to the training log.
    # PyTorch OOM reports use worker-local device IDs, often GPU 0 on every rank.
    {
        date -u
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv
    } > "$SPATIAL_GRPO_OUTPUT/gpu_preflight.log" 2>&1 || true
    echo "GPU memory and processes: $SPATIAL_GRPO_OUTPUT/gpu_preflight.log"
fi
set +e
"${command[@]}" 2>&1 | tee "$SPATIAL_GRPO_OUTPUT/train.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$SPATIAL_GRPO_OUTPUT/exit_code"
exit "$status"
