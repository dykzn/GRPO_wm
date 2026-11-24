#!/usr/bin/env bash
# --check is CPU-only and accepts an incomplete wrist download.
# --check-assets also verifies complete checkpoint checksums; starts no servers.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
export PROJECT_ROOT="$project_root"
export RLINF_BASE_ROOT="${RLINF_BASE_ROOT:-/data41/qianb2608/wan/RLinf_opd}"
runtime_base="${SPATIAL_WOVR_RUNTIME:-$project_root/runtime_artifacts/spatial_wovr_8gpu}"
mkdir -p "$runtime_base"
export SPATIAL_WOVR_RUN_DIR="$(mktemp -d "$runtime_base/run-XXXXXXXX")"
exec > >(tee "$SPATIAL_WOVR_RUN_DIR/launcher.log") 2>&1
echo "Startup log: $SPATIAL_WOVR_RUN_DIR/launcher.log"
export RLINF_ENV="${RLINF_ENV:-/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi}"
export COSMOS_PYTHON="${COSMOS_PYTHON:-/data73/qianb2608/grpo-world-model-runtime/venvs/cosmos-policy/bin/python}"
export COSMOS_BASE_ROOT="${COSMOS_BASE_ROOT:-$RLINF_BASE_ROOT/.tools/cosmos-policy}"
export LIBERO_ROOT="${LIBERO_ROOT:-$RLINF_BASE_ROOT/.venv/libero}"
export LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-$LIBERO_ROOT/libero/libero}"
export OPENPI_ROOT="${OPENPI_ROOT:-/data73/qianb2608/grpo-world-model-runtime/venvs/sources/openpi}"
export DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/data73/qianb2608/grpo-world-model-runtime/venvs/sources/diffsynth-studio}"
export WORLDPILOT_ROOT="$project_root/third_party/WorldPilot"
export PI05_CHECKPOINT_ROOT="${PI05_CHECKPOINT_ROOT:-$project_root/checkpoints/pi05/Pi05_few_shot_libero/global_step_12000}"
export WAN_SPATIAL_CHECKPOINT="${WAN_SPATIAL_CHECKPOINT:-$project_root/checkpoints/wovr/RLinf-Wan-LIBERO-Spatial}"
export SPATIAL_KIR_FRAME_POOL="${SPATIAL_KIR_FRAME_POOL:-$project_root/runtime_artifacts/kir/spatial_dual_view_first_close_pre16_v1}"
# The external download is usable before its convenience symlink is created.
export COSMOS_WRIST_CHECKPOINT="${COSMOS_WRIST_CHECKPOINT:-/data12/qianb2608/hf_models/Cosmos-WoVR-Wrist-LIBERO-Spatial-v3-step20000/iter_000020000}"
cosmos_assets="$project_root/../cosmos"
export COSMOS_VAE="${COSMOS_VAE:-$cosmos_assets/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth}"
export COSMOS_DATASET_STATS="${COSMOS_DATASET_STATS:-$cosmos_assets/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json}"
export COSMOS_T5_EMBEDDINGS="${COSMOS_T5_EMBEDDINGS:-$cosmos_assets/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$RLINF_BASE_ROOT/.assets/.cache/openpi}"
export OPENPI_DATA_HOME="$TOKENIZER_DIR"
run_stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-/data12/qianb2608/grpo_runs/spatial_wovr_pi05_12000_8gpu/$run_stamp}"
export CUDA_VISIBLE_DEVICES="${SPATIAL_WOVR_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RLINF_NODE_RANK=0 RAY_ADDRESS=local RAY_USAGE_STATS_ENABLED=0
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export RLINF_EGL_PHYSICAL_DEVICE_ID="${RLINF_EGL_PHYSICAL_DEVICE_ID:-1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_TF=0 USE_FLAX=0 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True
unset PYTORCH_NO_CUDA_MEMORY_CACHING
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MPLCONFIGDIR="$SPATIAL_WOVR_RUN_DIR/matplotlib"
export NUMBA_CACHE_DIR="$SPATIAL_WOVR_RUN_DIR/numba_cache"
export LIBERO_CONFIG_PATH="$SPATIAL_WOVR_RUN_DIR/libero_config"
if [[ -n "${SPATIAL_WOVR_TMP_ROOT:-}" ]]; then
  mkdir -p -- "$SPATIAL_WOVR_TMP_ROOT"
  temp_root="$(cd -- "$SPATIAL_WOVR_TMP_ROOT" && pwd -P)"
  # Ray appends /ray/session_<timestamp>_<pid>/sockets/plasma_store.
  # Budget a ten-digit PID and count bytes, including non-ASCII root names.
  socket_template="$temp_root/w-XXXXXXXX/ray/session_0000-00-00_00-00-00_000000_2147483647/sockets/plasma_store"
  socket_bytes="$(printf '%s' "$socket_template" | wc -c)"
  if (( socket_bytes >= 108 )); then
    echo "SPATIAL_WOVR_TMP_ROOT is too long: estimated Ray Unix socket path is $socket_bytes bytes (maximum 107). Use a shorter root such as /data41/qianb2608/dyk/r." >&2
    exit 1
  fi
  temp_dir="$(mktemp -d "$temp_root/w-XXXXXXXX")"
else
  temp_dir="$(mktemp -d /tmp/spatial-wovr8-XXXXXXXX)"
fi
export TMPDIR="$temp_dir" RAY_TMPDIR="$temp_dir"
echo "Temporary directory: $TMPDIR"
[[ -x "$RLINF_ENV/bin/python" ]] || { echo "Missing Python: $RLINF_ENV/bin/python" >&2; exit 1; }
exec "$RLINF_ENV/bin/python" "${SPATIAL_WOVR_LAUNCHER:-$script_dir/spatial_wovr_launch.py}" "$@"
