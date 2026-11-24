#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# shellcheck disable=SC1091
source "$script_dir/../scripts/env.sh"

project_root="$PROJECT_ROOT"
cosmos_root="$COSMOS_POLICY_ROOT"
worldpilot_root="$WORLDPILOT_ROOT"
rlinf_root="$RLINF_ROOT"
libero_root="$LIBERO_ROOT"
rlinf_env="$RLINF_ENV"
egl_root="$EGL_ROOT"
cosmos_python="${COSMOS_PYTHON:-python}"
wrist_checkpoint="${COSMOS_WRIST_CHECKPOINT:-$CHECKPOINT_ROOT/cosmos/wovr_wrist}"
wrist_config="${COSMOS_WRIST_CONFIG:-cosmos_predict2_2b_480p_libero_cosmos_wovr_wrist_full__inference_only}"
expected_cross_view="${COSMOS_WRIST_EXPECT_CROSS_VIEW:-false}"
if [[ "$expected_cross_view" != "true" && "$expected_cross_view" != "false" ]]; then
    echo "COSMOS_WRIST_EXPECT_CROSS_VIEW must be true or false" >&2
    exit 2
fi
use_cosmos_server="${HYBRID_USE_COSMOS_SERVER:-true}"
if [[ "$use_cosmos_server" != "true" && "$use_cosmos_server" != "false" ]]; then
    echo "HYBRID_USE_COSMOS_SERVER must be true or false" >&2
    exit 2
fi
skip_gpu_memory_check="${HYBRID_SKIP_GPU_MEMORY_CHECK:-false}"
if [[ "$skip_gpu_memory_check" != "true" && "$skip_gpu_memory_check" != "false" ]]; then
    echo "HYBRID_SKIP_GPU_MEMORY_CHECK must be true or false" >&2
    exit 2
fi
official_checkpoint="${COSMOS_OFFICIAL_CHECKPOINT:-$COSMOS_CHECKPOINT_ROOT/Cosmos-Policy-LIBERO-Predict2-2B.pt}"
dataset_root="${HYBRID_COSMOS_DATASET_ROOT:-$DATA_ROOT/cosmos-policy}"
dataset_stats="$dataset_root/success_only/dataset_statistics.json"
t5_embeddings="$dataset_root/success_only/t5_embeddings.pkl"
wovr_root="${HYBRID_WAN_CKPT:-$CHECKPOINT_ROOT/wovr/RLinf-Wan-LIBERO-Object}"
grpo_config="${HYBRID_GRPO_CONFIG:-wan_libero_object_grpo_openpi_pi05_hybrid_wovr_wrist}"
extra_hydra_overrides_file="${HYBRID_EXTRA_HYDRA_OVERRIDES_FILE:-}"
pi05_checkpoint_root="${HYBRID_PI05_ROOT:-$CHECKPOINT_ROOT/pi05/RLinf-Pi05-LIBERO-SFT}"
resume_dir="${HYBRID_RESUME_DIR:-}"
experiment_name="${HYBRID_EXPERIMENT_NAME:-wovr_grpo_hybrid_wrist_pi05_sft}"
gpu_id="${HYBRID_GPU:-7}"
# Same-GPU Cosmos + learner is supported by default.  Set HYBRID_COSMOS_GPU
# only when an explicit split is desired.
cosmos_gpu_id="${HYBRID_COSMOS_GPU:-$gpu_id}"
port="${HYBRID_COSMOS_PORT:-8055}"
value_port="${HYBRID_VALUE_PORT:-8056}"
value_enabled="${HYBRID_VALUE_ENABLED:-false}"
ray_port="${HYBRID_RAY_PORT:-29592}"
# Bound Ray's host-side worker pool.  The machine exposes hundreds of logical
# CPUs, but this single-GPU embodied job only needs a small number of workers;
# leaving Ray's default can reserve tens of GB of host RAM before rollout.
ray_num_cpus="${HYBRID_RAY_NUM_CPUS:-16}"
steps="${HYBRID_STEPS:-10}"
save_inter="${HYBRID_SAVE_INTER:-5}"
screen_name="${HYBRID_SCREEN:-wovr-grpo-hybrid-wrist-gpu7}"
result_base="${HYBRID_RESULT_BASE:-$RESULTS_ROOT/wovr_grpo_hybrid_wrist}"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

server_run() {
    local result_root="${HYBRID_RESULT_ROOT:?missing result root}"
    export CUDA_VISIBLE_DEVICES="$cosmos_gpu_id"
    export PYTHONPATH="$cosmos_root:$worldpilot_root:${PYTHONPATH:-}"
    export COSMOS_DISABLE_CUDNN=1
    export COSMOS_DISTRIBUTED_BACKEND=gloo
    export WANDB_MODE=disabled
    export COSMOS_WOVR_ACTION_DROPOUT_RATE="${HYBRID_ACTION_DROPOUT:-0.1}"
    cd "$worldpilot_root"
    exec "$cosmos_python" -m cosmos_bridge.cosmos_server \
        --cosmos_root "$cosmos_root" \
        --cosmos_config "$wrist_config" \
        --cosmos_ckpt "$wrist_checkpoint" \
        --cosmos_dataset_stats "$dataset_stats" \
        --cosmos_t5_embeddings "$t5_embeddings" \
        --suite libero --chunk_size 8 --action_dim 7 --proprio_dim 9 \
        --host 127.0.0.1 --port "$port" \
        >>"$result_root/cosmos_server.log" 2>&1
}

value_server_run() {
    local result_root="${HYBRID_RESULT_ROOT:?missing result root}"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export PYTHONPATH="$cosmos_root:$worldpilot_root:${PYTHONPATH:-}"
    export COSMOS_DISABLE_CUDNN=1
    export COSMOS_DISTRIBUTED_BACKEND=gloo
    export WANDB_MODE=disabled
    cd "$worldpilot_root"
    exec "$cosmos_python" -m cosmos_bridge.cosmos_server \
        --cosmos_root "$cosmos_root" \
        --cosmos_config cosmos_predict2_2b_480p_libero__inference_only \
        --cosmos_ckpt "$official_checkpoint" \
        --cosmos_dataset_stats "$dataset_stats" \
        --cosmos_t5_embeddings "$t5_embeddings" \
        --suite libero --chunk_size 16 --action_dim 7 --proprio_dim 9 \
        --host 127.0.0.1 --port "$value_port" \
        >>"$result_root/value_server.log" 2>&1
}

train_inside() {
    local result_root="${HYBRID_RESULT_ROOT:?missing result root}"
    local -a extra_hydra_overrides=()
    local -a resume_overrides=()
    if [[ -n "$extra_hydra_overrides_file" ]]; then
        [[ -f "$extra_hydra_overrides_file" ]] || {
            echo "Missing Hydra override file: $extra_hydra_overrides_file" >&2
            exit 2
        }
        mapfile -t extra_hydra_overrides < <(
            sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$extra_hydra_overrides_file"
        )
    fi
    if [[ -n "$resume_dir" ]]; then
        [[ -d "$resume_dir" ]] || {
            echo "Missing GRPO resume directory: $resume_dir" >&2
            exit 2
        }
        resume_overrides+=("runner.resume_dir=$resume_dir")
    fi
    cd "$rlinf_root"
    source "$rlinf_env/bin/activate"
    export RLINF_ROOT="$rlinf_root"
    export EMBODIED_PATH="$rlinf_root/examples/embodiment"
    export LIBERO_ROOT="$libero_root"
    export LIBERO_CONFIG_PATH="$libero_root/libero/libero"
    export PYTHONPATH="$libero_root:$rlinf_root:$worldpilot_root:$cosmos_root:${PYTHONPATH:-}"
    export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export CUDA_VISIBLE_DEVICES=0
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export RAY_memory_monitor_refresh_ms=0 PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export MPLCONFIGDIR=/tmp/wovr-grpo-hybrid-wrist-matplotlib
    mkdir -p "$MPLCONFIGDIR"

    ray start --head --port="$ray_port" --num-cpus="$ray_num_cpus" --num-gpus=1 --memory=461708984320
    trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

    python examples/embodiment/train_embodied_agent.py \
        --config-name "$grpo_config" \
        'cluster.component_placement={actor\,env\,rollout:0}' \
        "pi05_checkpoint_root=$pi05_checkpoint_root" \
        "runner.logger.log_path=$result_root" \
        "runner.logger.experiment_name=$experiment_name" \
        "runner.max_steps=$steps" \
        "runner.save_interval=$save_inter" \
        "++env.train.cosmos_host=127.0.0.1" \
        "++env.train.cosmos_port=$port" \
        "++env.train.cosmos_value_enabled=$value_enabled" \
        "++env.train.cosmos_value_host=127.0.0.1" \
        "++env.train.cosmos_value_port=$value_port" \
        "env.train.wan_wm_hf_ckpt_path=$wovr_root" \
        "env.train.video_cfg.video_base_dir=$result_root/video/train" \
        "env.eval.video_cfg.video_base_dir=$result_root/video/eval" \
        "${extra_hydra_overrides[@]}" \
        "${resume_overrides[@]}" \
        >"$result_root/train.log" 2>&1
}

coordinator() {
    local result_root="${HYBRID_RESULT_ROOT:?missing result root}"
    exec > >(tee -a "$result_root/coordinator.log") 2>&1
    printf 'STARTING\n' >"$result_root/status"
    local server_pid=""
    if [[ "$use_cosmos_server" == "true" ]]; then
        "$script_path" --server &
        server_pid="$!"
    else
        echo "Cosmos server disabled for this run; using Wan + LIBERO state bridge only."
    fi
    local value_server_pid=""
    if [[ "$value_enabled" == "true" ]]; then
        if "$cosmos_python" -c "import sys; sys.path[:0]=['$cosmos_root','$worldpilot_root']; from cosmos_bridge.cosmos_client import CosmosClient; d=CosmosClient(host='127.0.0.1',port=$value_port).query_metadata(); assert not d.get('wovr_action_contract',False) and int(d.get('chunk_size',-1))==16 and d.get('checkpoint_load_audit',{}).get('verified') is True" >/dev/null 2>&1; then
            echo "Reusing external Cosmos value server on 127.0.0.1:$value_port"
        else
            "$script_path" --value-server &
            value_server_pid="$!"
        fi
    fi
    cleanup() {
        if [[ -n "$server_pid" ]]; then
            kill "$server_pid" 2>/dev/null || true
            wait "$server_pid" 2>/dev/null || true
        fi
        if [[ -n "$value_server_pid" ]]; then
            kill "$value_server_pid" 2>/dev/null || true
            wait "$value_server_pid" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM
    trap 'printf "FAILED\n" >"$result_root/status"' ERR

    local ready=0
    if [[ "$use_cosmos_server" == "false" ]]; then
        ready=1
    else
        for _ in $(seq 1 240); do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            if "$cosmos_python" -c "import sys; sys.path[:0]=['$cosmos_root','$worldpilot_root']; from cosmos_bridge.cosmos_client import CosmosClient; d=CosmosClient(host='127.0.0.1',port=$port).query_metadata(); assert d.get('wovr_action_contract',False) is True and int(d.get('chunk_size',-1))==8 and bool(d.get('cross_view_primary_context',False)) is ('$expected_cross_view' == 'true') and d.get('checkpoint_load_audit',{}).get('verified') is True" >/dev/null 2>&1; then
                ready=1
                break
            fi
            sleep 2
        done
    fi
    if ((ready == 0)); then
        printf 'SERVER_FAILED\n' >"$result_root/status"
        tail -100 "$result_root/cosmos_server.log" || true
        exit 1
    fi
    if [[ "$value_enabled" == "true" ]]; then
        local value_ready=0
        for _ in $(seq 1 240); do
            if [[ -n "$value_server_pid" ]] && ! kill -0 "$value_server_pid" 2>/dev/null; then
                break
            fi
            if "$cosmos_python" -c "import sys; sys.path[:0]=['$cosmos_root','$worldpilot_root']; from cosmos_bridge.cosmos_client import CosmosClient; d=CosmosClient(host='127.0.0.1',port=$value_port).query_metadata(); assert not d.get('wovr_action_contract',False) and int(d.get('chunk_size',-1))==16 and d.get('checkpoint_load_audit',{}).get('verified') is True" >/dev/null 2>&1; then
                value_ready=1
                break
            fi
            sleep 2
        done
        if ((value_ready == 0)); then
            printf 'VALUE_SERVER_FAILED\n' >"$result_root/status"
            tail -100 "$result_root/value_server.log" || true
            exit 1
        fi
    fi
    printf 'SERVER_READY\n' >"$result_root/status"

    if [[ "$use_cosmos_server" == "true" ]]; then
        if [[ "$skip_gpu_memory_check" == "true" ]]; then
            echo "Skipping nvidia-smi resident-memory check for GPU${cosmos_gpu_id} (HYBRID_SKIP_GPU_MEMORY_CHECK=true)."
        else
            local loaded_mb
            loaded_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$cosmos_gpu_id" | tr -d '[:space:]')"
            echo "Cosmos wrist server resident memory on GPU${cosmos_gpu_id}: ${loaded_mb} MiB"
            max_loaded="${HYBRID_MAX_LOADED_GPU_MEMORY_MIB:-40000}"
            if [[ -z "$loaded_mb" || "$loaded_mb" -gt "$max_loaded" ]]; then
                echo "Refusing to load actor/env: Cosmos already occupies ${loaded_mb:-unknown} MiB" >&2
                exit 1
            fi
        fi
    fi

    export EGL_GPU="$gpu_id"
    export EGL_DOCKER_NETWORK=host
    export EGL_SHM_SIZE=32g
    export EGL_EXTRA_MOUNT="${EGL_EXTRA_MOUNT:-$DATA_ROOT}"
    "$egl_root/run_with_docker_egl.sh" --gpu "$gpu_id" -- \
        bash -lc "export HYBRID_RESULT_ROOT='$result_root' HYBRID_COSMOS_PORT='$port' HYBRID_VALUE_PORT='$value_port' HYBRID_VALUE_ENABLED='$value_enabled' HYBRID_RAY_PORT='$ray_port' HYBRID_RAY_NUM_CPUS='$ray_num_cpus' HYBRID_STEPS='$steps' HYBRID_SAVE_INTER='$save_inter' HYBRID_WAN_CKPT='$wovr_root' HYBRID_GRPO_CONFIG='$grpo_config' HYBRID_EXTRA_HYDRA_OVERRIDES_FILE='$extra_hydra_overrides_file' HYBRID_USE_COSMOS_SERVER='$use_cosmos_server' HYBRID_COSMOS_GPU='$cosmos_gpu_id' HYBRID_PI05_ROOT='$pi05_checkpoint_root' HYBRID_RESUME_DIR='$resume_dir' HYBRID_EXPERIMENT_NAME='$experiment_name'; exec bash '$script_path' --train-inside"
    printf 'DONE\n' >"$result_root/status"
}

case "${1:-}" in
    --server) server_run ;;
    --value-server) value_server_run ;;
    --train-inside) train_inside ;;
    --coordinator) coordinator ;;
    "")
        if screen -ls 2>/dev/null | grep -q "\.${screen_name}[[:space:]]"; then
            echo "Screen already exists: $screen_name" >&2
            exit 1
        fi
        required_artifacts=(
            "$wovr_root/model-00001.safetensors"
            "$wovr_root/Wan2.2_VAE.pth"
        )
        if [[ "$use_cosmos_server" == "true" ]]; then
            required_artifacts+=(
                "$wrist_checkpoint/model/.metadata"
                "$dataset_stats"
                "$t5_embeddings"
            )
        fi
        for required in "${required_artifacts[@]}"; do
            [[ -s "$required" ]] || { echo "Missing artifact: $required" >&2; exit 1; }
        done
        pi05_artifact_found=false
        for pi05_artifact in \
            "$pi05_checkpoint_root/model.safetensors" \
            "$pi05_checkpoint_root/model_state_dict/full_weights.pt" \
            "$pi05_checkpoint_root/actor/model_state_dict/full_weights.pt"; do
            if [[ -s "$pi05_artifact" ]]; then
                pi05_artifact_found=true
                break
            fi
        done
        [[ "$pi05_artifact_found" == true ]] || {
            echo "Missing Pi05 checkpoint under: $pi05_checkpoint_root" >&2
            echo "Expected model.safetensors or model_state_dict/full_weights.pt" >&2
            exit 1
        }
        if [[ "$skip_gpu_memory_check" == "true" ]]; then
            echo "Skipping nvidia-smi preflight for GPU${gpu_id} (HYBRID_SKIP_GPU_MEMORY_CHECK=true)."
        else
            max_existing="${HYBRID_MAX_EXISTING_GPU_MEMORY_MIB:-20000}"
            used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" | tr -d '[:space:]')"
            [[ -n "$used_mb" && "$used_mb" -le "$max_existing" ]] || { echo "GPU$gpu_id has too much memory in use: ${used_mb:-unknown} MiB (limit ${max_existing})" >&2; exit 1; }
        fi
        if [[ -n "${HYBRID_RESULT_ROOT:-}" ]]; then
            if [[ -e "$HYBRID_RESULT_ROOT" ]]; then
                echo "Refusing to reuse an existing GRPO result directory: $HYBRID_RESULT_ROOT" >&2
                exit 4
            fi
            export HYBRID_RESULT_ROOT
        else
            run_id="$(date '+%Y%m%d_%H%M%S')"
            export HYBRID_RESULT_ROOT="$result_base/$run_id"
        fi
        mkdir -p "$HYBRID_RESULT_ROOT"
        printf 'STARTING\n' >"$HYBRID_RESULT_ROOT/status"
        screen -dmS "$screen_name" bash -lc "export HYBRID_RESULT_ROOT='$HYBRID_RESULT_ROOT' HYBRID_GPU='$gpu_id' HYBRID_COSMOS_GPU='$cosmos_gpu_id' HYBRID_COSMOS_PORT='$port' HYBRID_VALUE_PORT='$value_port' HYBRID_VALUE_ENABLED='$value_enabled' HYBRID_RAY_PORT='$ray_port' HYBRID_RAY_NUM_CPUS='$ray_num_cpus' HYBRID_STEPS='$steps' HYBRID_SAVE_INTER='$save_inter' HYBRID_WAN_CKPT='$wovr_root' HYBRID_GRPO_CONFIG='$grpo_config' HYBRID_EXTRA_HYDRA_OVERRIDES_FILE='$extra_hydra_overrides_file' HYBRID_USE_COSMOS_SERVER='$use_cosmos_server' HYBRID_SKIP_GPU_MEMORY_CHECK='$skip_gpu_memory_check' HYBRID_COSMOS_DATASET_ROOT='$dataset_root' COSMOS_WRIST_CHECKPOINT='$wrist_checkpoint' COSMOS_WRIST_CONFIG='$wrist_config' COSMOS_WRIST_EXPECT_CROSS_VIEW='$expected_cross_view' HYBRID_PI05_ROOT='$pi05_checkpoint_root' HYBRID_RESUME_DIR='$resume_dir' HYBRID_EXPERIMENT_NAME='$experiment_name'; exec bash '$script_path' --coordinator"
        echo "screen=$screen_name"
        echo "result_root=$HYBRID_RESULT_ROOT"
        ;;
    *) echo "Usage: $0 [--server|--value-server|--train-inside|--coordinator]" >&2; exit 2 ;;
esac
