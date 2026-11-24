"""Prepare and supervise single-node Spatial dual-view GRPO on four/eight GPUs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WRIST_CONFIG = "cosmos_predict2_2b_480p_libero_cosmos_wovr_wrist_full__inference_only"


def child_log_error(message: str, log: Path, *, returncode=None) -> str:
    """Keep the actual child exception visible without reading an entire log."""
    try:
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        tail = "\n".join(tail.splitlines()[-40:])
    except OSError as error:
        tail = f"(log unavailable: {type(error).__name__})"
    code = f" (exit={returncode})" if returncode is not None else ""
    return f"{message}{code}; child log: {log}\n--- last child log lines ---\n{tail}"


def link_sources(destination: Path, sources: list[Path]) -> None:
    """Merge source trees using links, with later sources taking precedence.

    Never modify the external installations. Real directories allow selected
    modules to override the installed source without replacing its scheduler,
    actor, or native policy loader. Each launch gets its own destination.
    """
    destination.mkdir(parents=True, exist_ok=True)
    children: dict[str, list[Path]] = {}
    for source in sources:
        if source.is_dir():
            for child in source.iterdir():
                if child.name not in {"__pycache__", ".git"}:
                    children.setdefault(child.name, []).append(child)
    for name, candidates in children.items():
        selected = candidates[-1]
        target = destination / name
        if selected.is_dir() and len(candidates) > 1:
            link_sources(target, candidates)
        else:
            target.symlink_to(selected.resolve(), target_is_directory=selected.is_dir())


def make_endpoints(gpus: str, first_port: int) -> list[dict]:
    """Map environment ranks to one service per visible GPU, in visible order."""
    devices = [part.strip() for part in gpus.split(",")]
    world_size = len(devices)
    if (world_size not in (4, 8) or any(not part.isdecimal() for part in devices)
            or len({int(part) for part in devices}) != world_size):
        raise ValueError("SPATIAL_WOVR_GPUS must specify four or eight distinct numeric GPU IDs")
    if not 1024 <= first_port <= 65536 - world_size:
        raise ValueError(f"SPATIAL_COSMOS_PORT must be in [1024, {65536 - world_size}]")
    return [{"host": "127.0.0.1", "port": first_port + rank} for rank in range(world_size)]


def placement_world_size(cfg) -> int:
    placement = dict(cfg.cluster.component_placement)
    for count in (4, 8):
        if placement == {"actor,env,rollout": f"0-{count - 1}"}:
            return count
    raise ValueError("This launcher requires actor, env, rollout colocated on 0-3 or 0-7")


def prepare_runtime(run_dir: Path) -> None:
    """Combine the working installed backend with the modified world models."""
    project = Path(os.environ["PROJECT_ROOT"])
    baseline = Path(os.environ["RLINF_BASE_ROOT"])
    snapshot = project / "third_party/RLinf/rlinf"
    cosmos_base = Path(os.environ["COSMOS_BASE_ROOT"])
    for required in [
        baseline / "rlinf/workers/env/env_worker.py",
        cosmos_base / "cosmos_policy/_src/predict2/datasets/utils.py",
        cosmos_base / "cosmos_policy/datasets/libero_dataset.py",
    ]:
        if not required.is_file():
            raise FileNotFoundError(f"Missing runtime source: {required}")
    rlinf_root = run_dir / "code/rlinf"
    rlinf_root.mkdir(parents=True)
    for name in ["pyproject.toml", "examples", "LICENSE", "requirements", "ray_utils"]:
        source = baseline / name
        if source.exists():
            (rlinf_root / name).symlink_to(source, target_is_directory=source.is_dir())
    # Build the package structure for the environment and runner overrides.
    # Scheduler, GRPO actor, and native Pi05 loading stay together as installed.
    rlinf_package = rlinf_root / "rlinf"
    rlinf_package.mkdir(parents=True)
    for source in (baseline / "rlinf").iterdir():
        if source.name in {"envs", "runners", "__pycache__"}:
            continue
        (rlinf_package / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    from spatial_wovr_runtime import wait_for_env_offload_before_update
    from spatial_wovr_communication import install_collective_overlay

    collective_overlay = install_collective_overlay(rlinf_package, baseline, link_sources)

    runner_root = rlinf_package / "runners"
    link_sources(runner_root, [baseline / "rlinf/runners"])
    runner_overlay = runner_root / "embodied_runner.py"
    runner_source = runner_overlay.read_text()
    runner_overlay.unlink()
    runner_overlay.write_text(wait_for_env_offload_before_update(runner_source))
    env_root = rlinf_package / "envs"
    env_root.mkdir()
    for source in (baseline / "rlinf/envs").iterdir():
        if source.name in {"world_model", "wrappers", "__init__.py", "action_utils.py", "__pycache__"}:
            continue
        (env_root / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    for name in ["__init__.py", "action_utils.py"]:
        (env_root / name).symlink_to(snapshot / "envs" / name)
    link_sources(env_root / "world_model", [baseline / "rlinf/envs/world_model", snapshot / "envs/world_model"])
    from spatial_wovr_video import install_video_overlay

    wrapper_root = env_root / "wrappers"
    link_sources(wrapper_root, [baseline / "rlinf/envs/wrappers"])
    video_overlay = install_video_overlay(
        wrapper_root, baseline / "rlinf/envs/wrappers/record_video.py"
    )
    cosmos_root = run_dir / "code/cosmos"
    link_sources(cosmos_root / "cosmos_policy", [
        cosmos_base / "cosmos_policy", project / "third_party/cosmos-policy/cosmos_policy",
    ])
    os.environ["RLINF_ROOT"] = str(rlinf_root)
    os.environ["COSMOS_POLICY_ROOT"] = str(cosmos_root)
    os.environ["COSMOS_LOCAL_INFERENCE_CHECKPOINT"] = os.environ["COSMOS_WRIST_CHECKPOINT"]
    os.environ["EMBODIED_PATH"] = str(baseline / "examples/embodiment")
    paths = [
        SCRIPT_DIR, rlinf_root, cosmos_root, Path(os.environ["WORLDPILOT_ROOT"]),
        Path(os.environ["LIBERO_ROOT"]), Path(os.environ["OPENPI_ROOT"]) / "src",
        Path(os.environ["DIFFSYNTH_ROOT"]),
    ]
    os.environ["PYTHONPATH"] = ":".join(map(str, paths))
    # Current process was started before PYTHONPATH was set.
    sys.path[:0] = list(map(str, paths))
    benchmark = Path(os.environ["LIBERO_BENCHMARK_ROOT"])
    for suffix in ["assets", "bddl_files/libero_spatial", "init_files/libero_spatial"]:
        if not (benchmark / suffix).is_dir():
            raise FileNotFoundError(benchmark / suffix)
    import yaml

    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"), "assets": str(benchmark / "assets"),
        "datasets": str(benchmark.parent / "datasets"),
    }))
    (run_dir / "source_paths.json").write_text(json.dumps({
        "rlinf_backend": str(baseline), "world_model_overlay": str(snapshot / "envs/world_model"),
        "collective_overlay": str(collective_overlay) if collective_overlay else None,
        "collective_broadcast_reraises": collective_overlay is not None,
        "cosmos_snapshot": str(project / "third_party/cosmos-policy"),
        "cosmos_fallback": str(cosmos_base), "pythonpath": list(map(str, paths)),
        "runner_overlay": str(runner_overlay),
        "runner_overlay_sha256": sha256(runner_overlay),
        "record_video_overlay": str(video_overlay),
        "record_video_overlay_sha256": sha256(video_overlay),
        "wait_for_env_offload_before_update": True,
        "gpu_count": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")),
    }, indent=2) + "\n")


def validate_batch(cfg) -> int:
    """Reject incompatible placement, action contracts, and minibatch sizes."""
    train, actor = cfg.env.train, cfg.actor
    if cfg.runner.get("use_training_pipeline", False) or cfg.runner.get("overlap_env_bootstrap", False):
        raise ValueError("World-model memory handoff requires non-pipeline training and overlap_env_bootstrap=false")
    world_size = placement_world_size(cfg)
    if int(cfg.rollout.pipeline_stage_num) != 1:
        raise ValueError("Per-GPU wrist services require pipeline_stage_num=1")
    if actor.global_batch_size % (actor.micro_batch_size * world_size):
        raise ValueError(f"global batch must divide micro batch * {world_size} GPUs")
    if train.total_num_envs % (world_size * cfg.algorithm.group_size):
        raise ValueError("Each GPU must receive whole GRPO groups")
    if train.max_steps_per_rollout_epoch % 8:
        raise ValueError("Rollout length must be divisible by the eight-action horizon")
    samples = train.total_num_envs * train.rollout_epoch * (train.max_steps_per_rollout_epoch // 8)
    if samples % actor.global_batch_size:
        raise ValueError(f"{samples} rollout chunk samples do not divide global batch {actor.global_batch_size}")
    if not (train.use_cosmos_wrist and train.cosmos_wrist_use_wovr_contract):
        raise ValueError("This run requires Wan + strict Cosmos wrist")
    cosmos_batch = train.get("cosmos_batch_size", 2)
    if isinstance(cosmos_batch, bool) or not isinstance(cosmos_batch, int) or cosmos_batch < 1:
        raise ValueError("cosmos_batch_size must be a positive integer")
    video_stride = train.get("video_cfg", {}).get("record_every_n_episodes", 1)
    if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
        raise ValueError("record_every_n_episodes must be a positive integer")
    if train.get("skip_successful_env_generation", False):
        if (not train.get("mask_after_first_success", False)
                or train.auto_reset or train.ignore_terminations):
            raise ValueError("Skipping successful environments requires first-success masking, manual resets and terminations")
        unsupported = (
            "use_libero_state_bridge", "use_policy_state", "cosmos_value_enabled",
            "cosmos_use_predicted_proprio", "cosmos_wrist_cross_view_primary_history",
        )
        if any(train.get(field, False) for field in unsupported):
            raise ValueError("Skipping successful environments supports only stateless strict v3 wrist without a simulator bridge, value/proprio or cross-view history")
    if train.get("mask_after_first_success", False):
        if train.auto_reset or train.ignore_terminations:
            raise ValueError("First-success masking requires manual resets and terminations")
        if cfg.algorithm.reward_type != "action_level" or cfg.algorithm.logprob_type == "chunk_level":
            raise ValueError("First-success masking requires per-action rewards and log probabilities")
    from spatial_wovr_kl import validate_kl_config
    from spatial_wovr_hallucination import validate_hallucination_trust_config
    from spatial_wovr_hybrid import validate_hybrid_config

    validate_kl_config(cfg.algorithm)
    validate_hallucination_trust_config(train.get("hallucination_trust", None))
    validate_hybrid_config(cfg, world_size=world_size)
    if train.get("use_kir_frame_pool", False):
        if train.use_libero_state_bridge or train.use_policy_state:
            raise ValueError("Offline KIR requires the simulator bridge and policy state to be disabled")
        if train.auto_reset or train.ignore_terminations:
            raise ValueError("Offline KIR requires grouped manual resets and success masking")
        fraction = float(train.kir_fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("kir_fraction must be between zero and one")
    elif not train.use_libero_state_bridge:
        raise ValueError("Dual-view resets require a frame pool or LIBERO state bridge")
    for model in [actor.model, cfg.rollout.model]:
        if model.num_action_chunks != 8 or model.openpi.action_horizon != 10:
            raise ValueError("Pi05 12k must retain horizon=10 and execute action_chunk=8")
        if model.openpi.action_chunk != 8 or model.openpi.discrete_state_input:
            raise ValueError("Unexpected Pi05 execution/state-token contract")
    if train.chunk != 8 or train.cosmos_wrist_action_horizon != 8:
        raise ValueError("Both world models require eight actions")
    if not train.cosmos_wrist_primary_reference or train.cosmos_wrist_cross_view_primary_history:
        raise ValueError("The v3 checkpoint needs primary reference, without v4 cross-view history")
    if train.cosmos_gripper_action_mode != "already_cosmos" or train.openpi_gripper_action_mode != "libero_01_to_signed":
        raise ValueError("Pi05 gripper must be converted exactly once at the environment boundary")
    if train.cosmos_wrist_input_camera_space != "rlinf_runtime_rot180":
        raise ValueError("Unexpected camera orientation")
    if train.cosmos_use_predicted_proprio or train.cosmos_value_enabled:
        raise ValueError("The strict wrist checkpoint has no proprio/value prediction head")
    profile = actor.get("gpu_memory_profile", {})
    if profile.get("enabled", False) and profile.get("expected_optimizer_updates") == 1:
        if (samples != actor.global_batch_size or cfg.algorithm.update_epoch != 1
                or cfg.runner.max_steps != 1 or cfg.runner.max_epochs != 1
                or cfg.runner.save_interval > 0 or cfg.runner.val_check_interval > 0):
            raise ValueError("One-update GPU peak probe requires one global batch, one update/step, and disabled save/eval")
    if int(cfg.runner.val_check_interval) > 0:
        evaluation = cfg.env.eval
        if (evaluation.get("env_type", "libero") != "libero"
                or evaluation.get("task_suite_name", "libero_spatial") != "libero_spatial"
                or evaluation.get("episodes_per_task", 0) != 50
                or int(evaluation.group_size) != 1
                or not evaluation.auto_reset or evaluation.ignore_terminations
                or not evaluation.get("is_eval", False)
                or not evaluation.get("use_fixed_reset_state_ids", False)):
            raise ValueError("Periodic evaluation requires real LIBERO Spatial, 50 unique trials/task, fixed auto-resets and success terminations")
        if evaluation.get("specific_reset_id") is not None or evaluation.get("task_id_filter") is not None:
            raise ValueError("Periodic evaluation must include all 10 Spatial tasks")
        if evaluation.total_num_envs % world_size or evaluation.total_num_envs < world_size:
            raise ValueError(f"Real evaluation environment count must divide {world_size} GPUs")
        rank_trials = math.ceil(500 / world_size)
        waves = math.ceil(rank_trials / (evaluation.total_num_envs // world_size))
        budget = evaluation.rollout_epoch * evaluation.max_steps_per_rollout_epoch
        if budget < waves * evaluation.max_episode_steps:
            raise ValueError(f"Real evaluation needs at least {waves} waves to finish all 500 trials")
        if cfg.runner.save_interval <= 0 or cfg.runner.save_interval % cfg.runner.val_check_interval:
            raise ValueError("Checkpoint interval must be a positive multiple of evaluation interval")
    return samples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def wrist_records(checkpoint: Path) -> list[tuple[Path, dict]]:
    """Check every DCP shard, including the final bytes of an active download."""
    manifest = json.loads((checkpoint.parent / "checkpoint_manifest.json").read_text())
    if checkpoint.name != manifest["checkpoint"]:
        raise ValueError("COSMOS_WRIST_CHECKPOINT must point to iter_000020000")
    return [(checkpoint.parent / item["path"], item) for item in manifest["files"]]


def validate_wrist_runtime(runtime: dict) -> None:
    """Match the released v3 model's recorded inference contract."""
    expected = {
        "cosmos_config": WRIST_CONFIG, "checkpoint_relative_path": "iter_000020000",
        "task_suite": "libero_spatial", "action_contract": "wovr13",
        "action_chunk": 8, "action_dim": 7, "primary_reference": True,
        "cross_view_primary_context": False, "runtime_camera_space": "rlinf_runtime_rot180",
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise ValueError(f"Wrist runtime contract differs at {key}: {runtime.get(key)!r}, expected {value!r}")


def check_assets(cfg, full: bool, run_dir: Path, *, verify_wrist: bool = True) -> bool:
    """Verify resident assets; permit pending wrist files only in --check."""
    policy = Path(cfg.pi05_checkpoint_root)
    weight = policy / "actor/model_state_dict/full_weights.pt"
    manifest = json.loads((policy.parent / "model_manifest.json").read_text())
    policy_record = next(row for row in manifest["checkpoints"] if row["step"] == 12000)
    records = [(weight, policy_record["size_bytes"], policy_record["sha256"])]
    json.loads((policy / "physical-intelligence/libero/norm_stats.json").read_text())
    wan = Path(cfg.env.train.wan_wm_hf_ckpt_path)
    for filename, size in [
        ("model-00001.safetensors", 10050753584), ("Wan2.2_VAE.pth", 2818839170),
        ("resnet_rm.pth", 44784610),
    ]:
        path = wan / filename
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Incomplete Wan asset: {path}")
    rm_cfg = cfg.env.train.get("reward_model", None)
    if rm_cfg is not None and rm_cfg.get("from_pretrained", None):
        rm_path = Path(str(rm_cfg.from_pretrained))
        if not rm_path.is_file() or rm_path.stat().st_size == 0:
            raise FileNotFoundError(f"Configured reward_model.from_pretrained not found: {rm_path}")
    if cfg.env.train.get("use_kir_frame_pool", False):
        from kir_frame_pool import FramePool
        from rlinf.envs.world_model.dual_view_kir import validate_kir_config

        validate_kir_config(cfg.env.train)
        pool = FramePool(Path(cfg.env.train.kir_frame_pool_path))
        pool.check_sources()
        if pool.manifest["chunk"] != 8 or pool.manifest["summary"]["tasks"] != 10:
            raise ValueError("The Spatial KIR launcher requires all ten tasks and eight-action chunks")
        (run_dir / "kir_pool_check.json").write_text(json.dumps({
            "pool_path": str(pool.root), "pool_sha256": pool.manifest["pool_sha256"],
            "summary": pool.manifest["summary"],
            "selection": pool.manifest.get("selection"),
            "task_deadlines": pool.task_success_deadlines(cfg.env.train.success_horizon_grace_steps)
            if cfg.env.train.get("success_horizon_grace_steps") is not None else None,
            "kir_fraction": float(cfg.env.train.kir_fraction) if cfg.env.train.enable_kir else 0.0,
            "source_fingerprints_checked": True,
        }, indent=2) + "\n")
    elif not any(p for p in (wan / "dataset").glob("*.npy") if "_kir" not in p.name):
        raise FileNotFoundError(f"No non-KIR initial trajectories in {wan / 'dataset'}")
    for key in ["COSMOS_VAE", "TOKENIZER_DIR"]:
        path = Path(os.environ[key])
        if key == "TOKENIZER_DIR":
            path /= "paligemma_tokenizer.model"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        if key == "COSMOS_VAE" and path.stat().st_size != 507609880:
            raise ValueError(f"Incomplete Cosmos VAE: {path}")
    records.extend([
        (Path(os.environ["COSMOS_DATASET_STATS"]), 2372, "5b119a98ad7824507ddff3c6c7507ca244261ee416dcfa623876702160c580d3"),
        (Path(os.environ["COSMOS_T5_EMBEDDINGS"]), 41957938, "8a03499676c6c196127c577144b9fd09bb02c30f9caf058adbcef09bb99ad8f5"),
    ])
    pending = []
    if verify_wrist:
        checkpoint = Path(os.environ["COSMOS_WRIST_CHECKPOINT"])
        try:
            for path, record in wrist_records(checkpoint):
                if not path.is_file() or path.stat().st_size != record["size"]:
                    pending.append(str(path))
                else:
                    records.append((path, record["size"], record["sha256"]))
        except FileNotFoundError as exc:
            pending.append(str(exc.filename))
        for name in ["runtime_config.json", "training_summary.json"]:
            if not (checkpoint.parent / name).is_file():
                pending.append(str(checkpoint.parent / name))
        runtime_config = checkpoint.parent / "runtime_config.json"
        if runtime_config.is_file():
            validate_wrist_runtime(json.loads(runtime_config.read_text()))
    verified = []
    for path, size, expected in records:
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Incomplete asset: {path}, expected {size} bytes")
        # Statistics are small enough to verify even during the code-only check.
        if full or size < 50 * 1024**2:
            if sha256(path) != expected:
                raise ValueError(f"SHA-256 mismatch: {path}")
            verified.append(str(path))
    (run_dir / "asset_check.json").write_text(json.dumps({
        "full_checksums": full, "verified": verified, "pending_wrist_files": pending,
    }, indent=2) + "\n")
    if pending:
        print(f"Wrist download pending: {len(pending)} files; code checks may proceed.", flush=True)
        if full:
            raise ValueError("Wrist checkpoint is incomplete; training was not started")
    return not pending


def check_imports(run_dir: Path) -> None:
    """Import both runtimes and resolve Cosmos configuration without CUDA."""
    from omegaconf import OmegaConf

    resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
    # The backend accesses is_lora unconditionally. Check both model configs,
    # including the rollout-only selection that training imports do not use.
    for role in ("actor", "rollout"):
        if not isinstance(resolved[role].model.is_lora, bool):
            raise ValueError(f"{role}.model.is_lora must be explicitly true or false")
    probes = [
        (sys.executable, "from rlinf.workers.env.env_worker import EnvWorker; "
         "from rlinf.envs.world_model.world_model_wan_env import WanEnv; "
         "from rlinf.models.embodiment.openpi import get_model; "
         "from spatial_wovr_actor import WorldModelFSDPActor; "
         "from rlinf.envs.action_utils import prepare_actions; import numpy as np; "
         "raw=np.zeros((2,8,7),dtype=np.float32); raw[0,:,-1]=1; "
         "a=prepare_actions(raw, 'wan_wm', 'openpi', 8, 7, wm_env_type='libero', "
         "env_cfg={'openpi_gripper_action_mode':'libero_01_to_signed'}); "
         "assert np.all(a[0,:,-1]==-1) and np.all(a[1,:,-1]==1); "
         "print('RLinf/Wan imports and gripper boundary OK'); "
         "import sys; from train_spatial_wovr import main; "
         f"sys.argv={['train_spatial_wovr.py', '--config-path', str(run_dir), '--config-name', 'resolved_config', '--cfg', 'job', '--resolve']!r}; "
         "main()"),
        (os.environ["COSMOS_PYTHON"], "from cosmos_policy.config.config import make_config; "
         "from cosmos_policy._src.imaginaire.utils.config_helper import override; "
         "from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint; "
         f"cfg=override(make_config(), ['--', 'experiment={WRIST_CONFIG}', "
         "'model.config.tokenizer.vae_pth=' + __import__('os').environ['COSMOS_VAE']]); "
         "assert cfg.model.config.tokenizer.vae_pth == __import__('os').environ['COSMOS_VAE']; "
         "assert cfg.checkpoint.load_path == __import__('os').environ['COSMOS_LOCAL_INFERENCE_CHECKPOINT']; "
         "print('Cosmos inference config and local VAE override OK')"),
    ]
    for index, (python, code) in enumerate(probes):
        log = run_dir / f"import_check_{index}.log"
        with log.open("w") as stream:
            result = subprocess.run([python, "-c", code], stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        if result.returncode:
            print(log.read_text()[-8000:], file=sys.stderr)
            raise RuntimeError(f"Runtime import check failed: {log}")
        print(f"Runtime {index + 1}/2 imports passed: {log}", flush=True)
    if resolved.env.train.get("use_kir_frame_pool", False):
        log = run_dir / "kir_reset_check.log"
        with log.open("w") as stream:
            result = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "kir_reset_check.py"),
                "--config", str(run_dir / "resolved_config.yaml"),
                "--output", str(run_dir / "kir_reset_smoke.json"),
            ], stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            print(log.read_text()[-8000:], file=sys.stderr)
            raise RuntimeError(f"KIR CPU reset/chunk check failed: {log}")
        print(f"CPU KIR reset/chunk check passed: {log}", flush=True)


def validate_server_metadata(metadata: dict, checkpoint: Path) -> None:
    """Reject a stale server, wrong checkpoint, or incompatible action model."""
    identity = metadata.get("checkpoint_identity", {})
    if Path(identity.get("resolved_path", "")).resolve() != checkpoint.resolve():
        raise ValueError("Cosmos service loaded another checkpoint")
    if not metadata.get("wovr_action_contract") or metadata.get("chunk_size") != 8:
        raise ValueError("Cosmos service does not implement the eight-action WoVR contract")
    if metadata.get("action_dim") != 7 or metadata.get("cross_view_primary_context"):
        raise ValueError("Cosmos service does not match the v3 wrist model")
    if metadata.get("checkpoint_load_audit", {}).get("verified") is not True:
        raise ValueError("Cosmos checkpoint load was not verified")
    manifest = json.loads((checkpoint.parent / "checkpoint_manifest.json").read_text())
    expected_tree = manifest["tree_sha256"]
    if "files" in manifest:
        # Packaged updates record sha256sum-style manifest identities. The
        # serving helper hashes relative path, size, and file digest instead.
        # Derive that serving identity from the already verified file records.
        records = sorted(
            (Path(row["path"]).relative_to(checkpoint.name).as_posix(), row)
            for row in manifest["files"]
        )
        digest = hashlib.sha256()
        for relative, row in records:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(row["size"]).encode("ascii"))
            digest.update(b"\0")
            digest.update(row["sha256"].encode("ascii"))
            digest.update(b"\n")
        expected_tree = digest.hexdigest()
    if identity.get("tree_sha256") != expected_tree:
        raise ValueError("Cosmos service checkpoint identity differs from the release manifest")


def metadata_payload(response: dict) -> dict:
    """Decode the WebSocket server's inference response envelope."""
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise ValueError(f"Cosmos metadata request failed: {response}")
    if not isinstance(response.get("data"), dict):
        raise ValueError("Cosmos metadata response has no data mapping")
    return response["data"]


def query_metadata(endpoint: dict) -> dict:
    from deployment.model_server.tools import msgpack_numpy
    from websockets.sync.client import connect

    options = {"open_timeout": 2, "close_timeout": 1, "max_size": None}
    if "proxy" in inspect.signature(connect).parameters:
        options["proxy"] = None
    with connect(f"ws://{endpoint['host']}:{endpoint['port']}", **options) as connection:
        connection.recv(timeout=2)  # Initial server greeting.
        connection.send(msgpack_numpy.packb({"mode": "metadata"}))
        response = connection.recv(timeout=2)
        if isinstance(response, str):
            raise ValueError(response)
        return metadata_payload(msgpack_numpy.unpackb(response))


def stop_process(process: subprocess.Popen) -> None:
    """Terminate only a process group created by this launch."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def train(cfg, endpoints: list[dict], run_dir: Path) -> int:
    """Start one owned wrist service per GPU and one private RLinf job."""
    import torch

    torch.cuda.memory._set_allocator_settings(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    world_size = placement_world_size(cfg)
    if torch.cuda.device_count() != world_size:
        raise RuntimeError(f"Requires {world_size} visible CUDA GPUs; sees {torch.cuda.device_count()}. Use the GPU terminal.")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    output = Path(cfg.runner.logger.log_path)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "gpu_preflight.log").open("w") as gpu_log:
        for index in range(world_size):
            gpu_log.write(f"GPU {devices[index]}: {torch.cuda.get_device_name(index)}\n")
        gpu_log.flush()
        try:
            subprocess.run([
                "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                "--format=csv",
            ], stdout=gpu_log, stderr=subprocess.STDOUT, check=False, timeout=15)
        except FileNotFoundError:
            pass
    for name in ["resolved_config.yaml", "asset_check.json", "source_paths.json"]:
        (output / name).symlink_to(run_dir / name)
    if cfg.actor.get("shared_wan_pool", {}).get("enabled", False):
        (output / "shared_wan_pool").symlink_to(run_dir / "shared_wan_pool", target_is_directory=True)
    for endpoint in endpoints:
        with socket.socket() as sock:
            sock.bind((endpoint["host"], endpoint["port"]))
    processes = []
    streams = []
    memory_monitor = None
    profile_memory = bool(cfg.actor.get("gpu_memory_profile", {}).get("enabled", False))
    monitor_memory = profile_memory or os.environ.get("SPATIAL_WOVR_MONITOR_MEMORY") == "1"

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    for signum in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(signum, interrupted)
    try:
        if monitor_memory:
            stream = (output / "gpu_memory_monitor.log").open("w")
            streams.append(stream)
            memory_monitor = subprocess.Popen([
                sys.executable, str(SCRIPT_DIR / "monitor_gpu_memory.py"),
                "--gpus", ",".join(devices), "--interval", "0.2",
                "--run-dir", str(output), "--output", str(output / "gpu_peak/device"),
            ], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        for rank, endpoint in enumerate(endpoints):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank], WANDB_MODE="disabled",
                       COSMOS_DISTRIBUTED_BACKEND="gloo", COSMOS_WOVR_ACTION_DROPOUT_RATE="0.1")
            env["COSMOS_DISABLE_CUDNN"] = os.environ.get("COSMOS_DISABLE_CUDNN", "1")
            pool = cfg.actor.get("shared_wan_pool", {})
            if pool.get("enabled") and pool.get("peer_allocator_gib"):
                env["SPATIAL_COSMOS_ALLOCATOR"] = json.dumps({
                    "budget_gib": float(pool.peer_allocator_gib.cosmos), "rank": rank,
                    "log_dir": str(pool.log_dir),
                })
            # Each independent Cosmos process owns its distributed rendezvous.
            for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                env.pop(key, None)
            with socket.socket() as rendezvous:
                rendezvous.bind(("127.0.0.1", 0))
                rendezvous_port = rendezvous.getsockname()[1]
            env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rendezvous_port),
                       RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
            stream = (output / f"cosmos_rank{rank}.log").open("w")
            streams.append(stream)
            process = subprocess.Popen([
                os.environ["COSMOS_PYTHON"], "-m", "cosmos_bridge.cosmos_server",
                "--cosmos_root", os.environ["COSMOS_POLICY_ROOT"], "--cosmos_config", WRIST_CONFIG,
                "--cosmos_ckpt", os.environ["COSMOS_WRIST_CHECKPOINT"],
                "--cosmos_vae", os.environ["COSMOS_VAE"],
                "--cosmos_dataset_stats", os.environ["COSMOS_DATASET_STATS"],
                "--cosmos_t5_embeddings", os.environ["COSMOS_T5_EMBEDDINGS"],
                "--suite", "libero", "--chunk_size", "8", "--action_dim", "7", "--proprio_dim", "9",
                "--host", endpoint["host"], "--port", str(endpoint["port"]),
            ], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
        ready = set()
        deadline = time.monotonic() + float(os.environ.get("SPATIAL_COSMOS_START_TIMEOUT", "1200"))
        last_report = 0.0
        while len(ready) != world_size:
            if time.monotonic() > deadline:
                pending = [rank for rank in range(world_size) if rank not in ready]
                details = "\n".join(child_log_error(f"Wrist server rank {rank} not ready",
                                   output / f"cosmos_rank{rank}.log") for rank in pending)
                raise TimeoutError(f"Cosmos startup timed out; ready ranks={sorted(ready)}\n{details}")
            for rank, process in enumerate(processes):
                if process.poll() is not None:
                    raise RuntimeError(child_log_error(f"Wrist server rank {rank} exited",
                                       output / f"cosmos_rank{rank}.log", returncode=process.returncode))
                if rank in ready:
                    continue
                try:
                    metadata = query_metadata(endpoints[rank])
                except (OSError, TimeoutError):
                    continue
                validate_server_metadata(metadata, Path(os.environ["COSMOS_WRIST_CHECKPOINT"]))
                (output / f"cosmos_rank{rank}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                ready.add(rank)
            if time.monotonic() - last_report > 20:
                print(f"Wrist services ready: {len(ready)}/{world_size}. Logs: {output}", flush=True)
                last_report = time.monotonic()
            if len(ready) != world_size:
                time.sleep(2)
        command = [sys.executable, str(SCRIPT_DIR / "train_spatial_wovr.py"),
                   "--config-path", str(run_dir), "--config-name", "resolved_config",
                   f"hydra.run.dir={output}", "hydra.output_subdir=null", "hydra.job.chdir=false"]
        # The resolved file is standalone; it contains no Hydra defaults.
        stream = (output / "train.log").open("w")
        streams.append(stream)
        learner = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(learner)
        print(f"GRPO started. Training log: {output / 'train.log'}", flush=True)
        while learner.poll() is None:
            for rank, process in enumerate(processes[:-1]):
                if process.poll() is not None:
                    raise RuntimeError(child_log_error(f"Wrist service rank {rank} exited during training",
                                       output / f"cosmos_rank{rank}.log", returncode=process.returncode))
            time.sleep(2)
        (output / "exit_code").write_text(str(learner.returncode) + "\n")
        if learner.returncode:
            print(child_log_error("Policy driver exited", output / "train.log",
                                  returncode=learner.returncode), file=sys.stderr, flush=True)
        return learner.returncode
    finally:
        for process in reversed(processes):
            stop_process(process)
        if memory_monitor is not None:
            stop_process(memory_monitor)
        for stream in streams:
            stream.close()
        if profile_memory:
            from spatial_wovr_gpu_peak import write_peak_summary

            write_peak_summary(output, world_size=world_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="CPU config/import checks; allow pending wrist download")
    mode.add_argument("--check-assets", action="store_true", help="Also require and hash complete wrist weights")
    args, overrides = parser.parse_known_args()
    run_dir = Path(os.environ["SPATIAL_WOVR_RUN_DIR"])
    endpoints = make_endpoints(os.environ["CUDA_VISIBLE_DEVICES"], int(os.environ.get("SPATIAL_COSMOS_PORT", "18059")))
    # OmegaConf's oc.decode accepts its own flow mapping syntax; JSON's
    # double-quoted keys are not accepted by that interpolation grammar.
    os.environ["SPATIAL_COSMOS_ENDPOINTS"] = "[" + ", ".join(
        f"{{host: {endpoint['host']}, port: {endpoint['port']}}}" for endpoint in endpoints
    ) + "]"
    prepare_runtime(run_dir)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(SCRIPT_DIR / "config"), version_base="1.1"):
        cfg = compose(config_name="spatial_wovr_pi05_8gpu", overrides=overrides)
    from spatial_wovr_memory_guard import configure_recovery

    configure_recovery(cfg)
    samples = validate_batch(cfg)
    from spatial_wovr_shared_pool import configure_shared_pool

    configure_shared_pool(cfg, run_dir)
    world_size = placement_world_size(cfg)
    if len(endpoints) != world_size:
        raise ValueError(f"Visible GPU count {len(endpoints)} differs from configured placement {world_size}")
    if list(cfg.env.train.cosmos_endpoints) != endpoints:
        raise ValueError("Configured Cosmos endpoints differ from the services owned by this launcher")
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
    print(f"Config: {run_dir / 'resolved_config.yaml'}", flush=True)
    print(f"{world_size} GPU GRPO: micro={cfg.actor.micro_batch_size}, accumulation={cfg.actor.global_batch_size // (world_size * cfg.actor.micro_batch_size)}, global={cfg.actor.global_batch_size}, rollout chunks={samples}", flush=True)
    check_assets(cfg, full=not args.check, run_dir=run_dir)
    check_imports(run_dir)
    if args.check or args.check_assets:
        print("CPU checks passed. No GPU workers or training started.", flush=True)
        return 0
    return train(cfg, endpoints, run_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
