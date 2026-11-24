#!/usr/bin/env python3
"""Collect, prepare, train and validate a versioned Spatial world-model pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def environment() -> None:
    from wm_eval import configure_environment

    configure_environment()
    os.environ.setdefault(
        "PI05_CHECKPOINT_ROOT",
        str(PROJECT / "checkpoints/pi05/Pi05_few_shot_libero/global_step_12000"),
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault(
        "OPENPI_DATA_HOME", os.environ["RLINF_BASE_ROOT"] + "/.assets/.cache/openpi"
    )
    os.environ.setdefault("TOKENIZER_DIR", os.environ["OPENPI_DATA_HOME"])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def gpu_ids(value: str, count: int = 4) -> list[str]:
    values = [part.strip() for part in value.split(",")]
    if (
        len(values) != count
        or any(not v.isdecimal() for v in values)
        or len({int(v) for v in values}) != count
    ):
        raise ValueError(f"Choose exactly {count} distinct physical GPU IDs")
    return [str(int(v)) for v in values]


def gpu_pool(value: str) -> list[str]:
    count = len(value.split(","))
    if count not in (4, 8):
        raise ValueError("Choose four or eight distinct physical GPU IDs")
    return gpu_ids(value, count=count)


def require_idle(devices: list[str]) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    used = {
        row.split(",")[0].strip(): int(row.split(",")[1].strip())
        for row in result.stdout.splitlines()
        if row.strip()
    }
    busy = {
        device: used.get(device)
        for device in devices
        if device not in used or used[device] > 1024
    }
    if busy:
        raise RuntimeError(
            f"Requested GPUs are unavailable or busy (MiB): {busy}; no job was started"
        )


def collection_environment(device: str) -> dict:
    # EGL enumerates physical devices independently of CUDA_VISIBLE_DEVICES.
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        RLINF_EGL_PHYSICAL_DEVICE_ID=device,
        MUJOCO_EGL_DEVICE_ID=device,
    )
    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        env.pop(key, None)
    return env


def runtime(output: Path) -> None:
    from wm_eval import prepare_code_runtime

    output.mkdir(parents=True, exist_ok=True)
    prepare_code_runtime(output)


def worker_command(args, model: str, num_processes: int = 2) -> list[str]:
    from wm_update_memory import MemorySettings

    python = (
        str(Path(os.environ["RLINF_ENV"]) / "bin/python")
        if model == "wan"
        else os.environ["COSMOS_PYTHON"]
    )
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={num_processes}",
        str(SCRIPT),
        "train-worker",
        "--model",
        model,
        "--manifest",
        str(args.manifest.resolve()),
        "--output",
        str((args.output / model).resolve()),
        "--settings",
        str(args.settings.resolve()),
        "--steps",
        str(args.steps),
        "--save-interval",
        str(args.save_interval),
        "--accumulation",
        str(args.accumulation),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]
    if args.resume:
        command += ["--resume", str(args.resume.resolve())]
    if model == "cosmos":
        command += ["--cosmos-recipe", getattr(args, "cosmos_recipe", "auto")]
    command += MemorySettings.from_args(args).command_args()
    return command


def training_plan(args) -> list[dict]:
    """Split a four/eight-GPU pool into independent concurrent model groups."""
    from dataclasses import asdict
    from cosmos_recipe_train import select_recipe
    from wm_update_memory import MemorySettings

    count = len(args.gpus.split(","))
    ranks = getattr(args, "ranks_per_model", None) or (4 if count == 8 else 2)
    if ranks not in (2, 4, 8) or count not in (ranks, 2 * ranks):
        raise ValueError("GPU count must match one model group or two complete groups")
    if args.model == "both" and count != 2 * ranks:
        raise ValueError("Both models require two disjoint complete GPU groups")
    devices = gpu_ids(args.gpus, count=count)
    if count == ranks:
        assignments = {args.model: devices}
    else:
        assignments = {"wan": devices[:ranks], "cosmos": devices[ranks:]}
    models = ["wan", "cosmos"] if args.model == "both" else [args.model]
    return [
        {
            "model": model,
            "gpus": assignments[model],
            "num_processes": ranks,
            "accumulation": args.accumulation,
            "effective_batch": ranks * args.accumulation,
            "memory_reservation": asdict(MemorySettings.from_args(args)),
            **(
                {"cosmos_recipe": select_recipe(args)}
                if model == "cosmos" else {}
            ),
            "command": worker_command(args, model, ranks),
            "log": str(args.output / f"train_{model}.log"),
        }
        for model in models
    ]


def training_environment(devices: list[str]) -> dict:
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(devices),
        COSMOS_DISTRIBUTED_BACKEND="nccl",
        # Omitting the threshold keeps automatic garbage collection disabled.
        # PyTorch 2.7 rejects an explicitly supplied 0.0 during CUDA init.
        PYTORCH_CUDA_ALLOC_CONF="backend:native,expandable_segments:True",
        # Compile in the training rank; auxiliary compiler processes otherwise
        # create CUDA contexts on device zero before LOCAL_RANK is selected.
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    for key in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    ]:
        env.pop(key, None)
    return env


def prepare(args) -> None:
    from wm_update_data import build_manifest, inventory, sha256

    collection = json.loads((args.collection / "collection.json").read_text())
    rows = []
    for row in collection["episodes"]:
        actual = inventory(Path(row["path"]), verify_rgb=args.verify_rgb)
        if actual != row:
            raise ValueError(f"Collection episode changed: {row['path']}")
        rows.append(actual)
    if args.replay_root:
        from wm_eval_data import NAME, inventory_episode

        for path in sorted(args.replay_root.glob("*suite=libero_spatial--*.hdf5")):
            match = NAME.search(path.name)
            if not match:
                continue
            row = inventory_episode(path, match, args.verify_rgb)
            # Legacy data can have N observations and N actions; only N-1
            # transitions have recorded successors. Never train on its tail.
            row.update(
                actions=min(row["actions"], row["observations"] - 1),
                role="old",
                source_schema="legacy-cosmos-libero",
                sha256=sha256(path),
                suite="libero_spatial",
            )
            rows.append(row)
    report = build_manifest(
        rows, args.output, val_fraction=args.val_fraction, seed=args.seed
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


def inspect_data(args) -> None:
    from wm_update_data import PairedWindowDataset, load_manifest, sha256

    manifest = load_manifest(args.manifest)
    if args.verify_files:
        for row in manifest["episodes"]:
            if sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"Episode checksum mismatch: {row['path']}")
    result = {
        "manifest_sha256": manifest["manifest_sha256"],
        "summary": manifest["summary"],
        "windows": [],
    }
    for split in ["train", "val"]:
        data = PairedWindowDataset(str(args.manifest), split=split)
        for i in range(min(args.samples, len(data))):
            window = data[i]
            result["windows"].append(
                {
                    "split": split,
                    "episode": window["episode"]["trajectory_id"],
                    "step": window["step"],
                    "frames": window["frame_indices"].tolist(),
                    "incoming_actions": window["action_indices"].tolist(),
                }
            )
    print(json.dumps(result, indent=2), flush=True)


def check(args) -> int:
    from dataclasses import asdict
    from wm_update_memory import MemorySettings

    required = {
        "rlinf_python": Path(os.environ["RLINF_ENV"]) / "bin/python",
        "cosmos_python": Path(os.environ["COSMOS_PYTHON"]),
        "wan_training_source": Path(os.environ["DIFFSYNTH_ROOT"])
        / "examples/wanvideo/model_training/train_rlinf.py",
        "wan_weights": Path(os.environ["WAN_SPATIAL_CHECKPOINT"])
        / "model-00001.safetensors",
        "wan_vae": Path(os.environ["WAN_SPATIAL_CHECKPOINT"]) / "Wan2.2_VAE.pth",
        "wrist_model": Path(os.environ["COSMOS_WRIST_CHECKPOINT"]) / "model/.metadata",
        "cosmos_vae": Path(os.environ["COSMOS_VAE"]),
        "cosmos_stats": Path(os.environ["COSMOS_DATASET_STATS"]),
    }
    report = {
        key: {"path": str(path), "present": path.is_file()}
        for key, path in required.items()
    }
    if args.check_runtime:
        runtime(args.output)
        for model, python in [
            ("wan", str(required["rlinf_python"])),
            ("cosmos", str(required["cosmos_python"])),
        ]:
            code = (
                "from wm_update_wan import native_module; import os; from pathlib import Path; "
                "native_module(Path(os.environ['DIFFSYNTH_ROOT'])); print('Wan native trainer imports OK')"
                if model == "wan"
                else "from cosmos_recipe_train import make_training_config; from types import SimpleNamespace; "
                "from pathlib import Path; "
                "cfg=make_training_config(SimpleNamespace(steps=2000,accumulation=8,seed=42,lr=1e-6,"
                f"resume=None,save_interval=200,settings=Path({str(args.settings)!r}),cosmos_recipe='v3-reconstructed')); "
                "assert cfg.model.config.state_t==4; assert cfg.trainer.grad_accum_iter==8; "
                "from cosmos_recipe_data import RecipeSettings; import json; "
                f"assert cfg.optimizer.weight_decay==RecipeSettings.from_settings(json.loads(Path({str(args.settings)!r}).read_text())).weight_decay; "
                "from cosmos_policy.models.temporal_action_audit import select_latent_training_loss; "
                "print('Cosmos native training config and loss imports OK')"
            )
            log = args.output / f"{model}_imports.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    [python, "-c", code],
                    check=False,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            report[model + "_imports"] = {
                "present": result.returncode == 0,
                "log": str(log),
            }
    ready = all(row["present"] for row in report.values())
    print(
        json.dumps(
            {
                "ready": ready,
                "checks": report,
                "memory_reservation": asdict(MemorySettings.from_args(args)),
                "note": "CPU check only; GPU backward peak and end-to-end training remain to be measured",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if ready else 2


def main() -> int:
    from wm_update_memory import MemorySettings, add_memory_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_memory_arguments(parser)
    parser.add_argument(
        "command",
        choices=[
            "check",
            "collect",
            "collect-worker",
            "prepare",
            "inspect",
            "train",
            "probe",
            "train-worker",
            "package",
            "evaluate",
            "compare",
        ],
    )
    parser.add_argument("--gpus", default=os.environ.get("WM_UPDATE_GPUS", "0,1,2,3"))
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "runtime_artifacts/wm_update_check"
    )
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--verify-rgb", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--model", choices=["wan", "cosmos", "both"], default="both")
    parser.add_argument(
        "--cosmos-recipe",
        choices=["auto", "legacy", "v3-reconstructed"],
        default="auto",
        help="Fresh Cosmos uses the reconstructed v3 recipe; auto keeps an existing checkpoint's recipe on resume",
    )
    parser.add_argument(
        "--ranks-per-model",
        type=int,
        choices=[2, 4, 8],
        help="Defaults to four ranks for an eight-GPU pool, otherwise two",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=PROJECT / "experiments/config/spatial_wm_update.json",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument(
        "--accumulation",
        type=int,
        default=int(os.environ.get("WM_UPDATE_ACCUMULATION", "8")),
        help="Micro-batches per optimizer update per rank; ranks x accumulation = effective batch",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--wan-checkpoint", type=Path)
    parser.add_argument("--cosmos-checkpoint", type=Path)
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.05)
    args = parser.parse_args()
    MemorySettings.from_args(args)
    environment()
    args.output = args.output.resolve()
    if (
        min(
            args.steps,
            args.save_interval,
            args.accumulation,
            args.per_task,
            args.samples,
        )
        < 1
        or args.lr <= 0
    ):
        parser.error(
            "steps, intervals, accumulation, per-task, samples and lr must be positive"
        )
    if args.command == "check":
        return check(args)
    if args.command in {"collect", "collect-worker"}:
        from wm_update_collect import (
            collect_shard,
            collection_plan,
            merge_collection,
            policy_config,
        )
        from wm_update_data import atomic_json

        if args.command == "collect-worker":
            if not 0 <= args.shard_index < args.num_shards:
                parser.error("Invalid collection shard index")
            collect_shard(args)
            return 0
        if not args.policy_config or not args.policy_checkpoint:
            parser.error("collect requires --policy-config and --policy-checkpoint")
        devices = gpu_pool(args.gpus)
        plan = collection_plan(
            policy_config=args.policy_config,
            policy_checkpoint=args.policy_checkpoint,
            per_task=args.per_task,
            seed=args.seed,
            max_steps=args.max_steps,
            settle_steps=args.settle_steps,
            mode=args.mode,
        )
        policy_config(plan)
        if args.dry_run:
            print(json.dumps({"plan": plan, "gpus": devices}, indent=2))
            return 0
        require_idle(devices)
        args.output.mkdir(parents=True, exist_ok=True)
        config_snapshot = args.output / "policy_config.yaml"
        config_bytes = args.policy_config.read_bytes()
        if config_snapshot.exists() and config_snapshot.read_bytes() != config_bytes:
            raise ValueError("Existing collection has another policy configuration")
        config_snapshot.write_bytes(config_bytes)
        plan["policy_config"] = str(config_snapshot)
        plan_path = args.output / "collection_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError(
                "Existing collection uses another policy or protocol; choose a new output"
            )
        atomic_json(plan_path, plan)
        runtime(args.output)
        from wm_eval import supervise

        commands = []
        for rank, device in enumerate(devices):
            command = [
                sys.executable,
                str(SCRIPT),
                "collect-worker",
                "--output",
                str(args.output),
                "--shard-index",
                str(rank),
                "--num-shards",
                str(len(devices)),
            ]
            env = collection_environment(device)
            commands.append((command, env, args.output / f"collect_gpu{device}.log"))
        supervise(commands)
        print(
            json.dumps(merge_collection(args.output, len(devices))["summary"], indent=2)
        )
    elif args.command == "prepare":
        if not args.collection:
            parser.error("prepare requires --collection")
        prepare(args)
    elif args.command == "inspect":
        if not args.manifest:
            parser.error("inspect requires --manifest")
        inspect_data(args)
    elif args.command in {"train", "probe", "train-worker"}:
        if not args.manifest:
            parser.error("training requires --manifest")
        from wm_update_data import load_manifest

        load_manifest(args.manifest)
        if args.command == "train-worker":
            if args.model == "both":
                parser.error("One model per torchrun group")
            if args.model == "wan":
                from wm_update_wan import train
            else:
                from cosmos_recipe_train import select_recipe, train

                # Check recipe compatibility before reserving memory/loading CUDA.
                select_recipe(args)
            from wm_update_memory import train_with_memory

            train_with_memory(args, args.model, train)
            return 0
        if args.resume and args.model == "both":
            parser.error("Resume one model at a time with --model wan or cosmos")
        if args.command == "probe":
            args.steps, args.save_interval = 1, 1
        jobs = training_plan(args)
        for job in jobs:
            if job.get("cosmos_recipe") == "v3-reconstructed":
                from cosmos_recipe_train import effective_settings

                job["cosmos_recipe_settings"] = effective_settings(args).identity()
        if args.dry_run:
            print(json.dumps({"execution": "parallel", "jobs": jobs}, indent=2))
            return 0
        require_idle([device for job in jobs for device in job["gpus"]])
        for job in jobs:
            destination = args.output / job["model"]
            if (destination / "training_identity.json").exists() and not args.resume:
                raise FileExistsError(
                    f"Existing training output requires explicit --resume: {destination}"
                )
        runtime(args.output)
        from wm_eval import supervise
        from wm_update_data import atomic_json

        atomic_json(
            args.output / f"training_plan_{args.model}.json",
            {"execution": "parallel", "jobs": jobs},
        )
        for job in jobs:
            print(
                f"{job['model']}: GPUs={','.join(job['gpus'])}; ranks={job['num_processes']}; "
                f"accumulation={job['accumulation']}; effective_batch={job['effective_batch']}; "
                f"reserve_memory={job['memory_reservation']['enabled']}; "
                f"headroom={job['memory_reservation']['headroom_gib']} GiB; "
                f"recipe={job.get('cosmos_recipe', 'native-wan')}; "
                f"log={job['log']}",
                flush=True,
            )
        # A single supervisor starts both torchrun groups before waiting. Each
        # uses an independent rendezvous and only its own CUDA_VISIBLE_DEVICES.
        supervise(
            [
                (job["command"], training_environment(job["gpus"]), Path(job["log"]))
                for job in jobs
            ]
        )
    else:
        from wm_update_accept import compare, evaluate, package_pair

        if args.command == "package":
            if (
                not args.manifest
                or not args.wan_checkpoint
                or not args.cosmos_checkpoint
            ):
                parser.error(
                    "package requires --manifest, --wan-checkpoint, --cosmos-checkpoint"
                )
            package_pair(args)
        elif args.command == "evaluate":
            if not args.pair and not args.manifest:
                parser.error(
                    "evaluate requires --pair or --manifest for the baseline models"
                )
            if not args.dry_run:
                require_idle(gpu_pool(args.gpus))
            evaluate(args)
        else:
            if not args.baseline or not args.candidate:
                parser.error(
                    "compare requires --baseline and --candidate evaluation directories"
                )
            compare(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
