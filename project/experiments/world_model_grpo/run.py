#!/usr/bin/env python3
"""Validate and launch the Wan -> Cosmos -> WoVR -> GRPO pipeline.

The existing ``run_wovr_grpo_hybrid_wrist.sh`` remains the low-level launcher.
This file is the collaboration-facing entry point: it makes the artifact
contract explicit, rewrites only a run-local Hydra override file, and records
the exact run specification next to the outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TASKS = (0, 3, 5, 9)
DEFAULT_CONFIG = "wan_libero_goal_grpo_openpi_pi05_hybrid_wovr_wrist"
DEFAULT_OVERRIDES = Path("experiments/grpo_task039_samewm_continue_overrides.txt")
DEFAULT_DATASET_ROOT = Path("data/cosmos-policy")


class ValidationError(ValueError):
    """Raised when a run specification cannot be launched safely."""


def parse_task_ids(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated task list and preserve its declared order."""
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValidationError(f"Invalid task list: {raw!r}") from exc
    if not values:
        raise ValidationError("At least one task id is required")
    if any(value < 0 for value in values):
        raise ValidationError(f"Task ids must be non-negative: {values}")
    if len(set(values)) != len(values):
        raise ValidationError(f"Task ids must be unique: {values}")
    return values


def parse_task_assignment(value: str) -> tuple[int, ...]:
    """Parse the list value used by a Hydra task filter."""
    values = tuple(
        int(part.strip()) for part in value.strip("[] ").split(",") if part.strip()
    )
    return values


def assignment_value(text: str, key: str) -> str | None:
    """Return the value of the first plain ``key=value`` override."""
    pattern = re.compile(rf"(?m)^\+?{re.escape(key)}\s*=\s*(.*?)\s*$")
    match = pattern.search(text)
    return match.group(1) if match else None


def extract_resume_step(path: Path) -> int | None:
    """Find ``global_step_N`` in a checkpoint path or one of its parents."""
    for item in (path, *path.parents):
        match = re.search(r"global_step_(\d+)", item.name)
        if match:
            return int(match.group(1))
    return None


def resolve_from_project(value: str | Path, project_root: Path) -> Path:
    """Resolve an override path relative to the project when needed."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def require_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"{label} is missing or empty: {path}")


def require_dir(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"{label} is missing: {path}")


def validate_dual_view_config(config_path: Path) -> None:
    """Reject the historical single-view config at the collaboration boundary.

    The current experiment is deliberately a dual-view policy: Wan supplies
    the primary view and Cosmos-WoVR supplies the wrist view.  This check is
    intentionally independent of checkpoint paths so a stale Hydra config
    cannot silently turn the run into the old Wan-only experiment.
    """
    text = config_path.read_text()
    image_inputs = re.findall(
        r"(?m)^\s*num_images_in_input\s*:\s*(\d+)\s*(?:#.*)?$", text
    )
    if len(image_inputs) < 2 or any(value != "2" for value in image_inputs):
        raise ValidationError(
            "The current world-model GRPO contract requires dual-view Pi05 "
            "inputs: every num_images_in_input entry must be 2"
        )

    required_true = ("use_cosmos_wrist", "cosmos_wrist_use_wovr_contract")
    for key in required_true:
        match = re.search(
            rf"(?m)^\s*{re.escape(key)}\s*:\s*(true|True)\s*(?:#.*)?$",
            text,
        )
        if match is None:
            raise ValidationError(
                f"The current dual-view contract requires {key}: true in "
                f"{config_path}"
            )

    # The current Cosmos wrist client uses the shadow LIBERO bootstrap for
    # reset views/task metadata.  Do not accept a config that explicitly turns
    # this current runtime dependency off; removing it requires a separate
    # bootstrap refactor.
    bridge_match = re.search(
        r"(?m)^\s*use_libero_state_bridge\s*:\s*(true|false|True|False)\s*(?:#.*)?$",
        text,
    )
    if bridge_match is not None and bridge_match.group(1).lower() == "false":
        raise ValidationError(
            "The current Cosmos-WoVR wrist path requires "
            "use_libero_state_bridge: true; removing the bridge is a separate "
            "bootstrap refactor"
        )


def check_project_layout(project_root: Path, errors: list[str]) -> None:
    for relative in (
        "third_party/RLinf",
        "third_party/WorldPilot",
        "third_party/cosmos-policy",
    ):
        require_dir(project_root / relative, f"project component {relative}", errors)
    egl_root = Path(os.environ.get("EGL_ROOT", project_root / "third_party/egl_docker"))
    require_file(egl_root / "run_with_docker_egl.sh", "EGL launcher", errors)
    libero_root = Path(os.environ.get("LIBERO_ROOT", project_root / "external/LIBERO"))
    require_dir(libero_root, "LIBERO component", errors)


def check_artifacts(
    *,
    project_root: Path,
    wan_checkpoint: Path,
    cosmos_checkpoint: Path,
    cosmos_dataset_root: Path,
    pi05_root: Path,
    initial_image_path: Path,
    resume_dir: Path | None,
    grpo_config: str,
    overrides_path: Path,
) -> None:
    """Validate the files consumed by the low-level launcher and RLinf."""
    errors: list[str] = []
    check_project_layout(project_root, errors)

    require_file(
        wan_checkpoint / "model-00001.safetensors",
        "merged Wan checkpoint",
        errors,
    )
    require_file(wan_checkpoint / "Wan2.2_VAE.pth", "Wan VAE", errors)
    require_file(
        cosmos_checkpoint / "model/.metadata",
        "Cosmos wrist model metadata",
        errors,
    )
    require_file(
        cosmos_dataset_root / "success_only/dataset_statistics.json",
        "Cosmos dataset statistics",
        errors,
    )
    require_file(
        cosmos_dataset_root / "success_only/t5_embeddings.pkl",
        "Cosmos T5 embeddings",
        errors,
    )
    require_dir(initial_image_path, "Wan initial-image dataset", errors)
    require_file(overrides_path, "base Hydra override file", errors)

    pi05_candidates = (
        pi05_root / "model.safetensors",
        pi05_root / "model_state_dict/full_weights.pt",
        pi05_root / "actor/model_state_dict/full_weights.pt",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in pi05_candidates):
        errors.append(
            "Pi05 checkpoint is missing; expected one of: "
            + ", ".join(str(path) for path in pi05_candidates)
        )

    config_path = (
        project_root
        / "third_party/RLinf/examples/embodiment/config"
        / f"{grpo_config}.yaml"
    )
    require_file(config_path, "GRPO Hydra config", errors)
    if config_path.is_file():
        validate_dual_view_config(config_path)
    if resume_dir is not None:
        require_dir(resume_dir, "GRPO resume directory", errors)

    if errors:
        raise ValidationError("\n".join(f"- {error}" for error in errors))


def rewrite_overrides(
    base_text: str,
    *,
    tasks: tuple[int, ...],
    initial_image_path: Path,
) -> str:
    """Create a run-local override file without modifying the shared file."""
    task_value = "[" + ",".join(str(task) for task in tasks) + "]"
    replacements = {
        "env.train.task_id_filter": task_value,
        "env.eval.task_id_filter": task_value,
        "env.train.initial_image_path": str(initial_image_path),
    }
    seen: set[str] = set()
    output: list[str] = []
    assignment_pattern = re.compile(
        r"^(?P<prefix>\s*\+?)(?P<key>env\.(?:train|eval)\."
        r"(?:task_id_filter|initial_image_path))\s*=.*$"
    )
    for line in base_text.splitlines():
        match = assignment_pattern.match(line)
        if match and match.group("key") in replacements:
            key = match.group("key")
            output.append(f"{match.group('prefix')}{key}={replacements[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key in ("env.train.task_id_filter", "env.eval.task_id_filter"):
        if key not in seen:
            output.append(f"+{key}={task_value}")
    if "env.train.initial_image_path" not in seen:
        output.append(f"+env.train.initial_image_path={initial_image_path}")
    return "\n".join(output).rstrip() + "\n"


def validate_override_contract(
    text: str,
    *,
    tasks: tuple[int, ...],
    allow_custom_contract: bool,
) -> Path:
    """Check task selection and return the configured initial-image dataset."""
    expected_tasks = tasks
    for key in ("env.train.task_id_filter", "env.eval.task_id_filter"):
        value = assignment_value(text, key)
        if value is None:
            raise ValidationError(f"Override file does not define {key}")
        try:
            actual_tasks = parse_task_assignment(value)
        except ValueError as exc:
            raise ValidationError(f"Invalid {key} value: {value!r}") from exc
        if actual_tasks != expected_tasks:
            raise ValidationError(
                f"{key}={actual_tasks} does not match requested tasks={expected_tasks}"
            )

    for key in (
        "env.train.use_cosmos_wrist",
        "env.eval.use_cosmos_wrist",
        "env.train.cosmos_wrist_use_wovr_contract",
        "env.eval.cosmos_wrist_use_wovr_contract",
        "env.train.use_libero_state_bridge",
        "env.eval.use_libero_state_bridge",
    ):
        value = assignment_value(text, key)
        if value is not None and value.strip().lower() in {"false", "0", "no"}:
            raise ValidationError(
                f"Current dual-view Wan + Cosmos-WoVR runtime cannot disable {key}"
            )

    initial_image = assignment_value(text, "env.train.initial_image_path")
    if not initial_image:
        raise ValidationError("Override file does not define env.train.initial_image_path")

    # Gripper protocol is checkpoint-specific.  Accept both contracts, but
    # reject typos and an explicitly contradictory train/eval pair.  The
    # task039's two checkpoint-specific Pi05 variants convert at the model
    # boundary and keep the environment boundary as a no-op, matching the
    # previously validated action path; signed checkpoints can keep their
    # existing already_signed override.
    env_modes = [
        assignment_value(text, "env.train.openpi_gripper_action_mode"),
        assignment_value(text, "env.eval.openpi_gripper_action_mode"),
    ]
    output_modes = [
        assignment_value(text, "actor.model.openpi_data.output_gripper_action_mode"),
        assignment_value(text, "rollout.model.openpi_data.output_gripper_action_mode"),
    ]
    allowed_env_modes = {"libero_01_to_signed", "already_signed", "none"}
    allowed_output_modes = {
        "raw_01",
        "libero_01_to_signed",
        "already_signed",
    }
    if any(mode is not None and mode not in allowed_env_modes for mode in env_modes):
        raise ValidationError(
            "openpi_gripper_action_mode must be libero_01_to_signed, already_signed, or none"
        )
    if any(
        mode is not None and mode not in allowed_output_modes
        for mode in output_modes
    ):
        raise ValidationError(
            "output_gripper_action_mode must be raw_01, libero_01_to_signed, "
            "or already_signed"
        )
    if all(mode is not None for mode in env_modes) and env_modes[0] != env_modes[1]:
        raise ValidationError("Train/eval OpenPI gripper modes must match")
    if all(mode is not None for mode in output_modes) and output_modes[0] != output_modes[1]:
        raise ValidationError("Actor/rollout OpenPI output gripper modes must match")

    if not allow_custom_contract:
        required = {
            "env.train.use_policy_state": "false",
            "env.train.enable_kir": "false",
            "env.train.cosmos_wrist_primary_reference": "true",
            "env.train.cosmos_wrist_input_camera_space": "rlinf_runtime_rot180",
            "env.eval.use_policy_state": "false",
            "env.eval.cosmos_wrist_primary_reference": "true",
            "env.eval.cosmos_wrist_input_camera_space": "rlinf_runtime_rot180",
        }
        missing = [
            f"{key}={value}"
            for key, value in required.items()
            if assignment_value(text, key) != value
        ]
        if missing:
            raise ValidationError(
                "Override contract mismatch; pass --allow-custom-contract only "
                "for a deliberate variant. Missing/mismatched: " + ", ".join(missing)
            )

    return Path(initial_image)


def query_gpu_memory(gpu_id: int) -> int:
    """Return currently used GPU memory in MiB."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValidationError(f"Unable to query GPU{gpu_id} with nvidia-smi") from exc
    try:
        return int(result.stdout.strip().splitlines()[0].strip())
    except (IndexError, ValueError) as exc:
        raise ValidationError(
            f"Unexpected nvidia-smi output for GPU{gpu_id}: {result.stdout!r}"
        ) from exc


def git_revision(repo: Path) -> dict[str, Any]:
    """Collect a best-effort revision/dirty marker for the run manifest."""
    info: dict[str, Any] = {"path": str(repo), "revision": None, "dirty": None}
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return info
    info.update(revision=revision, dirty=bool(status.strip()))
    return info


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    project_default = Path(
        os.environ.get(
            "PROJECT_ROOT",
            str(Path(__file__).resolve().parents[2]),
        )
    ).expanduser()
    parser = argparse.ArgumentParser(
        description="Validate and launch world-model simulation followed by GRPO."
    )
    parser.add_argument("--project-root", type=Path, default=project_default)
    parser.add_argument("--wan-ckpt", type=Path, required=True)
    parser.add_argument("--cosmos-ckpt", type=Path, required=True)
    parser.add_argument(
        "--pi05-root",
        type=Path,
        default=project_default / "checkpoints/pi05/RLinf-Pi05-LIBERO-SFT",
    )
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument(
        "--additional-steps",
        type=int,
        help="Number of new GRPO steps after --resume-dir.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Target global step; required for a fresh run.",
    )
    parser.add_argument("--tasks", default=",".join(map(str, DEFAULT_TASKS)))
    parser.add_argument("--overrides", type=Path, default=project_default / DEFAULT_OVERRIDES)
    parser.add_argument("--initial-image-path", type=Path)
    parser.add_argument(
        "--cosmos-dataset-root",
        type=Path,
        default=project_default
        / Path(
            os.environ.get("HYBRID_COSMOS_DATASET_ROOT", str(DEFAULT_DATASET_ROOT))
        ),
    )
    parser.add_argument("--grpo-config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--launcher",
        type=Path,
        default=project_default / "experiments/run_wovr_grpo_hybrid_wrist.sh",
    )
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("HYBRID_GPU", "6")))
    parser.add_argument("--cosmos-gpu", type=int)
    parser.add_argument("--cosmos-port", type=int, default=8120)
    parser.add_argument("--ray-port", type=int, default=29620)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument(
        "--cosmos-config",
        default="cosmos_predict2_2b_480p_libero_cosmos_wovr_wrist_full__inference_only",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--allow-custom-contract", action="store_true")
    parser.add_argument("--max-existing-gpu-memory-mib", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv))


def make_run_root(args: argparse.Namespace, project_root: Path, tasks: tuple[int, ...]) -> Path:
    if args.result_root is not None:
        return args.result_root.expanduser().resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_tag = "-".join(str(task) for task in tasks)
    return (
        project_root
        / "results/05_GRPO"
        / f"world_model_grpo_{timestamp}_tasks{task_tag}_gpu{args.gpu}"
    )


def compute_target_step(args: argparse.Namespace) -> tuple[int | None, int | None]:
    if args.additional_steps is not None and args.additional_steps <= 0:
        raise ValidationError("--additional-steps must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValidationError("--max-steps must be positive")
    if args.resume_step is not None and args.resume_step < 0:
        raise ValidationError("--resume-step must be non-negative")
    if args.resume_dir is None:
        if args.resume_step is not None:
            raise ValidationError("--resume-step requires --resume-dir")
        if args.additional_steps is not None:
            raise ValidationError("--additional-steps requires --resume-dir")
        if args.max_steps is None:
            raise ValidationError("Fresh runs require --max-steps")
        return None, args.max_steps

    resume_step = args.resume_step
    if resume_step is None:
        resume_step = extract_resume_step(args.resume_dir)
    if resume_step is None:
        raise ValidationError(
            "Could not infer resume step from path; pass --resume-step explicitly"
        )
    if args.max_steps is not None and args.additional_steps is not None:
        raise ValidationError("Use only one of --max-steps and --additional-steps")
    if args.max_steps is not None:
        target_step = args.max_steps
    elif args.additional_steps is not None:
        target_step = resume_step + args.additional_steps
    else:
        raise ValidationError(
            "Resumed runs require --additional-steps or an explicit --max-steps"
        )
    if target_step <= resume_step:
        raise ValidationError(
            f"Target global step {target_step} must be larger than resume step {resume_step}"
        )
    return resume_step, target_step


def build_manifest(
    *,
    args: argparse.Namespace,
    project_root: Path,
    result_root: Path,
    tasks: tuple[int, ...],
    resume_step: int | None,
    target_step: int,
    initial_image_path: Path,
    generated_overrides: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "Wan primary simulation -> Cosmos wrist simulation -> WoVR reward -> GRPO",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "prepared",
        "tasks": list(tasks),
        "steps": {
            "resume_global_step": resume_step,
            "target_global_step": target_step,
            "additional_steps": (
                target_step - resume_step if resume_step is not None else target_step
            ),
        },
        "artifacts": {
            "wan_checkpoint": str(args.wan_ckpt.resolve()),
            "cosmos_wrist_checkpoint": str(args.cosmos_ckpt.resolve()),
            "pi05_checkpoint": str(args.pi05_root.resolve()),
            "grpo_resume_dir": (
                str(args.resume_dir.resolve()) if args.resume_dir is not None else None
            ),
            "cosmos_dataset_root": str(args.cosmos_dataset_root.resolve()),
            "initial_image_path": str(initial_image_path.resolve()),
            "base_overrides": str(args.overrides.resolve()),
            "generated_overrides": str(generated_overrides),
        },
        "runtime": {
            "learner_gpu": args.gpu,
            "cosmos_gpu": args.cosmos_gpu,
            "cosmos_port": args.cosmos_port,
            "ray_port": args.ray_port,
            "grpo_config": args.grpo_config,
            "cosmos_config": args.cosmos_config,
            "save_interval": args.save_interval,
        },
        "code_revisions": {
            "RLinf": git_revision(project_root / "third_party/RLinf"),
            "WorldPilot": git_revision(project_root / "third_party/WorldPilot"),
            "cosmos-policy": git_revision(project_root / "third_party/cosmos-policy"),
        },
        "result_root": str(result_root),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    project_root = args.project_root.expanduser().resolve()
    args.wan_ckpt = args.wan_ckpt.expanduser().resolve()
    args.cosmos_ckpt = args.cosmos_ckpt.expanduser().resolve()
    args.pi05_root = args.pi05_root.expanduser().resolve()
    args.overrides = args.overrides.expanduser().resolve()
    args.launcher = args.launcher.expanduser().resolve()
    args.cosmos_dataset_root = args.cosmos_dataset_root.expanduser().resolve()
    if args.cosmos_gpu is None:
        args.cosmos_gpu = args.gpu

    try:
        tasks = parse_task_ids(args.tasks)
        resume_step, target_step = compute_target_step(args)
        if args.resume_dir is not None:
            args.resume_dir = args.resume_dir.expanduser().resolve()

        base_text = args.overrides.read_text()
        configured_initial = assignment_value(
            base_text, "env.train.initial_image_path"
        )
        initial_image_path = (
            resolve_from_project(args.initial_image_path, project_root)
            if args.initial_image_path is not None
            else resolve_from_project(configured_initial, project_root)
            if configured_initial and not configured_initial.startswith("${oc.env:")
            else (args.wan_ckpt / "dataset").resolve()
            if (args.wan_ckpt / "dataset").is_dir()
            else None
        )
        if initial_image_path is None:
            raise ValidationError(
                "Pass --initial-image-path or define env.train.initial_image_path"
            )

        generated_text = rewrite_overrides(
            base_text,
            tasks=tasks,
            initial_image_path=initial_image_path,
        )
        validate_override_contract(
            generated_text,
            tasks=tasks,
            allow_custom_contract=args.allow_custom_contract,
        )
        check_artifacts(
            project_root=project_root,
            wan_checkpoint=args.wan_ckpt,
            cosmos_checkpoint=args.cosmos_ckpt,
            cosmos_dataset_root=args.cosmos_dataset_root,
            pi05_root=args.pi05_root,
            initial_image_path=initial_image_path,
            resume_dir=args.resume_dir,
            grpo_config=args.grpo_config,
            overrides_path=args.overrides,
        )
        if not args.launcher.is_file():
            raise ValidationError(f"Launcher is missing: {args.launcher}")
        if not args.dry_run:
            for gpu_id in sorted({args.gpu, args.cosmos_gpu}):
                used = query_gpu_memory(gpu_id)
                if used > args.max_existing_gpu_memory_mib:
                    raise ValidationError(
                        f"GPU{gpu_id} currently uses {used} MiB, above the "
                        f"limit {args.max_existing_gpu_memory_mib} MiB"
                    )
    except (OSError, ValidationError) as exc:
        print(f"world_model_grpo: validation failed: {exc}", file=sys.stderr)
        return 2

    result_root = make_run_root(args, project_root, tasks)
    if result_root.exists():
        print(f"world_model_grpo: refusing to reuse existing result root: {result_root}", file=sys.stderr)
        return 4
    experiment_name = args.experiment_name or (
        f"world_model_grpo_tasks{'-'.join(map(str, tasks))}_gpu{args.gpu}"
    )

    manifest_path = result_root / "run_manifest.json"
    generated_overrides_path = result_root / "hydra_overrides.txt"
    manifest = build_manifest(
        args=args,
        project_root=project_root,
        result_root=result_root,
        tasks=tasks,
        resume_step=resume_step,
        target_step=target_step,
        initial_image_path=initial_image_path,
        generated_overrides=generated_overrides_path,
    )
    manifest["runtime"]["experiment_name"] = experiment_name

    env_updates = {
        "PROJECT_ROOT": str(project_root),
        "HYBRID_RESULT_ROOT": str(result_root),
        "HYBRID_GPU": str(args.gpu),
        "HYBRID_COSMOS_GPU": str(args.cosmos_gpu),
        "HYBRID_COSMOS_PORT": str(args.cosmos_port),
        "HYBRID_RAY_PORT": str(args.ray_port),
        "HYBRID_STEPS": str(target_step),
        "HYBRID_SAVE_INTER": str(args.save_interval),
        "HYBRID_WAN_CKPT": str(args.wan_ckpt),
        "HYBRID_GRPO_CONFIG": args.grpo_config,
        "HYBRID_EXTRA_HYDRA_OVERRIDES_FILE": str(generated_overrides_path),
        "HYBRID_USE_COSMOS_SERVER": "true",
        "HYBRID_COSMOS_DATASET_ROOT": str(args.cosmos_dataset_root),
        "HYBRID_PI05_ROOT": str(args.pi05_root),
        "HYBRID_RESUME_DIR": str(args.resume_dir) if args.resume_dir else "",
        "HYBRID_EXPERIMENT_NAME": experiment_name,
        "COSMOS_WRIST_CHECKPOINT": str(args.cosmos_ckpt),
        "COSMOS_WRIST_CONFIG": args.cosmos_config,
        "COSMOS_WRIST_EXPECT_CROSS_VIEW": "false",
    }
    launch_env = os.environ.copy()
    launch_env.update(env_updates)
    command = ["bash", str(args.launcher), "--coordinator"]

    if args.dry_run:
        manifest["status"] = "dry_run"
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print("generated Hydra overrides:")
        print(generated_text, end="")
        print("launch command:")
        print(
            shlex.join(
                [
                    f"{key}={value}"
                    for key, value in env_updates.items()
                ]
                + command
            )
        )
        return 0

    try:
        result_root.mkdir(parents=True)
        generated_overrides_path.write_text(generated_text)
        write_json(manifest_path, manifest)
        print(f"result_root={result_root}")
        print(f"manifest={manifest_path}")
        print(f"launching: {shlex.join(command)}")
        completed = subprocess.run(command, cwd=project_root, env=launch_env)
        manifest["status"] = "completed" if completed.returncode == 0 else "failed"
        manifest["exit_code"] = completed.returncode
        write_json(manifest_path, manifest)
        return completed.returncode
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["exit_code"] = 130
        write_json(manifest_path, manifest)
        return 130
    except OSError as exc:
        manifest["status"] = "failed_to_start"
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        print(f"world_model_grpo: launch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
