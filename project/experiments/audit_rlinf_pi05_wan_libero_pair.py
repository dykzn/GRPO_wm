#!/usr/bin/env python3
"""Paired RLinf-Pi05 -> real LIBERO vs Wan future-image diagnostic.

The policy actions are generated once by RLinf's Pi05 on a real LIBERO reset
and replayed both in the simulator and in Wan.  This separates action/policy
errors from the Wan image-transition error.  No Cosmos model or shadow policy
is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pathlib
import random
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numba

_numba_jit = numba.jit


def _jit_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _numba_jit(*args, **kwargs)


numba.jit = _jit_without_cache

import imageio.v2 as imageio
import numpy as np
import torch

# The EGL diagnostic image has a cuDNN initialization defect, while native
# CUDA kernels work normally.  Disable only cuDNN for this audit so Pi05's
# vision tower can run; this does not change the model or production training.
torch.backends.cudnn.enabled = False

_torch_load = torch.load


def _trusted_legacy_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_legacy_torch_load

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

PROJECT_ROOT = pathlib.Path(
    os.environ.get("PROJECT_ROOT", str(pathlib.Path(__file__).resolve().parents[1]))
).expanduser().resolve()
RLINF_ROOT = pathlib.Path(
    os.environ.get("RLINF_ROOT", str(PROJECT_ROOT / "third_party/RLinf"))
).expanduser().resolve()
WAN_ROOT = pathlib.Path(
    os.environ.get("WAN_ROOT", str(PROJECT_ROOT / "external/wan"))
).expanduser().resolve()
sys.path.insert(0, str(RLINF_ROOT))
sys.path.insert(0, str(WAN_ROOT))

CONDITION_FRAMES = 5
FUTURE_FRAMES = 8
ACTION_DIM = 7


def _ensure_rlinf_ray_compat() -> None:
    """Load a compatible Ray runtime when the Wan environment lacks Ray.

    The Wan/CUDA environment has the matching torch + flash-attn stack but no
    Ray package.  RLinf only needs Ray for its scheduler imports in this
    standalone diagnostic.  The shared simplevla environment contains Ray
    2.55.1 with the same Python ABI, so load that package without changing the
    active torch environment.  This is diagnostic-only and does not start a
    Ray cluster.
    """
    from importlib import metadata

    try:
        metadata.version("ray")
        return
    except metadata.PackageNotFoundError:
        pass

    ray_site = pathlib.Path(os.environ.get("RLINF_RAY_SITE", "")).expanduser()
    if not ray_site.is_dir():
        raise RuntimeError(f"Ray compatibility site-packages not found: {ray_site}")

    original_version = metadata.version
    sys.path.insert(0, str(ray_site))
    try:
        import ray  # noqa: F401
    finally:
        sys.path.remove(str(ray_site))

    def version(name: str) -> str:
        if name == "ray":
            return "2.55.1"
        return original_version(name)

    metadata.version = version


def _ensure_openpi_tqdm_compat() -> None:
    """Provide OpenPI's optional tqdm-loggable import for local checkpoints."""
    import datetime

    # Keep the active Python 3.10 CUDA/Torch environment intact for Wan, while
    # making the CPU-only OpenPI import dependencies available to RLinf-Pi05.
    # This directory is deliberately outside the project and is never used by
    # production training.
    openpi_deps = pathlib.Path("/tmp/rlinf_openpi_py310_deps")
    if openpi_deps.is_dir() and str(openpi_deps) not in sys.path:
        sys.path.append(str(openpi_deps))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # The checked-out OpenPI source uses datetime.UTC (Python 3.11 API),
    # while the CUDA environment is Python 3.10.
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc

    try:
        import tqdm_loggable.auto  # noqa: F401
    except ModuleNotFoundError:
        import types
        import tqdm

        package = types.ModuleType("tqdm_loggable")
        auto = types.ModuleType("tqdm_loggable.auto")
        auto.tqdm = tqdm.tqdm
        package.auto = auto
        sys.modules["tqdm_loggable"] = package
        sys.modules["tqdm_loggable.auto"] = auto

    # OpenPI's generic training-data module imports LeRobot at module import
    # time, although RLinf inference never constructs a LeRobot dataset.  The
    # CUDA environment intentionally does not carry the Python 3.11 LeRobot
    # wheel, so provide an import-only stub for this diagnostic process.  Any
    # accidental dataset construction still fails loudly instead of silently
    # changing the data path.
    if "lerobot.common.datasets.lerobot_dataset" not in sys.modules:
        import types

        for name in ("lerobot", "lerobot.common", "lerobot.common.datasets"):
            module = sys.modules.setdefault(name, types.ModuleType(name))
            if not hasattr(module, "__path__"):
                module.__path__ = []
        dataset_module = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
        sys.modules[dataset_module.__name__] = dataset_module


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def state_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def proprio_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Return the native LIBERO proprio contract used by rollout HDF5 files.

    ``state_from_obs`` is the compact 8-D axis-angle state used as a Pi05
    policy input.  The temporal world-model datasets historically store the
    raw 9-D proprio vector instead: gripper qpos, EEF position, and EEF
    quaternion.  Keep both representations explicit so they cannot be
    silently substituted for one another during rollout conversion.
    """

    return np.concatenate(
        [
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
        ]
    ).astype(np.float32)


def rot180(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[::-1, ::-1])


def fliplr(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.fliplr(image))


def flipud(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(image))


def mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255.0
    )


def make_env(task: benchmark.Task) -> OffScreenRenderEnv:
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )


def settle_env(
    env: OffScreenRenderEnv, initial_state: np.ndarray, settle_steps: int
) -> dict[str, np.ndarray]:
    env.reset()
    obs = env.set_init_state(initial_state)
    dummy = [0.0] * 6 + [-1.0]
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(dummy)
    return obs


def policy_observation(
    obs: dict[str, np.ndarray], instruction: str
) -> dict[str, Any]:
    return {
        "main_images": torch.from_numpy(rot180(obs["agentview_image"])[None]),
        "wrist_images": torch.from_numpy(
            rot180(obs["robot0_eye_in_hand_image"])[None]
        ),
        "extra_view_images": None,
        "states": torch.from_numpy(state_from_obs(obs)[None]),
        "task_descriptions": [instruction],
    }


def labelled(image: np.ndarray, title: str, size: int = 256) -> np.ndarray:
    canvas = Image.fromarray(image).convert("RGB").resize((size, size))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size, 24), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return np.asarray(canvas)


def save_video(
    path: pathlib.Path,
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    frames = []
    for index, (truth, pred) in enumerate(zip(target, prediction), start=1):
        frames.append(
            np.concatenate(
                [
                    labelled(truth, f"LIBERO h{index}"),
                    labelled(pred, f"Wan h{index}"),
                ],
                axis=1,
            )
        )
    imageio.mimwrite(
        path,
        frames,
        fps=2,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )


def save_contact_sheet(
    path: pathlib.Path,
    reference: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    tiles = [
        labelled(reference, "reference", 128),
        *[labelled(x, f"L h{i + 1}", 128) for i, x in enumerate(target)],
        *[labelled(x, f"W h{i + 1}", 128) for i, x in enumerate(prediction)],
    ]
    columns = FUTURE_FRAMES
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * 128, columns * 128, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, columns)
        sheet[row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128] = tile
    canvas = Image.new("RGB", (sheet.shape[1], sheet.shape[0] + 26), "white")
    canvas.paste(Image.fromarray(sheet), (0, 26))
    ImageDraw.Draw(canvas).text((4, 5), title, fill="black")
    canvas.save(path)


def output_to_uint8(output: Any) -> np.ndarray:
    samples = []
    for sample in output:
        frames = np.stack([np.asarray(frame, dtype=np.uint8) for frame in sample])
        expected = (CONDITION_FRAMES + FUTURE_FRAMES, 256, 256, 3)
        if frames.shape != expected:
            raise RuntimeError(f"Unexpected Wan output {frames.shape}, expected {expected}")
        samples.append(frames)
    return np.stack(samples, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-path", type=pathlib.Path, required=True)
    parser.add_argument("--wan-model-path", type=pathlib.Path, required=True)
    parser.add_argument("--vae-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--trial-id", type=int, default=11)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=15,
        help="Match RLinf LiberoEnv.reset(), which settles with 15 zero actions.",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument(
        "--memory-source",
        choices=("reference", "libero_future"),
        default="reference",
        help=(
            "Wan's four-frame memory. 'reference' is the causal inference "
            "contract; 'libero_future' is an explicit leakage control only."
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.rollout.model.model_path = str(args.pi05_path)
    cfg.rollout.model.num_action_chunks = FUTURE_FRAMES
    cfg.rollout.model.openpi.action_chunk = FUTURE_FRAMES
    cfg.rollout.model.openpi.num_steps = 10

    _ensure_rlinf_ray_compat()
    _ensure_openpi_tqdm_compat()
    from rlinf.models.embodiment.openpi import get_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pi05 = get_model(cfg.rollout.model).to("cuda").eval()

    task_ids = args.task_ids
    if task_ids is None:
        task_ids = [6 if args.task_id is None else args.task_id]
    suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_cls()
    paired_cases = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if args.trial_id >= len(initial_states):
            raise IndexError(
                f"trial-id {args.trial_id} out of range for task {task_id}; "
                f"available={len(initial_states)}"
            )
        initial_state = initial_states[args.trial_id]
        case_seed = args.seed + int(task_id)
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)

        env = make_env(task)
        try:
            obs = settle_env(env, initial_state, args.settle_steps)
            reference_r = rot180(obs["agentview_image"])
            policy_obs = policy_observation(obs, task.language)
            with torch.inference_mode():
                policy_action, _ = pi05.predict_action_batch(
                    env_obs=policy_obs, mode="eval", compute_values=False
                )
            actions = (
                policy_action[0, :FUTURE_FRAMES, :ACTION_DIM]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            libero_future = []
            libero_states = [state_from_obs(obs)]
            for action in actions:
                obs, _, _, _ = env.step(action.tolist())
                libero_future.append(rot180(obs["agentview_image"]))
                libero_states.append(state_from_obs(obs))
        finally:
            env.close()
        paired_cases.append(
            {
                "task_id": int(task_id),
                "instruction": task.language,
                "reference_r": reference_r,
                "libero_future": np.stack(libero_future),
                "libero_states": np.stack(libero_states),
                "actions": actions,
                "initial_state_sha256": hashlib.sha256(
                    np.asarray(initial_state).tobytes()
                ).hexdigest(),
                "seed": int(case_seed),
            }
        )

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(args.wan_model_path), offload_device="cpu"),
            ModelConfig(path=str(args.vae_path), offload_device="cpu"),
        ],
    )
    pipe.dit.to("cuda")
    pipe.vae.to("cuda")

    report = {
        "task_suite": args.task_suite,
        "task_ids": [int(x) for x in task_ids],
        "trial_id": args.trial_id,
        "pi05": str(args.pi05_path.resolve()),
        "wan_model": str(args.wan_model_path.resolve()),
        "vae": str(args.vae_path.resolve()),
        "memory_source": args.memory_source,
        "cases": {},
    }

    for case in paired_cases:
        task_id = case["task_id"]
        reference_r = case["reference_r"]
        libero_future = case["libero_future"]
        actions = case["actions"]
        sentinel = np.zeros((CONDITION_FRAMES, ACTION_DIM), dtype=np.float32)
        sentinel[:, -1] = -1.0
        wan_actions = np.concatenate([sentinel, actions], axis=0)
        if args.memory_source == "reference":
            memory_r = np.repeat(reference_r[None], 4, axis=0)
        elif args.memory_source == "libero_future":
            # Deliberately non-causal control: this should only be used to
            # detect accidental future-frame leakage in an earlier audit.
            memory_r = np.ascontiguousarray(libero_future[:4])
        else:  # pragma: no cover - argparse restricts this value
            raise ValueError(args.memory_source)
        variants = {
            "runtime_direct": {
                "reference": reference_r,
                "memory": memory_r,
                "output_to_runtime": lambda x: x,
            },
            "explicit_runtime_to_wan_boundary": {
                "reference": fliplr(reference_r),
                "memory": np.stack([fliplr(x) for x in memory_r]),
                "output_to_runtime": fliplr,
            },
            "explicit_flipud": {
                "reference": flipud(reference_r),
                "memory": np.stack([flipud(x) for x in memory_r]),
                "output_to_runtime": flipud,
            },
            "explicit_rot180": {
                "reference": rot180(reference_r),
                "memory": np.stack([rot180(x) for x in memory_r]),
                "output_to_runtime": rot180,
            },
        }
        case_report = {
            "task_id": task_id,
            "instruction": case["instruction"],
            "seed": case["seed"],
            "action_shape": list(actions.shape),
            "actions": actions.astype(float).tolist(),
            "action_contract": {
                "pi05_output_shape": list(actions.shape),
                "wan_future_action_shape": list(actions.shape),
                "libero_action_shape": list(actions.shape),
                "wan_future_equals_libero": True,
                "max_abs_wan_future_minus_libero": 0.0,
                "action_min": actions.min(axis=0).astype(float).tolist(),
                "action_max": actions.max(axis=0).astype(float).tolist(),
                "action_mean": actions.mean(axis=0).astype(float).tolist(),
            },
            "initial_state_sha256": case["initial_state_sha256"],
            "variants": {},
        }
        task_dir = args.output_dir / f"task_{task_id}_trial{args.trial_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        np.save(task_dir / "reference.npy", reference_r.astype(np.uint8))
        np.save(task_dir / "libero_future.npy", libero_future.astype(np.uint8))
        np.save(task_dir / "pi05_actions.npy", actions.astype(np.float32))
        np.save(
            task_dir / "libero_states.npy",
            case["libero_states"].astype(np.float32),
        )
        for name, variant in variants.items():
            # Keep the diffusion seed identical so the only change is the
            # image-boundary convention.
            variant_seed = case["seed"]
            random.seed(variant_seed)
            np.random.seed(variant_seed)
            torch.manual_seed(variant_seed)
            torch.cuda.manual_seed_all(variant_seed)
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                generated = pipe(
                    seed=variant_seed,
                    tiled=False,
                    input_image=[Image.fromarray(variant["reference"])],
                    input_image4=[
                        [Image.fromarray(frame) for frame in variant["memory"]]
                    ],
                    action=torch.from_numpy(wan_actions[None]).to(
                        device="cuda", dtype=torch.float32
                    ),
                    height=256,
                    width=256,
                    num_frames=CONDITION_FRAMES + FUTURE_FRAMES,
                    num_inference_steps=args.num_inference_steps,
                    cfg_scale=1.0,
                    progress_bar_cmd=lambda values: values,
                    batch_size=1,
                )
            generated_w = output_to_uint8(generated)[0][CONDITION_FRAMES:]
            generated_r = np.stack(
                [variant["output_to_runtime"](frame) for frame in generated_w]
            )
            variant_dir = args.output_dir / f"task_{task_id}_trial{args.trial_id}" / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            save_video(
                variant_dir / "libero_vs_wan.mp4",
                libero_future,
                generated_r,
                f"task {task_id} {name}",
            )
            save_contact_sheet(
                variant_dir / "contact_sheet.png",
                reference_r,
                libero_future,
                generated_r,
                f"task {task_id} {name}",
            )
            horizon = [mae(generated_r[i], libero_future[i]) for i in range(FUTURE_FRAMES)]
            alignment_targets = {
                "offset_minus1_current_frame": np.concatenate(
                    [reference_r[None], libero_future[:-1]], axis=0
                ),
                "offset_0_next_frame": libero_future,
                "offset_plus1_two_steps_later": libero_future[1:],
            }
            alignment_mae = {
                name: [
                    mae(generated_r[i], target[i])
                    for i in range(min(len(generated_r), len(target)))
                ]
                for name, target in alignment_targets.items()
            }
            case_report["variants"][name] = {
                "horizon_mae": horizon,
                "mean_mae": float(np.mean(horizon)),
                "temporal_alignment_mae": alignment_mae,
            }
        report["cases"][str(task_id)] = case_report

    variant_names = list(next(iter(report["cases"].values()))["variants"])
    for name in variant_names:
        means = [
            case["variants"][name]["mean_mae"]
            for case in report["cases"].values()
        ]
        curves = [
            case["variants"][name]["horizon_mae"]
            for case in report["cases"].values()
        ]
        report[name] = {
            "num_cases": len(means),
            "mean_mae": float(np.mean(means)),
            "horizon_mae": np.mean(np.asarray(curves), axis=0).tolist(),
        }

    (args.output_dir / "rlinf_pi05_wan_libero_pair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
