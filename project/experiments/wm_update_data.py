"""Paired LIBERO truth data and a shared action/frame clock for WM updates.

Observation i is the state BEFORE action i. Action i produces observation i+1.
Only complete eight-action targets are sampled; future frames are never padded.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

SCHEMA = "spatial-wm-truth-v1"
MANIFEST_SCHEMA = "spatial-wm-update-v1"
CHUNK = 8
VIEWS = ("primary_images_jpeg", "wrist_images_jpeg")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def training_code_identity() -> str:
    root = Path(__file__).parent
    names = ["wm_update_data.py", "wm_update_wan.py", "wm_update_cosmos.py"]
    return digest({name: sha256(root / name) for name in names})


def file_identity(path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def signed_actions(actions) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected finite executed actions [N,7]")
    if not np.isin(values[:, -1], [-1, 1]).all():
        raise ValueError("Executed gripper actions must already be signed -1/+1")
    return values


def write_episode(
    path: Path,
    *,
    primary,
    wrist,
    actions,
    states,
    proprio,
    successes,
    rewards,
    dones,
    metadata: dict,
) -> None:
    """Atomically commit one complete truth episode, including its successor RGB."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    actions = signed_actions(actions)
    count = len(actions)
    if count < 1 or any(len(x) != count + 1 for x in [primary, wrist, states, proprio]):
        raise ValueError("N actions require N+1 paired RGB/state/proprio observations")
    if any(len(x) != count for x in [successes, rewards, dones]):
        raise ValueError("Success, reward and done must label every executed action")
    for values in [states, proprio, rewards]:
        if not np.isfinite(np.asarray(values)).all():
            raise ValueError("Nonfinite simulator data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # Keep the HDF5 encoder independent of OS descriptors while JPEG encoding
    # can trigger cleanup of recently closed simulator/rendering objects.
    image = io.BytesIO()
    try:
        with h5py.File(image, "w") as file:
            for key, frames in zip(VIEWS, [primary, wrist], strict=True):
                dataset = file.create_dataset(
                    key, (count + 1,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                shape = np.asarray(frames[0]).shape
                for index, frame in enumerate(frames):
                    frame = np.asarray(frame)
                    if (
                        frame.dtype != np.uint8
                        or frame.shape != shape
                        or frame.ndim != 3
                        or frame.shape[-1] != 3
                    ):
                        raise ValueError(
                            "RGB must be synchronized uint8 [H,W,3] frames of constant size"
                        )
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(
                        buffer, format="JPEG", quality=95, subsampling=0
                    )
                    dataset[index] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            for key, values in {
                "actions": actions,
                "sim_states": states,
                "proprio": proprio,
                "successes": successes,
                "rewards": rewards,
                "dones": dones,
            }.items():
                file.create_dataset(key, data=np.asarray(values), compression="gzip")
            file.create_dataset("observation_step", data=np.arange(count + 1))
            file.create_dataset("action_step", data=np.arange(count))
            file.attrs.update(
                schema=SCHEMA,
                image_orientation="rlinf_runtime_rot180",
                task_description=metadata["task"],
                fps=20,
                action_alignment="action[i]: observation[i] -> observation[i+1]",
                action_gripper_mode="already_signed",
                complete=True,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
        # Open the destination only after HDF5 has fully closed its in-memory
        # image, then commit the complete episode with the existing rename.
        with temporary.open("wb") as stream:
            stream.write(image.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        image.close()
        temporary.unlink(missing_ok=True)


@contextmanager
def open_episode(path: Path):
    """Use sequential filesystem IO; HDF5's random reads stay in memory."""
    with io.BytesIO(Path(path).read_bytes()) as image, h5py.File(image, "r") as file:
        yield file


def inventory(path: Path, *, role: str = "new", verify_rgb: bool = False) -> dict:
    """Validate our truth format and fingerprint it for immutable manifests."""
    path = Path(path).resolve()
    with open_episode(path) as file:
        if file.attrs.get("schema") != SCHEMA or not file.attrs.get("complete", False):
            raise ValueError(f"Incomplete or unknown truth episode: {path}")
        metadata = json.loads(file.attrs["metadata_json"])
        actions = signed_actions(file["actions"][:])
        count = len(actions)
        if any(
            len(file[key]) != count + 1
            for key in [*VIEWS, "sim_states", "proprio", "observation_step"]
        ):
            raise ValueError(f"Broken N+1 observation contract: {path}")
        if any(
            len(file[key]) != count
            for key in ["successes", "rewards", "dones", "action_step"]
        ):
            raise ValueError(f"Broken per-action labels: {path}")
        if not np.array_equal(file["observation_step"][:], np.arange(count + 1)):
            raise ValueError("Observation indices must be contiguous")
        if not np.array_equal(file["action_step"][:], np.arange(count)):
            raise ValueError("Action indices must be contiguous")
        if verify_rgb:
            for index in range(count + 1):
                shapes = [decode(file[key][index]).shape for key in VIEWS]
                if shapes[0] != shapes[1]:
                    raise ValueError(
                        "Paired cameras must use the same acquisition resolution"
                    )
        success = bool(np.any(file["successes"][:]))
        if success != bool(metadata["success"]):
            raise ValueError("Episode success disagrees with simulator step labels")
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
        "trajectory_id": metadata["episode_id"],
        "task_id": int(metadata["task_id"]),
        "task": metadata["task"],
        "suite": metadata["suite"],
        "initial_state_id": int(metadata["initial_state_id"]),
        "policy_id": metadata["policy_id"],
        "seed": int(metadata["seed"]),
        "success": success,
        "actions": count,
        "observations": count + 1,
        "horizon": count // CHUNK * CHUNK,
        "role": role,
        "source_camera_space": "rlinf_runtime_rot180",
        "source_schema": SCHEMA,
    }


def decode(value) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(value))) as image:
        return np.asarray(image.convert("RGB"))


def unchanged(row: dict) -> None:
    stat = Path(row["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
        raise ValueError(f"Episode changed after manifest creation: {row['path']}")


def split_group(row: dict) -> str:
    if row.get("initial_state_id") is not None:
        return f"{row.get('suite', 'libero_spatial')}:{row['task_id']}:{row['initial_state_id']}"
    return f"unknown-state:{row['trajectory_id']}"


def build_manifest(
    rows: list[dict], output: Path, *, val_fraction: float = 0.2, seed: int = 20261021
) -> dict:
    """Split complete episodes, keeping repeated known initial states together."""
    if not 0 < val_fraction < 1 or not rows:
        raise ValueError("Need episodes and a validation fraction in (0,1)")
    identities, files = set(), set()
    for row in rows:
        if row["trajectory_id"] in identities or row["sha256"] in files:
            raise ValueError("Duplicate trajectory ID or identical file in update data")
        identities.add(row["trajectory_id"])
        files.add(row["sha256"])
        if row["horizon"] < 8 or row.get("role") not in {"new", "old"}:
            raise ValueError(
                "Each episode needs at least eight observed transitions and role new/old"
            )
        unchanged(row)
    validation = set()
    for task in sorted({row["task_id"] for row in rows}):
        groups = sorted(
            {split_group(row) for row in rows if row["task_id"] == task},
            key=lambda group: digest([seed, group]),
        )
        if len(groups) < 2:
            raise ValueError(
                f"Task {task} needs at least two distinct episode/state groups"
            )
        count = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
        validation.update(groups[:count])
    episodes = [
        dict(
            row,
            split="val" if split_group(row) in validation else "train",
            split_group=split_group(row),
        )
        for row in rows
    ]
    if not any(row["role"] == "new" and row["split"] == "train" for row in episodes):
        raise ValueError("Update training requires new policy data")
    result = {
        "schema": MANIFEST_SCHEMA,
        "suite": "libero_spatial",
        "seed": seed,
        "val_fraction": val_fraction,
        "chunk": CHUNK,
        "action_alignment": "action[i] produces observation[i+1]",
        "conditioning": "fixed o0 + four recent observations + eight successor targets",
        "split_status": "episode-held-out; known initial states grouped; external pretraining overlap unknown",
        "episodes": episodes,
        "summary": {
            split: {
                "episodes": sum(r["split"] == split for r in episodes),
                "success": sum(r["split"] == split and r["success"] for r in episodes),
            }
            for split in ["train", "val"]
        },
    }
    result["manifest_sha256"] = digest(result)
    output = Path(output)
    if output.exists() and json.loads(output.read_text()) != result:
        raise FileExistsError(f"Manifest is frozen; choose a new output: {output}")
    atomic_json(output, result)
    return result


def load_manifest(path: Path) -> dict:
    result = json.loads(Path(path).read_text())
    payload = {key: value for key, value in result.items() if key != "manifest_sha256"}
    if result.get("schema") != MANIFEST_SCHEMA or result.get(
        "manifest_sha256"
    ) != digest(payload):
        raise ValueError("Unknown or modified update manifest")
    train = {r["split_group"] for r in result["episodes"] if r["split"] == "train"}
    val = {r["split_group"] for r in result["episodes"] if r["split"] == "val"}
    if train & val:
        raise ValueError("Training/validation state-group leakage")
    return result


def window_indices(step: int, action_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB and incoming-action indices; -1 denotes an open no-op."""
    if not 0 <= step <= action_count - CHUNK:
        raise ValueError("Window would pad unobserved future transitions")
    memory = np.maximum(0, np.arange(step - 3, step + 1))
    frames = np.concatenate([[0], memory, np.arange(step + 1, step + 9)])
    incoming = np.concatenate([[-1], memory - 1, np.arange(step, step + 8)])
    return frames, incoming


def validation_indices(length: int, rank: int, world: int) -> list[int]:
    if length < 1 or not 0 <= rank < world:
        raise ValueError("Invalid validation shard")
    count = min(4, length)
    return (
        np.linspace(0, length - 1, count * world)
        .round()
        .astype(int)[rank::world]
        .tolist()
    )


def read_window(row: dict, step: int) -> dict:
    unchanged(row)
    indices, action_indices = window_indices(step, row["actions"])
    with open_episode(Path(row["path"])) as file:
        views = [
            np.stack([decode(file[key][int(i)]) for i in indices]) for key in VIEWS
        ]
        actions = np.zeros((13, 7), dtype=np.float32)
        actions[:, -1] = -1
        for i, source in enumerate(action_indices):
            if source >= 0:
                actions[i] = file["actions"][int(source)]
    if row["source_camera_space"] == "cosmos_model_flipud":
        views = [np.ascontiguousarray(np.flip(view, axis=-2)) for view in views]
    elif row["source_camera_space"] != "rlinf_runtime_rot180":
        raise ValueError("Unknown camera space")
    return {
        "primary": views[0],
        "wrist": views[1],
        "actions": actions,
        "frame_indices": indices,
        "action_indices": action_indices,
        "task": row["task"],
        "episode": row,
        "step": step,
    }


class PairedWindowDataset:
    """Task/role/outcome-balanced sampling, with optional pre-grasp emphasis."""

    def __init__(
        self,
        manifest: str,
        split: str = "train",
        samples: int = 40000,
        seed: int = 20261021,
        new_fraction: float = 0.5,
        keyframe_probability: float = 0.25,
        reset_probability: float = 0.05,
    ):
        self.manifest = load_manifest(Path(manifest))
        self.rows = [r for r in self.manifest["episodes"] if r["split"] == split]
        if not self.rows or samples < 1 or split not in {"train", "val"}:
            raise ValueError("Empty or invalid dataset split")
        if any(
            not 0 <= p <= 1
            for p in [new_fraction, keyframe_probability, reset_probability]
        ):
            raise ValueError("Invalid sampling probabilities")
        self.split, self.samples, self.seed, self.epoch = split, samples, seed, 0
        self.new_fraction = new_fraction
        self.keyframe_probability, self.reset_probability = (
            keyframe_probability,
            reset_probability,
        )

    def __len__(self):
        return self.samples if self.split == "train" else len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )
        if self.split == "val":
            row = self.rows[index % len(self.rows)]
        else:
            roles = {r["role"] for r in self.rows}
            role = (
                ("new" if rng.random() < self.new_fraction else "old")
                if len(roles) == 2
                else next(iter(roles))
            )
            rows = [r for r in self.rows if r["role"] == role]
            task = rng.choice(sorted({r["task_id"] for r in rows}))
            rows = [r for r in rows if r["task_id"] == task]
            successes = {r["success"] for r in rows}
            outcome = (
                bool(rng.integers(2)) if len(successes) == 2 else next(iter(successes))
            )
            rows = [r for r in rows if r["success"] == outcome]
            row = rows[int(rng.integers(len(rows)))]
        maximum = row["actions"] - CHUNK
        step = int(rng.integers(maximum + 1))
        if self.split == "train":
            if rng.random() < self.reset_probability:
                step = 0
            elif rng.random() < self.keyframe_probability:
                with open_episode(Path(row["path"])) as file:
                    grip = np.asarray(file["actions"][:, -1])
                closed = np.flatnonzero((grip[:-1] > 0) & (grip[1:] > 0))
                if len(closed):
                    start = max(0, int(closed[0]) - 16)
                    step = int(
                        rng.integers(min(start, maximum), min(start + 32, maximum) + 1)
                    )
        return read_window(row, step)


class WanUpdateDataset(PairedWindowDataset):
    def __getitem__(self, index: int) -> dict:
        import torch

        window = super().__getitem__(index)
        video = [
            Image.fromarray(frame).resize((256, 256), Image.Resampling.BILINEAR)
            for frame in window["primary"]
        ]
        return {
            "video": video,
            "input_image": [video[0]],
            "reference_image": [video[0]],
            "action": torch.from_numpy(window["actions"]),
            "prompt": window["task"],
        }


class CosmosUpdateDataset(PairedWindowDataset):
    """Native Cosmos action-conditioned training batch, matching v3 inference."""

    def __init__(
        self, *args, stats_path: str, primary_reference: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stats = json.loads(Path(stats_path).read_text())
        self.primary_reference = primary_reference
        for key in ["actions_min", "actions_max"]:
            if np.asarray(self.stats[key]).shape != (7,):
                raise ValueError(
                    "Cosmos statistics must contain seven action dimensions"
                )

    def __getitem__(self, index: int) -> dict:
        import torch
        from world_model_grpo.cosmos_wrist_adapter import (
            _preprocess_wrist_model_space,
            _runtime_to_cosmos_model,
        )

        window = super().__getitem__(index)
        rgb = window["wrist"].copy()
        if self.primary_reference:
            rgb[0] = window["primary"][0]
        rgb = np.stack(
            [
                _runtime_to_cosmos_model(_preprocess_wrist_model_space(f, (224, 224)))
                for f in rgb
            ]
        )
        minimum = np.asarray(self.stats["actions_min"], dtype=np.float32)
        maximum = np.asarray(self.stats["actions_max"], dtype=np.float32)
        denominator = np.where(maximum == minimum, 1.0, maximum - minimum)
        actions = np.clip(2 * (window["actions"] - minimum) / denominator - 1, -1, 1)
        actions[0] = [0, 0, 0, 0, 0, 0, -1]
        # The strict v3 net replaces text cross-attention with action tokens.
        # Keep the native conditioner shape, without loading an unused T5 model.
        return {
            "dataset_name": "video_data",
            "video": torch.from_numpy(rgb.transpose(3, 0, 1, 2).copy()),
            "action": torch.from_numpy(actions),
            "t5_text_embeddings": torch.zeros(512, 1024),
            "fps": 20,
            "padding_mask": torch.zeros(1, 224, 224),
            "num_conditional_frames": 2,
            "__key__": f"{window['episode']['trajectory_id']}:{window['step']}",
        }
