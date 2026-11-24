"""Episode alignment and statistics for local Spatial world-model evaluation."""

from __future__ import annotations

import hashlib
from functools import wraps
import io
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
from PIL import Image

NAME = re.compile(r"suite=(libero_spatial)--.*--task=(\d+)--ep=(\d+)--success=(True|False)")
CHUNK = 8


def horizon_for(observations: int, actions: int) -> int:
    """Keep only actions with recorded successor observations, in full chunks."""
    return max(0, min(observations - 1, actions) // CHUNK * CHUNK)


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def retry_io(function):
    """Retry transient NFS reads, without dropping errors from a running score."""
    @wraps(function)
    def call(*args, **kwargs):
        for attempt in range(3):
            try:
                return function(*args, **kwargs)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2 ** attempt))
    return call


@retry_io
def inventory_episode(path: Path, match, verify_frames: bool = False) -> dict:
    """Inspect the streams, optionally decoding every scored frame."""
    with h5py.File(io.BytesIO(path.read_bytes()), "r") as file:
        required = ["primary_images_jpeg", "wrist_images_jpeg", "actions"]
        if any(key not in file for key in required):
            raise ValueError("missing real RGB/action stream")
        counts = [len(file[key]) for key in required]
        if len(set(counts[:2])) != 1:
            raise ValueError(f"Camera streams have different lengths: {path}")
        actions = np.asarray(file["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid actions: {path}")
        horizon = horizon_for(counts[0], counts[2])
        if horizon < 16:
            raise ValueError("fewer than 16 future frames")
        # Equal actions from different initial states are different
        # trajectories. Include RGB as well as the actions in the identity.
        digest = hashlib.sha256(actions.tobytes())
        for key in ["primary_images_jpeg", "wrist_images_jpeg"]:
            digest.update(bytes(file[key][0]))
        label = file.attrs["task_description"]
        if isinstance(label, bytes):
            label = label.decode()
        # Released Cosmos JPEG data uses flipud camera space. An explicit
        # different source attribute must not silently inherit that choice.
        space = str(file.attrs.get("image_orientation", "cosmos"))
        if space not in {"cosmos", "cosmos_model_flipud", "openpi", "rlinf_runtime_rot180"}:
            raise ValueError(f"Unrecognized image_orientation={space!r}: {path}")
        row = {
            "path": str(path.resolve()), "name": path.name,
            "trajectory_id": digest.hexdigest(), "task_id": int(match[2]),
            "success": match[4] == "True", "task": str(label),
            "observations": counts[0], "actions": counts[2], "horizon": horizon,
            "unused_tail_observations": counts[0] - 1 - horizon,
            "source_camera_space": "cosmos_model_flipud" if space.startswith("cosmos") else "rlinf_runtime_rot180",
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        }
    if verify_frames:
        read_episode(row)
    return row


def prepare_manifest(root: Path, output: Path, limit: int = 0, verify_frames: bool = False) -> dict:
    """Inventory complete real episodes without padding or assuming a test split."""
    rows, excluded, seen = [], [], set()
    for index, path in enumerate(sorted(root.glob("*suite=libero_spatial--*.hdf5"))):
        match = NAME.search(path.name)
        if not match:
            continue
        try:
            row = inventory_episode(path, match, verify_frames)
        except (OSError, ValueError) as exc:
            excluded.append({"path": str(path), "reason": str(exc).splitlines()[0]})
            print(f"Excluded unreadable/invalid episode: {path.name}", flush=True)
            continue
        if row["trajectory_id"] in seen:
            excluded.append({"path": str(path), "reason": "duplicate trajectory"})
            continue
        seen.add(row["trajectory_id"])
        rows.append(row)
        if (index + 1) % 50 == 0:
            print(f"Inventoried Spatial RGB/actions: {index + 1} files, usable={len(rows)}", flush=True)
    if limit:
        # Round-robin tasks for a useful small smoke; never select only task 0.
        pools = {task: [row for row in rows if row["task_id"] == task] for task in range(10)}
        rows = [pool[index] for index in range(max((len(p) for p in pools.values()), default=0))
                for pool in pools.values() if index < len(pool)][:limit]
    if not rows:
        raise ValueError(f"No usable Spatial real trajectories: {root}")
    payload = {
        "protocol": "local-variable-length-autoregressive-v1", "suite": "libero_spatial",
        "split_status": "training_overlap_unknown; local diagnostic, not verified held-out",
        "source": str(root.resolve()), "chunk": CHUNK,
        "future_indices": "1..horizon; action i predicts observation i+1",
        "conditioning": "only real initial RGB; repeat initial memory, then predicted RGB only",
        "padding": "none; drop at most seven unmatched tail steps",
        "full_rgb_decode_verified": verify_frames,
        "episodes": rows, "excluded": excluded,
        "summary": {
            "episodes": len(rows), "future_frames": sum(row["horizon"] for row in rows),
            "min_horizon": min(row["horizon"] for row in rows),
            "max_horizon": max(row["horizon"] for row in rows),
            "per_task": {str(task): sum(row["task_id"] == task for row in rows) for task in range(10)},
            "success": sum(row["success"] for row in rows),
        },
    }
    payload["manifest_sha256"] = digest_json(payload)
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError(f"Existing manifest differs; use a new output path: {output}")
    atomic_json(output, payload)
    return payload


@retry_io
def read_episode(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode aligned RGB in runtime camera space, retaining o0 for conditioning."""
    path = Path(row["path"])
    if path.stat().st_size != row["size"] or path.stat().st_mtime_ns != row["mtime_ns"]:
        raise ValueError(f"Episode changed after manifest creation: {path}")
    contents = path.read_bytes()
    if len(contents) != row["size"]:
        raise ValueError(f"Episode size changed while reading: {path}")
    if row.get("sha256") and hashlib.sha256(contents).hexdigest() != row["sha256"]:
        raise ValueError(f"Episode checksum changed after manifest creation: {path}")
    # Decode from an in-memory file, as in the training data reader. HDF5's
    # native file driver has returned EIO for valid files on this NFS mount.
    with h5py.File(io.BytesIO(contents), "r") as file:
        horizon = row["horizon"]
        views = []
        for key in ["primary_images_jpeg", "wrist_images_jpeg"]:
            frames = [np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB"))
                      for value in file[key][:horizon + 1]]
            rgb = np.stack(frames)
            if row["source_camera_space"] == "cosmos_model_flipud":
                rgb = np.flip(rgb, axis=-2)
            views.append(np.ascontiguousarray(rgb))
        actions = np.asarray(file["actions"][:horizon], dtype=np.float32)
    return views[0], views[1], actions


def resize_rgb(frames: np.ndarray, side: int) -> np.ndarray:
    return np.stack([np.asarray(Image.fromarray(frame).resize((side, side), Image.Resampling.BILINEAR))
                     for frame in frames]).astype(np.uint8)


class FeatureMoments:
    """Accumulate feature statistics without storing all frame embeddings."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all() or not len(values):
            raise ValueError("Expected a nonempty finite feature matrix")
        mean = values.mean(axis=0)
        centered = values - mean
        m2 = centered.T @ centered
        if self.mean is None:
            self.mean, self.m2, self.count = mean, m2, len(values)
            return
        if self.mean.shape != mean.shape:
            raise ValueError("Feature dimension changed")
        total = self.count + len(values)
        delta = mean - self.mean
        self.m2 += m2 + np.outer(delta, delta) * (self.count * len(values) / total)
        self.mean += delta * (len(values) / total)
        self.count = total

    def statistics(self):
        if self.count < 2:
            raise ValueError("Distribution metrics need at least two feature samples")
        return self.mean, self.m2 / (self.count - 1)


def frechet_distance(real: FeatureMoments, predicted: FeatureMoments) -> float:
    """Use a symmetric PSD square root, including singular small-sample cases."""
    from scipy.linalg import eigh

    mean_a, cov_a = real.statistics()
    mean_b, cov_b = predicted.statistics()
    if mean_a.shape != mean_b.shape:
        raise ValueError("Feature dimension mismatch")
    eig, vectors = eigh((cov_a + cov_a.T) / 2, check_finite=True)
    root_a = (vectors * np.sqrt(np.maximum(eig, 0))) @ vectors.T
    product = root_a @ cov_b @ root_a
    trace_root = np.sqrt(np.maximum(eigh((product + product.T) / 2, eigvals_only=True), 0)).sum()
    result = np.square(mean_a - mean_b).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_root
    # A negative result beyond floating-point cancellation is a numerical failure.
    tolerance = 1e-7 * max(1.0, np.trace(cov_a) + np.trace(cov_b))
    if result < -tolerance:
        raise ValueError(f"Negative Fréchet distance: {result}")
    return float(max(0, result))
