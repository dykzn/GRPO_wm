#!/usr/bin/env python3
"""Build a compact, deterministic metadata manifest for Cosmos LIBERO rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = "2026-07-29.2"
FILENAME_RE = re.compile(
    r"^episode_data--suite=(?P<suite>[^-]+)--.*--task=(?P<task>\d+)--"
    r"ep=(?P<episode>\d+)--success=(?P<success>True|False)--regen_demo\.hdf5$"
)
EXPECTED_SUITE_COUNTS = {
    "libero_10": 500,
    "libero_goal": 500,
    "libero_object": 500,
    "libero_spatial": 500,
}


def _trajectory_sha256(actions: np.ndarray, proprio: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, array in (("actions", actions), ("proprio", proprio)):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def filename_sha256(episodes: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(episode["file_path"] for episode in episodes).encode("utf-8")
    ).hexdigest()


def metadata_sha256(episodes: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(
            "|".join(
                [
                    episode["file_path"],
                    str(episode["size_bytes"]),
                    str(episode["mtime_ns"]),
                    episode["suite"],
                    str(episode["task_id"]),
                    str(episode["episode_id"]),
                    str(episode["success"]),
                    str(episode["num_steps"]),
                    str(episode["is_jpeg"]),
                    episode["trajectory_sha256"],
                    episode["task_description"],
                ]
            )
            for episode in episodes
        ).encode("utf-8")
    ).hexdigest()


def inspect_episode(file_path: str) -> dict:
    path = Path(file_path)
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected Cosmos LIBERO rollout filename: {path.name}")

    with h5py.File(path, "r") as episode_file:
        if "primary_images_jpeg" in episode_file:
            image_key = "primary_images_jpeg"
            is_jpeg = True
        elif "primary_images" in episode_file:
            image_key = "primary_images"
            is_jpeg = False
        else:
            raise KeyError(f"No primary image dataset in {path}")

        num_steps = int(len(episode_file[image_key]))
        task_description = str(episode_file.attrs.get("task_description", ""))
        success = bool(episode_file.attrs.get("success", False))
        lengths = {
            key: int(len(episode_file[key]))
            for key in ("actions", "proprio", image_key)
        }
        actions = episode_file["actions"][:]
        proprio = episode_file["proprio"][:]
        wrist_key = "wrist_images_jpeg" if is_jpeg else "wrist_images"
        lengths[wrist_key] = int(len(episode_file[wrist_key]))

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Misaligned episode arrays in {path}: {lengths}")
    if actions.shape != (num_steps, 7):
        raise ValueError(
            f"Expected actions [{num_steps},7] in {path}, got {actions.shape}."
        )
    if proprio.shape != (num_steps, 9):
        raise ValueError(
            f"Expected proprio [{num_steps},9] in {path}, got {proprio.shape}."
        )
    if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
        raise ValueError(f"Non-finite action/proprio values in {path}.")
    if num_steps < 23:
        raise ValueError(
            f"Episode {path} has {num_steps} steps; an unpadded temporal "
            "window at t=6 requires at least 23."
        )
    filename_success = match.group("success") == "True"
    if success != filename_success:
        raise ValueError(
            f"Success mismatch for {path}: filename={filename_success}, HDF5={success}"
        )
    if not task_description:
        raise ValueError(f"Missing task_description in {path}")

    return {
        "file_path": path.name,
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "suite": match.group("suite"),
        "task_id": int(match.group("task")),
        "episode_id": int(match.group("episode")),
        "success": success,
        "task_description": task_description,
        "num_steps": num_steps,
        "is_jpeg": is_jpeg,
        # This identity is independent of the filename and keeps duplicate
        # robot trajectories in the same train/calibration/holdout split.
        "trajectory_sha256": _trajectory_sha256(actions, proprio),
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(payload, temp_file, indent=2, sort_keys=True)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    rollout_data_dir = Path(args.rollout_data_dir).resolve()
    output_path = Path(args.output).resolve()
    files = sorted(str(path) for path in rollout_data_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes found in {rollout_data_dir}")
    if not 1 <= args.max_workers <= 8:
        raise ValueError("--max-workers must be in [1,8] to bound host I/O and process use.")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        episodes = []
        for inspected_count, episode in enumerate(
            executor.map(inspect_episode, files, chunksize=1), start=1
        ):
            episodes.append(episode)
            if inspected_count % 50 == 0 or inspected_count == len(files):
                print(
                    f"[manifest] inspected {inspected_count}/{len(files)} episodes",
                    flush=True,
                )

    filename_digest = filename_sha256(episodes)
    metadata_digest = metadata_sha256(episodes)
    suite_counts = {}
    for episode in episodes:
        suite_counts[episode["suite"]] = suite_counts.get(episode["suite"], 0) + 1
    if suite_counts != EXPECTED_SUITE_COUNTS:
        raise ValueError(
            "Cosmos LIBERO rollout suite counts are incomplete or unexpected: "
            f"expected {EXPECTED_SUITE_COUNTS}, got {dict(sorted(suite_counts.items()))}."
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "rollout_data_dir": str(rollout_data_dir),
        "episode_count": len(episodes),
        "total_steps": sum(episode["num_steps"] for episode in episodes),
        "filename_sha256": filename_digest,
        "metadata_sha256": metadata_digest,
        "suite_counts": dict(sorted(suite_counts.items())),
        "episodes": episodes,
    }
    atomic_write_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "episode_count": payload["episode_count"],
                "total_steps": payload["total_steps"],
                "filename_sha256": filename_digest,
                "metadata_sha256": metadata_digest,
                "suite_counts": payload["suite_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
