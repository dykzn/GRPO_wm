from pathlib import Path

import h5py
import numpy as np
import pytest

from build_libero_cosmos_episode_manifest import inspect_episode


def _write_episode(path: Path, *, success: bool = True, num_steps: int = 23) -> None:
    with h5py.File(path, "w") as episode_file:
        episode_file.attrs["success"] = success
        episode_file.attrs["task_description"] = "pick up the bowl"
        episode_file.create_dataset("actions", data=np.zeros((num_steps, 7), dtype=np.float32))
        episode_file.create_dataset("proprio", data=np.zeros((num_steps, 9), dtype=np.float32))
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        primary = episode_file.create_dataset("primary_images_jpeg", (num_steps,), dtype=jpeg_dtype)
        wrist = episode_file.create_dataset("wrist_images_jpeg", (num_steps,), dtype=jpeg_dtype)
        for frame_idx in range(num_steps):
            primary[frame_idx] = np.asarray([frame_idx], dtype=np.uint8)
            wrist[frame_idx] = np.asarray([frame_idx], dtype=np.uint8)


def test_manifest_episode_inspection(tmp_path: Path):
    path = tmp_path / (
        "episode_data--suite=libero_object--2025_08_03-18_55_45--"
        "task=2--ep=17--success=True--regen_demo.hdf5"
    )
    _write_episode(path)
    entry = inspect_episode(str(path))
    assert entry["file_path"] == path.name
    assert entry["size_bytes"] == path.stat().st_size
    assert entry["mtime_ns"] == path.stat().st_mtime_ns
    assert entry["suite"] == "libero_object"
    assert entry["task_id"] == 2
    assert entry["episode_id"] == 17
    assert entry["success"] is True
    assert entry["task_description"] == "pick up the bowl"
    assert entry["num_steps"] == 23
    assert entry["is_jpeg"] is True
    assert len(entry["trajectory_sha256"]) == 64


def test_manifest_rejects_misaligned_arrays(tmp_path: Path):
    path = tmp_path / (
        "episode_data--suite=libero_goal--2025_08_03-18_55_46--"
        "task=1--ep=3--success=True--regen_demo.hdf5"
    )
    _write_episode(path)
    with h5py.File(path, "a") as episode_file:
        del episode_file["actions"]
        episode_file.create_dataset("actions", data=np.zeros((20, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="Misaligned"):
        inspect_episode(str(path))


def test_trajectory_identity_changes_with_robot_data(tmp_path: Path):
    first = tmp_path / (
        "episode_data--suite=libero_goal--2025_08_03-18_55_46--"
        "task=1--ep=3--success=True--regen_demo.hdf5"
    )
    second = tmp_path / (
        "episode_data--suite=libero_goal--2025_08_03-18_55_47--"
        "task=1--ep=4--success=True--regen_demo.hdf5"
    )
    _write_episode(first)
    _write_episode(second)
    with h5py.File(second, "a") as episode_file:
        actions = episode_file["actions"][:]
        actions[0, 0] = 1.0
        del episode_file["actions"]
        episode_file.create_dataset("actions", data=actions)
    assert (
        inspect_episode(str(first))["trajectory_sha256"]
        != inspect_episode(str(second))["trajectory_sha256"]
    )
