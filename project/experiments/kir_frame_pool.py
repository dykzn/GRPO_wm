"""Build and read a shared, timestamp-aligned Spatial dual-camera frame pool.

Only indices are materialized. RGB, actions and optional proprio remain in
their original HDF5 files. Starts can follow a regular grid or precede the
first stable gripper-close command; neither proves that a grasp succeeded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
from PIL import Image

VERSION = 1
NAME = re.compile(r"suite=libero_spatial--.*--task=(\d+)--ep=(\d+)--success=(True|False)")
VIEWS = ("primary_images_jpeg", "wrist_images_jpeg")
DEFAULT_SOURCE = Path("/data41/qianb2608/wan/hf_home/LIBERO-Cosmos-Policy/all_episodes")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runtime_artifacts/kir/spatial_dual_view_v1"
DEFAULT_GRASP_OUTPUT = DEFAULT_OUTPUT.with_name("spatial_dual_view_first_close_pre16_v1")


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def unchanged(row: dict) -> None:
    if fingerprint(Path(row["path"])) != {key: row[key] for key in ("path", "size", "mtime_ns")}:
        raise ValueError(f"Source changed after indexing: {row['path']}")


def transient_io(exc) -> bool:
    return (
        isinstance(exc, OSError)
        and exc.errno in (errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.ECONNRESET)
    ) or any(token in str(exc).lower() for token in ("file read failed", "input/output error", "errno = 5"))


def retry_io(function, *args):
    for attempt in range(4):
        try:
            return function(*args)
        except (OSError, RuntimeError) as exc:
            if not transient_io(exc) or attempt == 3:
                raise
            time.sleep(0.5 * 2**attempt)


def camera_space(file) -> str:
    value = file.attrs.get("image_orientation", "cosmos")
    if isinstance(value, bytes):
        value = value.decode()
    if value in ("cosmos", "cosmos_model_flipud"):
        return "cosmos_model_flipud"
    if value in ("openpi", "rlinf_runtime_rot180"):
        return "rlinf_runtime_rot180"
    raise ValueError(f"Unknown image_orientation: {value!r}")


def decode_rgb(value) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(value))) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def scan_episode(path: Path) -> dict:
    match = NAME.search(path.name)
    if not match:
        raise ValueError(f"Not a Spatial episode: {path}")
    row = fingerprint(path)
    with h5py.File(path, "r") as file:
        if any(key not in file for key in (*VIEWS, "actions")):
            raise ValueError(f"Missing RGB/action stream: {path}")
        count = len(file[VIEWS[0]])
        if count < 9 or len(file[VIEWS[1]]) != count:
            raise ValueError(f"Unpaired or too-short camera streams: {path}")
        actions = np.asarray(file["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid actions: {path}")
        if len(actions) not in (count - 1, count):
            raise ValueError(f"Ambiguous action/observation alignment: {path}")
        transitions = min(count - 1, len(actions))
        task = file.attrs["task_description"]
        if isinstance(task, bytes):
            task = task.decode()
        success = match[3] == "True"
        if "success" in file.attrs:
            attr = str(file.attrs["success"]).lower()
            if attr not in ("true", "false", "1", "0") or (attr in ("true", "1")) != success:
                raise ValueError(f"Success attribute disagrees with filename: {path}")
        space = camera_space(file)
        content = hashlib.sha256(str(task).encode() + space.encode() + actions.tobytes())
        shapes = {}
        for view in VIEWS:
            content.update(view.encode())
            shape = None
            for value in file[view]:
                encoded = bytes(value)
                content.update(len(encoded).to_bytes(8, "little"))
                content.update(encoded)
                rgb = decode_rgb(encoded)
                if shape is None:
                    shape = rgb.shape
                if rgb.shape != shape:
                    raise ValueError(f"Camera shape changed within {path}: {view}")
            shapes[view] = list(shape)
        proprio_dim = None
        if "proprio" in file:
            proprio = np.asarray(file["proprio"], dtype=np.float32)
            if proprio.ndim != 2 or len(proprio) != count or not np.isfinite(proprio).all():
                raise ValueError(f"Unaligned or invalid proprio: {path}")
            proprio_dim = proprio.shape[1]
            content.update(proprio.tobytes())
        row.update(
            task_id=int(match[1]), episode_number=int(match[2]), task=str(task),
            success=success, observations=count, actions=len(actions), transitions=transitions,
            unmatched_final_actions=len(actions) - transitions,
            source_camera_space=space, image_shapes=shapes, source_proprio_dim=proprio_dim,
            content_sha256=content.hexdigest(), full_rgb_decode_verified=True,
        )
    unchanged(row)
    return row


def scan_cached(path: Path, cache: Path) -> dict:
    identity = fingerprint(path)
    destination = cache / (digest_json(identity) + ".json")
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("scan_version") == VERSION and saved.get("fingerprint") == identity:
            return saved["episode"]
    row = retry_io(scan_episode, path)
    atomic_json(destination, {"scan_version": VERSION, "fingerprint": identity, "episode": row})
    return row


@contextmanager
def build_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def build_pool(source: Path, output: Path, stride: int = 8, chunk: int = 8, workers: int = 4) -> dict:
    source, output = source.resolve(), output.resolve()
    if stride < 1 or chunk != 8 or workers < 1:
        raise ValueError("Require positive stride/workers and the production eight-action chunk")
    paths = sorted(source.glob("*suite=libero_spatial--*.hdf5"))
    if not paths:
        raise ValueError(f"No Spatial source episodes: {source}")
    with build_lock(output):
        if (output / "manifest.json").exists():
            pool = FramePool(output)
            manifest = pool.manifest
            if (manifest["source"], manifest["stride"], manifest["chunk"]) != (str(source), stride, chunk):
                raise ValueError("Existing pool has different settings; choose another output directory")
            if sorted(manifest["source_paths"]) != sorted(map(str, paths)):
                raise ValueError("Source episode inventory changed; choose another output directory")
            pool.check_sources()
            print("Existing complete frame pool verified; nothing rewritten.", flush=True)
            return manifest
        cache = output / "scan_cache"
        cache.mkdir(exist_ok=True)
        episodes, excluded, seen = [], [], {}
        scanned, failures = {}, {}

        def scan_result(path):
            try:
                return path, scan_cached(path, cache), None
            except Exception as exc:
                return path, None, exc

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (path, row, error) in enumerate(executor.map(scan_result, paths), 1):
                if error is not None:
                    failures[path] = error
                else:
                    scanned[path] = row
                if index % 25 == 0 or index == len(paths):
                    print(f"Scanned paired RGB/actions: {index}/{len(paths)}, verified={len(scanned)}, pending={len(failures)}", flush=True)
        # A second pass closes all earlier HDF5 handles before retrying NFS
        # failures. Other valid episodes remain cached even when one is bad.
        for path, error in list(failures.items()):
            if transient_io(error):
                print(f"Retrying source after scan pass: {path.name}", flush=True)
                try:
                    scanned[path] = retry_io(scan_cached, path, cache)
                    del failures[path]
                except Exception as exc:
                    failures[path] = exc
        if failures:
            report = [{"path": str(path), "error": str(error)} for path, error in failures.items()]
            atomic_json(output / "scan_failures.json", report)
            raise ValueError(f"{len(failures)} source episodes failed verification; see {output / 'scan_failures.json'}")
        (output / "scan_failures.json").unlink(missing_ok=True)
        for path in paths:
            row = scanned[path]
            digest = row["content_sha256"]
            if digest in seen:
                if seen[digest] != row["success"]:
                    raise ValueError(f"Conflicting success labels for identical trajectory: {path}")
                excluded.append({key: row[key] for key in ("path", "size", "mtime_ns")} | {"reason": "duplicate trajectory content"})
            else:
                seen[digest] = row["success"]
                row["episode_index"] = len(episodes)
                episodes.append(row)
        frame_episode, frame_t, action_t, next_t = [], [], [], []
        start_episode, start_t, start_kind = [], [], []
        for row in episodes:
            count, transitions, episode = row["observations"], row["transitions"], row["episode_index"]
            frame_episode.extend([episode] * count)
            frame_t.extend(range(count))
            action_t.extend(t if t < transitions else -1 for t in range(count))
            next_t.extend(t + 1 if t < transitions else -1 for t in range(count))
            starts = [0] + list(range(((4 + stride - 1) // stride) * stride, transitions - chunk + 1, stride))
            start_episode.extend([episode] * len(starts))
            start_t.extend(starts)
            start_kind.extend([0] + [1] * (len(starts) - 1))
        arrays = {key: np.asarray(value, dtype=np.int32) for key, value in {
            "frame_episode": frame_episode, "frame_t": frame_t, "action_t": action_t,
            "next_t": next_t, "start_episode": start_episode, "start_t": start_t,
            "start_kind": start_kind,
        }.items()}
        temporary = output / "indices.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / "indices.npz")
        per_task = {}
        for task in sorted({row["task_id"] for row in episodes}):
            rows = [row for row in episodes if row["task_id"] == task]
            ids = [row["episode_index"] for row in rows]
            per_task[str(task)] = {
                "episodes": len(rows), "success": sum(row["success"] for row in rows),
                "paired_frames": sum(row["observations"] for row in rows),
                "kir_candidates": int(np.sum(np.isin(arrays["start_episode"], ids) & (arrays["start_kind"] == 1))),
            }
        manifest = {
            "schema": "spatial-dual-view-kir-frame-pool", "version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
            "source_paths": list(map(str, paths)), "stride": stride, "chunk": chunk,
            "alignment": "action[t] drives paired observation[t] -> observation[t+1]",
            "alignment_verification": "stream lengths, source ordering and complete RGB decode; no independent timestamps in source",
            "output_camera_space": "rlinf_runtime_rot180", "reference_frame": 0,
            "candidate_selection": "regular grid; task-critical status is not annotated",
            "start_kind": {"0": "episode_initial", "1": "intermediate_candidate"},
            "conditioning": {
                "wan": "primary[0]; primary[t-3:t+1]; actions[t-4:t]",
                "cosmos_v3": "primary[0]; wrist[0] + wrist[t-2:t+1]; actions[t-3:t]",
                "initial": "repeat frame 0; dummy zero actions with open signed gripper=-1",
                "future": "never exposed by the default reset reader",
            },
            "split_status": "training candidates; source overlaps local wm_metrics diagnostics; not a held-out evaluation set",
            "full_rgb_decode_verified": True, "episodes": episodes, "excluded": excluded,
            "indices_sha256": sha256_file(output / "indices.npz"),
            "summary": {
                "episodes": len(episodes), "tasks": len(per_task),
                "success": sum(row["success"] for row in episodes),
                "failure": sum(not row["success"] for row in episodes),
                "paired_frames": len(frame_t), "valid_transitions": sum(row["transitions"] for row in episodes),
                "unmatched_final_actions": sum(row["unmatched_final_actions"] for row in episodes),
                "initial_starts": len(episodes), "kir_candidates": int(np.sum(arrays["start_kind"] == 1)),
                "starts": len(start_t), "per_task": per_task,
            },
        }
        manifest["pool_sha256"] = digest_json(manifest)
        atomic_json(output / "manifest.json", manifest)
        return manifest


def gripper_close_events(gripper: np.ndarray, stable_frames: int = 3) -> tuple[list[int], list[int]]:
    """Find negative-to-positive commands, requiring stable open/closed runs.

    An event at t means action[t] is the first close command. Observation[t]
    precedes its execution under the frame pool's action/observation contract.
    """
    gripper = np.asarray(gripper)
    if stable_frames < 1 or gripper.ndim != 1 or not np.isfinite(gripper).all():
        raise ValueError("Require a finite gripper vector and positive stable_frames")
    if np.any(np.abs(gripper) > 1.000001):
        raise ValueError("Gripper commands must use the signed [-1, 1] convention")
    raw = (np.flatnonzero((gripper[:-1] < 0) & (gripper[1:] > 0)) + 1).tolist()
    stable = [
        t for t in raw
        if t >= stable_frames and t + stable_frames <= len(gripper)
        and np.all(gripper[t-stable_frames:t] < 0)
        and np.all(gripper[t:t+stable_frames] > 0)
    ]
    return raw, stable


def build_grasp_pool(base_root: Path, output: Path, backtrack: int = 16, stable_frames: int = 3) -> dict:
    """Replace grid starts with one pre-first-close start per eligible episode.

    Reuse a fully verified frame table without duplicating or rescanning RGB.
    Never clamp an early start or substitute a later grasp attempt.
    """
    base_root, output = base_root.resolve(), output.resolve()
    if output == base_root or backtrack < 1 or stable_frames < 1:
        raise ValueError("Require a separate output and positive backtrack/stable_frames")
    base = FramePool(base_root)
    base.check_sources()
    if not base.manifest.get("full_rgb_decode_verified"):
        raise ValueError("The base pool must have complete paired RGB verification")
    selection = {
        "method": "first_stable_gripper_close",
        "backtrack_frames": backtrack,
        "stable_frames": stable_frames,
        "gripper_action_column": 6,
        "open_command": -1,
        "close_command": 1,
        "minimum_history_frames": 4,
        "minimum_future_transitions": base.manifest["chunk"],
        "event_alignment": "action[t_close] is the first close command; start=t_close-backtrack",
        "missing_or_ineligible_event": "initial only; no clamping or later-attempt substitution",
        "interpretation": "grasp-attempt heuristic; not a successful-grasp or per-frame failure annotation",
        "base_pool_sha256": base.manifest["pool_sha256"],
    }
    with build_lock(output):
        if (output / "manifest.json").exists():
            existing = FramePool(output)
            if existing.manifest.get("selection") != selection:
                raise ValueError("Existing grasp pool has different settings; choose another output directory")
            existing.check_sources()
            print("Existing complete grasp pool verified; nothing rewritten.", flush=True)
            return existing.manifest

        def read_event(row):
            unchanged(row)
            with h5py.File(row["path"], "r") as file:
                raw, stable = gripper_close_events(file["actions"][:, 6], stable_frames)
            unchanged(row)
            return raw, stable

        starts = {key: [] for key in ("start_episode", "start_t", "start_kind", "start_event_t")}
        events = []
        for row in base.episodes:
            raw, stable = retry_io(read_event, row)
            close_t = stable[0] if stable else None
            t = close_t - backtrack if close_t is not None else None
            status = "selected"
            if close_t is None:
                status = "no_stable_close"
            elif t < 4:
                status = "insufficient_history"
            elif t + base.manifest["chunk"] > row["transitions"]:
                status = "insufficient_future"
            starts["start_episode"].append(row["episode_index"])
            starts["start_t"].append(0)
            starts["start_kind"].append(0)
            starts["start_event_t"].append(-1)
            if status == "selected":
                starts["start_episode"].append(row["episode_index"])
                starts["start_t"].append(t)
                starts["start_kind"].append(1)
                starts["start_event_t"].append(close_t)
            events.append({
                "episode_index": row["episode_index"], "task_id": row["task_id"],
                "episode_success": row["success"], "raw_close_events": raw,
                "stable_close_events": stable, "first_close_action_t": close_t,
                "requested_start_t": t, "status": status,
            })
        base.check_sources()
        arrays = {key: base.indices[key] for key in ("frame_episode", "frame_t", "action_t", "next_t")}
        arrays.update({key: np.asarray(value, dtype=np.int32) for key, value in starts.items()})
        temporary = output / "indices.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / "indices.npz")
        selected = [row for row in events if row["status"] == "selected"]
        summary = dict(base.manifest["summary"])
        summary.update(kir_candidates=len(selected), starts=len(arrays["start_t"]))
        summary["per_task"] = {
            task: dict(counts, kir_candidates=sum(row["task_id"] == int(task) for row in selected))
            for task, counts in summary["per_task"].items()
        }
        manifest = dict(base.manifest)
        manifest.update(
            created_utc=datetime.now(timezone.utc).isoformat(), stride=None,
            candidate_selection=f"first stable gripper-close command minus {backtrack} frames",
            start_kind={"0": "episode_initial", "1": "before_first_gripper_close"},
            selection=selection, selection_events=events, summary=summary,
            selection_summary={
                "selected_episodes": len(selected),
                "selected_success_episodes": sum(row["episode_success"] for row in selected),
                "selected_failure_episodes": sum(not row["episode_success"] for row in selected),
                "skipped_episodes": len(events) - len(selected),
                "skip_reasons": {reason: sum(row["status"] == reason for row in events) for reason in (
                    "no_stable_close", "insufficient_history", "insufficient_future",
                )},
            },
            indices_sha256=sha256_file(output / "indices.npz"),
        )
        manifest.pop("pool_sha256")
        manifest["pool_sha256"] = digest_json(manifest)
        atomic_json(output / "manifest.json", manifest)
        return manifest


class FramePool:
    """CPU reader; supplies initialization only, without advancing any simulator."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        payload = dict(self.manifest)
        expected = payload.pop("pool_sha256")
        if digest_json(payload) != expected or self.manifest["version"] != VERSION:
            raise ValueError("Frame-pool manifest checksum/version mismatch")
        if sha256_file(self.root / "indices.npz") != self.manifest["indices_sha256"]:
            raise ValueError("Frame-pool index checksum mismatch")
        with np.load(self.root / "indices.npz", allow_pickle=False) as saved:
            self.indices = {key: saved[key] for key in saved.files}
        self.episodes = self.manifest["episodes"]

    def check_sources(self):
        for row in [*self.episodes, *self.manifest["excluded"]]:
            retry_io(unchanged, row)

    def __len__(self):
        return len(self.indices["start_t"])

    def task_success_deadlines(self, grace_steps: int) -> dict[int, int]:
        """Absolute action deadlines for tasks with at least one successful source episode."""
        longest = {}
        for row in self.episodes:
            if row["success"]:
                task = row["task_id"]
                longest[task] = max(longest.get(task, 0), row["actions"])
        return {task: steps + grace_steps for task, steps in sorted(longest.items())}

    def sample_groups(
        self, groups: int, *, kir_fraction: float, group_size: int = 8,
        seed=None, eligible_starts=None,
    ):
        """Task-balanced groups with an explicit KIR ratio; all branches share one start."""
        if groups < 1 or group_size < 1 or not 0 <= kir_fraction <= 1:
            raise ValueError("Invalid GRPO group count, size or KIR fraction")
        rng = np.random.default_rng(seed)
        kinds = self.indices["start_kind"]
        if eligible_starts is not None:
            kinds = np.where(eligible_starts, kinds, -1)
        task_ids = np.array([row["task_id"] for row in self.episodes])[self.indices["start_episode"]]
        selected = []
        for _ in range(groups):
            kind = int(rng.random() < kir_fraction)
            eligible = np.flatnonzero(kinds == kind)
            if not len(eligible):
                raise ValueError(f"No available starts of requested kind {kind}")
            task = rng.choice(np.unique(task_ids[eligible]))
            selected.append(int(rng.choice(eligible[task_ids[eligible] == task])))
        return np.repeat(np.array(selected, dtype=np.int32)[:, None], group_size, axis=1)

    def read_start(self, start_id: int, *, include_future: bool = False) -> dict:
        if not 0 <= start_id < len(self):
            raise IndexError(start_id)
        episode = int(self.indices["start_episode"][start_id])
        t = int(self.indices["start_t"][start_id])
        row = self.episodes[episode]
        return retry_io(self._read_start, row, t, start_id, include_future)

    def _read_start(self, row, t, start_id, include_future):
        unchanged(row)
        primary_times = list(range(t - 3, t + 1)) if t else [0] * 4
        wrist_times = [0] + (list(range(t - 2, t + 1)) if t else [0] * 3)
        needed = {0, *primary_times, *wrist_times}
        if include_future:
            needed.update(range(t + 1, t + 9))
        with h5py.File(row["path"], "r") as file:
            views = []
            for name in VIEWS:
                frames = {index: decode_rgb(file[name][index]) for index in sorted(needed)}
                if row["source_camera_space"] == "cosmos_model_flipud":
                    frames = {index: np.ascontiguousarray(np.flip(rgb, axis=1)) for index, rgb in frames.items()}
                views.append(frames)
            primary, wrist = views
            contexts = []
            for count in (4, 3):
                context = np.zeros((5, 7), dtype=np.float32)
                context[:, -1] = -1
                if t:
                    context[-count:] = np.asarray(file["actions"][t-count:t], dtype=np.float32)
                contexts.append(context)
            result = {
                "start_id": start_id, "episode_index": row["episode_index"], "frame_index": t,
                "task_id": row["task_id"], "task": row["task"], "episode_success": row["success"],
                "camera_space": "rlinf_runtime_rot180", "source_camera_space": row["source_camera_space"],
                "history_indices": {"wan_primary": primary_times, "cosmos_v3_wrist": wrist_times},
                "observation": {"main_images": primary[t].copy(), "wrist_images": wrist[t].copy()},
                "wan": {"reference": primary[0], "memory": np.stack([primary[i] for i in primary_times]), "context_actions": contexts[0]},
                "cosmos_v3": {"primary_reference": primary[0], "wrist_memory": np.stack([wrist[i] for i in wrist_times]), "context_actions": contexts[1]},
            }
            if include_future:
                result["audit_future"] = {
                    "actions": np.asarray(file["actions"][t:t+8], dtype=np.float32),
                    "primary": np.stack([primary[i] for i in range(t+1, t+9)]),
                    "wrist": np.stack([wrist[i] for i in range(t+1, t+9)]),
                }
        unchanged(row)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "select-grasp", "check"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-pool", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backtrack", type=int, default=16)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_GRASP_OUTPUT if args.command == "select-grasp" else DEFAULT_OUTPUT
    if args.command == "build":
        build_pool(args.source, args.output, stride=args.stride, workers=args.workers)
    elif args.command == "select-grasp":
        build_grasp_pool(args.base_pool, args.output, args.backtrack, args.stable_frames)
    pool = FramePool(args.output)
    pool.check_sources()
    candidates = pool.indices["start_kind"] == 1
    failures = np.array([not row["success"] for row in pool.episodes])[pool.indices["start_episode"]]
    chosen = []
    for mask in (~candidates, candidates, candidates & failures):
        ids = np.flatnonzero(mask)
        if len(ids):
            chosen.append(int(ids[0]))
    audit = []
    for start_id in dict.fromkeys(chosen):
        sample = pool.read_start(start_id, include_future=True)
        audit.append({
            key: sample[key] for key in ("start_id", "episode_index", "frame_index", "task_id", "episode_success", "history_indices")
        } | {"future_frames": len(sample["audit_future"]["primary"]), "future_actions": len(sample["audit_future"]["actions"])})
        if "start_event_t" in pool.indices and pool.indices["start_kind"][start_id] == 1:
            close_t = int(pool.indices["start_event_t"][start_id])
            distance = close_t - sample["frame_index"]
            if distance != pool.manifest["selection"]["backtrack_frames"]:
                raise ValueError("Grasp start does not match its configured pre-close offset")
            audit[-1].update(first_close_action_t=close_t, frames_before_close=distance)
    validation = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "pool_sha256": pool.manifest["pool_sha256"],
        "all_source_fingerprints_checked": True,
        "indices_sha256_checked": True, "sample_checks": audit,
    }
    atomic_json(args.output / "validation.json", validation)
    print(json.dumps({"output": str(args.output.resolve()), "summary": pool.manifest["summary"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
