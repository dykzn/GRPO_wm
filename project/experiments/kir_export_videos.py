"""Export paired source recordings from KIR starts to their recorded ends.

These clips inspect offline reset data; no world-model predictions are made.
Playback FPS is a display setting, since the HDF5 files have no timestamps.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
from kir_frame_pool import (
    DEFAULT_GRASP_OUTPUT,
    VIEWS,
    FramePool,
    atomic_json,
    decode_rgb,
    retry_io,
    unchanged,
)
from PIL import Image, ImageDraw, ImageFont


def length_rows(pool: FramePool) -> list[dict]:
    rows = []
    for sid in np.flatnonzero(pool.indices["start_kind"] == 1):
        episode = pool.episodes[int(pool.indices["start_episode"][sid])]
        start = int(pool.indices["start_t"][sid])
        total = episode["observations"]
        rows.append(
            {
                "start_id": int(sid),
                "episode_index": episode["episode_index"],
                "task_id": episode["task_id"],
                "episode_number": episode["episode_number"],
                "success": episode["success"],
                "start_frame": start,
                "first_close_action_frame": int(pool.indices["start_event_t"][sid]),
                "end_frame": total - 1,
                "original_frames": total,
                "remaining_frames": total - start,
                "removed_frames": start,
                "frame_reduction_percent": 100 * start / total,
                "original_transitions": episode["transitions"],
                "remaining_transitions": episode["transitions"] - start,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"episodes": 0}
    original = np.array([row["original_frames"] for row in rows])
    removed = np.array([row["removed_frames"] for row in rows])
    transitions = np.array([row["original_transitions"] for row in rows])
    return {
        "episodes": len(rows),
        "original_frames": int(original.sum()),
        "remaining_frames": int((original - removed).sum()),
        "removed_frames": int(removed.sum()),
        "mean_original_frames": float(original.mean()),
        "mean_remaining_frames": float((original - removed).mean()),
        "mean_removed_frames": float(removed.mean()),
        "aggregate_frame_reduction_percent": float(
            100 * removed.sum() / original.sum()
        ),
        "mean_episode_frame_reduction_percent": float(
            np.mean(100 * removed / original)
        ),
        "mean_original_transitions": float(transitions.mean()),
        "mean_remaining_transitions": float((transitions - removed).mean()),
        "aggregate_transition_reduction_percent": float(
            100 * removed.sum() / transitions.sum()
        ),
    }


def decode_clip(episode: dict, start: int) -> tuple[list[np.ndarray], np.ndarray]:
    unchanged(episode)
    with h5py.File(episode["path"], "r") as file:
        views = []
        for name in VIEWS:
            frames = np.stack([decode_rgb(value) for value in file[name][start:]])
            if episode["source_camera_space"] == "cosmos_model_flipud":
                frames = np.ascontiguousarray(frames[:, :, ::-1])
            views.append(frames)
        gripper = np.asarray(file["actions"][start:, 6])
    unchanged(episode)
    return views, gripper


def render_frame(views, i, episode, row, font) -> np.ndarray:
    canvas = Image.new("RGB", (768, 512), "#161b22")
    draw = ImageDraw.Draw(canvas)
    outcome = "SUCCESS" if row["success"] else "FAILURE"
    draw.text(
        (10, 5),
        f"SOURCE RECORDING | task {row['task_id']} | episode {row['episode_number']} | final {outcome}",
        font=font,
        fill="white",
    )
    draw.text(
        (10, 25),
        f"Start={row['start_frame']} | first close action={row['first_close_action_frame']} | end={row['end_frame']}",
        font=font,
        fill="#c9d1d9",
    )
    for line, label in enumerate(textwrap.wrap(episode["task"], 94)):
        draw.text((10, 45 + line * 19), label, font=font, fill="#c9d1d9")
    for view, name in enumerate(("PRIMARY CAMERA", "WRIST CAMERA")):
        x = view * 384
        canvas.paste(
            Image.fromarray(views[view][i]).resize(
                (384, 384), Image.Resampling.BICUBIC
            ),
            (x, 96),
        )
        draw.rectangle((x + 4, 100, x + 158, 120), fill="#161b22")
        draw.text((x + 8, 101), name, font=font, fill="white")
    current = row["start_frame"] + i
    offset = current - row["first_close_action_frame"]
    event = "FIRST CLOSE COMMAND" if offset == 0 else f"close offset {offset:+d} frames"
    draw.text(
        (10, 482),
        f"Source frame {current}/{row['end_frame']} | clip {i + 1}/{row['remaining_frames']} | {event}",
        font=font,
        fill="#ffd166" if offset == 0 else "#c9d1d9",
    )
    draw.rectangle(
        (0, 507, int(768 * (i + 1) / row["remaining_frames"]) - 1, 511), fill="#58a6ff"
    )
    return np.asarray(canvas)


def export_clip(pool: FramePool, row: dict, output: Path, fps: float) -> dict:
    episode = pool.episodes[row["episode_index"]]
    views, _ = retry_io(decode_clip, episode, row["start_frame"])
    count = row["remaining_frames"]
    assert len(views[0]) == len(views[1]) == count
    packet = pool.read_start(row["start_id"])
    np.testing.assert_array_equal(views[0][0], packet["observation"]["main_images"])
    np.testing.assert_array_equal(views[1][0], packet["observation"]["wrist_images"])
    outcome = "success" if row["success"] else "failure"
    path = (
        output
        / f"task{row['task_id']:02d}_ep{row['episode_number']:04d}_{outcome}_f{row['start_frame']:04d}_to{row['end_frame']:04d}.mp4"
    )
    temporary = path.with_name(path.stem + ".partial.mp4")
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 13)
        if font_path.exists()
        else ImageFont.load_default()
    )
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (768, 512),
        fps=fps,
        codec="libx264",
        quality=None,
        ffmpeg_log_level="error",
        output_params=[
            "-crf",
            "18",
            "-preset",
            "fast",
            "-movflags",
            "+faststart",
            "-threads",
            "2",
        ],
    )
    expected = {}
    try:
        writer.send(None)
        try:
            for i in range(count):
                frame = render_frame(views, i, episode, row, font)
                if i in (0, count - 1):
                    expected[i] = frame
                writer.send(frame)
        finally:
            writer.close()
        decoder = imageio_ffmpeg.read_frames(str(temporary))
        metadata = next(decoder)
        errors = {}
        decoded = 0
        for i, raw in enumerate(decoder):
            if i in expected:
                actual = np.frombuffer(raw, np.uint8).reshape(512, 768, 3)
                errors[str(i)] = float(
                    np.abs(
                        actual.astype(np.int16) - expected[i].astype(np.int16)
                    ).mean()
                )
            decoded += 1
        if decoded != count or abs(metadata["fps"] - fps) > 0.01:
            raise ValueError("Encoded clip changed the frame count or playback rate")
        if len(errors) != len(expected) or max(errors.values()) > 10:
            raise ValueError(
                "Encoded clip's first/last image disagrees with the source"
            )
        unchanged(episode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(
        row,
        video=str(path.resolve()),
        task=episode["task"],
        source=episode["path"],
        playback_fps=fps,
        duration_seconds=count / fps,
        decoded_frames=decoded,
        first_last_encoding_mean_absolute_errors=errors,
        content="paired real source frames from start to recorded end; no model predictions",
    )
    print(
        f"Exported task {row['task_id']} {outcome}: {count} frames, {count / fps:.2f}s -> {path}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_GRASP_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", default="0,4,7")
    parser.add_argument("--fps", type=float, default=20)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    pool = FramePool(args.pool)
    pool.check_sources()
    if pool.manifest.get("selection", {}).get("method") != "first_stable_gripper_close":
        parser.error("This exporter requires a first-gripper-close selected pool")
    output = args.output or pool.root / "videos_to_end"
    output.mkdir(parents=True, exist_ok=True)
    rows = length_rows(pool)
    if not rows:
        parser.error("No intermediate starts in the pool")
    statistics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": str(pool.root.resolve()),
        "pool_sha256": pool.manifest["pool_sha256"],
        "selection": pool.manifest["selection"],
        "definition": "Original observations 0..N-1; retained observations start..N-1 inclusive. These are source lengths, not a measured training speedup.",
        "all": summarize(rows),
        "success": summarize([row for row in rows if row["success"]]),
        "failure": summarize([row for row in rows if not row["success"]]),
        "per_task": {
            str(t): summarize([row for row in rows if row["task_id"] == t])
            for t in sorted({row["task_id"] for row in rows})
        },
    }
    atomic_json(output / "trajectory_length_stats.json", statistics)
    with (output / "trajectory_lengths.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tasks = list(dict.fromkeys(int(value) for value in args.tasks.split(",")))
    videos = []
    for task in tasks:
        for success in (True, False):
            chosen = next(
                (
                    row
                    for row in rows
                    if row["task_id"] == task and row["success"] == success
                ),
                None,
            )
            if chosen is not None:
                videos.append(export_clip(pool, chosen, output, args.fps))
    pool.check_sources()
    atomic_json(
        output / "videos_manifest.json",
        {
            "pool_sha256": pool.manifest["pool_sha256"],
            "selection": "first indexed episode per requested task and final outcome",
            "playback_fps_is_display_setting": True,
            "videos": videos,
        },
    )
    cards = []
    for video in videos:
        name = Path(video["video"]).name
        outcome = "成功" if video["success"] else "失败"
        cards.append(
            f'<article><h2>任务 {video["task_id"]} · {outcome} · episode {video["episode_number"]}</h2><video controls preload="metadata" src="{html.escape(name)}"></video><p>{html.escape(video["task"])}</p><p>源帧 {video["start_frame"]}–{video["end_frame"]}，共 {video["remaining_frames"]} 帧，{video["duration_seconds"]:.2f} 秒</p></article>'
        )
    stats = statistics["all"]
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>KIR 起点至轨迹末尾</title>
<style>body{{font:16px system-ui;background:#0d1117;color:#e6edf3;margin:24px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}article{{background:#161b22;padding:16px;border-radius:12px}}video{{width:100%}}h2{{font-size:18px}}p{{line-height:1.5;color:#c9d1d9}}a{{color:#58a6ff}}</style>
<h1>第一次有效闭合前 {pool.manifest["selection"]["backtrack_frames"]} 帧至记录末尾</h1>
<p>以下为真实源轨迹截取：左侧主相机，右侧腕部相机。包含全部剩余帧，按 {args.fps:g} FPS 播放。视频不包含世界模型预测。</p>
<p>{stats["episodes"]} 条轨迹平均 {stats["mean_original_frames"]:.3f} → {stats["mean_remaining_frames"]:.3f} 帧，平均减少 {stats["mean_removed_frames"]:.3f} 帧，总帧数减少 {stats["aggregate_frame_reduction_percent"]:.3f}%。这不是训练耗时的实测变化。</p>
<p><a href="trajectory_length_stats.json">统计报告</a> · <a href="trajectory_lengths.csv">逐轨迹 CSV</a> · <a href="videos_manifest.json">视频清单</a></p><main>{"".join(cards)}</main></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "statistics": stats,
                "video_count": len(videos),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
