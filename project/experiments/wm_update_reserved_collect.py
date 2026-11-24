"""Run the existing collector with a reusable cache owned by each policy worker."""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path


def reserved_collect(args, collect):
    import torch
    import wm_update_collect
    from wm_update_memory import MemorySettings, TrainingMemoryReservation, memory_hooks

    settings = MemorySettings.from_args(args)
    if not settings.enabled:
        return collect(args)
    os.environ.update(LOCAL_RANK="0", RANK="0", LOCAL_WORLD_SIZE="1")
    torch.cuda.set_device(0)
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    pool = TrainingMemoryReservation(
        torch, args.output / "memory" / f"gpu_{device}",
        settings, "policy_collection",
    )
    with memory_hooks(torch, pool):
        pool.initialize()
        original_episode = wm_update_collect.collect_episode
        completed = 0

        def observed_episode(*episode_args, **kwargs):
            nonlocal completed
            data = original_episode(*episode_args, **kwargs)
            completed += 1
            if completed == 1 or completed % settings.log_every == 0:
                pool.record("after_policy_episode", completed_episodes=completed)
            return data

        wm_update_collect.collect_episode = observed_episode
        try:
            collect(args)
            pool.record("collection_complete")
        finally:
            wm_update_collect.collect_episode = original_episode


def main():
    import wm_update
    import wm_update_collect

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,expandable_segments:True"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    # The parent's existing sharding/merge logic launches this same entry point.
    wm_update.SCRIPT = Path(__file__).resolve()
    original = wm_update_collect.collect_shard
    wm_update_collect.collect_shard = lambda args: reserved_collect(args, original)
    result = wm_update.main()
    if len(sys.argv) > 1 and sys.argv[1] == "train-worker":
        retain_training_cache_until_handoff()
    return result


def owner_alive(identity):
    try:
        content = Path(f"/proc/{identity['pid']}/stat").read_text()
        fields = content[content.rfind(")")+2:].split()
        return fields[0] != "Z" and int(fields[19]) == identity["start_ticks"]
    except FileNotFoundError:
        return False


def retain_training_cache_until_handoff():
    """Keep the real training process's reusable cache until both models finish."""
    directory = os.environ.get("WM_UPDATE_HANDOFF_ROOT")
    if not directory:
        return
    import torch
    from wm_update_data import atomic_json

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model = sys.argv[sys.argv.index("--model")+1]
    rank = int(os.environ["RANK"])
    torch.cuda.synchronize()
    atomic_json(root / f"{model}_rank{rank}.json", dict(
        model=model, rank=rank, pid=os.getpid(),
        allocated_gib=torch.cuda.memory_allocated()/2**30,
        reserved_gib=torch.cuda.memory_reserved()/2**30,
        visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        phase="training_complete_cache_retained",
    ))
    identity = json.loads(os.environ["WM_UPDATE_HANDOFF_OWNER"])
    print(f"{model} rank={rank}: training complete; retaining reusable cache until both models finish", flush=True)
    while not (root / "release.json").exists():
        if not owner_alive(identity):
            raise RuntimeError("Training handoff supervisor exited")
        time.sleep(.5)


if __name__ == "__main__":
    raise SystemExit(main())
