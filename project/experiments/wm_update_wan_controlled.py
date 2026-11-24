"""Four-rank native Wan training with a checkpoint request at an update boundary.

The native dataset, RNG, loss, optimizer and checkpoint/export functions stay
unchanged. After resuming an existing checkpoint, an operational hook can end
the run at the next complete update. The native final-step branch then writes
the full model/Adam/RNG state and inference export before the worker retains
its reusable CUDA cache for the eight-rank handoff.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


class CheckpointRequest:
    def __init__(self, args, restored_step, manifest, torch, dist, reset, *, device=None):
        self.args, self.step, self.manifest = args, restored_step, manifest
        self.torch, self.dist, self.reset = torch, dist, reset
        self.device = device if device is not None else torch.device("cuda", torch.cuda.current_device())

    def begin_update(self):
        flag = 0
        if self.dist.get_rank() == 0:
            path = self.args.control_dir / "checkpoint_request.json"
            if path.is_file():
                try:
                    request = json.loads(path.read_text())
                    flag = 1 if (request.get("token") == self.args.control_token
                        and request.get("manifest_sha256") == self.manifest) else -1
                except (OSError, ValueError):
                    flag = -1
        # One rank makes the decision. All FSDP ranks must validate and save at
        # the same update, even if filesystem visibility differs temporarily.
        value = self.torch.tensor([flag], dtype=self.torch.int32, device=self.device)
        self.dist.broadcast(value, src=0)
        flag = int(value.item())
        if flag < 0:
            raise ValueError("Wan checkpoint request does not match this run and manifest")
        self.step += 1
        if flag:
            self.args.steps = min(self.args.steps, self.step)
        self.reset()


def controlled_train(args):
    import torch
    import torch.distributed as dist
    import wm_update_wan as native

    original_load = native.load_state
    reset = torch.cuda.reset_peak_memory_stats

    def load_and_arm(path, model, optimizer, *, identity):
        step = original_load(path, model, optimizer, identity=identity)
        control = CheckpointRequest(args, step, identity["manifest"], torch, dist, reset)
        torch.cuda.reset_peak_memory_stats = control.begin_update
        return step

    native.load_state = load_and_arm
    try:
        native.train(args)
    finally:
        native.load_state = original_load
        torch.cuda.reset_peak_memory_stats = reset


def retain_cache(args):
    import torch
    from stop_spatial_grpo import read_process
    from wm_update_data import atomic_json
    from wm_update_reserved_collect import owner_alive

    rank = int(os.environ["RANK"])
    own = read_process(os.getpid())
    torch.cuda.synchronize()
    atomic_json(args.control_dir / f"finished_rank{rank}.json", dict(
        pid=own.pid, start_ticks=own.started, rank=rank, step=args.steps,
        checkpoint=str(args.output / f"step_{args.steps:06d}"),
        reserved_gib=torch.cuda.memory_reserved()/2**30,
        allocated_gib=torch.cuda.memory_allocated()/2**30,
        phase="checkpoint_complete_cache_retained",
    ))
    owner = json.loads(args.owner_identity)
    while not (args.control_dir / "release.json").exists():
        if not owner_alive(owner):
            raise RuntimeError("Wan checkpoint handoff supervisor exited")
        time.sleep(.5)


def main():
    from wm_update import environment
    from wm_update_memory import add_memory_arguments, train_with_memory

    parser = argparse.ArgumentParser(description=__doc__)
    add_memory_arguments(parser)
    for name in ("manifest", "settings", "output", "resume", "control-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--control-token", required=True)
    parser.add_argument("--owner-identity", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 4 or args.accumulation != 4:
        parser.error("The controlled continuation requires four ranks and accumulation four")
    environment()
    train_with_memory(args, "wan", controlled_train)
    retain_cache(args)


if __name__ == "__main__":
    main()
