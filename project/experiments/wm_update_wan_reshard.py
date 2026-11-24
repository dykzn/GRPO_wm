"""Resume an existing four-rank Wan DCP checkpoint on eight ranks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def validate_resume(record: dict, identity: dict, world: int) -> None:
    previous = dict(record["identity"])
    current = dict(identity)
    old_accumulation = previous.pop("accumulation")
    new_accumulation = current.pop("accumulation")
    if previous != current:
        raise ValueError("Resharding requires identical data, sources, code and settings")
    if (record["world_size"], world) != (4, 8):
        raise ValueError("This handoff supports only four to eight ranks")
    if old_accumulation * 4 != new_accumulation * 8:
        raise ValueError("Resharding must preserve the effective batch size")


def load_resharded_state(path: Path, model, optimizer, *, identity: dict) -> int:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    from wm_update_wan import restore_rng

    record = json.loads((path / "complete.json").read_text())
    rank, world = dist.get_rank(), dist.get_world_size()
    validate_resume(record, identity, world)
    for old_rank in range(record["world_size"]):
        if not (path / f"rng_rank{old_rank}.pt").is_file():
            raise ValueError("Source checkpoint is missing a rank RNG state")
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    # DCP reads the old layout into the new DTensor layout for parameters and
    # Adam moments. The optimizer step is restored together with its moments.
    dcp.load(state, checkpoint_id=path / "state")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    step = int(record["step"])
    if rank < record["world_size"]:
        restore_rng(torch.load(path / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False))
    else:
        seed = int(identity["seed"]) + rank + 1_000_003 * step
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed(seed)
    dist.barrier()
    if rank == 0:
        print(json.dumps({"status": "reshard_restored", "step": step,
                          "old_world_size": 4, "world_size": 8,
                          "accumulation": identity["accumulation"],
                          "effective_batch": 8 * identity["accumulation"],
                          "rng": "old rank streams restored; new ranks seeded deterministically; not bitwise four-rank continuation"}), flush=True)
    return step


def main():
    from wm_update import environment
    from wm_update_memory import add_memory_arguments, train_with_memory
    import wm_update_wan

    parser = argparse.ArgumentParser(description=__doc__)
    add_memory_arguments(parser)
    for name in ("manifest", "output", "settings", "resume"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 8 or args.accumulation != 2:
        parser.error("Use eight ranks and accumulation two to preserve batch sixteen")
    environment()
    wm_update_wan.load_state = load_resharded_state
    train_with_memory(args, "wan", wm_update_wan.train)


if __name__ == "__main__":
    main()
