"""Exercise four-to-eight FSDP/DCP model and Adam state migration on CPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.tensor import DTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_update_data import atomic_json
from wm_update_distributed_smoke import make_model
from wm_update_wan import save_state
from wm_update_wan_reshard import load_resharded_state


def full_state(model, optimizer):
    def dense(value):
        if isinstance(value, DTensor):
            return value.full_tensor().detach().clone()
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: dense(item) for key, item in value.items()}
        if isinstance(value, list):
            return [dense(item) for item in value]
        if isinstance(value, tuple):
            return tuple(dense(item) for item in value)
        return value
    return dense(get_state_dict(model, optimizer))


def assert_state(actual, expected, *, rtol, atol):
    torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual[1]["state"], expected[1]["state"], rtol=rtol, atol=atol)
    assert actual[1]["param_groups"] == expected[1]["param_groups"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save4", "restore8"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    world, rank = dist.get_world_size(), dist.get_rank()
    assert world == (4 if args.mode == "save4" else 8)
    accumulation = 16 // world
    torch.manual_seed(7)
    model = make_model(True, world)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    identity = {"manifest": "reshard-smoke", "seed": 7, "accumulation": accumulation}

    def update(step):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            index = (step * accumulation + micro) * world + rank
            inputs = torch.tensor([[index / 32 + 1, 2, 3]])
            (model(inputs).square().mean() / accumulation).backward()
        optimizer.step()

    if args.mode == "save4":
        update(0)
        save_state(args.output / "source", model, optimizer, step=1, identity=identity)
        state = full_state(model, optimizer)
        if rank == 0:
            torch.save(state, args.output / "before.pt")
        update(1)
        state = full_state(model, optimizer)
        if rank == 0:
            torch.save(state, args.output / "expected.pt")
    else:
        assert load_resharded_state(args.output / "source", model, optimizer, identity=identity) == 1
        state = full_state(model, optimizer)
        if rank == 0:
            expected = torch.load(args.output / "before.pt", weights_only=False)
            assert_state(state, expected, rtol=0, atol=0)
        update(1)
        state = full_state(model, optimizer)
        if rank == 0:
            expected = torch.load(args.output / "expected.pt", weights_only=False)
            assert_state(state, expected, rtol=1e-5, atol=1e-7)
        save_state(args.output / "eight_rank_save", model, optimizer, step=2, identity=identity)
        if rank == 0:
            atomic_json(args.output / "result.json", {
                "status": "passed", "source_world": 4, "target_world": 8,
                "effective_batch": 16, "model_and_optimizer_restored_exactly": True,
                "next_update_matches": "rtol=1e-5 atol=1e-7", "eight_rank_checkpoint_saved": True,
                "cuda_initialized": torch.cuda.is_initialized(),
            })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
