"""CPU ranks: real backward, distributed save, restore, next update."""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_update_data import atomic_json
from wm_update_wan import load_state, save_state


class ModulatedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        # Like Wan modulation: only rank 0 has a dimension-0 tensor shard.
        self.modulation = torch.nn.Parameter(torch.ones(1, 3))

    def forward(self, values):
        return self.linear(values) * self.modulation


class ShardedToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3).requires_grad_(False)
        self.blocks = torch.nn.Sequential(ModulatedBlock(), ModulatedBlock())
        self.head = torch.nn.Linear(3, 1)

    def forward(self, values):
        return self.head(self.blocks(self.frozen(values)))


def make_model(fsdp, world):
    if not fsdp:
        return DistributedDataParallel(torch.nn.Linear(3, 1))
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    module = ShardedToy()
    return FSDP(
        module,
        device_id=torch.device("cpu"),
        device_mesh=init_device_mesh("cpu", (world,)),
        use_orig_params=True,
        ignored_modules=[module.frozen],
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={ModulatedBlock}
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fsdp", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("This smoke is CPU-only; set CUDA_VISIBLE_DEVICES='' explicitly")
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(7)
    model = make_model(args.fsdp, world)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.tensor([[1.0 + rank, 2, 3]])

    def update(module, opt):
        opt.zero_grad()
        module(inputs).square().mean().backward()
        opt.step()

    update(model, optimizer)
    identity = {"manifest": "distributed-smoke", "world_size": world}
    save_state(args.output / "checkpoint", model, optimizer, step=1, identity=identity)
    update(model, optimizer)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    restored = make_model(args.fsdp, world)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    step = load_state(
        args.output / "checkpoint", restored, restored_optimizer, identity=identity
    )
    assert step == 1
    update(restored, restored_optimizer)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    assert not torch.cuda.is_initialized()
    dist.barrier()
    if rank == 0:
        report = {
            "world_size": world,
            "parallelism": "fsdp_dtensor" if args.fsdp else "ddp",
            "backward_updates": 2,
            "restored_step": step,
            "next_update_bitwise_equal": True,
            "cuda_initialized": False,
        }
        atomic_json(args.output / "result.json", report)
        print(report, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
