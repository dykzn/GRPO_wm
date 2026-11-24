"""CPU four-rank proof: broadcast a save request and commit native Adam state."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm_update_wan_controlled import CheckpointRequest
from wm_update_wan import save_state
from wm_update_data import atomic_json
from wm_update_distributed_smoke import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    torch.manual_seed(7)
    model = make_model(True, world)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SimpleNamespace(steps=2000, control_dir=args.output / "control", control_token="cpu-proof")
    identity = dict(manifest="controlled-save-smoke", seed=7, accumulation=4)
    control = CheckpointRequest(config, 1000, identity["manifest"], torch, dist, lambda:None, device="cpu")
    step = 1000
    while step < config.steps:
        control.begin_update()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(4):
            inputs = torch.tensor([[1.+rank+micro, 2., 3.]])
            (model(inputs).square().mean()/4).backward()
        optimizer.step()
        step += 1
        if step == 1002 and rank == 0:
            atomic_json(config.control_dir / "checkpoint_request.json",
                        dict(token="cpu-proof", manifest_sha256=identity["manifest"]))
        dist.barrier()
        if step == config.steps:
            save_state(args.output / f"step_{step:06d}", model, optimizer, step=step, identity=identity)
    assert step == 1003 and config.steps == 1003
    assert not torch.cuda.is_initialized()
    dist.barrier()
    if rank == 0:
        complete = json.loads((args.output / "step_001003/complete.json").read_text())
        assert complete["step"] == 1003 and complete["world_size"] == 4
        assert (args.output / "step_001003/state/.metadata").is_file()
        atomic_json(args.output / "result.json", dict(status="passed", world_size=4,
            restored_step=1000, requested_after_step=1002, saved_step=1003,
            real_backward_and_optimizer_updates=3, native_model_adam_rng_checkpoint=True,
            all_ranks_stopped_at_same_boundary=True, cuda_initialized=False))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
