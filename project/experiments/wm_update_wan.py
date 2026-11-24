"""Sharded Wan updates with native WoVR loss and complete DCP resume state."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from wm_update_data import (
    WanUpdateDataset,
    atomic_json,
    file_identity,
    load_manifest,
    training_code_identity,
)


def rng_state() -> dict:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


@contextmanager
def fixed_rng(seed: int):
    """Make validation noise repeatable without perturbing training randomness."""
    import torch

    state = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed(seed)
        yield
    finally:
        restore_rng(state)


def save_state(path: Path, model, optimizer, *, step: int, identity: dict) -> None:
    """All ranks save model, optimizer and rank-local RNG; commit marker last."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    rank = dist.get_rank() if dist.is_initialized() else 0
    path.mkdir(parents=True, exist_ok=True)
    if (path / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed checkpoint: {path}")
    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=path / "state",
    )
    torch.save(rng_state(), path / f"rng_rank{rank}.pt")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        atomic_json(
            path / "complete.json",
            {
                "step": step,
                "identity": identity,
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            },
        )
    if dist.is_initialized():
        dist.barrier()


def load_state(path: Path, model, optimizer, *, identity: dict) -> int:
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    record = json.loads((path / "complete.json").read_text())
    world = dist.get_world_size() if dist.is_initialized() else 1
    if record["identity"] != identity or record["world_size"] != world:
        raise ValueError(
            "Resume requires the same manifest, hyperparameters and GPU count"
        )
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=path / "state")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    restore_rng(
        torch.load(path / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
    )
    return int(record["step"])


def native_module(root: Path):
    path = root / "examples/wanvideo/model_training/train_rlinf.py"
    spec = importlib.util.spec_from_file_location("spatial_wmevo_native_wan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanTrainingModule


def train(args) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        MixedPrecision,
        StateDictType,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    manifest = load_manifest(args.manifest)
    cfg = json.loads(args.settings.read_text())
    identity = {
        "manifest": manifest["manifest_sha256"],
        "model": "wan",
        "config": cfg,
        "source": str(Path(os.environ["WAN_SPATIAL_CHECKPOINT"]).resolve()),
        "accumulation": args.accumulation,
        "lr": args.lr,
        "seed": args.seed,
    }
    identity["code_sha256"] = training_code_identity()
    identity["source_files"] = [
        file_identity(Path(os.environ["WAN_SPATIAL_CHECKPOINT"]) / name)
        for name in ["model-00001.safetensors", "Wan2.2_VAE.pth"]
    ]
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.enabled = False
    root = Path(os.environ["WAN_SPATIAL_CHECKPOINT"])
    constructor = native_module(Path(os.environ["DIFFSYNTH_ROOT"]))
    module = constructor(
        model_paths=json.dumps(
            [str(root / "model-00001.safetensors"), str(root / "Wan2.2_VAE.pth")]
        ),
        trainable_models="dit",
        extra_inputs="input_image,action",
        use_gradient_checkpointing=True,
        context_noise_sigma=0.0,
        static_video_prob=cfg["static_video_probability"],
    )
    # FSDP stores FP32 master parameters and computes in BF16. Frozen VAE stays
    # outside sharding and follows the same BF16 encoding path as inference.
    module.pipe.dit.float()
    module.pipe.vae.requires_grad_(False).to("cuda", dtype=torch.bfloat16)
    module.pipe.device = torch.device("cuda", local_rank)
    wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(module.pipe.dit.blocks[0])},
    )
    model = FSDP(
        module,
        auto_wrap_policy=wrap,
        ignored_modules=[module.pipe.vae],
        device_id=local_rank,
        # DeviceMesh makes DCP use DTensor. Legacy ShardedTensor.is_meta fails
        # on ranks with no local shard for parameters such as [1, 6, dim].
        device_mesh=init_device_mesh("cuda", (world,)),
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        limit_all_gathers=True,
        sync_module_states=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=cfg["weight_decay"],
    )
    dataset = WanUpdateDataset(str(args.manifest), **cfg["sampling"])
    validation = WanUpdateDataset(str(args.manifest), split="val", seed=args.seed)
    step = (
        load_state(args.resume, model, optimizer, identity=identity)
        if args.resume
        else 0
    )
    if step >= args.steps:
        raise ValueError("--steps must exceed the restored optimizer step")
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json(
            args.output / "training_identity.json", dict(identity, world_size=world)
        )
    try:
        while step < args.steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started, loss_sum = time.perf_counter(), 0.0
            for micro in range(args.accumulation):
                index = (step * args.accumulation + micro) * world + rank
                # Synchronize sharded gradients each micro-step: FSDP.no_sync
                # would retain full unsharded gradients across accumulation.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(dataset[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Wan training loss")
                (loss / args.accumulation).backward()
                loss_sum += float(loss.detach()) / args.accumulation
            grad_norm = model.clip_grad_norm_(1.0)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            values = torch.tensor(
                [
                    loss_sum,
                    float(grad_norm),
                    torch.cuda.max_memory_allocated() / 2**30,
                    torch.cuda.max_memory_reserved() / 2**30,
                ],
                device="cuda",
            )
            dist.all_reduce(values[:1])
            dist.all_reduce(values[1:], op=dist.ReduceOp.MAX)
            record = {
                "step": step,
                "loss": float(values[0]) / world,
                "grad_norm": float(values[1]),
                "peak_allocated_gib": float(values[2]),
                "peak_reserved_gib": float(values[3]),
                "seconds": time.perf_counter() - started,
                "effective_batch": world * args.accumulation,
            }
            if step % args.save_interval == 0 or step == args.steps:
                # Every rank executes the same number of validation forwards.
                model.eval()
                with (
                    fixed_rng(args.seed),
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    from wm_update_data import validation_indices

                    losses = [
                        model(validation[index]).float()
                        for index in validation_indices(len(validation), rank, world)
                    ]
                    val_loss = torch.stack(losses).mean()
                    dist.all_reduce(val_loss)
                    record["validation_loss"] = float(val_loss) / world
                checkpoint = args.output / f"step_{step:06d}"
                save_state(checkpoint, model, optimizer, step=step, identity=identity)
                # Export a native runtime package, while preserving full DCP
                # state separately for continuation.
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    state = model.state_dict()
                if rank == 0:
                    from safetensors.torch import save_file

                    runtime = checkpoint / "runtime"
                    runtime.mkdir(exist_ok=True)
                    dit = {
                        k.removeprefix("pipe.dit."): v.to(torch.bfloat16).contiguous()
                        for k, v in state.items()
                        if k.startswith("pipe.dit.")
                    }
                    if not dit:
                        raise RuntimeError("Empty Wan runtime export")
                    save_file(dit, str(runtime / "model-00001.safetensors"))
                    (runtime / "Wan2.2_VAE.pth").symlink_to(
                        (root / "Wan2.2_VAE.pth").resolve()
                    )
                    atomic_json(
                        runtime / "wm_update.json",
                        {
                            "model": "wan",
                            "step": step,
                            "manifest_sha256": manifest["manifest_sha256"],
                        },
                    )
                del state
                dist.barrier()
                record["update_peak_allocated_gib"] = record["peak_allocated_gib"]
                record["update_peak_reserved_gib"] = record["peak_reserved_gib"]
                complete_peak = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated() / 2**30,
                        torch.cuda.max_memory_reserved() / 2**30,
                    ],
                    device="cuda",
                )
                dist.all_reduce(complete_peak, op=dist.ReduceOp.MAX)
                record["peak_allocated_gib"] = float(complete_peak[0])
                record["peak_reserved_gib"] = float(complete_peak[1])
            if rank == 0:
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
    finally:
        dist.destroy_process_group()
