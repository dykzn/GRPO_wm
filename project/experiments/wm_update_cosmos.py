"""Spatial wrist adaptation using Cosmos' native loss, trainer and checkpointer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from wm_update_data import (
    CosmosUpdateDataset,
    atomic_json,
    file_identity,
    load_manifest,
    training_code_identity,
)


class UpdateSampler:
    """Rank-local indices derived from completed optimizer updates."""

    def __init__(self, start: int, stop: int, accumulation: int, rank: int, world: int):
        self.start, self.stop, self.accumulation = start, stop, accumulation
        self.rank, self.world = rank, world

    def __iter__(self):
        return iter(
            range(
                self.start * self.accumulation * self.world + self.rank,
                self.stop * self.accumulation * self.world + self.rank,
                self.world,
            )
        )

    def __len__(self):
        return (self.stop - self.start) * self.accumulation

    def set_epoch(self, epoch: int) -> None:
        # Native trainer requires this hook. Indices already encode the exact
        # update cursor and must not be reshuffled when an epoch changes.
        pass


def make_training_config(args):
    """Build a new run without editing any installed or existing experiment."""
    from cosmos_policy._src.imaginaire.utils.config_helper import override
    from cosmos_policy.config.config import make_config

    cfg = override(
        make_config(),
        [
            "--",
            "experiment=cosmos_predict2_2b_480p_libero_cosmos_wovr_wrist_full",
            f"model.config.tokenizer.vae_pth={os.environ['COSMOS_VAE']}",
        ],
    )
    cfg.trainer.max_iter = args.steps
    cfg.trainer.grad_accum_iter = args.accumulation
    cfg.trainer.distributed_parallelism = "ddp"
    cfg.trainer.seed = args.seed
    cfg.trainer.run_validation = True
    cfg.trainer.run_validation_on_start = False
    cfg.trainer.validation_iter = args.save_interval
    cfg.trainer.max_val_iter = 4
    cfg.trainer.logging_iter = 1
    # The inherited experiment also registers expensive text/video sampling
    # callbacks. Those need unrelated caption datasets and can allocate a
    # second generation pipeline; this update uses a separate video evaluator.
    from cosmos_policy._src.imaginaire.lazy_config import LazyDict

    essential = {
        "grad_clip",
        "low_prec",
        "iter_speed",
        "heart_beat",
        "device_monitor",
        "manual_gc",
    }
    cfg.trainer.callbacks = LazyDict(
        {key: value for key, value in cfg.trainer.callbacks.items() if key in essential}
    )
    cfg.optimizer.lr = args.lr
    # Keep the schedule independent of --steps so extending a resumed run
    # does not silently change its learning-rate history.
    cfg.scheduler.cycle_lengths = [100000000000000]
    cfg.scheduler.warm_up_steps = [100]
    cfg.scheduler.f_start = [0.1]
    cfg.scheduler.f_max = [1.0]
    cfg.scheduler.f_min = [1.0]
    cfg.checkpoint.load_path = str(
        args.resume or Path(os.environ["COSMOS_WRIST_CHECKPOINT"])
    )
    cfg.checkpoint.load_training_state = bool(args.resume)
    cfg.checkpoint.strict_resume = True
    cfg.checkpoint.save_iter = args.save_interval
    cfg.checkpoint.dcp_async_mode_enabled = False
    cfg.checkpoint.save_to_object_store.enabled = False
    cfg.checkpoint.load_from_object_store.enabled = False
    cfg.model.config.use_lora = False
    cfg.job.project, cfg.job.group, cfg.job.name = (
        "spatial_wmevo",
        "cosmos_wrist_v3",
        "train",
    )
    cfg.job.wandb_mode = "disabled"
    return cfg


def config_snapshot(value):
    """Record native config values, including partial callables, for auditing."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from functools import partial

    import attrs
    from omegaconf import ListConfig

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, partial):
        return {
            "_target_": "functools.partial",
            "func": config_snapshot(value.func),
            "args": config_snapshot(value.args),
            "keywords": config_snapshot(value.keywords or {}),
        }
    if isinstance(value, Mapping):
        return {str(key): config_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, ListConfig)):
        return [config_snapshot(item) for item in value]
    if isinstance(value, type) or callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    if is_dataclass(value):
        return {
            field.name: config_snapshot(getattr(value, field.name))
            for field in fields(value)
        }
    if attrs.has(type(value)):
        return {
            field.name: config_snapshot(getattr(value, field.name))
            for field in attrs.fields(type(value))
        }
    return str(value)


def save_master_weights(path: Path, optimizer) -> None:
    """FusedAdam stores FP32 masters outside its normal optimizer state_dict."""
    import torch

    if not getattr(optimizer, "master_weights", False):
        return
    groups = getattr(optimizer, "param_groups_master", None)
    if groups is None:
        raise RuntimeError(
            "FusedAdam master weights were not initialized before saving"
        )
    payload = {
        "schema": "cosmos-master-v1",
        "groups": [
            [p.detach().to(device="cpu", dtype=torch.float32) for p in group["params"]]
            for group in groups
        ],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_master_weights(path: Path, optimizer) -> None:
    """Restore full precision masters instead of recreating them from BF16."""
    import torch

    if not getattr(optimizer, "master_weights", False):
        return
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "cosmos-master-v1" or len(payload["groups"]) != len(
        optimizer.param_groups
    ):
        raise ValueError("Incompatible Cosmos master-weight checkpoint")
    restored = []
    for saved, group in zip(payload["groups"], optimizer.param_groups, strict=True):
        if len(saved) != len(group["params"]):
            raise ValueError("Cosmos master parameter-group length changed")
        masters = []
        for weight, param in zip(saved, group["params"], strict=True):
            if weight.dtype != torch.float32 or weight.shape != param.shape:
                raise ValueError("Cosmos master parameter shape or precision changed")
            masters.append(weight.to(device=param.device, dtype=torch.float32))
        restored.append({"params": masters})
    optimizer.param_groups_master = restored


def identity_for(args) -> dict:
    return {
        "manifest": load_manifest(args.manifest)["manifest_sha256"],
        "model": "cosmos",
        "settings": json.loads(args.settings.read_text()),
        "lr": args.lr,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "source": str(Path(os.environ["COSMOS_WRIST_CHECKPOINT"]).resolve()),
        "code_sha256": training_code_identity(),
        "assets": [
            file_identity(Path(os.environ[key]))
            for key in ["COSMOS_VAE", "COSMOS_DATASET_STATS"]
        ],
    }


def train(args) -> None:
    from functools import partial

    import torch
    import torch.distributed as dist
    import yaml
    from cosmos_policy._src.imaginaire.lazy_config import instantiate
    from cosmos_policy._src.imaginaire.utils import distributed
    from cosmos_policy._src.imaginaire.utils.callback import Callback
    from cosmos_policy._src.imaginaire.utils.context_managers import (
        distributed_init,
        model_init,
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.utils.data import DataLoader
    from wm_update_wan import fixed_rng, restore_rng, rng_state

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cudnn.enabled = False
    os.environ["COSMOS_DISTRIBUTED_BACKEND"] = "nccl"
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output)
    with distributed_init():
        distributed.init()
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = make_training_config(args)
    cfg.validate()
    cfg.freeze()
    identity = identity_for(args)
    args.output.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume:
        resume = json.loads((args.resume / "wm_update_complete.json").read_text())
        if resume["identity"] != identity or resume["world_size"] != world:
            raise ValueError(
                "Cosmos resume requires the same data, hyperparameters and GPU count"
            )
        for name in ["model", "optim", "scheduler", "trainer"]:
            if not (args.resume / name / ".metadata").exists():
                raise ValueError(f"Incomplete native Cosmos resume state: {name}")
        if not (args.resume / "master_weights.pt").is_file():
            raise ValueError(
                "Incomplete Cosmos resume state: missing FP32 master weights"
            )
        start = int(resume["step"])
    if start >= args.steps:
        raise ValueError("--steps must exceed the restored optimizer step")
    settings = json.loads(args.settings.read_text())
    data = CosmosUpdateDataset(
        str(args.manifest),
        stats_path=os.environ["COSMOS_DATASET_STATS"],
        primary_reference=True,
        **settings["sampling"],
    )
    validation = CosmosUpdateDataset(
        str(args.manifest),
        split="val",
        seed=args.seed,
        stats_path=os.environ["COSMOS_DATASET_STATS"],
        primary_reference=True,
    )

    class ValidationSampler:
        def __iter__(self):
            from wm_update_data import validation_indices

            return iter(validation_indices(len(validation), rank, world))

        def __len__(self):
            return min(4, len(validation))

    loader = DataLoader(
        data,
        batch_size=1,
        sampler=UpdateSampler(start, args.steps, args.accumulation, rank, world),
        num_workers=0,
    )
    val_loader = DataLoader(
        validation, batch_size=1, sampler=ValidationSampler(), num_workers=0
    )
    trainer = cfg.trainer.type(cfg)
    # Native checkpointers otherwise prefer latest_checkpoint.txt even when a
    # particular --resume was requested. Only touch this run's in-memory loader.
    trainer.checkpointer._read_latest_checkpoint_file = lambda: None
    native_load = trainer.checkpointer.load

    def load_with_master(model, optimizer, scheduler, grad_scaler):
        iteration = native_load(model, optimizer, scheduler, grad_scaler)
        if args.resume:
            restore_master_weights(args.resume / "master_weights.pt", optimizer)
        return iteration

    trainer.checkpointer.load = load_with_master
    with model_init():
        model = instantiate(cfg.model)
    block_type = type(model.net.blocks[0])
    apply_activation_checkpointing(
        model.net,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda module: isinstance(module, block_type),
    )

    class UpdateCallback(Callback):
        """Add rank RNG, atomic completion markers and measured update peaks."""

        current_step = None

        def on_train_start(self, model, iteration=0):
            if iteration != start:
                raise ValueError(
                    f"Native trainer restored step {iteration}, expected {start}"
                )
            if args.resume:
                restore_rng(
                    torch.load(
                        args.resume / f"rng_rank{rank}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                )

        def on_training_step_start(self, model, data_batch, iteration=0):
            if self.current_step != iteration:
                self.current_step = iteration
                torch.cuda.reset_peak_memory_stats()
                self.started = time.perf_counter()

        def on_training_step_batch_end(
            self, model, data_batch, output_batch, loss, iteration=0
        ):
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite Cosmos training loss")

        def on_training_step_end(
            self, model, data_batch, output_batch, loss, iteration=0
        ):
            values = torch.tensor(
                [
                    float(loss.detach()),
                    torch.cuda.max_memory_allocated() / 2**30,
                    torch.cuda.max_memory_reserved() / 2**30,
                ],
                device="cuda",
            )
            dist.all_reduce(values[:1])
            dist.all_reduce(values[1:], op=dist.ReduceOp.MAX)
            if rank == 0:
                record = {
                    "step": iteration,
                    "loss": float(values[0]) / world,
                    "peak_allocated_gib": float(values[1]),
                    "peak_reserved_gib": float(values[2]),
                    "seconds": time.perf_counter() - self.started,
                    "effective_batch": world * args.accumulation,
                }
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

    callback = UpdateCallback()
    callback.config, callback.trainer = cfg, trainer
    trainer.callbacks._callbacks.append(callback)
    native_save = trainer.checkpointer.save

    def save_with_rng(model, optimizer, scheduler, grad_scaler, *, iteration):
        native_save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        trainer.checkpointer.finalize()
        checkpoint = Path(cfg.job.path_local) / "checkpoints" / f"iter_{iteration:09d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            save_master_weights(checkpoint / "master_weights.pt", optimizer)
        torch.save(rng_state(), checkpoint / f"rng_rank{rank}.pt")
        dist.barrier()
        if rank == 0:
            for name in ["model", "optim", "scheduler", "trainer"]:
                if not (checkpoint / name / ".metadata").is_file():
                    raise RuntimeError(
                        f"Cosmos did not save complete native state: {checkpoint / name}"
                    )
            atomic_json(
                checkpoint / "wm_update_complete.json",
                {
                    "step": iteration,
                    "identity": identity,
                    "world_size": world,
                    "master_weights": "master_weights.pt",
                },
            )
        dist.barrier()

    trainer.checkpointer.save = save_with_rng

    def repeatable_validation(model, dataloader_val, iteration):
        # Cosmos' generic validation_step expects caption/guidance fields and
        # generates text-conditioned videos. Evaluate the native temporal
        # denoising loss here; full autoregressive evaluation is a separate job.
        from cosmos_policy._src.imaginaire.utils import misc

        previous = model.training
        model.eval()
        try:
            with (
                fixed_rng(args.seed),
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                losses = []
                for index, batch in enumerate(dataloader_val):
                    if index >= 4:
                        break
                    batch = misc.to(batch, device="cuda")
                    batch = {
                        key: value.to(dtype=torch.bfloat16)
                        if torch.is_tensor(value) and value.is_floating_point()
                        else value
                        for key, value in batch.items()
                    }
                    _, loss = model.training_step(batch, iteration)
                    losses.append(loss.float())
                value = torch.stack(losses).mean()
                dist.all_reduce(value)
                if rank == 0:
                    with (args.output / "validation.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {"step": iteration, "loss": float(value) / world}
                            )
                            + "\n"
                        )
        finally:
            model.train(previous)

    trainer.validate = repeatable_validation
    if rank == 0:
        atomic_json(
            args.output / "training_identity.json", dict(identity, world_size=world)
        )
        (args.output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config_snapshot(cfg), sort_keys=False)
        )
    trainer.train(model, loader, val_loader)
    dist.destroy_process_group()
