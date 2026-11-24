"""Select a versioned Cosmos recipe while retaining exact legacy resume identity.

The native trainer, diffusion/SDE, loss, FP32 master weights, RNG and DCP state
are reused. Scoped overrides change only the dataset, optimizer weight decay
and the recorded recipe identity. No installed Cosmos sources are modified.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

from cosmos_recipe_data import (
    RECIPE_VERSION,
    RecipeSettings,
    ReconstructedCosmosDataset,
)
from wm_update_data import atomic_json, digest, sha256

CORRECTED = "v3-reconstructed"
LEGACY = "legacy"


def select_recipe(args) -> str:
    """Old optimizer states stay on their original data/configuration recipe."""
    selected = getattr(args, "cosmos_recipe", "auto")
    if selected not in {"auto", LEGACY, CORRECTED}:
        raise ValueError(f"Unknown Cosmos recipe: {selected}")
    previous = None
    if args.resume:
        marker = Path(args.resume) / "wm_update_complete.json"
        identity = json.loads(marker.read_text())["identity"]
        version = identity.get("cosmos_recipe", {}).get("version")
        if version is not None and version != RECIPE_VERSION:
            raise ValueError(f"Unsupported saved Cosmos recipe version: {version}")
        previous = CORRECTED if version else LEGACY
    if selected == "auto":
        return previous or CORRECTED
    if previous is not None and previous != selected:
        raise ValueError(
            "Cannot resume optimizer state across Cosmos recipes. For a new recipe, "
            "omit --resume and set COSMOS_WRIST_CHECKPOINT to model weights in a fresh output."
        )
    return selected


def recipe_code_identity() -> str:
    root = Path(__file__).parent
    return digest(
        {
            name: sha256(root / name)
            for name in [
                "cosmos_recipe_data.py",
                "cosmos_recipe_train.py",
            ]
        }
    )


def effective_settings(args) -> RecipeSettings:
    return RecipeSettings.from_settings(json.loads(Path(args.settings).read_text()))


def apply_training_settings(cfg, recipe: RecipeSettings):
    cfg.optimizer.weight_decay = recipe.weight_decay
    return cfg


def checkpoint_model_inventory(checkpoint: Path) -> dict:
    """Validate referenced DCP bytes on CPU before allocating model memory."""
    from torch.distributed.checkpoint import FileSystemReader

    model = Path(checkpoint).resolve() / "model"
    metadata_path = model / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cosmos initializer is missing model/.metadata: {checkpoint}"
        )
    metadata = FileSystemReader(model).read_metadata()
    references = metadata.storage_data or {}
    if not references or not metadata.state_dict_metadata:
        raise ValueError(f"Empty Cosmos model checkpoint: {model}")
    sizes = {}
    for item in references.values():
        relative = Path(item.relative_path)
        path = model / relative
        if relative.is_absolute() or ".." in relative.parts or not path.is_file():
            raise FileNotFoundError(f"Missing or invalid Cosmos model shard: {path}")
        if path not in sizes:
            sizes[path] = path.stat().st_size
        if (
            item.offset < 0
            or item.length < 0
            or sizes[path] < item.offset + item.length
        ):
            raise ValueError(f"Truncated Cosmos model shard: {path}")
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "model_metadata_sha256": sha256(metadata_path),
        "state_entries": len(metadata.state_dict_metadata),
        "referenced_shards": {path.name: size for path, size in sorted(sizes.items())},
    }


def model_load_coverage(state_dict: dict, metadata) -> dict:
    """Require every tensor requested by the native model, including actions."""
    import torch

    expected = {
        key: tensor
        for key, tensor in state_dict.items()
        if torch.is_tensor(tensor) and not key.endswith("._extra_state")
    }
    if not expected:
        raise ValueError("Cosmos model load requested no tensor parameters")
    saved = metadata.state_dict_metadata
    missing = sorted(set(expected) - set(saved))
    mismatched = {
        key: {
            "model": list(tensor.shape),
            "checkpoint": list(getattr(saved[key], "size", [])),
        }
        for key, tensor in expected.items()
        if key in saved
        and tuple(tensor.shape) != tuple(getattr(saved[key], "size", []))
    }
    if missing or mismatched:
        raise ValueError(
            "Incomplete Cosmos model initialization: "
            f"missing={missing[:12]}, shape_mismatches={dict(list(mismatched.items())[:12])}"
        )
    return {
        "requested_tensors": len(expected),
        "matched_tensors": len(expected),
        "matched_elements": sum(tensor.numel() for tensor in expected.values()),
        "action_tensors": sorted(key for key in expected if ".action_mlp" in key),
        "missing_tensors": [],
        "shape_mismatches": {},
    }


def guarded_model_load(
    native_load,
    state_dict,
    storage_reader,
    load_planner,
    *,
    checkpoint: Path,
    audit_path: Path,
):
    """Reject partial model loads, then delegate unchanged native DCP copying."""
    actual = Path(storage_reader.path).resolve()
    expected = (Path(checkpoint) / "model").resolve()
    if actual != expected:
        raise ValueError(
            f"Cosmos checkpoint path changed: expected {expected}, got {actual}"
        )
    report = dict(
        model_load_coverage(state_dict, storage_reader.read_metadata()),
        checkpoint=str(Path(checkpoint).resolve()),
        metadata_sha256=sha256(actual / ".metadata"),
        status="checked_before_load",
    )
    atomic_json(audit_path, report)
    try:
        result = native_load(state_dict, storage_reader, load_planner)
    except BaseException as error:
        atomic_json(
            audit_path, dict(report, status="failed", error=type(error).__name__)
        )
        raise
    atomic_json(audit_path, dict(report, status="loaded"))
    return result


@contextmanager
def strict_native_model_load(args, *, backend=None):
    if backend is None:
        from cosmos_policy._src.predict2.checkpointer import dcp as backend

    previous = backend.dcp_load_state_dict
    backend.dcp_load_state_dict = partial(
        guarded_model_load,
        previous,
        checkpoint=Path(args.resume or os.environ["COSMOS_WRIST_CHECKPOINT"]),
        audit_path=Path(args.output)
        / "recipe"
        / f"model_load_rank_{os.environ.get('RANK', '0')}.json",
    )
    try:
        yield
    finally:
        backend.dcp_load_state_dict = previous


@contextmanager
def isolated_loader_rng():
    """Keep DataLoader iterator creation out of the restored model CPU RNG."""
    import torch
    from torch.utils.data import DataLoader

    previous = DataLoader.__iter__

    @wraps(previous)
    def iterate(loader):
        dataset = loader.dataset
        if isinstance(dataset, ReconstructedCosmosDataset) and loader.generator is None:
            rank = int(os.environ.get("RANK", "0"))
            split_tag = 1 if dataset.split == "val" else 0
            seed = (dataset.seed + 100003 * rank + 73013 + split_tag) % (2**63 - 1)
            loader.generator = torch.Generator(device="cpu").manual_seed(seed)
        return previous(loader)

    DataLoader.__iter__ = iterate
    try:
        yield
    finally:
        DataLoader.__iter__ = previous


def make_training_config(args):
    from wm_update_cosmos import make_training_config as native_config

    cfg = native_config(args)
    if select_recipe(args) == CORRECTED:
        apply_training_settings(cfg, effective_settings(args))
    return cfg


def augmented_identity(base: dict, recipe: RecipeSettings, cfg) -> dict:
    return dict(
        base,
        cosmos_recipe=recipe.identity(),
        recipe_code_sha256=recipe_code_identity(),
        model_load_policy="require_all_requested_tensor_names_and_shapes_v1",
        data_loader_rng="dedicated_CPU_generator; dataset samples depend on rank cursor only",
        effective_action_dropout=float(
            cfg.model.config.conditioner.action.dropout_rate
        ),
    )


@contextmanager
def corrected_training(args):
    import wm_update_cosmos as legacy

    recipe = effective_settings(args)
    native_config, native_identity = legacy.make_training_config, legacy.identity_for
    resolved = {}

    def config(config_args):
        cfg = apply_training_settings(native_config(config_args), recipe)
        resolved["config"] = cfg
        return cfg

    def identity(identity_args):
        return augmented_identity(
            native_identity(identity_args), recipe, resolved["config"]
        )

    native_dataset = legacy.CosmosUpdateDataset
    legacy.CosmosUpdateDataset = partial(ReconstructedCosmosDataset, recipe=recipe)
    legacy.make_training_config, legacy.identity_for = config, identity
    try:
        yield
    finally:
        legacy.CosmosUpdateDataset = native_dataset
        legacy.make_training_config, legacy.identity_for = (
            native_config,
            native_identity,
        )


def train(args) -> None:
    import wm_update_cosmos as legacy

    selected = select_recipe(args)
    print(
        f"Cosmos training recipe: {selected}; optimizer_resume={bool(args.resume)}",
        flush=True,
    )
    if selected == LEGACY:
        legacy.train(args)
        return
    # Record before expensive model loading so a startup failure still has an
    # auditable recipe. Each rank gets its own file, avoiding write races.
    recipe = effective_settings(args)
    source = Path(args.resume or os.environ["COSMOS_WRIST_CHECKPOINT"])
    model_inventory = checkpoint_model_inventory(source)
    # Bind before importing native Cosmos/Inductor dependencies. Their device
    # discovery must use this rank's GPU, and the shared allocator hook must
    # reserve memory before any model initializes.
    import torch

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    atomic_json(
        Path(args.output) / "recipe" / f"rank_{os.environ.get('RANK', '0')}.json",
        dict(
            recipe=recipe.identity(),
            code_sha256=recipe_code_identity(),
            initialization="full_optimizer_resume"
            if args.resume
            else "model_weights_only_new_optimizer",
            source=str(Path(os.environ["COSMOS_WRIST_CHECKPOINT"]).resolve()),
            loaded_model=model_inventory,
        ),
    )
    with (
        corrected_training(args),
        strict_native_model_load(args),
        isolated_loader_rng(),
    ):
        legacy.train(args)
