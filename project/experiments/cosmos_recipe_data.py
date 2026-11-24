"""Versioned reconstruction of the released wrist-v3 data recipe.

The release confirms the frame/action contract and augmentation probabilities,
but does not include the memory-corruption or static-video implementation.
Those two transforms are deliberately named and recorded as reconstructions.
The legacy data module remains unchanged for exact old-checkpoint resumes.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import numpy as np
from PIL import Image, ImageFilter

from wm_update_data import CosmosUpdateDataset, PairedWindowDataset, read_window

RECIPE_VERSION = "wrist-v3-reconstructed-v1"
NOOP = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


@dataclass(frozen=True)
class RecipeSettings:
    memory_degrade_prob: float = 0.5
    static_video_probability: float = 0.05
    use_image_aug: bool = True
    use_stronger_image_aug: bool = True
    weight_decay: float = 0.01

    @classmethod
    def from_settings(cls, settings: dict) -> "RecipeSettings":
        overrides = settings.get("cosmos_recipe", {})
        if not isinstance(overrides, dict):
            raise ValueError("cosmos_recipe must be a settings object")
        allowed = {field.name for field in fields(cls)}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unknown Cosmos recipe settings: {sorted(unknown)}")
        values = {
            key: settings[key]
            for key in ["static_video_probability", "weight_decay"]
            if key in settings
        }
        recipe = cls(**dict(values, **overrides))
        for key in ["memory_degrade_prob", "static_video_probability"]:
            value = getattr(recipe, key)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be in [0,1]")
        if not math.isfinite(recipe.weight_decay) or recipe.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for key in ["use_image_aug", "use_stronger_image_aug"]:
            if not isinstance(getattr(recipe, key), bool):
                raise ValueError(f"{key} must be a boolean")
        return recipe

    def identity(self) -> dict:
        return {
            "version": RECIPE_VERSION,
            **asdict(self),
            "layout": "primary0,wrist0,wrist[t-2:t+1],wrist[t+1:t+9]",
            "action_contract": "incoming_action; fixed references use normalized sentinel",
            "early_history": "t=0 bootstrap; t=1,2 resampled to t>=3 or short-episode bootstrap",
            "memory_degradation": "reconstructed_downsample_blur_pixel_noise_v1; slots2:5 only",
            "static_video": "reconstructed_future_hold_v1; preserve history, hold current wrist and gripper",
            "image_aug": "native_dataset_utils.apply_image_aug; shared transform across all13 frames",
            "validation_augmentation": False,
            "original_augmentation_implementation_available": False,
        }


@contextmanager
def fixed_cpu_rng(seed: int):
    """Isolate CPU image augmentation from model/noise RNG, without CUDA calls."""
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            # torch.manual_seed also touches CUDA generators; this does not.
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def native_image_aug(video, *, stronger: bool, seed: int):
    with fixed_cpu_rng(seed):
        from cosmos_policy.datasets.dataset_utils import apply_image_aug

        return apply_image_aug(video, stronger=stronger)


def degrade_memory_frames(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild pixel-space corruption; not the native diffusion target noising."""
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Memory images must be uint8 [T,H,W,3]")
    height, width = images.shape[1:3]
    scale, blur, sigma = rng.uniform(0.5, 0.9), rng.uniform(0.2, 0.8), rng.uniform(1, 4)
    low_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    result = []
    for frame in images:
        image = Image.fromarray(frame).resize(low_size, Image.Resampling.BILINEAR)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        image = image.filter(ImageFilter.GaussianBlur(blur))
        pixels = np.asarray(image, dtype=np.float32)
        pixels += rng.normal(0, sigma, pixels.shape).astype(np.float32)
        result.append(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
    return np.stack(result)


def aligned_window(window: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore the fixed wrist anchor and its missing predecessor sentinel."""
    rgb, actions = window["wrist"].copy(), window["actions"].copy()
    rgb[1], rgb[0] = rgb[0].copy(), window["primary"][0]
    actions[:2] = NOOP
    missing = np.asarray(window["action_indices"]) < 0
    missing[:2] = True
    return rgb, actions, missing


class ReconstructedCosmosDataset(CosmosUpdateDataset):
    """Native Cosmos batch with fixed anchors and explicit train-only transforms."""

    def __init__(self, *args, recipe: RecipeSettings | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.primary_reference:
            raise ValueError(
                "The reconstructed wrist-v3 recipe requires primary_reference=True"
            )
        self.recipe = recipe or RecipeSettings()
        for key in ["actions_min", "actions_max"]:
            if not np.isfinite(np.asarray(self.stats[key])).all():
                raise ValueError("Cosmos action statistics must be finite")
        if np.any(
            np.asarray(self.stats["actions_max"])
            < np.asarray(self.stats["actions_min"])
        ):
            raise ValueError("Cosmos action maxima must not be below minima")

    def __getitem__(self, index: int) -> dict:
        import torch
        from world_model_grpo.cosmos_wrist_adapter import (
            _preprocess_wrist_model_space,
            _runtime_to_cosmos_model,
        )

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index), 73013])
        )
        window = PairedWindowDataset.__getitem__(self, index)
        if 0 < window["step"] < 3:
            maximum = window["episode"]["actions"] - 8
            step = int(rng.integers(3, maximum + 1)) if maximum >= 3 else 0
            window = read_window(window["episode"], step)
        rgb, actions, missing = aligned_window(window)
        static = (
            self.split == "train"
            and rng.random() < self.recipe.static_video_probability
        )
        if static:
            # Real observed history remains causal. Only the future is made
            # stationary, using zero delta pose and the last held gripper.
            rgb[5:] = rgb[4]
            actions[5:, :6] = 0
            actions[5:, 6] = actions[4, 6]
        rgb = np.stack(
            [
                _runtime_to_cosmos_model(
                    _preprocess_wrist_model_space(frame, (224, 224))
                )
                for frame in rgb
            ]
        )
        video = torch.from_numpy(rgb.transpose(3, 0, 1, 2).copy())
        if self.split == "train" and self.recipe.use_image_aug:
            video = native_image_aug(
                video,
                stronger=self.recipe.use_stronger_image_aug,
                seed=int(rng.integers(0, 2**63 - 1)),
            )
        if self.split == "train" and rng.random() < self.recipe.memory_degrade_prob:
            # Shared image augmentation above also changes targets. This
            # additional corruption never touches references or target truth.
            recent = video[:, 2:5].permute(1, 2, 3, 0).numpy()
            corrupted = degrade_memory_frames(recent, rng)
            video[:, 2:5] = torch.from_numpy(corrupted.transpose(3, 0, 1, 2).copy())
        minimum = np.asarray(self.stats["actions_min"], dtype=np.float32)
        maximum = np.asarray(self.stats["actions_max"], dtype=np.float32)
        denominator = np.where(maximum == minimum, 1.0, maximum - minimum)
        actions = np.clip(2 * (actions - minimum) / denominator - 1, -1, 1)
        actions[missing] = NOOP  # Sentinel is in normalized/model action space.
        return {
            "dataset_name": "video_data",
            "video": video.contiguous(),
            "action": torch.from_numpy(actions),
            "t5_text_embeddings": torch.zeros(512, 1024),
            "fps": 20,
            "padding_mask": torch.zeros(1, 224, 224),
            "num_conditional_frames": 2,
            "__key__": f"{window['episode']['trajectory_id']}:{window['step']}",
        }
