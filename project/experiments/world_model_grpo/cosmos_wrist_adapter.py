"""Contract-level Cosmos-WoVR wrist adapter for online WorldSample.

The serial Wan scorer owns primary-video generation.  This module owns only
the strict Cosmos temporal wrist contract already used by ``WanEnv``:

    real runtime o0 + wrist o0
        -> fixed reference/four-frame memory + sentinel actions
        -> query_temporal_future_batch(actions[8,7])
        -> generated wrist clip
        -> roll wrist memory and the previous four action rows

It intentionally has no Cosmos model dependency.  A real
``WorldPilot.cosmos_bridge.cosmos_client.CosmosClient`` or a shape-checking
fake can be injected, which makes the camera/action/metadata boundary
testable before a server is started.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Any, Protocol, Sequence

import numpy as np


class TemporalCosmosClient(Protocol):
    """Subset of CosmosClient used by the strict WoVR wrist route."""

    def query_metadata(self) -> dict[str, Any]: ...

    def query_temporal_future_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class CosmosWristAdapterConfig:
    host: str = "127.0.0.1"
    port: int = 8055
    worldpilot_root: pathlib.Path | None = None
    image_size: tuple[int, int] = (224, 224)
    chunk_size: int = 8
    action_dim: int = 7
    primary_reference: bool = False
    cross_view_primary_history: bool = False
    guidance: float = 1.0
    future_denoising_steps: int = 5
    actions_are_normalized: bool = False
    gripper_action_mode: str = "already_cosmos"


@dataclasses.dataclass
class CosmosWristState:
    """Mutable per-request WoVR conditioning buffers, in Cosmos model space."""

    reference_primary: np.ndarray
    memory_primary: np.ndarray
    condition_actions: np.ndarray
    cross_view_reference_primary: np.ndarray | None = None
    cross_view_memory_primary: np.ndarray | None = None


def _as_rgb_frame(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} must contain one o0 frame, got {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {array.shape}")
    return np.ascontiguousarray(array)


def _resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    height, width = int(size[0]), int(size[1])
    image = Image.fromarray(_as_rgb_frame(frame, "image"))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.ascontiguousarray(
        np.asarray(image.resize((width, height), resampling), dtype=np.uint8)
    )


def _preprocess_wrist_model_space(
    frame: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Match the WoVR wrist center-crop preprocessing used by ``WanEnv``."""

    if size[0] != size[1]:
        raise ValueError(f"WoVR adapter expects a square image size, got {size}")
    side = int(size[0])
    values = _as_rgb_frame(frame, "wrist image")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        # A minimal shell can still run the contract smoke.  The actual
        # Cosmos-policy environment has torch and takes the exact native path.
        resized = _resize_rgb(values, (side, side))
        crop_size = int(side * 0.9**0.5)
        crop = (side - crop_size) // 2
        cropped = resized[crop : crop + crop_size, crop : crop + crop_size]
        return _resize_rgb(cropped, (side, side))

    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).float()
    target = (side, side)
    if tuple(tensor.shape[-2:]) != target:
        tensor = functional.interpolate(
            tensor, size=target, mode="bilinear", align_corners=False
        )
    crop_size = int(side * 0.9**0.5)
    crop = (side - crop_size) // 2
    tensor = tensor[..., crop : crop + crop_size, crop : crop + crop_size]
    tensor = functional.interpolate(
        tensor,
        size=target,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        tensor.round()
        .clamp(0, 255)
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def _preprocess_primary_cross_view(
    frame: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Match the unaugmented primary cross-view stream used by the wrist model."""

    values = _as_rgb_frame(frame, "primary cross-view image")
    if tuple(values.shape[:2]) == tuple(size):
        return values
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        return _resize_rgb(values, size)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).float()
    tensor = functional.interpolate(
        tensor,
        size=tuple(size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        tensor.round()
        .clamp(0, 255)
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def _runtime_to_cosmos_model(frame: np.ndarray) -> np.ndarray:
    """RLinf runtime rot180 -> Cosmos model flipud is one horizontal flip."""

    return np.ascontiguousarray(np.flip(np.asarray(frame, dtype=np.uint8), axis=-2))


def _cosmos_model_to_runtime(frame: np.ndarray) -> np.ndarray:
    return _runtime_to_cosmos_model(frame)


def _convert_gripper_actions(actions: np.ndarray, mode: str) -> np.ndarray:
    values = np.ascontiguousarray(actions, dtype=np.float32).copy()
    normalized = str(mode).lower()
    if normalized in {"libero_01_to_cosmos", "libero01", "qpos01"}:
        values[..., -1] = 2.0 * np.clip(values[..., -1], 0.0, 1.0) - 1.0
    elif normalized in {"already_cosmos", "minus_plus_one", "none"}:
        pass
    else:
        raise ValueError(
            "gripper_action_mode must be libero_01_to_cosmos or already_cosmos"
        )
    return values


def _sentinel_actions(
    batch_size: int, action_dim: int, gripper_action_mode: str
) -> np.ndarray:
    normalized = str(gripper_action_mode).lower()
    if normalized in {"libero_01_to_cosmos", "libero01", "qpos01"}:
        open_value = 0.0
    elif normalized in {"already_cosmos", "minus_plus_one", "none"}:
        open_value = -1.0
    else:
        raise ValueError(
            "gripper_action_mode must be libero_01_to_cosmos or already_cosmos"
        )
    actions = np.zeros((batch_size, 5, action_dim), dtype=np.float32)
    actions[:, :, -1] = open_value
    return actions


class CosmosWristAdapter:
    """Strict WoVR wrist adapter with injectable Cosmos client."""

    def __init__(
        self,
        config: CosmosWristAdapterConfig,
        *,
        client: TemporalCosmosClient | None = None,
    ) -> None:
        self.config = config
        if config.chunk_size != 8 or config.action_dim != 7:
            raise ValueError(
                "strict WoVR wrist currently requires chunk_size=8 and action_dim=7"
            )
        if config.cross_view_primary_history and not config.primary_reference:
            raise ValueError(
                "cross-view primary history requires primary_reference=True"
            )
        if config.future_denoising_steps <= 0:
            raise ValueError("future_denoising_steps must be positive")
        if not 0.0 <= float(config.guidance) <= 5.0:
            raise ValueError("guidance must be in [0,5]")
        self.client = client or self._connect_client(config)
        metadata = self.client.query_metadata()
        self.metadata = dict(metadata)
        if not bool(metadata.get("wovr_action_contract", False)):
            raise RuntimeError(
                "Cosmos wrist endpoint is not a strict WoVR action contract: "
                f"metadata={metadata}"
            )
        if int(metadata.get("chunk_size", -1)) != config.chunk_size:
            raise RuntimeError(
                "Cosmos wrist chunk size mismatch: "
                f"endpoint={metadata.get('chunk_size')}, expected={config.chunk_size}"
            )
        endpoint_action_dim = metadata.get("action_dim")
        if endpoint_action_dim is not None and int(endpoint_action_dim) != config.action_dim:
            raise RuntimeError(
                "Cosmos wrist action dimension mismatch: "
                f"endpoint={endpoint_action_dim}, expected={config.action_dim}"
            )
        endpoint_cross_view = bool(metadata.get("cross_view_primary_context", False))
        if endpoint_cross_view != config.cross_view_primary_history:
            raise RuntimeError(
                "Cosmos wrist cross-view mismatch: "
                f"endpoint={endpoint_cross_view}, requested={config.cross_view_primary_history}"
            )

    @staticmethod
    def _connect_client(config: CosmosWristAdapterConfig) -> TemporalCosmosClient:
        if config.worldpilot_root is None:
            raise ValueError("worldpilot_root is required when no client is injected")
        root = pathlib.Path(config.worldpilot_root).resolve()
        if not root.is_dir():
            raise ValueError(f"worldpilot_root does not exist: {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from cosmos_bridge.cosmos_client import CosmosClient

        return CosmosClient(host=config.host, port=config.port)

    def initial_state(
        self,
        primary_o0: Any,
        wrist_o0: Any,
        *,
        batch_size: int,
    ) -> CosmosWristState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        primary = _as_rgb_frame(primary_o0, "primary_o0")
        wrist = _as_rgb_frame(wrist_o0, "wrist_o0")
        primary_wrist_space = _runtime_to_cosmos_model(
            _preprocess_wrist_model_space(primary, self.config.image_size)
        )
        wrist_space = _runtime_to_cosmos_model(
            _preprocess_wrist_model_space(wrist, self.config.image_size)
        )
        if self.config.primary_reference:
            reference = primary_wrist_space
            memory = np.repeat(wrist_space[None], 4, axis=0)
        else:
            reference = wrist_space
            memory = np.repeat(wrist_space[None], 4, axis=0)
        reference_batch = np.repeat(reference[None], batch_size, axis=0)
        memory_batch = np.repeat(memory[None], batch_size, axis=0)

        cross_reference = None
        cross_memory = None
        if self.config.cross_view_primary_history:
            cross_reference = _runtime_to_cosmos_model(
                _preprocess_primary_cross_view(primary, self.config.image_size)
            )
            cross_reference = np.repeat(cross_reference[None], batch_size, axis=0)
            cross_memory = np.repeat(cross_reference[:, None], 4, axis=1)
        return CosmosWristState(
            reference_primary=np.ascontiguousarray(reference_batch),
            memory_primary=np.ascontiguousarray(memory_batch),
            condition_actions=_sentinel_actions(
                batch_size,
                self.config.action_dim,
                self.config.gripper_action_mode,
            ),
            cross_view_reference_primary=cross_reference,
            cross_view_memory_primary=cross_memory,
        )

    def query_chunk(
        self,
        state: CosmosWristState,
        actions: np.ndarray,
        *,
        languages: Sequence[str],
        batch_seed: int,
    ) -> np.ndarray:
        action_batch = np.ascontiguousarray(actions, dtype=np.float32)
        if action_batch.ndim != 3 or action_batch.shape[1:] != (
            self.config.chunk_size,
            self.config.action_dim,
        ):
            raise ValueError(
                "Cosmos wrist actions must have shape "
                f"[B,{self.config.chunk_size},{self.config.action_dim}], got "
                f"{action_batch.shape}"
            )
        if not np.isfinite(action_batch).all():
            raise ValueError("Cosmos wrist actions contain non-finite values")
        batch_size = int(action_batch.shape[0])
        if len(languages) != batch_size:
            raise ValueError(
                f"expected {batch_size} languages, got {len(languages)}"
            )
        if state.reference_primary.shape[0] != batch_size:
            raise ValueError(
                "Cosmos wrist state/action batch mismatch: "
                f"{state.reference_primary.shape[0]} vs {batch_size}"
            )
        expected_reference = (
            batch_size,
            self.config.image_size[0],
            self.config.image_size[1],
            3,
        )
        expected_memory = (batch_size, 4, *expected_reference[1:])
        if state.reference_primary.shape != expected_reference:
            raise ValueError(
                "Cosmos wrist reference shape mismatch: "
                f"{state.reference_primary.shape} vs {expected_reference}"
            )
        if state.memory_primary.shape != expected_memory:
            raise ValueError(
                "Cosmos wrist memory shape mismatch: "
                f"{state.memory_primary.shape} vs {expected_memory}"
            )
        if state.condition_actions.shape != (batch_size, 5, self.config.action_dim):
            raise ValueError(
                "Cosmos wrist context action shape mismatch: "
                f"{state.condition_actions.shape} vs "
                f"{(batch_size, 5, self.config.action_dim)}"
            )
        cosmos_actions = _convert_gripper_actions(
            action_batch, self.config.gripper_action_mode
        )
        history_flags = np.ones((batch_size, 5), dtype=bool)
        kwargs: dict[str, Any] = {
            "langs": list(languages),
            "actions_are_normalized": self.config.actions_are_normalized,
            "flip_images": False,
            "batch_seed": int(batch_seed),
            "future_denoising_steps": int(self.config.future_denoising_steps),
            "guidance": float(self.config.guidance),
            "history_images_are_model_space": history_flags,
            # The state buffer is kept in policy/source action space so the
            # rolling update remains auditable.  The Cosmos client boundary,
            # however, must receive the same converted gripper convention as
            # the future action chunk (native WanEnv does this conversion for
            # both arrays).
            "context_actions": _convert_gripper_actions(
                state.condition_actions, self.config.gripper_action_mode
            ),
        }
        if self.config.cross_view_primary_history:
            if (
                state.cross_view_reference_primary is None
                or state.cross_view_memory_primary is None
            ):
                raise RuntimeError(
                    "cross-view primary history is enabled but the state "
                    "buffers are not initialized"
                )
            if state.cross_view_reference_primary.shape != expected_reference:
                raise ValueError(
                    "Cosmos wrist cross-view reference shape mismatch: "
                    f"{state.cross_view_reference_primary.shape} vs "
                    f"{expected_reference}"
                )
            if state.cross_view_memory_primary.shape != expected_memory:
                raise ValueError(
                    "Cosmos wrist cross-view memory shape mismatch: "
                    f"{state.cross_view_memory_primary.shape} vs "
                    f"{expected_memory}"
                )
            kwargs.update(
                {
                    "cross_view_reference_primary": state.cross_view_reference_primary,
                    "cross_view_memory_primary": state.cross_view_memory_primary,
                    "cross_view_history_images_are_model_space": history_flags,
                }
            )
        response = self.client.query_temporal_future_batch(
            state.reference_primary,
            state.memory_primary,
            cosmos_actions,
            **kwargs,
        )
        if not isinstance(response, dict):
            raise RuntimeError(
                "Cosmos wrist response must be a mapping, got "
                f"{type(response)!r}"
            )
        if "future_primary_clip" not in response:
            raise RuntimeError("Cosmos wrist response misses future_primary_clip")
        clip_model = np.asarray(response["future_primary_clip"], dtype=np.uint8)
        expected = (
            batch_size,
            self.config.chunk_size,
            self.config.image_size[0],
            self.config.image_size[1],
            3,
        )
        if clip_model.shape != expected:
            raise RuntimeError(
                f"Cosmos wrist clip shape {clip_model.shape} != expected {expected}"
            )
        if self.config.primary_reference:
            state.memory_primary[:, 1:4] = clip_model[:, -3:]
            state.condition_actions[:, 2:5] = action_batch[:, -3:]
        else:
            state.memory_primary[...] = clip_model[:, -4:]
            state.condition_actions[:, 1:5] = action_batch[:, -4:]
        # Return runtime-space frames for video/debugging. Keep model-space
        # frames in state.memory_primary for the next request.
        return np.ascontiguousarray(_cosmos_model_to_runtime(clip_model))

    def update_primary_history(
        self, state: CosmosWristState, primary_clip_runtime: np.ndarray
    ) -> None:
        if not self.config.cross_view_primary_history:
            return
        clip = np.asarray(primary_clip_runtime, dtype=np.uint8)
        if clip.ndim != 5 or clip.shape[1] < 4 or clip.shape[-1] != 3:
            raise ValueError(
                "primary clip must have shape [B,T,H,W,3] with T>=4, got "
                f"{clip.shape}"
            )
        flattened = clip[:, -4:].reshape(-1, *clip.shape[2:])
        # Match WanEnv's update path: convert the runtime camera space before
        # the cross-view resize.  (The reset path has no temporal flattening
        # and follows its corresponding native initialization order.)
        flattened = _runtime_to_cosmos_model(flattened)
        resized = np.stack(
            [
                _preprocess_primary_cross_view(frame, self.config.image_size)
                for frame in flattened
            ],
            axis=0,
        ).reshape(
            clip.shape[0],
            4,
            self.config.image_size[0],
            self.config.image_size[1],
            3,
        )
        if state.cross_view_memory_primary is None:
            raise RuntimeError(
                "cross-view primary history is enabled but the state has no "
                "cross_view_memory_primary buffer"
            )
        state.cross_view_memory_primary[...] = resized

    def close(self) -> None:
        self.client.close()


__all__ = [
    "CosmosWristAdapter",
    "CosmosWristAdapterConfig",
    "CosmosWristState",
    "TemporalCosmosClient",
]
