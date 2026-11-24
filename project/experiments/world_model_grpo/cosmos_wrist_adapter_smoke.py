#!/usr/bin/env python3
"""CPU contract smoke for the strict Cosmos-WoVR wrist adapter.

This deliberately uses a fake client.  It verifies the boundary that can be
checked without a resident Cosmos server: model-space camera transforms,
``[B,8,7]`` actions, converted context actions, cross-view primary history,
rolling memory, and metadata rejection.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Running this file by path places ``experiments/world_model_grpo`` rather than
# the project root on sys.path; make the smoke executable in that mode too.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.world_model_grpo.cosmos_wrist_adapter import (
    CosmosWristAdapter,
    CosmosWristAdapterConfig,
)


class FakeCosmosClient:
    def __init__(self, *, cross_view: bool = True) -> None:
        self.cross_view = cross_view
        self.calls: list[tuple[tuple[np.ndarray, ...], dict[str, object]]] = []
        self.closed = False

    def query_metadata(self) -> dict[str, object]:
        return {
            "wovr_action_contract": True,
            "chunk_size": 8,
            "action_dim": 7,
            "cross_view_primary_context": self.cross_view,
        }

    def query_temporal_future_batch(self, *args: object, **kwargs: object):
        arrays = tuple(np.asarray(value).copy() for value in args)
        self.calls.append((arrays, dict(kwargs)))
        batch_size = int(arrays[0].shape[0])
        # Distinct temporal values make memory rolling observable while
        # keeping the fake deterministic and independent of model weights.
        clip = np.zeros((batch_size, 8, 224, 224, 3), dtype=np.uint8)
        for frame_index in range(8):
            clip[:, frame_index] = frame_index + 10
        return {"future_primary_clip": clip}

    def close(self) -> None:
        self.closed = True


def main() -> None:
    batch_size = 8
    primary = np.zeros((256, 256, 3), dtype=np.uint8)
    primary[:, :128] = [11, 22, 33]
    wrist = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist[:, :128] = [44, 55, 66]
    client = FakeCosmosClient(cross_view=True)
    adapter = CosmosWristAdapter(
        CosmosWristAdapterConfig(
            primary_reference=True,
            cross_view_primary_history=True,
            gripper_action_mode="libero_01_to_cosmos",
        ),
        client=client,
    )
    state = adapter.initial_state(primary, wrist, batch_size=batch_size)
    assert state.reference_primary.shape == (batch_size, 224, 224, 3)
    assert state.memory_primary.shape == (batch_size, 4, 224, 224, 3)
    assert state.condition_actions.shape == (batch_size, 5, 7)
    assert np.all(state.condition_actions[..., -1] == 0.0)
    # The source primary has a red left half; the single runtime->Cosmos
    # horizontal flip must place it on the model-space right half.
    assert state.reference_primary[:, :, :96].mean() < 10.0
    assert state.reference_primary[:, :, -96:].mean() > 5.0
    assert state.cross_view_reference_primary is not None
    assert state.cross_view_memory_primary is not None

    actions = np.zeros((batch_size, 8, 7), dtype=np.float32)
    actions[..., -1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    wrist_clip = adapter.query_chunk(
        state,
        actions,
        languages=["open the drawer"] * batch_size,
        batch_seed=123,
    )
    assert wrist_clip.shape == (batch_size, 8, 224, 224, 3)
    assert len(client.calls) == 1
    positional, kwargs = client.calls[0]
    assert positional[0].shape == (batch_size, 224, 224, 3)
    assert positional[1].shape == (batch_size, 4, 224, 224, 3)
    np.testing.assert_allclose(positional[2][..., -1], 2.0 * actions[..., -1] - 1.0)
    context = np.asarray(kwargs["context_actions"])
    assert context.shape == (batch_size, 5, 7)
    # The first chunk has only the five reset sentinels.  The actions just
    # queried are stored in the rolling state and become context on chunk 2.
    assert np.all(context[:, :, -1] == -1.0)
    assert np.asarray(kwargs["history_images_are_model_space"]).shape == (
        batch_size,
        5,
    )
    assert np.asarray(kwargs["cross_view_memory_primary"]).shape == (
        batch_size,
        4,
        224,
        224,
        3,
    )
    actions_second = actions.copy()
    actions_second[..., -1] = 0.25
    adapter.query_chunk(
        state,
        actions_second,
        languages=["open the drawer"] * batch_size,
        batch_seed=124,
    )
    assert len(client.calls) == 2
    _, second_kwargs = client.calls[1]
    second_context = np.asarray(second_kwargs["context_actions"])
    assert np.all(second_context[:, :2, -1] == -1.0)
    np.testing.assert_allclose(
        second_context[:, 2:, -1], 2.0 * actions[:, -3:, -1] - 1.0
    )
    # Primary Wan output is runtime-space and becomes the next Cosmos
    # cross-view memory only after the wrist query for this chunk.
    primary_clip = np.zeros((batch_size, 8, 256, 256, 3), dtype=np.uint8)
    primary_clip[:, -4:, :, :128] = [77, 88, 99]
    adapter.update_primary_history(state, primary_clip)
    assert state.cross_view_memory_primary.shape == (
        batch_size,
        4,
        224,
        224,
        3,
    )
    assert state.cross_view_memory_primary[:, :, :, :96].mean() < 20.0
    assert state.cross_view_memory_primary[:, :, :, -96:].mean() > 20.0

    adapter.close()
    assert client.closed

    try:
        CosmosWristAdapter(
            CosmosWristAdapterConfig(
                primary_reference=True,
                cross_view_primary_history=True,
            ),
            client=FakeCosmosClient(cross_view=False),
        )
    except RuntimeError as exc:
        assert "cross-view mismatch" in str(exc)
    else:
        raise AssertionError("metadata cross-view mismatch was not rejected")

    print("cosmos_wrist_adapter_smoke: PASS")


if __name__ == "__main__":
    main()
