#!/usr/bin/env python3
"""CPU-only contract smoke for the online WorldSample bridge.

This test deliberately does not import LIBERO, Ray, OpenPI, Wan, or Cosmos.
It verifies the ordering/shape layer that must pass before any GPU process is
started.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.world_model_grpo.online_bridge import (
    CounterfactualTransition,
    CounterfactualVariant,
    OnlineSyntheticQueue,
    RealPPOTransition,
    RealPPOTrajectory,
    SyntheticRequest,
    SyntheticResult,
    combine_paired_loss,
    renoise_nft_x0,
)


def main() -> None:
    x0 = torch.zeros(1, 16, 32)
    xcur, step_indices, timesteps = renoise_nft_x0(
        x0,
        num_steps=10,
        num_variants=8,
        generator=torch.Generator().manual_seed(7),
    )
    assert xcur.shape == (8, 16, 32)
    assert step_indices.shape == (8,)
    assert timesteps.shape == (8,)

    forward_inputs = {
        "chains": torch.zeros(1, 2, 16, 32),
        "denoise_inds": torch.zeros(1, 2, dtype=torch.long),
        "model_action": torch.zeros(1, 16 * 32),
        "action": torch.zeros(1, 8 * 7),
        "nft_x0": x0,
    }
    transition = RealPPOTransition(
        observation={"main_images": torch.zeros(1, 2, 2, 3)},
        forward_inputs=forward_inputs,
        action=torch.zeros(1, 8, 7),
        prev_logprobs=torch.zeros(1, 8, 8),
        prev_values=torch.zeros(1, 1),
        reward=0.0,
        done=False,
    )
    source = RealPPOTrajectory(
        trajectory_id="smoke-trajectory",
        source_step=0,
        weight_version=3,
        initial_observation=transition.observation,
        transitions=(transition,),
    )
    source.validate(
        action_chunk=8,
        action_dim=7,
        model_action_horizon=16,
        model_action_dim=32,
    )
    no_nft_inputs = {
        key: value for key, value in forward_inputs.items() if key != "nft_x0"
    }
    RealPPOTransition(
        observation=transition.observation,
        forward_inputs=no_nft_inputs,
        action=transition.action,
        prev_logprobs=transition.prev_logprobs,
        prev_values=transition.prev_values,
        reward=transition.reward,
        done=transition.done,
    ).validate(require_nft_state=False)

    branch_transition = CounterfactualTransition(
        transition_index=0,
        observation=transition.observation,
        nft_xcur=xcur[:1],
        x0_target=x0,
        nft_step_index=int(step_indices[0]),
        timestep=float(timesteps[0]),
        actions=torch.zeros(1, 8, 7),
        forward_inputs={
            "nft_xcur": xcur[:1],
            "nft_x0_target": x0,
            "nft_step_index": step_indices[:1],
            "nft_v_old": torch.zeros(1, 8, 32),
            "nft_timestep": timesteps[:1],
            "nft_noise_level": torch.full((1,), 0.3),
        },
    )
    branch = CounterfactualVariant(
        variant_index=0,
        transitions=(branch_transition,),
        action_sequence=torch.zeros(1, 8, 7),
    )
    branch.validate(
        action_chunk=8,
        action_dim=7,
        model_action_horizon=16,
        model_action_dim=32,
    )
    try:
        CounterfactualVariant(
            variant_index=0,
            transitions=(dataclasses.replace(
                branch_transition, x0_target=torch.ones_like(x0)
            ),),
            action_sequence=torch.zeros(1, 8, 7),
        ).validate(
            action_chunk=8,
            action_dim=7,
            model_action_horizon=16,
            model_action_dim=32,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("counterfactual target mismatch was not rejected")
    request = SyntheticRequest(
        request_id="smoke-request",
        trajectory_id=source.trajectory_id,
        source_step=source.source_step,
        weight_version=source.weight_version,
        initial_observation=source.initial_observation,
        variants=(
            branch,
        ),
    )
    request.validate(
        expected_variants=1,
        action_chunk=8,
        action_dim=7,
        model_action_horizon=16,
        model_action_dim=32,
    )

    queue = OnlineSyntheticQueue(max_pending=2, min_lag=1)
    try:
        queue.submit(
            request,
            lambda item: SyntheticResult(
                request_id=item.request_id,
                trajectory_id=item.trajectory_id,
                source_step=item.source_step,
                weight_version=item.weight_version,
                rewards=(0.25,),
                wm_name="cpu-fake",
            ),
        )
        assert queue.poll(0) == []
        result = queue.poll(1)
        assert len(result) == 1 and result[0].is_success
    finally:
        queue.close()

    # A completed future with the wrong provenance must never be paired with
    # the actor's request.  The queue converts it to a failed result carrying
    # the request's own identifiers so the caller cannot consume wrong data.
    bad_request = dataclasses.replace(request, request_id="bad-request")
    bad_queue = OnlineSyntheticQueue(max_pending=1, min_lag=1)
    try:
        bad_queue.submit(
            bad_request,
            lambda item: SyntheticResult(
                request_id="wrong-request",
                trajectory_id=item.trajectory_id,
                source_step=item.source_step,
                weight_version=item.weight_version,
                rewards=(0.25,),
                wm_name="cpu-fake",
            ),
        )
        bad_result = bad_queue.poll(1)
        assert len(bad_result) == 1
        assert not bad_result[0].is_success
        assert bad_result[0].request_id == bad_request.request_id
    finally:
        bad_queue.close()

    ppo_loss = torch.tensor(2.0, requires_grad=True)
    nft_loss = torch.tensor(3.0, requires_grad=True)
    combine_paired_loss(ppo_loss, nft_loss, 0.5).backward()
    assert ppo_loss.grad.item() == 1.0
    assert nft_loss.grad.item() == 0.5
    print("online_bridge_smoke: PASS")


if __name__ == "__main__":
    main()
