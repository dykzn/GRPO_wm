"""Small, framework-independent pieces of the online WorldSample bridge.

The distributed runner is intentionally not imported here.  This module owns
the data contract that must remain true regardless of whether the world model
is executed in a Ray worker, a local process, or a standalone smoke:

    PPO real rollout -> retain PPO/NFT state -> 1xK re-noise -> WM score
    -> consume the result at a later actor update.

Keeping this contract separate makes it possible to test the ordering and
shape invariants on CPU before starting a world-model process.  The serial
reference implements Wan primary scoring and can optionally attach the strict
Cosmos-WoVR wrist stream to the same request/result contract. In particular,
the queue never creates a new rollout from a checkpoint; every synthetic
request carries the source real trajectory id and its original observation
anchor.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import threading
from collections import Counter
from typing import Any, Callable, Mapping

import torch


def detach_tree_to_cpu(value: Any) -> Any:
    """Detach a nested tensor tree without mutating the caller's object."""

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_tree_to_cpu(item) for item in value)
    return value


def _as_batch_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim == 0:
        raise ValueError(f"{name} must have a batch dimension, got scalar")
    return value


@dataclasses.dataclass(frozen=True)
class RealPPOTransition:
    """One chunk transition retained from the real simulator rollout.

    ``forward_inputs`` is intentionally the exact dictionary produced by
    ``predict_action_batch`` after CPU detachment.  It therefore contains the
    PPO observations/chains/logprob context and, when the online collection
    flag is enabled, ``nft_x0`` as well.
    """

    observation: Any
    forward_inputs: Mapping[str, Any]
    action: torch.Tensor
    prev_logprobs: torch.Tensor
    prev_values: torch.Tensor
    reward: torch.Tensor | float
    done: torch.Tensor | bool

    def validate(
        self,
        *,
        require_nft_state: bool = True,
        action_chunk: int | None = None,
        action_dim: int | None = None,
        model_action_horizon: int | None = None,
        model_action_dim: int | None = None,
    ) -> None:
        if not isinstance(self.forward_inputs, Mapping):
            raise TypeError("forward_inputs must be a mapping")
        required = {"chains", "denoise_inds", "model_action", "action"}
        missing = sorted(required.difference(self.forward_inputs))
        if missing:
            raise ValueError(f"real PPO transition misses keys: {missing}")
        if require_nft_state and "nft_x0" not in self.forward_inputs:
            raise ValueError(
                "online synthetic branch requires forward_inputs['nft_x0']; "
                "enable collect_online_nft_state on the PPO rollout model"
            )

        action = _as_batch_tensor(self.action, "action")
        if action_chunk is not None or action_dim is not None:
            if action.ndim != 3:
                raise ValueError(
                    "real action must have shape [B, action_chunk, action_dim], "
                    f"got {tuple(action.shape)}"
                )
            if action_chunk is not None and action.shape[1] != action_chunk:
                raise ValueError(
                    f"expected real action chunk {action_chunk}, got {action.shape[1]}"
                )
            if action_dim is not None and action.shape[2] != action_dim:
                raise ValueError(
                    f"expected real action dim {action_dim}, got {action.shape[2]}"
                )
        logprobs = _as_batch_tensor(self.prev_logprobs, "prev_logprobs")
        values = _as_batch_tensor(self.prev_values, "prev_values")
        batch_sizes = {
            "action": action.shape[0],
            "prev_logprobs": logprobs.shape[0],
            "prev_values": values.shape[0],
        }
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(f"real PPO fields disagree on batch size: {batch_sizes}")
        if not torch.isfinite(action.float()).all():
            raise ValueError("real action contains non-finite values")
        if not torch.isfinite(logprobs.float()).all():
            raise ValueError("real PPO logprob contains non-finite values")
        if not torch.isfinite(values.float()).all():
            raise ValueError("real PPO value contains non-finite values")

        model_action = _as_batch_tensor(
            self.forward_inputs["model_action"], "forward_inputs.model_action"
        )
        if model_action.shape[0] != action.shape[0]:
            raise ValueError(
                "model_action and PPO fields have different batch sizes: "
                f"{model_action.shape[0]} vs {action.shape[0]}"
            )
        if model_action_horizon is not None or model_action_dim is not None:
            if model_action.ndim != 2:
                raise ValueError(
                    "forward_inputs.model_action must be flattened [B, H*D] "
                    f"for horizon validation, got {tuple(model_action.shape)}"
                )
            if model_action_dim is not None and model_action.shape[1] % model_action_dim:
                raise ValueError(
                    "forward_inputs.model_action flattened size is not divisible "
                    f"by model action dim {model_action_dim}: {model_action.shape[1]}"
                )
            inferred_horizon = (
                model_action.shape[1] // model_action_dim
                if model_action_dim is not None
                else model_action.shape[1]
            )
            if (
                model_action_horizon is not None
                and inferred_horizon != model_action_horizon
            ):
                raise ValueError(
                    "forward_inputs.model_action horizon disagrees with the "
                    f"configured horizon {model_action_horizon}: {inferred_horizon}"
                )
        if require_nft_state:
            x0 = _as_batch_tensor(self.forward_inputs["nft_x0"], "nft_x0")
            if x0.shape[0] != action.shape[0]:
                raise ValueError(
                    "nft_x0 and PPO fields have different batch sizes: "
                    f"{x0.shape[0]} vs {action.shape[0]}"
                )
            if model_action_horizon is not None or model_action_dim is not None:
                if x0.ndim != 3:
                    raise ValueError(
                        "nft_x0 must have shape [B,H,D], got "
                        f"{tuple(x0.shape)}"
                    )
                if (
                    model_action_horizon is not None
                    and x0.shape[1] != model_action_horizon
                ):
                    raise ValueError(
                        "nft_x0 horizon disagrees with model_action: "
                        f"{x0.shape[1]} vs {model_action_horizon}"
                    )
                if model_action_dim is not None and x0.shape[2] != model_action_dim:
                    raise ValueError(
                        "nft_x0 model dim disagrees with model_action: "
                        f"{x0.shape[2]} vs {model_action_dim}"
                    )
            if not torch.isfinite(x0.float()).all():
                raise ValueError("nft_x0 contains non-finite values")


@dataclasses.dataclass(frozen=True)
class RealPPOTrajectory:
    """A source trajectory and its exact online provenance."""

    trajectory_id: str
    source_step: int
    weight_version: int
    initial_observation: Any
    transitions: tuple[RealPPOTransition, ...]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def validate(
        self,
        *,
        require_nft_state: bool = True,
        action_chunk: int | None = None,
        action_dim: int | None = None,
        model_action_horizon: int | None = None,
        model_action_dim: int | None = None,
    ) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        if self.source_step < 0:
            raise ValueError(f"source_step must be non-negative, got {self.source_step}")
        if self.initial_observation is None:
            raise ValueError("the synthetic branch must retain the real o0 anchor")
        if not self.transitions:
            raise ValueError("cannot synthesize from an empty real trajectory")
        for transition in self.transitions:
            transition.validate(
                require_nft_state=require_nft_state,
                action_chunk=action_chunk,
                action_dim=action_dim,
                model_action_horizon=model_action_horizon,
                model_action_dim=model_action_dim,
            )


@dataclasses.dataclass(frozen=True)
class CounterfactualTransition:
    """One re-noised state/action pair inside a synthetic branch."""

    transition_index: int
    observation: Any
    nft_xcur: torch.Tensor
    # The clean action state actually denoised and sent to the WM.  This is
    # distinct from the source real transition's nft_x0: the synthetic reward
    # must supervise this counterfactual target.
    x0_target: torch.Tensor
    nft_step_index: int
    timestep: float
    actions: torch.Tensor
    forward_inputs: Mapping[str, Any]

    def validate(
        self,
        *,
        action_chunk: int | None = None,
        action_dim: int | None = None,
        model_action_horizon: int | None = None,
        model_action_dim: int | None = None,
    ) -> None:
        if self.transition_index < 0:
            raise ValueError("transition_index must be non-negative")
        for name, value in (
            ("nft_xcur", self.nft_xcur),
            ("x0_target", self.x0_target),
            ("actions", self.actions),
        ):
            _as_batch_tensor(value, name)
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"{name} contains non-finite values")
        synthetic_batch_sizes = {
            "nft_xcur": self.nft_xcur.shape[0],
            "x0_target": self.x0_target.shape[0],
            "actions": self.actions.shape[0],
        }
        if len(set(synthetic_batch_sizes.values())) != 1:
            raise ValueError(
                "synthetic fields disagree on batch size: "
                f"{synthetic_batch_sizes}"
            )
        if model_action_horizon is not None or model_action_dim is not None:
            for name, value in (
                ("nft_xcur", self.nft_xcur),
                ("x0_target", self.x0_target),
            ):
                if value.ndim != 3:
                    raise ValueError(
                        f"{name} must have shape [B,H,D], got {tuple(value.shape)}"
                    )
                if (
                    model_action_horizon is not None
                    and value.shape[1] != model_action_horizon
                ):
                    raise ValueError(
                        f"{name} horizon {value.shape[1]} != "
                        f"configured {model_action_horizon}"
                    )
                if model_action_dim is not None and value.shape[2] != model_action_dim:
                    raise ValueError(
                        f"{name} model dim {value.shape[2]} != "
                        f"configured {model_action_dim}"
                    )
        if action_chunk is not None or action_dim is not None:
            if self.actions.ndim != 3:
                raise ValueError(
                    "synthetic actions must have shape [B, action_chunk, action_dim], "
                    f"got {tuple(self.actions.shape)}"
                )
            if action_chunk is not None and self.actions.shape[1] != action_chunk:
                raise ValueError(
                    f"expected synthetic action chunk {action_chunk}, got {self.actions.shape[1]}"
                )
            if action_dim is not None and self.actions.shape[2] != action_dim:
                raise ValueError(
                    f"expected synthetic action dim {action_dim}, got {self.actions.shape[2]}"
                )
        if not 0.0 <= float(self.timestep) <= 1.0:
            raise ValueError(f"timestep must be in [0, 1], got {self.timestep}")
        if self.nft_step_index < 0:
            raise ValueError("nft_step_index must be non-negative")
        if "nft_xcur" not in self.forward_inputs:
            raise ValueError("synthetic forward_inputs must carry nft_xcur")
        if "nft_x0_target" not in self.forward_inputs:
            raise ValueError(
                "synthetic forward_inputs must carry the counterfactual x0 target"
            )
        if "nft_step_index" not in self.forward_inputs:
            raise ValueError("synthetic forward_inputs must carry nft_step_index")
        for key, expected in (
            ("nft_xcur", self.nft_xcur),
            ("nft_x0_target", self.x0_target),
        ):
            value = _as_batch_tensor(self.forward_inputs[key], key)
            if value.shape != expected.shape or not torch.allclose(
                value.float().cpu(), expected.float().cpu()
            ):
                raise ValueError(
                    f"{key} disagrees with the explicit counterfactual transition field"
                )
        for key in ("nft_v_old", "nft_timestep", "nft_noise_level"):
            if key not in self.forward_inputs:
                raise ValueError(
                    f"synthetic forward_inputs must carry {key} for lagged NFT"
                )
            value = _as_batch_tensor(self.forward_inputs[key], key)
            if value.shape[0] != self.nft_xcur.shape[0]:
                raise ValueError(
                    f"{key} and nft_xcur disagree on batch size: "
                    f"{value.shape[0]} vs {self.nft_xcur.shape[0]}"
                )
            if key == "nft_v_old" and model_action_dim is not None:
                if value.ndim != 3 or value.shape[1] != (action_chunk or value.shape[1]):
                    raise ValueError(
                        "nft_v_old must have shape [B, action_chunk, model_dim], "
                        f"got {tuple(value.shape)}"
                    )
                if value.shape[2] != model_action_dim:
                    raise ValueError(
                        f"nft_v_old model dim {value.shape[2]} != "
                        f"configured {model_action_dim}"
                    )
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"{key} contains non-finite values")
        step_indices = self.forward_inputs["nft_step_index"].reshape(-1)
        if step_indices.numel() != self.nft_xcur.shape[0] or not torch.all(
            step_indices.long()
            == torch.full_like(step_indices.long(), self.nft_step_index)
        ):
            raise ValueError(
                "forward_inputs.nft_step_index disagrees with nft_step_index"
            )
        timestep_values = self.forward_inputs["nft_timestep"].reshape(-1).float()
        if not torch.allclose(
            timestep_values,
            torch.full_like(timestep_values, float(self.timestep)),
        ):
            raise ValueError("forward_inputs.nft_timestep disagrees with timestep")


@dataclasses.dataclass(frozen=True)
class CounterfactualVariant:
    """One full-trajectory branch built from 1xK re-noised transitions."""

    variant_index: int
    transitions: tuple[CounterfactualTransition, ...]
    action_sequence: torch.Tensor

    def validate(
        self,
        *,
        action_chunk: int | None = None,
        action_dim: int | None = None,
        model_action_horizon: int | None = None,
        model_action_dim: int | None = None,
    ) -> None:
        if self.variant_index < 0:
            raise ValueError("variant_index must be non-negative")
        if not self.transitions:
            raise ValueError("a synthetic branch must contain one transition")
        for transition in self.transitions:
            transition.validate(
                action_chunk=action_chunk,
                action_dim=action_dim,
                model_action_horizon=model_action_horizon,
                model_action_dim=model_action_dim,
            )
        _as_batch_tensor(self.action_sequence, "action_sequence")
        if self.action_sequence.ndim != 3:
            raise ValueError(
                "action_sequence must have shape [num_decisions, action_chunk, action_dim], "
                f"got {tuple(self.action_sequence.shape)}"
            )
        if self.action_sequence.shape[0] != len(self.transitions):
            raise ValueError(
                "action_sequence and transition count disagree: "
                f"{self.action_sequence.shape[0]} vs {len(self.transitions)}"
            )
        if action_chunk is not None and self.action_sequence.shape[1] != action_chunk:
            raise ValueError(
                f"expected synthetic action chunk {action_chunk}, got {self.action_sequence.shape[1]}"
            )
        if action_dim is not None and self.action_sequence.shape[2] != action_dim:
            raise ValueError(
                f"expected synthetic action dim {action_dim}, got {self.action_sequence.shape[2]}"
            )
        if not torch.isfinite(self.action_sequence.float()).all():
            raise ValueError("action_sequence contains non-finite values")
        transition_actions = torch.cat(
            [transition.actions for transition in self.transitions], dim=0
        )
        if transition_actions.shape != self.action_sequence.shape or not torch.allclose(
            transition_actions.float().cpu(), self.action_sequence.float().cpu()
        ):
            raise ValueError(
                "action_sequence disagrees with the per-transition synthetic actions"
            )


@dataclasses.dataclass(frozen=True)
class SyntheticRequest:
    """World-model work submitted directly from a real PPO trajectory."""

    request_id: str
    trajectory_id: str
    source_step: int
    weight_version: int
    initial_observation: Any
    variants: tuple[CounterfactualVariant, ...]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def validate(
        self,
        *,
        expected_variants: int | None = None,
        action_chunk: int | None = None,
        action_dim: int | None = None,
        model_action_horizon: int | None = None,
        model_action_dim: int | None = None,
    ) -> None:
        if not self.request_id or not self.trajectory_id:
            raise ValueError("request_id and trajectory_id must be non-empty")
        if self.source_step < 0:
            raise ValueError("source_step must be non-negative")
        if self.weight_version < 0:
            raise ValueError("weight_version must be non-negative")
        if self.initial_observation is None:
            raise ValueError("SyntheticRequest lost the source real o0")
        if not self.variants:
            raise ValueError("SyntheticRequest must contain at least one variant")
        if expected_variants is not None and len(self.variants) != expected_variants:
            raise ValueError(
                f"expected {expected_variants} variants, got {len(self.variants)}"
            )
        variant_indices = [variant.variant_index for variant in self.variants]
        if len(set(variant_indices)) != len(variant_indices):
            raise ValueError(
                "synthetic variants contain duplicate variant indices: "
                f"{variant_indices}"
            )
        if sorted(variant_indices) != list(range(len(self.variants))):
            raise ValueError(
                "synthetic variant indices must be exactly "
                f"0..{len(self.variants) - 1}, got {variant_indices}"
            )
        decision_counts: set[int] = set()
        reference_transition_indices: tuple[int, ...] | None = None
        for variant in self.variants:
            variant.validate(
                action_chunk=action_chunk,
                action_dim=action_dim,
                model_action_horizon=model_action_horizon,
                model_action_dim=model_action_dim,
            )
            decision_counts.add(len(variant.transitions))
            transition_indices = tuple(
                transition.transition_index for transition in variant.transitions
            )
            expected_indices = tuple(range(len(variant.transitions)))
            if transition_indices != expected_indices:
                raise ValueError(
                    "synthetic transition indices must be contiguous from zero, "
                    f"got {transition_indices}"
                )
            if reference_transition_indices is None:
                reference_transition_indices = transition_indices
            elif transition_indices != reference_transition_indices:
                raise ValueError(
                    "synthetic variants do not share the same transition indices: "
                    f"{reference_transition_indices} vs {transition_indices}"
                )
        if len(decision_counts) > 1:
            raise ValueError(
                "synthetic variants have different decision counts: "
                f"{sorted(decision_counts)}"
            )


@dataclasses.dataclass(frozen=True)
class SyntheticResult:
    """World-model result returned to the actor-side lagged consumer."""

    request_id: str
    trajectory_id: str
    source_step: int
    weight_version: int
    rewards: tuple[float, ...]
    wm_name: str
    status: str = "completed"
    error: str | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "completed" and self.error is None

    def validate(self, *, expected_variants: int | None = None) -> None:
        if not self.request_id or not self.trajectory_id:
            raise ValueError("synthetic result ids must be non-empty")
        if self.source_step < 0:
            raise ValueError("synthetic result source_step must be non-negative")
        if self.weight_version < 0:
            raise ValueError("synthetic result weight_version must be non-negative")
        if (
            self.is_success
            and expected_variants is not None
            and len(self.rewards) != expected_variants
        ):
            raise ValueError(
                f"expected {expected_variants} rewards, got {len(self.rewards)}"
            )
        if self.is_success:
            tensor = torch.as_tensor(self.rewards, dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError("world-model rewards contain non-finite values")


def _validate_result_provenance(
    request: SyntheticRequest, result: SyntheticResult
) -> None:
    """Reject a result that cannot be paired with its submitted request."""

    expected = {
        "request_id": request.request_id,
        "trajectory_id": request.trajectory_id,
        "source_step": request.source_step,
        "weight_version": request.weight_version,
    }
    actual = {
        "request_id": result.request_id,
        "trajectory_id": result.trajectory_id,
        "source_step": result.source_step,
        "weight_version": result.weight_version,
    }
    mismatches = {
        key: (expected[key], actual[key])
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(
            "world-model result provenance does not match request: "
            f"{mismatches}"
        )


def renoise_nft_x0(
    nft_x0: torch.Tensor,
    *,
    num_steps: int,
    num_variants: int = 8,
    generator: torch.Generator | None = None,
    step_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the requested 1xK re-noised states from one real ``nft_x0``.

    Returns ``(xcur, step_indices, timesteps)``.  The first dimension is the
    variant dimension; all remaining dimensions are copied from ``nft_x0``.
    The random noise is independent per variant and is never used to create a
    new simulator observation.
    """

    x0 = _as_batch_tensor(nft_x0, "nft_x0")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if num_variants <= 0:
        raise ValueError(f"num_variants must be positive, got {num_variants}")
    if x0.shape[0] != 1:
        raise ValueError(
            "renoise_nft_x0 expects one source transition at a time, "
            f"got batch shape {tuple(x0.shape)}"
        )

    device = x0.device
    dtype = x0.dtype
    if step_indices is None:
        step_indices = torch.randint(
            0, num_steps, (num_variants,), device=device, generator=generator
        )
    else:
        step_indices = step_indices.to(device=device, dtype=torch.long).reshape(-1)
        if step_indices.numel() != num_variants:
            raise ValueError(
                f"step_indices has {step_indices.numel()} entries, "
                f"expected {num_variants}"
            )
        if bool(((step_indices < 0) | (step_indices >= num_steps)).any()):
            raise ValueError("step_indices contains an out-of-range value")

    schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    timesteps = schedule[step_indices]
    noise = torch.randn(
        (num_variants, *x0.shape[1:]),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    x0_variants = x0.expand(num_variants, *x0.shape[1:])
    xcur = (1.0 - timesteps.reshape(-1, *([1] * (x0.ndim - 1)))) * x0_variants
    xcur = xcur + timesteps.reshape(-1, *([1] * (x0.ndim - 1))) * noise
    return xcur, step_indices, timesteps


def combine_paired_loss(
    ppo_loss: torch.Tensor, nft_loss: torch.Tensor, rho: float
) -> torch.Tensor:
    """Return ``L_PPO + rho * L_NFT`` with explicit finite-value checks."""

    if not torch.is_tensor(ppo_loss) or not torch.is_tensor(nft_loss):
        raise TypeError("ppo_loss and nft_loss must be tensors")
    if not torch.isfinite(ppo_loss).all() or not torch.isfinite(nft_loss).all():
        raise FloatingPointError("paired objective received a non-finite loss")
    if not torch.isfinite(torch.as_tensor(rho, dtype=ppo_loss.dtype)):
        raise ValueError(f"rho must be finite, got {rho}")
    return ppo_loss + float(rho) * nft_loss


class OnlineSyntheticQueue:
    """Bounded asynchronous queue with explicit one-update lag semantics.

    ``submit`` receives a request built from a real rollout and a callable
    world-model scorer.  ``poll(consumer_step)`` only returns completed work
    whose source step is at most ``consumer_step - min_lag``.  Thus the common
    setting ``min_lag=1`` enforces the intended ``t -> t+1`` consumption while
    allowing the WM to run in the background.

    The queue is deliberately agnostic to Ray.  A Ray worker can be wrapped by
    passing a blocking remote-result getter as ``scorer``; a local smoke can
    pass an ordinary function.  Exceptions become explicit failed results so
    the actor can skip only the synthetic term and continue PPO.
    """

    def __init__(
        self,
        *,
        max_pending: int = 2,
        min_lag: int = 1,
        max_workers: int = 1,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if min_lag < 0:
            raise ValueError("min_lag must be non-negative")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_pending = int(max_pending)
        self.min_lag = int(min_lag)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="worldsample",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, tuple[SyntheticRequest, concurrent.futures.Future]] = {}
        self._seen_results: set[str] = set()
        self._metrics: Counter[str] = Counter()

    def submit(
        self,
        request: SyntheticRequest,
        scorer: Callable[[SyntheticRequest], SyntheticResult],
    ) -> str:
        request.validate()
        with self._lock:
            if request.request_id in self._futures or request.request_id in self._seen_results:
                raise ValueError(f"duplicate online synthetic request: {request.request_id}")
            if len(self._futures) >= self.max_pending:
                self._metrics["submit_rejected_capacity"] += 1
                raise RuntimeError(
                    "online synthetic queue is full; increase max_pending or "
                    "skip synthetic consumption for this update"
                )
            future = self._executor.submit(scorer, request)
            self._futures[request.request_id] = (request, future)
            self._metrics["submitted"] += 1
        return request.request_id

    def poll(
        self, consumer_step: int, *, exact_lag: bool = True
    ) -> list[SyntheticResult]:
        """Return completed results in source-step order.

        The online method uses ``exact_lag=True``: a result from source t is
        consumable only at t+``min_lag``.  A late result is dropped once it is
        complete instead of silently turning the method into variable-lag
        replay.  Callers that explicitly want variable-lag replay may pass
        ``exact_lag=False``.
        """

        if consumer_step < 0:
            raise ValueError("consumer_step must be non-negative")
        eligible: list[tuple[str, SyntheticRequest, concurrent.futures.Future]] = []
        stale: list[tuple[str, SyntheticRequest, concurrent.futures.Future]] = []
        with self._lock:
            for request_id, (request, future) in self._futures.items():
                due_step = request.source_step + self.min_lag
                if exact_lag and due_step < consumer_step and future.done():
                    stale.append((request_id, request, future))
                elif (
                    (exact_lag and due_step == consumer_step)
                    or (not exact_lag and due_step <= consumer_step)
                ) and future.done():
                    eligible.append((request_id, request, future))
        eligible.sort(key=lambda item: (item[1].source_step, item[0]))

        for request_id, _request, future in stale:
            # The future is complete, so removing it cannot leave a running
            # world-model call behind. Its result is intentionally not
            # returned to the actor because the exact consumption slot passed.
            future.cancel()
            with self._lock:
                self._futures.pop(request_id, None)
                self._seen_results.add(request_id)
                self._metrics["late_dropped"] += 1

        results: list[SyntheticResult] = []
        for request_id, request, future in eligible:
            try:
                result = future.result()
                if not isinstance(result, SyntheticResult):
                    raise TypeError(
                        "world-model scorer returned "
                        f"{type(result)!r}, expected SyntheticResult"
                    )
                result.validate(expected_variants=len(request.variants))
                _validate_result_provenance(request, result)
            except Exception as exc:  # noqa: BLE001 - convert worker failure to data
                result = SyntheticResult(
                    request_id=request.request_id,
                    trajectory_id=request.trajectory_id,
                    source_step=request.source_step,
                    weight_version=request.weight_version,
                    rewards=(),
                    wm_name="unknown",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._metrics["failed"] += 1
            else:
                self._metrics["completed"] += 1

            with self._lock:
                self._futures.pop(request_id, None)
                self._seen_results.add(request_id)
            results.append(result)
        self._metrics["polled"] += len(results)
        return results

    def pending(self) -> tuple[SyntheticRequest, ...]:
        with self._lock:
            return tuple(item[0] for item in self._futures.values())

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                **dict(self._metrics),
                "pending": len(self._futures),
            }

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


__all__ = [
    "CounterfactualTransition",
    "CounterfactualVariant",
    "OnlineSyntheticQueue",
    "RealPPOTrajectory",
    "RealPPOTransition",
    "SyntheticRequest",
    "SyntheticResult",
    "combine_paired_loss",
    "detach_tree_to_cpu",
    "renoise_nft_x0",
]
