"""Exercise native Cosmos generation/solver code with a small CPU-only net."""

from __future__ import annotations
import __future__

import ast
import math
import sys
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from einops import rearrange

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(PROJECT / "experiments"),
    str(PROJECT / "third_party/WorldPilot"),
    str(PROJECT / "third_party/cosmos-policy"),
]

from cosmos_bridge.cosmos_client import CosmosClient
from cosmos_bridge.cosmos_server import CosmosAdvisorPolicy, CosmosCfg
from cosmos_policy._src.imaginaire.functional.multi_step import (
    get_multi_step_fn,
    is_multi_step_fn_supported,
)
from cosmos_policy._src.imaginaire.functional.runge_kutta import (
    get_runge_kutta_fn,
    is_runge_kutta_fn_supported,
)
from cosmos_policy._src.imaginaire.modules.denoiser_scaling import RectifiedFlowScaling
from cosmos_policy._src.imaginaire.utils.denoise_prediction import DenoisePrediction
from spatial_wovr_noise import (
    compact_cosmos_denoiser,
    cosmos_active_sample_mask_supported,
    validate_active_sample_mask,
)

COSMOS = PROJECT / "third_party/cosmos-policy/cosmos_policy"


def native_function(relative_path, class_name, name, namespace):
    """Run the checked-out method itself, retaining its real source location."""
    path = COSMOS / relative_path
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(  # noqa: S102 -- exercise checked-out native methods against CPU fixtures
        ast.Module(body=[function], type_ignores=[]),
        str(path), "exec", flags=__future__.annotations.compiler_flag,
    ), namespace)
    return namespace[name]


class DataType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class Condition:
    _is_broadcasted: bool = False
    crossattn_emb: torch.Tensor | None = None
    data_type: DataType = DataType.VIDEO
    padding_mask: torch.Tensor | None = None
    fps: torch.Tensor | None = None
    use_video_condition: bool = True
    gt_frames: torch.Tensor | None = None
    condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None
    action: torch.Tensor | None = None
    img_context_emb: torch.Tensor | None = None

    @property
    def is_video(self):
        return self.data_type == DataType.VIDEO

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self) if not field.name.startswith("_")}

    def edit_data_type(self, data_type):
        return replace(self, data_type=data_type)

    def set_video_condition(self, *, gt_frames, num_conditional_frames, **kwargs):
        del kwargs
        mask = torch.zeros_like(gt_frames[:, :1])
        for sample, count in enumerate(num_conditional_frames):
            mask[sample, :, :int(count)] = 1
        return replace(self, gt_frames=gt_frames, condition_video_input_mask_B_C_T_H_W=mask)

    def edit_for_inference(self, **kwargs):
        del kwargs
        return self


def make_model(*, fail_net=False):
    """Only learned net/conditioning/VAE are tiny fixtures; solver is native."""
    ns = {
        "__name__": "cosmos_policy._src.imaginaire.modules.res_sampler",
        "torch": torch, "math": math,
        "log": SimpleNamespace(info=lambda *args: None),
        "get_multi_step_fn": get_multi_step_fn,
        "get_runge_kutta_fn": get_runge_kutta_fn,
        "is_multi_step_fn_supported": is_multi_step_fn_supported,
        "is_runge_kutta_fn_supported": is_runge_kutta_fn_supported,
        "SolverConfig": SimpleNamespace,
        "SolverTimestampConfig": SimpleNamespace,
        "SamplerConfig": SimpleNamespace,
    }
    sampler_path = "_src/imaginaire/modules/res_sampler.py"
    for name in ("get_rev_ts", "fori_loop", "differential_equation_solver"):
        native_function(sampler_path, None, name, ns)
    sampler_type = type("Sampler", (), {
        "__module__": ns["__name__"],
        "forward": native_function(sampler_path, "Sampler", "forward", ns),
        "_forward_impl": native_function(sampler_path, "Sampler", "_forward_impl", ns),
        "__call__": lambda self, *args, **kwargs: self.forward(*args, **kwargs),
    })
    noise_calls = []

    def arch_invariant_rand(shape, dtype, device, seed):
        # The native utility's RandomState algorithm, with a shape spy only.
        noise_calls.append(tuple(shape))
        noise = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
        return torch.from_numpy(noise).to(dtype=dtype, device=device)

    model_ns = {
        "__name__": "cosmos_policy._src.predict2.models.text2world_model",
        "torch": torch, "misc": SimpleNamespace(arch_invariant_rand=arch_invariant_rand),
    }
    generate = native_function(
        "_src/predict2/models/text2world_model.py", "DiffusionModel", "generate_samples_from_batch", model_ns
    )
    condition_ns = {
        "__name__": "cosmos_policy._src.predict2.action.models.action_conditioned_video2world_model",
        "torch": torch, "DataType": DataType,
        "NUM_CONDITIONAL_FRAMES_KEY": "num_conditional_frames",
        "parallel_state": SimpleNamespace(is_initialized=lambda: False),
    }
    get_x0 = native_function(
        "_src/predict2/action/models/action_conditioned_video2world_model.py",
        "ActionConditionedVideo2WorldModel", "get_x0_fn_from_batch", condition_ns,
    )
    denoise_ns = {
        "__name__": "cosmos_policy.models.temporal_action_conditioned_model",
        "torch": torch, "rearrange": rearrange, "DenoisePrediction": DenoisePrediction,
    }
    denoise = native_function(
        "models/temporal_action_conditioned_model.py", "CosmosWoVRVideo2WorldModel", "denoise", denoise_ns
    )
    model_type = type("CosmosWoVRVideo2WorldModel", (), {
        "__module__": denoise_ns["__name__"],
        "generate_samples_from_batch": generate, "get_x0_fn_from_batch": get_x0, "denoise": denoise,
    })
    model = model_type()
    model.training = False
    model.config = SimpleNamespace(
        state_t=4, state_ch=3, min_num_conditional_frames=2, max_num_conditional_frames=2,
        use_flowunipc_scheduler=False, sigma_conditional=0.0001,
        sigma_data=1.0, use_wan_fp32_strategy=False, denoise_replace_gt_frames=True,
    )
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.sde = SimpleNamespace(sigma_max=80.0, sigma_min=0.002)
    model.sampler = sampler_type()
    model.scaling = RectifiedFlowScaling()
    model.tokenizer = SimpleNamespace(get_latent_num_frames=lambda count: 4, spatial_compression_factor=1)
    model.input_data_key, model.input_image_key = "video", "images"
    model.is_image_batch = lambda batch: False
    model._augment_image_dim_inplace = lambda batch: None
    model._normalize_video_databatch_inplace = lambda batch: batch.update(video=batch["video"].float() / 127.5 - 1)
    model.broadcast_split_for_model_parallelsim = lambda *args: args
    model.encode_batches, model.decode_batches, model.net_batches = [], [], []
    model.noise_calls = noise_calls

    def encode(video):
        model.encode_batches.append(video.shape[0])
        return video[:, :, [0, 4, 8, 12]].clone()

    model.encode = encode
    model.get_data_and_condition = lambda batch: (None, model.encode(batch["video"]), None)

    def decode(latent):
        model.decode_batches.append(latent.shape[0])
        assert latent.shape[2] == 4
        return torch.cat([latent[:, :, :1], latent[:, :, 1:].repeat_interleave(4, dim=2)], dim=2)

    model.decode = decode

    def conditions(batch):
        condition = Condition(
            action=batch["action"], crossattn_emb=batch["t5_text_embeddings"],
            padding_mask=batch["padding_mask"], fps=batch["fps"],
        )
        return condition, replace(condition, action=torch.zeros_like(condition.action))

    model.conditioner = SimpleNamespace(get_condition_uncondition=conditions)

    class Net:
        training = False
        wovr_action_contract = True
        is_context_parallel_enabled = False
        cross_view_primary_context = False

        def __call__(self, *, x_B_C_T_H_W, action, **kwargs):
            del kwargs
            if fail_net:
                raise RuntimeError("injected network failure")
            model.net_batches.append(x_B_C_T_H_W.shape[0])
            action_signal = action.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
            return torch.tanh(x_B_C_T_H_W * 0.1 + action_signal)

    model.net = Net()
    return model


def make_batch(batch_size=4):
    return {
        "video": torch.arange(batch_size * 3 * 13 * 2 * 2).remainder(256).to(torch.uint8).reshape(batch_size, 3, 13, 2, 2),
        "action": torch.arange(batch_size * 13 * 7).reshape(batch_size, 13, 7).float() / 500,
        "t5_text_embeddings": torch.ones(batch_size, 2, 3),
        "padding_mask": torch.zeros(batch_size, 1, 2, 2),
        "fps": torch.full((batch_size,), 20.0),
        "num_conditional_frames": torch.full((batch_size,), 2),
    }


@pytest.mark.parametrize("mask", [[True, False, True, False], [False, True, False, True], [False, False, False, True]])
@pytest.mark.parametrize("guidance", [0.0, 1.5])
@pytest.mark.parametrize("seed", [7, 31])
def test_native_full_batch_noise_solver_and_active_predictions_unchanged(mask, guidance, seed):
    mask = np.asarray(mask, dtype=bool)
    baseline, optimized = make_model(), make_model()
    assert cosmos_active_sample_mask_supported(optimized)
    torch.manual_seed(123)
    expected = baseline.generate_samples_from_batch(make_batch(), n_sample=4, num_steps=5, guidance=guidance, seed=seed)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(123)
    with compact_cosmos_denoiser(optimized, mask):
        actual = optimized.generate_samples_from_batch(make_batch(), n_sample=4, num_steps=5, guidance=guidance, seed=seed)
    torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng, rtol=0, atol=0)
    assert baseline.noise_calls == optimized.noise_calls == [(4, 3, 4, 2, 2)]
    assert baseline.encode_batches == optimized.encode_batches == [4]
    assert set(baseline.net_batches) == {4}
    assert set(optimized.net_batches) == {int(mask.sum())}
    assert "denoise" not in optimized.__dict__
    assert cosmos_active_sample_mask_supported(optimized)


def test_compact_scope_restores_instance_after_network_failure():
    model = make_model(fail_net=True)
    original = model.denoise
    with pytest.raises(RuntimeError, match="injected network failure"), compact_cosmos_denoiser(
        model, [False, True, False, True]
    ):
        model.generate_samples_from_batch(make_batch(), n_sample=4, num_steps=5)
    assert model.denoise == original
    assert "denoise" not in model.__dict__
    assert not hasattr(model, "_spatial_wovr_compact_request")


@pytest.mark.parametrize("unsupported", ["training", "flowunipc", "context_parallel", "unknown_denoise"])
def test_unknown_or_unsupported_models_do_not_advertise_support(unsupported):
    model = make_model()
    if unsupported == "training":
        model.training = True
    elif unsupported == "flowunipc":
        model.config.use_flowunipc_scheduler = True
    elif unsupported == "context_parallel":
        model.net.is_context_parallel_enabled = True
    else:
        model.denoise = lambda *args: None
    assert not cosmos_active_sample_mask_supported(model)
    with pytest.raises(RuntimeError), compact_cosmos_denoiser(model, [False, True, False, True]):
        pytest.fail("unsupported protocol accepted")


def test_all_active_scope_keeps_native_method_and_rejects_overlap():
    model = make_model()
    original = model.denoise
    with compact_cosmos_denoiser(model, [True] * 4):
        assert model.denoise == original
    with (
        compact_cosmos_denoiser(model, [False, True, False, True]),
        pytest.raises(RuntimeError, match="Overlapping"),
        compact_cosmos_denoiser(model, [True] * 4),
    ):
        pytest.fail("overlapping scope accepted")


@pytest.mark.parametrize("value", [[1, 0], ["true", "false"], [[True, False]], [True], []])
def test_active_mask_is_strict_bool_vector(value):
    with pytest.raises(ValueError, match=r"bool\[B\]"):
        validate_active_sample_mask(value, 2)


def temporal_modules():
    decode_calls, build_calls = [], []
    ns = {"torch": torch, "np": np, "STATE_T": 4, "TOKENIZER_CHUNK_DURATION": 13,
          "FUTURE_PRIMARY_RAW_FRAME_INDICES": tuple(range(5, 13))}
    native_decode = native_function("experiments/robot/libero_temporal_utils.py", None, "decode_temporal_future", ns)

    def build(**kwargs):
        build_calls.append(kwargs)
        reference, memory = kwargs["reference_primary"], kwargs["memory_primary"]
        video = np.concatenate([reference[:, None], memory, np.repeat(memory[:, -1:], 8, axis=1)], axis=1)
        batch_size = reference.shape[0]
        batch = make_batch(batch_size)
        batch["video"] = torch.from_numpy(video.transpose(0, 4, 1, 2, 3).copy())
        batch["action"] = torch.from_numpy(np.concatenate([
            kwargs["normalized_context_actions"], kwargs["normalized_actions"]
        ], axis=1))
        return batch

    def decode(**kwargs):
        decode_calls.append(kwargs)
        return native_decode(**kwargs)

    temporal = SimpleNamespace(build_temporal_action_conditioned_batch=build, decode_temporal_future=decode)
    return {"cosmos_policy.experiments.robot.libero_temporal_utils": temporal}, build_calls, decode_calls


def request():
    return {
        "reference_primary": np.full((4, 2, 2, 3), 30, dtype=np.uint8),
        "memory_primary": np.arange(4 * 4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 4, 2, 2, 3),
        "actions": np.zeros((4, 8, 7), dtype=np.float32),
        "context_actions": np.zeros((4, 5, 7), dtype=np.float32),
        "langs": ["task"] * 4, "actions_are_normalized": True, "batch_seed": 17,
    }


def policy(model):
    return CosmosAdvisorPolicy(CosmosCfg(chunk_size=8), model, {})


def test_route_retains_full_noise_batch_decodes_only_active_and_freezes_inactive(monkeypatch):
    modules, builds, decodes = temporal_modules()
    baseline, optimized = make_model(), make_model()
    msg = request()
    mask = np.asarray([False, True, False, True])
    with patch.dict(sys.modules, modules):
        torch.manual_seed(99)
        expected = policy(baseline).predict_temporal_future_batch(**msg)
        monkeypatch.setenv("SPATIAL_WOVR_FAST_TRANSFER", "1")
        torch.manual_seed(99)
        actual = policy(optimized).predict_temporal_future_batch(**msg, active_sample_mask=mask)
    np.testing.assert_array_equal(actual["future_primary_clip"][mask], expected["future_primary_clip"][mask])
    frozen = np.repeat(msg["memory_primary"][~mask, -1:], 8, axis=1)
    np.testing.assert_array_equal(actual["future_primary_clip"][~mask], frozen)
    assert baseline.noise_calls == optimized.noise_calls == [(4, 3, 4, 2, 2)]
    assert [call["reference_primary"].shape[0] for call in builds] == [4, 4]
    assert baseline.decode_batches == [4]
    assert optimized.decode_batches == [2]
    assert decodes[-1]["select_future_before_cpu"] is True
    assert set(optimized.net_batches) == {2}


def test_route_all_inactive_does_not_generate_or_decode():
    modules, _, decodes = temporal_modules()
    model = make_model()
    msg = request()
    with patch.dict(sys.modules, modules):
        actual = policy(model).predict_temporal_future_batch(**msg, active_sample_mask=np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(actual["future_primary_clip"], np.repeat(msg["memory_primary"][:, -1:], 8, axis=1))
    assert not model.noise_calls and not model.encode_batches and not model.net_batches and not decodes


def test_route_shared_view_mask_keeps_original_expanded_noise_positions():
    modules, _, _ = temporal_modules()
    baseline, optimized = make_model(), make_model()
    msg = request()
    msg.update(reference_wrist=msg["reference_primary"], memory_wrist=255 - msg["memory_primary"])
    mask = np.asarray([True, False, False, True])
    with patch.dict(sys.modules, modules):
        torch.manual_seed(99)
        expected = policy(baseline).predict_temporal_future_batch(**msg)
        torch.manual_seed(99)
        actual = policy(optimized).predict_temporal_future_batch(**msg, active_sample_mask=mask)
    for name in ("future_primary_clip", "future_wrist_clip"):
        np.testing.assert_array_equal(actual[name][mask], expected[name][mask])
    assert baseline.noise_calls == optimized.noise_calls == [(8, 3, 4, 2, 2)]
    assert optimized.decode_batches == [4]


def test_metadata_reports_only_verified_loaded_protocol():
    model = make_model()
    assert policy(model).predict_action(mode="metadata")["active_sample_mask_supported"] is True
    model.config.use_flowunipc_scheduler = True
    assert policy(model).predict_action(mode="metadata")["active_sample_mask_supported"] is False


def test_client_forwards_mask_and_keeps_original_batch():
    client = CosmosClient.__new__(CosmosClient)
    client._client = SimpleNamespace(predict_action=Mock(return_value={"data": {"ok": True}}))
    mask = np.asarray([False, True, False, True])
    client.query_temporal_future_batch(**request(), active_sample_mask=mask)
    message = client._client.predict_action.call_args.args[0]
    np.testing.assert_array_equal(message["active_sample_mask"], mask)
    assert message["reference_primary"].shape[0] == 4
    client.query_temporal_future_batch(**request())
    assert "active_sample_mask" not in client._client.predict_action.call_args.args[0]
    with pytest.raises(ValueError, match=r"bool\[B\]"):
        client.query_temporal_future_batch(**request(), active_sample_mask=np.asarray([1, 0, 1, 0]))
