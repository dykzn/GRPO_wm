"""CPU integration checks for the actor-owned Wan service and memory budget."""

import ast
import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import test_spatial_wovr_active_generation as active_fixture
from spatial_wovr_shared_pool import (
    GIB,
    MIB,
    SharedCudaReservation,
    SharedWanOwner,
    SharedWanProxy,
    SharedWanServer,
    _cpu_wire,
    configure_shared_pool,
    job_device_memory,
    retain_process_cache,
    shared_reservation_target,
    start_actor_wan,
)
from test_spatial_grpo_memory import FakeCuda


@pytest.fixture
def settings(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="wan-pool-test-") as directory:
        monkeypatch.setenv("TMPDIR", directory)
        monkeypatch.setenv(
            "PYTORCH_CUDA_ALLOC_CONF", "backend:native,expandable_segments:True"
        )
        cfg = OmegaConf.create(
            {
                "cluster": {"num_nodes": 1},
                "actor": {
                    "enable_offload": False,
                    "memory_pool": {"enabled": True},
                    "shared_wan_pool": {
                        "enabled": True,
                        "target_job_gib": 75,
                        "device_headroom_gib": 3.5,
                        "rpc_timeout_seconds": 5,
                    },
                },
                "rollout": {"pipeline_stage_num": 1, "enable_offload": False},
                "env": {"train": {"enable_offload": True}},
                "runner": {"memory_guard": {"enabled": False}},
            }
        )
        configure_shared_pool(cfg, Path(directory) / "logs")
        yield cfg.actor.shared_wan_pool


def test_transport_rejects_even_cpu_tensors_and_object_arrays():
    for payload in (
        torch.ones(2),
        {"nested": [torch.zeros(1)]},
        np.array([object()], dtype=object),
    ):
        with pytest.raises(TypeError, match="CPU arrays/scalars"):
            _cpu_wire(payload)
    _cpu_wire({"arrays": [np.zeros((2, 3), dtype=np.float32)], "seed": 0})


@pytest.mark.parametrize(
    "field,value",
    [
        ("runner.memory_guard.enabled", True),
        ("actor.memory_pool.enabled", False),
        ("actor.enable_offload", True),
        ("env.train.enable_offload", False),
        ("cluster.num_nodes", 2),
        ("actor.shared_wan_pool.target_job_gib", float("nan")),
    ],
)
def test_incompatible_shared_config_fails_before_workers(settings, field, value):
    cfg = OmegaConf.create(
        {
            "actor": {
                "shared_wan_pool": settings,
                "memory_pool": {"enabled": True},
                "enable_offload": False,
            },
            "rollout": {"enable_offload": False, "pipeline_stage_num": 1},
            "cluster": {"num_nodes": 1},
            "env": {"train": {"enable_offload": True}},
            "runner": {},
        }
    )
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError):
        configure_shared_pool(cfg, "/unused")


def test_rpc_credentials_are_private_and_per_run(settings):
    root = Path(settings.rpc_root)
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "authkey").stat().st_mode & 0o777 == 0o600
    assert len((root / "authkey").read_bytes()) == 32


def test_service_is_actor_thread_and_preserves_real_environment_outputs(settings):
    # Construct before the fixture snapshots sys.modules: torch.optim lazily
    # registers distributed kernels which must not be imported a second time.
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    active_fixture.ActiveGenerationTests.setUpClass()
    case = active_fixture.ActiveGenerationTests("runTest")
    case.setUp()
    server = None
    try:
        control, remote = (
            case.environment(compact=True, fast=True),
            case.environment(compact=True, fast=True),
        )
        pipeline = remote.pipe
        pipeline.dit = pipeline.vae = SimpleNamespace(to=lambda device: pipeline.dit)
        phases, pids = [], []

        @contextmanager
        def execution():
            pids.append((os.getpid(), threading.current_thread().name))
            yield

        owner = SharedWanOwner(
            pipeline, "cpu", execution=execution, retain=phases.append
        )
        server = SharedWanServer(settings, 0, owner)
        proxy = SharedWanProxy(settings, 0)
        proxy.offload()  # Init offload precedes actor readiness and must be local.
        remote.pipe, remote._uses_shared_wan_pool = proxy, True
        proxy.onload()
        for _ in range(4):
            expected = control.chunk_step(case.actions())
            actual = remote.chunk_step(case.actions())
            for index in (1, 2, 3):
                torch.testing.assert_close(actual[index], expected[index])
            for key in ("main_images", "wrist_images", "states"):
                torch.testing.assert_close(actual[0][0][key], expected[0][0][key])
            torch.testing.assert_close(remote.current_obs, control.current_obs)
            torch.testing.assert_close(
                remote.condition_action, control.condition_action
            )
        # Compaction preserves original slot noise and the source seed=0.
        for actual_call, expected_call in zip(
            pipeline.calls, control.pipe.calls, strict=True
        ):
            torch.testing.assert_close(actual_call["noise"], expected_call["noise"])
            torch.testing.assert_close(actual_call["actions"], expected_call["actions"])
        with (
            pytest.raises(RuntimeError, match="offload acknowledgement"),
            owner.actor_phase("training"),
        ):
            pass
        proxy.offload()
        with owner.actor_phase("training"):
            before = model.weight.detach().clone()
            model(torch.ones(2, 3)).square().mean().backward()
            optimizer.step()
            assert model.weight.grad.isfinite().all()
            assert not torch.equal(model.weight, before)
        proxy.onload()
        proxy.offload()
        assert phases == [
            "wan_onload",
            "first_dual_view_chunk_complete",
            "wan_offload",
            "wan_onload",
            "wan_offload",
        ]
        assert pids and all(
            pid == os.getpid() and name == "wan-pool-0" for pid, name in pids
        )
    finally:
        if server is not None:
            server.close()
            assert not server.thread.is_alive()
        case.doCleanups()


def test_rpc_error_does_not_kill_service_or_silently_retry(settings):
    owner = SharedWanOwner(None, "cpu")
    server = SharedWanServer(settings, 0, owner)
    try:
        proxy = SharedWanProxy(settings, 0)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="GPU-resident Wan lease"):
                proxy.request("generate")
        assert server.thread.is_alive()
    finally:
        server.close()


def test_actor_phase_prevents_overlapping_wan_onload():
    entered = threading.Event()

    class Model:
        def to(self, device):
            entered.set()
            return self

    owner = SharedWanOwner(SimpleNamespace(dit=Model(), vae=Model()), "cpu")
    with owner.actor_phase("training"):
        thread = threading.Thread(
            target=lambda: owner.dispatch({"protocol": 1, "operation": "onload"})
        )
        thread.start()
        assert not entered.wait(timeout=0.05)
    thread.join(timeout=2)
    assert entered.is_set() and not thread.is_alive()


def test_fixed_frame_windows_preserve_variable_and_frozen_suffixes():
    cls = active_fixture.production_environment_class()
    env = cls.__new__(cls)
    env.condition_frame_length, env.chunk = 5, 8
    env.current_obs = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(1, 3, 1, 5, 2, 2)
    env._wan_frame_windows = torch.empty(2, 1, 3, 1, 13, 2, 2)
    env._wan_frame_write = 0
    expected = env.current_obs.clone()
    for size, frozen in [(8, False), (8, True), (0, True), (2, False), (17, False), (8, True)]:
        if frozen:
            frames = env.current_obs[:, :, 0, -1:].expand(-1, -1, size, -1, -1)
        else:
            frames = torch.arange(3 * size * 4, dtype=torch.float32).reshape(1, 3, size, 2, 2)
        expected = torch.cat((expected, frames.unsqueeze(2)), dim=3)[:, :, :, -13:].clone()
        env._append_wan_frames(frames)
        torch.testing.assert_close(env.current_obs, expected, atol=0, rtol=0)
        assert env.current_obs.untyped_storage().data_ptr() == env._wan_frame_windows.untyped_storage().data_ptr()


def test_target_counts_whole_job_and_does_not_shrink_when_foreign_job_appears():
    assert (
        shared_reservation_target(12 * GIB, 43 * GIB, 36 * GIB, 75 * GIB, 4 * GIB)
        == 51 * GIB
    )
    assert (
        shared_reservation_target(51 * GIB, 1 * GIB, 75 * GIB, 75 * GIB, 4 * GIB)
        == 51 * GIB
    )
    assert (
        shared_reservation_target(12 * GIB, 8 * GIB, 36 * GIB, 75 * GIB, 4 * GIB)
        == 16 * GIB
    )


def test_actual_retention_reuses_cache_across_wan_and_backward(settings, monkeypatch):
    class Cuda(FakeCuda):
        def synchronize(self, device=None):
            pass

        def stream(self, value):
            return nullcontext()

        def default_stream(self, device):
            return None

    cuda = Cuda(allocated=10 * GIB, reserved=12 * GIB, free=44 * GIB)
    torch_stub = SimpleNamespace(
        cuda=cuda, empty=cuda.empty, uint8="uint8", OutOfMemoryError=RuntimeError
    )
    monkeypatch.setenv("SPATIAL_WOVR_OUTPUT", "/own-run")
    monkeypatch.setattr(
        "spatial_wovr_shared_pool.job_device_memory",
        lambda device, run: {
            "job_used": 36 * GIB + cuda.reserved - 12 * GIB,
            "foreign_used": 0,
            "device_total": 80 * GIB,
            "device_used": 80 * GIB - cuda.free,
        },
    )
    pool = SharedCudaReservation(torch_stub, 0, settings, 0)
    pool.retain("initialized")
    assert cuda.reserved == 41 * GIB  # Task starts at 65 GiB.
    pool.retain("wan_onload")
    assert cuda.reserved == 41 * GIB  # Outside services have not warmed yet.
    pool.activate()
    pool.retain("first_dual_view_chunk_complete")
    reserved, free = cuda.reserved, cuda.free
    assert reserved == 51 * GIB
    assert cuda.allocated == 10 * GIB  # No live filler tensors remain.
    wan = cuda.empty(28 * GIB, "uint8", 0)
    del wan
    pool.retain("wan_offload")
    backward = cuda.empty(18 * GIB, "uint8", 0)
    del backward
    pool.retain("after_training")
    assert cuda.reserved == reserved and cuda.free == free and cuda.empty_calls == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_rollout_weight_sync_retains_warmed_cache_only_in_shared_mode(enabled):
    released = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            empty_cache=lambda: released.append("driver"), memory=SimpleNamespace()
        )
    )

    class NativeRollout:
        def init_worker(self):
            return "native_init"

        async def sync_model_from_actor(self):
            fake_torch.cuda.empty_cache()

    spec = importlib.util.spec_from_file_location(
        "shared_rollout_test", EXPERIMENTS / "spatial_wovr_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "rlinf.workers.rollout.hf.huggingface_worker": SimpleNamespace(
                MultiStepRolloutWorker=NativeRollout
            ),
        },
    ):
        spec.loader.exec_module(module)
    rollout = module.CachedWorldModelRollout()
    rollout.cfg = OmegaConf.create({"actor": {"shared_wan_pool": {"enabled": enabled}}})
    assert rollout.init_worker() == "native_init"
    asyncio.run(rollout.sync_model_from_actor())
    assert released == ([] if enabled else ["driver"])


def test_process_patch_keeps_both_empty_cache_aliases_local():
    calls = []
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            empty_cache=lambda: calls.append(1),
            memory=SimpleNamespace(empty_cache=lambda: calls.append(2)),
        )
    )
    retain_process_cache(fake)
    fake.cuda.empty_cache()
    fake.cuda.memory.empty_cache()
    assert calls == []


@pytest.mark.parametrize("device", [1, "cuda:1", torch.device("cuda:1")])
def test_gpu_mapping_counts_own_processes_only(monkeypatch, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3,5,1,6,2,4,0")

    def query(command, **kwargs):
        data = (
            "7, GPU-seven, 81920, 60000\n3, GPU-three, 81920, 70000\n"
            if "--query-gpu" in command[1]
            else """<nvidia_smi_log><gpu><uuid>GPU-three</uuid><processes>
        <process_info><pid>11</pid><used_memory>20000 MiB</used_memory><type>C</type></process_info>
        <process_info><pid>11</pid><used_memory>20000 MiB</used_memory><type>G</type></process_info>
        <process_info><pid>12</pid><used_memory>8000 MiB</used_memory><type>C</type></process_info>
        <process_info><pid>13</pid><used_memory>4096 MiB</used_memory><type>G</type></process_info>
        </processes></gpu></nvidia_smi_log>"""
        )
        return SimpleNamespace(stdout=data)

    with (
        patch("spatial_wovr_shared_pool.subprocess.run", side_effect=query),
        patch(
            "spatial_wovr_memory_guard.process_belongs_to_run",
            side_effect=lambda pid, run: pid in (11, 13),
        ),
    ):
        snapshot = job_device_memory(device, "/own")
    assert snapshot["job_used"] == 24096 * MIB
    assert snapshot["job_graphics_used"] == 24096 * MIB
    assert snapshot["foreign_used"] == 8000 * MIB


@pytest.mark.parametrize("device", [2, "cuda:2", torch.device("cuda:2")])
def test_owner_normalizes_native_actor_integer_device_without_cuda(settings, device):
    pipeline = SimpleNamespace()
    fake_torch = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(
            get_allocator_backend=lambda: "native", memory=SimpleNamespace()
        ),
        random=SimpleNamespace(fork_rng=lambda devices: nullcontext()),
    )
    actor = SimpleNamespace(
        device=device,
        _rank=2,
        cfg=OmegaConf.create(
            {"actor": {"shared_wan_pool": settings}, "env": {"train": {}}}
        ),
        log_info=lambda message: None,
    )
    from unittest.mock import Mock

    builder = Mock(return_value=pipeline)
    with (
        patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "rlinf.envs.world_model.world_model_wan_env": SimpleNamespace(
                    build_wan_pipeline=builder
                ),
            },
        ),
        patch("spatial_wovr_shared_pool.SharedCudaReservation") as reservation,
        patch("spatial_wovr_shared_pool.SharedWanServer"),
    ):
        start_actor_wan(actor)
    assert actor.device == device
    assert actor._shared_wan_owner.device == torch.device("cuda:2")
    assert reservation.call_args.args[1] == torch.device("cuda:2")
    assert builder.call_args.args[1] == "cuda:2"


def test_actor_sync_retains_cache_in_shared_mode():
    events = []

    class Base:
        async def sync_model_to_rollout(self):
            events.append("sync")

    spec = importlib.util.spec_from_file_location(
        "shared_actor_test", EXPERIMENTS / "spatial_wovr_actor.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"spatial_grpo_actor": SimpleNamespace(ReservedEmbodiedFSDPActor=Base)},
    ):
        spec.loader.exec_module(module)
    actor = module.WorldModelFSDPActor()
    actor._shared_wan_owner = SharedWanOwner(None, "cpu")
    actor.cfg = SimpleNamespace(runner={})
    actor._cuda_cache_reservation = SimpleNamespace(retain=events.append)
    with patch(
        "torch.cuda.empty_cache", side_effect=AssertionError("cache returned to driver")
    ):
        asyncio.run(actor.sync_model_to_rollout())
    assert events == ["sync", "after_weight_sync"]


def test_busy_gpu_entry_exits_before_launching_workers(tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text(
        "#!/bin/sh\nprintf '0, 70000\\n1, 70000\\n2, 70000\\n3, 70000\\n4, 70000\\n5, 70000\\n6, 70000\\n7, 70000\\n'\n"
    )
    fake.chmod(0o700)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        SPATIAL_WOVR_OUTPUT=str(tmp_path / "unstarted"),
    )
    result = subprocess.run(
        ["bash", str(EXPERIMENTS / "run_spatial_wovr_shared_pool_8gpu.sh")],
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "no workers launched" in result.stderr
    assert not (tmp_path / "unstarted").exists()


def test_cpu_pipeline_build_does_not_allocate_wan_on_cuda():
    path = (
        EXPERIMENTS.parent
        / "third_party/RLinf/rlinf/envs/world_model/world_model_wan_env.py"
    )
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "build_wan_pipeline"
    )
    moves, params = [], []

    def pretrained(**kwargs):
        params.append(kwargs)
        return SimpleNamespace(
            dit=SimpleNamespace(to=moves.append), vae=SimpleNamespace(to=moves.append)
        )

    namespace = {
        "torch": torch,
        "WanVideoPipeline": SimpleNamespace(from_pretrained=pretrained),
        "ModelConfig": lambda **kwargs: kwargs,
    }
    exec(  # noqa: S102 -- execute the real builder with CPU-only model doubles
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["build_wan_pipeline"](
        SimpleNamespace(model_path="dit", VAE_path="vae"),
        "cuda:0",
        initially_offloaded=True,
    )
    assert not moves
    assert params[0]["use_usp"] is False
    assert all(c["offload_device"] == "cpu" for c in params[0]["model_configs"])
