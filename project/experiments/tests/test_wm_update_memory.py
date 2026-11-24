"""Allocator budgets, real CPU backward/resume, and scoped training hooks."""

import argparse
import copy
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_spatial_grpo_memory import FakeCuda
from wm_update import training_environment, training_plan, worker_command
from wm_update_data import training_code_identity
from wm_update_memory import (
    GIB,
    MemorySettings,
    TrainingMemoryReservation,
    add_memory_arguments,
    device_process_snapshot,
    memory_hooks,
    physical_device,
    reservation_target,
)


def test_runtime_allocator_uses_default_disabled_gc_and_no_cuda_compile_workers():
    with patch.dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.0",
                    TORCHINDUCTOR_COMPILE_THREADS="8"):
        env = training_environment(["0", "1"])
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native,expandable_segments:True"
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


class MeasuredCuda(FakeCuda):
    def __init__(self):
        super().__init__()
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved
        self.resets = 0

    def current_device(self):
        return 0

    def get_allocator_backend(self):
        return "native"

    def stream(self, stream):
        return nullcontext()

    def default_stream(self, device):
        return "default"

    def max_memory_allocated(self, device):
        return self.peak_allocated

    def max_memory_reserved(self, device):
        return self.peak_reserved

    def reset_peak_memory_stats(self, device=None):
        self.resets += 1
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def empty(self, *args, **kwargs):
        result = super().empty(*args, **kwargs)
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        return result


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native,expandable_segments:True")
    cuda = MeasuredCuda()
    fake_torch = SimpleNamespace(
        cuda=cuda, empty=cuda.empty, uint8="uint8",
        OutOfMemoryError=torch.OutOfMemoryError,
    )
    # Match the allocator's optional OOM class to the fake implementation.
    from test_spatial_grpo_memory import FakeOutOfMemoryError
    fake_torch.OutOfMemoryError = FakeOutOfMemoryError
    with patch("wm_update_memory.device_process_snapshot", return_value={
        "foreign_pids": [], "physical_gpu": "5", "processes": [],
    }):
        yield TrainingMemoryReservation(fake_torch, tmp_path, MemorySettings(), "wan")


def rows(pool):
    return [json.loads(line) for line in pool.log.read_text().splitlines()]


def test_reservation_stays_in_allocator_and_training_reuses_it(pool):
    cuda = pool.cuda
    pool.initialize()
    assert cuda.allocated == 10 * GIB
    assert cuda.free == int(3.5 * GIB)
    assert cuda.total - cuda.free == int(76.5 * GIB)
    before = cuda.reserved
    activations = cuda.empty(25 * GIB, "uint8", 0)
    assert cuda.reserved == before
    assert cuda.free == int(3.5 * GIB)
    del activations
    assert cuda.allocated == 10 * GIB
    assert cuda.empty_calls == 0


def test_cache_warmup_does_not_inflate_native_training_tensor_peak(pool):
    cuda = pool.cuda
    pool.initialize()
    assert cuda.max_memory_allocated(0) == 10 * GIB
    first = rows(pool)[0]
    assert first["reservation_transient_peak_allocated_gib"] > 60
    pool.reset_step_peak()
    batch = cuda.empty(20 * GIB, "uint8", 0)
    del batch
    pool.has_backward = True
    pool.after_optimizer()
    assert rows(pool)[-1]["peak_allocated_gib"] == 30
    assert rows(pool)[-1]["allocated_gib"] == 10
    assert rows(pool)[-1]["reserved_gib"] > 60
    pool.reset_step_peak()
    assert cuda.max_memory_allocated(0) == 10 * GIB


def test_no_fill_runs_during_save_and_real_backward_metrics(pool):
    pool.initialize()
    count = len(pool.cuda.requests)
    for phase in ["after_first_forward", "after_first_backward", "before_checkpoint_save", "after_checkpoint_save"]:
        pool.record(phase)
    assert len(pool.cuda.requests) == count
    assert pool.cuda.empty_calls == 0


def test_foreign_process_is_rejected_before_reserving(pool):
    with patch("wm_update_memory.device_process_snapshot", return_value={"foreign_pids": [12345]}):
        with pytest.raises(RuntimeError, match="other processes"):
            pool.initialize()
    assert pool.cuda.requests == []


def test_telemetry_outage_keeps_the_owned_cache(pool):
    pool.initialize()
    reserved = pool.cuda.reserved
    pool.last_snapshot = float("-inf")
    with patch("wm_update_memory.device_process_snapshot", side_effect=subprocess.TimeoutExpired("nvidia-smi", 15)):
        pool.record("after_checkpoint_save")
    assert pool.cuda.reserved == reserved
    assert rows(pool)[-1]["telemetry_error"] == "TimeoutExpired"


def test_optional_reservation_oom_does_not_fail_training(pool):
    pool.cuda.race_oom = True
    pool.initialize()
    assert rows(pool)[0]["reservation_error"]
    assert pool.cuda.allocated == 10 * GIB
    assert pool.cuda.empty_calls == 0


def test_synthetic_optimizer_initialization_is_not_a_training_update(pool):
    pool.initialize()
    pool.after_optimizer()
    assert pool.optimizer_updates == 0
    pool.has_backward = True
    pool.after_optimizer()
    assert pool.optimizer_updates == 1


@pytest.mark.parametrize("settings", [MemorySettings(headroom_gib=80), MemorySettings(target_gib=80)])
def test_impossible_device_budget_fails_before_allocation(settings):
    with pytest.raises(ValueError):
        reservation_target(0, 80 * GIB, 80 * GIB, settings)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf")])
def test_bad_settings_fail_without_cuda_initialization(value):
    with pytest.raises(ValueError):
        MemorySettings(headroom_gib=value)


def test_rank_mapping_rejects_duplicate_cards(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,5")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(ValueError):
        physical_device(0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    assert physical_device(1) == "7"


def test_graphics_and_compute_processes_are_both_detected():
    xml = """<nvidia_smi_log><gpu><uuid>GPU-example</uuid><minor_number>5</minor_number>
    <processes><process_info><pid>10</pid><type>C</type></process_info>
    <process_info><pid>20</pid><type>G</type></process_info></processes>
    </gpu></nvidia_smi_log>"""
    with patch("wm_update_memory.subprocess.run", return_value=SimpleNamespace(stdout=xml)):
        snapshot = device_process_snapshot("GPU-example", 10)
    assert snapshot["foreign_pids"] == [20]


class RecordingPool:
    """Exercise real backward/checkpoint hooks while leaving CUDA uninitialized."""
    def __init__(self):
        self.events = []
        self.seen = set()
        self.has_backward = False
        self.optimizer_updates = 0

    def initialize(self):
        raise AssertionError("CPU smoke must not initialize CUDA")

    def record(self, phase, **kwargs):
        self.events.append(phase)

    def record_once(self, phase):
        if phase not in self.seen:
            self.seen.add(phase)
            self.record(phase)

    def reset_step_peak(self, device=None):
        self.record("peak_reset")

    def after_optimizer(self):
        if self.has_backward:
            self.optimizer_updates += 1
            self.has_backward = False
            self.record("after_optimizer")


def test_real_backward_rng_and_optimizer_are_identical_with_hooks():
    torch.manual_seed(123)
    initial = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.3), torch.nn.Linear(3, 1))
    inputs = torch.ones(4, 3)
    state = torch.get_rng_state()

    def update(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()  # Same synthetic state-initialization shape as DCP.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            optimizer.step()
        return copy.deepcopy(model.state_dict()), torch.get_rng_state()

    expected, expected_rng = update(copy.deepcopy(initial))
    torch.set_rng_state(state)
    pool = RecordingPool()
    legacy_identity = training_code_identity()
    native_backward, native_empty = torch.Tensor.backward, torch.cuda.empty_cache
    with memory_hooks(torch, pool):
        actual, actual_rng = update(copy.deepcopy(initial))
        torch.cuda.empty_cache()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert torch.equal(actual_rng, expected_rng)
    assert pool.optimizer_updates == 2
    assert pool.events.count("after_first_backward") == 1
    assert "explicit_empty_cache_suppressed" in pool.events
    assert torch.Tensor.backward is native_backward
    assert torch.cuda.empty_cache is native_empty
    assert training_code_identity() == legacy_identity
    assert not torch.cuda.is_initialized()


def test_real_dcp_and_runtime_export_round_trip_with_scoped_hooks(tmp_path):
    import torch.distributed.checkpoint as dcp
    import safetensors.torch as safe

    pool = RecordingPool()
    original = dcp.save
    state = {"weights": torch.arange(6, dtype=torch.float32)}
    with memory_hooks(torch, pool):
        dcp.save(state, checkpoint_id=tmp_path / "state")
        restored = {"weights": torch.zeros(6)}
        dcp.load(restored, checkpoint_id=tmp_path / "state")
        safe.save_file(state, str(tmp_path / "runtime.safetensors"))
    torch.testing.assert_close(restored["weights"], state["weights"], atol=0, rtol=0)
    torch.testing.assert_close(safe.load_file(str(tmp_path / "runtime.safetensors"))["weights"], state["weights"])
    assert "after_checkpoint_save" in pool.events
    assert "after_checkpoint_load" in pool.events
    assert "after_runtime_export" in pool.events
    assert dcp.save is original
    assert not torch.cuda.is_initialized()


def test_hooks_restore_all_apis_after_failure():
    pool = RecordingPool()
    before = (torch.Tensor.backward, torch.cuda.empty_cache, torch.cuda.memory.empty_cache, torch.cuda.reset_peak_memory_stats)
    with pytest.raises(RuntimeError, match="training error"):
        with memory_hooks(torch, pool):
            raise RuntimeError("training error")
    assert before == (torch.Tensor.backward, torch.cuda.empty_cache, torch.cuda.memory.empty_cache, torch.cuda.reset_peak_memory_stats)
    assert "training_failed" in pool.events


def test_worker_flags_and_eight_rank_resume_keep_memory_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("RLINF_ENV", "/runtime")
    monkeypatch.setenv("COSMOS_PYTHON", "/cosmos/python")
    args = SimpleNamespace(
        manifest=tmp_path / "manifest", output=tmp_path / "output", settings=tmp_path / "settings",
        steps=2000, save_interval=200, accumulation=2, lr=1e-6, seed=42,
        resume=tmp_path / "step_001400", model="wan", gpus="0,1,2,3,4,5,6,7",
        ranks_per_model=8, reserve_memory=True, memory_headroom_gib=4, memory_target_gib=75,
    )
    job = training_plan(args)[0]
    assert job["num_processes"] == 8
    assert job["effective_batch"] == 16
    assert job["memory_reservation"]["headroom_gib"] == 4
    command = worker_command(args, "cosmos", 4)
    assert "--reserve-memory" in command
    assert command[command.index("--memory-target-gib") + 1] == "75"
    assert "--resume" in command


def test_disabled_mode_is_explicit_and_propagates_to_workers(monkeypatch):
    parser = argparse.ArgumentParser()
    add_memory_arguments(parser)
    args = parser.parse_args(["--no-reserve-memory"])
    assert MemorySettings.from_args(args).command_args()[0] == "--no-reserve-memory"
