"""Do not replace live 4+4 training until committed checkpoints are available."""

import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_iterative_cosmos_metrics_wan8 as queue


def plan_at(root):
    original = root / "original"
    original.mkdir()
    root_output = root / "replacement"
    root_output.mkdir()
    return dict(output=str(root_output), source_output=str(original), wm_output=str(root / "wm"),
        source_plan=str(original / "plan.json"), python="/python", seed=20261021,
        gpus=[str(i) for i in range(8)], evaluation_gpus=[str(i) for i in range(4,8)],
        wan_gpus=[str(i) for i in range(4)], wan_control_token="this-run",
        manifest_sha256="current-split", evaluation_manifest_sha256="same-100",
        evaluation_summary=dict(episodes=100, future_frames=800),
        original_supervisor=dict(pid=100, start_ticks=200),
        original_training=dict(pid=101, start_ticks=201), original_notifier=dict(pid=102, start_ticks=202),
        original_groups=dict(wan=dict(pid=103,start_ticks=203),cosmos=dict(pid=104,start_ticks=204)))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cosmos_states(plan):
    for step in queue.SAVED_STEPS:
        root = queue.cosmos_checkpoint(plan, step)
        write(root / "wm_update_complete.json", dict(step=step, world_size=4, identity=dict(manifest="current-split")))
        for name in ("model", "optim", "scheduler", "trainer"):
            (root / name).mkdir()
            (root / name / ".metadata").touch()
        (root / "master_weights.pt").touch()
        for rank in range(4):
            (root / f"rng_rank{rank}.pt").touch()


def receipts(plan):
    root = Path(plan["source_output"]) / "training_handoff"
    for rank in range(4):
        write(root / f"cosmos_rank{rank}.json", dict(model="cosmos", rank=rank, pid=500+rank,
            phase="training_complete_cache_retained", reserved_gib=75))


def test_cosmos_checkpoint_marker_without_worker_completion_does_not_start_eval(tmp_path):
    plan = plan_at(tmp_path)
    cosmos_states(plan)
    assert not queue.cosmos_ready(plan)
    receipts(plan)
    worker = SimpleNamespace(uid=os.geteuid(), argv=("python", "--model", "cosmos"))
    with patch.object(queue, "read_process", return_value=worker):
        assert queue.cosmos_ready(plan)
        (queue.cosmos_checkpoint(plan, 1000) / "optim/.metadata").unlink()
        with pytest.raises(ValueError, match="native model/optimizer"):
            queue.cosmos_ready(plan)


def test_cosmos_rejects_checkpoint_from_another_iteration(tmp_path):
    plan = plan_at(tmp_path)
    cosmos_states(plan)
    write(queue.cosmos_checkpoint(plan, 1000) / "wm_update_complete.json",
          dict(step=1000, world_size=4, identity=dict(manifest="old-split")))
    with pytest.raises(ValueError, match="training data"):
        queue.cosmos_ready(plan)


def test_no_complete_exported_wan_checkpoint_keeps_existing_training_running(tmp_path):
    plan = plan_at(tmp_path)
    with patch.object(queue, "cosmos_ready", return_value=True), \
         patch.object(queue, "latest_wan_checkpoint", side_effect=ValueError("No complete, exported and logged Wan checkpoint is available")):
        assert queue.ready_checkpoint(plan) is None


def test_metrics_use_cosmos_four_gpus_while_wan_uses_its_other_four(tmp_path):
    plan = plan_at(tmp_path)
    evaluate = queue.evaluate_command(plan, 1000)
    assert Path(evaluate[1]).name == "wm_eval_reserved.py"
    assert evaluate[evaluate.index("--model") + 1] == "cosmos"
    assert evaluate[evaluate.index("--gpus") + 1] == "4,5,6,7"
    assert evaluate[evaluate.index("--warmup") + 1] == "3"
    write(Path(plan["source_plan"]), dict(output=plan["source_output"]))
    resume = tmp_path / "wan/step_001000"
    training = queue.wan_command(plan, resume)
    assert "--nproc_per_node=8" in training
    assert training[training.index("--accumulation") + 1] == "2"
    assert training[training.index("--steps") + 1] == "2000"
    assert training[training.index("--save-interval") + 1] == "1000"
    assert training[training.index("--resume") + 1] == str(resume)
    assert "--reserve-memory" in training
    assert queue.SAVED_STEPS == (1000, 2000)
    parallel = queue.wan_four_command(plan, resume)
    assert "--nproc_per_node=4" in parallel
    assert parallel[parallel.index("--accumulation") + 1] == "4"
    assert Path(parallel[parallel.index("--control-dir") + 1]).name == "wan_control"
    assert "--reserve-memory" in parallel


def test_evaluation_supervisor_sees_its_four_gpus_even_when_cpu_queue_is_hidden(tmp_path):
    plan = plan_at(tmp_path)
    original = dict(CUDA_VISIBLE_DEVICES="", WAN_SPATIAL_CHECKPOINT="/wan/source")
    environment = queue.evaluation_environment(plan, original, Path("/cosmos/step1000"))
    assert original["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert environment["COSMOS_WRIST_CHECKPOINT"] == "/cosmos/step1000"
    assert environment["WAN_SPATIAL_CHECKPOINT"] == "/wan/source"


def test_retry_accepts_complete_cosmos_after_the_old_workers_released_their_cache(tmp_path):
    plan = plan_at(tmp_path)
    cosmos_states(plan)
    plan["cosmos_already_released"] = True
    with patch.object(queue, "process_alive", return_value=False):
        assert queue.cosmos_ready(plan)
    with patch.object(queue, "process_alive", return_value=True), \
         pytest.raises(RuntimeError, match="GPUs to be released"):
        queue.cosmos_ready(plan)


def test_release_cosmos_pauses_only_cpu_controller_and_leaves_wan_running(tmp_path):
    plan = plan_at(tmp_path)
    process = SimpleNamespace(pid=100, started=200)
    alive = lambda record: record["pid"] != 104
    with patch.object(queue, "read_process", return_value=process), \
         patch.object(queue, "process_alive", side_effect=alive), \
         patch.object(queue, "signal_process") as send, \
         patch.object(queue, "stop_owned_tree") as stop:
        queue.release_cosmos(plan)
    send.assert_called_once_with(process, signal.SIGSTOP)
    stop.assert_not_called()
    root = Path(plan["output"])
    assert (root / "cosmos_cache_release.json").exists()
    assert not (Path(plan["source_output"]) / "training_handoff/release.json").exists()
    assert not (root / "ownership_transferred.json").exists()


def test_old_cpu_cleanup_is_paused_before_its_gpu_training_is_stopped(tmp_path):
    plan = plan_at(tmp_path)
    original = Path(plan["source_output"])
    for name in ("worker_identity.json", "primary_identity.json"):
        write(original / name, plan["original_supervisor"])
    checkpoint = tmp_path / "wan/step_001000"
    write(checkpoint / "complete.json", dict(step=1000))
    calls = []
    previous = SimpleNamespace(pid=100, started=200)
    own = SimpleNamespace(pid=os.getpid(), started=300)
    def lookup(pid):
        return previous if pid == 100 else own
    with patch.object(queue, "verify"), \
         patch.object(queue, "read_process", side_effect=lookup), \
         patch.object(queue, "signal_process", side_effect=lambda p,s:calls.append(("signal",p.pid,s))), \
         patch.object(queue, "stop_owned_tree", side_effect=lambda p,**kw:calls.append(("stop_tree",p["pid"]))), \
         patch.object(queue, "process_alive", return_value=False):
        lock = queue.take_ownership(plan, checkpoint)
        lock.close()
    assert calls == [("signal",100,signal.SIGSTOP),("stop_tree",101),
                     ("stop_tree",102),("signal",100,signal.SIGKILL)]
    assert (Path(plan["output"]) / "ownership_transferred.json").is_file()
    assert json.loads((original / "worker_identity.json").read_text())["pid"] == os.getpid()


def test_failed_companion_resumes_original_controller_before_ownership_transfer(tmp_path):
    plan = plan_at(tmp_path)
    root = Path(plan["output"])
    (root / "controller_paused.json").touch()
    process = SimpleNamespace(pid=100, started=200)
    with patch.object(queue, "read_process", return_value=process), \
         patch.object(queue, "signal_process") as send:
        queue.recover_original(plan)
        send.assert_called_once_with(process, signal.SIGCONT)
        send.reset_mock()
        (root / "ownership_transferred.json").touch()
        queue.recover_original(plan)
        send.assert_not_called()


def test_comparison_episodes_must_still_be_held_out_after_replay_mix():
    rows = [dict(trajectory_id=str(i), split_group=str(i), sha256=f"sha-{i}", split="val") for i in range(100)]
    evaluation = dict(episodes=rows)
    training = dict(episodes=[dict(row) for row in rows])
    queue.validate_holdout(evaluation, training)
    training["episodes"][0]["split"] = "train"
    with pytest.raises(ValueError, match="overlaps current training"):
        queue.validate_holdout(evaluation, training)


def test_comparison_requires_full_four_shard_same_protocol_metrics(tmp_path):
    plan = plan_at(tmp_path)
    root = Path(plan["output"])
    value = dict(model="cosmos", manifest_sha256="same-100", episodes=100, frames=800,
                 metric_shards=4, protocol="autoregressive", metric_backbones={"FID":"inception"},
                 metric_resolution=224, horizon_min=8, horizon_max=8,
                 FPS=7., LPIPS=.3, FID=70., FVD_variable_length=300., FloLPIPS=.31)
    for step in queue.SAVED_STEPS:
        write(root / f"video_metrics/cosmos_step{step}/cosmos_metrics.json", value)
    queue.summarize(plan)
    assert len(json.loads((root / "cosmos_comparison.json").read_text())["rows"]) == 2
    value["metric_shards"] = 8
    write(root / "video_metrics/cosmos_step2000/cosmos_metrics.json", value)
    with pytest.raises(ValueError, match="four-shard"):
        queue.summarize(plan)


def test_metrics_completion_requests_fresh_wan_checkpoint_before_releasing_cache(tmp_path):
    plan = plan_at(tmp_path)
    root = Path(plan["output"])
    checkpoint = root / "train4/wan/step_001234"
    write(checkpoint / "complete.json", dict(step=1234))
    calls = []
    class Child:
        def poll(self):
            return None
        def wait(self, timeout):
            request = json.loads((root / "wan_control/checkpoint_request.json").read_text())
            assert request["token"] == "this-run"
            assert (root / "wan_control/release.json").exists()
            calls.append("four-rank-cache-released")
            return 0
    def ready(_plan):
        assert (root / "wan_control/checkpoint_request.json").exists()
        assert not (root / "wan_control/release.json").exists()
        calls.append("new-checkpoint-complete")
        return checkpoint
    with patch.object(queue, "controlled_checkpoint", side_effect=ready):
        assert queue.finish_parallel_wan(plan, {}, dict(wan_child=Child())) == checkpoint
    assert calls == ["new-checkpoint-complete", "four-rank-cache-released"]


def test_request_is_observed_only_at_a_complete_update_boundary(tmp_path):
    import torch
    from wm_update_wan_controlled import CheckpointRequest
    args = SimpleNamespace(control_dir=tmp_path, control_token="this-run", steps=2000)
    class Dist:
        @staticmethod
        def get_rank(): return 0
        @staticmethod
        def broadcast(value, src): assert src == 0
    resets = []
    control = CheckpointRequest(args, 1000, "current-split", torch, Dist,
                                lambda:resets.append("begin"), device="cpu")
    control.begin_update()
    control.begin_update()
    assert args.steps == 2000
    write(tmp_path / "checkpoint_request.json", dict(token="this-run",manifest_sha256="current-split"))
    control.begin_update()
    assert args.steps == 1003
    assert len(resets) == 3
    write(tmp_path / "checkpoint_request.json", dict(token="different-run",manifest_sha256="current-split"))
    with pytest.raises(ValueError, match="does not match"):
        control.begin_update()
