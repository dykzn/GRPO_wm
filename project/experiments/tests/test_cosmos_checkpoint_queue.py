"""A queued multi-checkpoint evaluation must not race the existing Wan workflow."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_cosmos_checkpoints as queue


class CheckpointQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "wan"
        self.source.mkdir()
        self.output = self.root / "eval"
        self.output.mkdir()
        self.plan = {
            "output": str(self.output), "upstream": str(self.source),
            "upstream_identity": {"pid": 100, "start_ticks": 200},
            "upstream_target": 2000, "upstream_training_manifest_sha256": "train",
            "python": sys.executable, "gpus": list(map(str, range(8))),
            "seed": 42, "inference_steps": 5, "warmup_chunks": 3,
            "checkpoints": [{"step": 200, "path": "unused"}, {"step": 600, "path": "unused"}],
            "baseline_metrics": str(self.root / "baseline.json"), "protocol_note": "variable-length",
        }

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def completed_wan(self):
        path = self.source / "train8/wan/step_002000"
        self.write(self.source / "wan_training_complete.json", {
            "status": "complete", "step": 2000, "checkpoint": str(path),
        })
        self.write(path / "complete.json", {
            "step": 2000, "world_size": 8, "identity": {"manifest": "train"},
        })
        self.write(path / "runtime/wm_update.json", {"step": 2000, "manifest_sha256": "train"})
        for name in ("state/.metadata", "runtime/model-00001.safetensors", "runtime/Wan2.2_VAE.pth"):
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"state")
        return path

    def test_active_wan_and_its_followup_metrics_block_cosmos(self):
        self.write(self.source / "state.json", {"status": "running", "stage": "wan_train8"})
        with patch.object(queue, "process_alive", return_value=True):
            self.assertFalse(queue.upstream_ready(self.plan))
            self.completed_wan()
            self.write(self.source / "state.json", {"status": "running", "stage": "wan_updated_metrics"})
            self.assertFalse(queue.upstream_ready(self.plan))

    def test_success_waits_for_the_final_child_to_release(self):
        self.completed_wan()
        self.write(self.source / "state.json", {"status": "complete"})
        self.write(self.source / "active_child.json", {"pid": 300, "start_ticks": 400})
        with patch.object(queue, "process_alive", side_effect=[False, True]):
            self.assertFalse(queue.upstream_ready(self.plan))
        with patch.object(queue, "process_alive", return_value=False):
            self.assertTrue(queue.upstream_ready(self.plan))

    def test_failure_after_wan_training_does_not_discard_cosmos_evaluation(self):
        self.completed_wan()
        self.write(self.source / "state.json", {"status": "failed", "error": "Wan video metric failed"})
        with patch.object(queue, "process_alive", return_value=False):
            self.assertTrue(queue.upstream_ready(self.plan))

    def immediate_plan(self):
        self.plan.update(start_condition="stopped_predecessor", boot_id="local-boot",
                         stop_request=str(self.source / "stop_requested.json"))
        request = {
            "request": "user_requested_stop_wan_start_cosmos", "boot_id": "local-boot",
            "upstream_output": str(self.source), "upstream_identity": self.plan["upstream_identity"],
            "tracked": [self.plan["upstream_identity"], {"pid": 300, "start_ticks": 400}],
        }
        self.write(self.source / "stop_requested.json", request)
        self.write(self.source / "state.json", {"status": "failed", "error": "Signal 15"})
        return request

    def test_explicit_stop_allows_cosmos_without_a_final_wan_checkpoint(self):
        self.immediate_plan()
        self.assertFalse((self.source / "wan_training_complete.json").exists())
        with patch.object(queue, "process_alive", return_value=False):
            self.assertTrue(queue.upstream_ready(self.plan))

    def test_immediate_evaluation_waits_for_recorded_orphan_workers(self):
        self.immediate_plan()
        with patch.object(queue, "process_alive", side_effect=[False, True]):
            self.assertFalse(queue.upstream_ready(self.plan))

    def test_immediate_evaluation_rejects_wrong_node_and_empty_release_record(self):
        request = self.immediate_plan()
        for override in ({"boot_id": "other-node"}, {"tracked": []}, {"request": "unknown"}):
            with self.subTest(override=override):
                self.write(self.source / "stop_requested.json", {**request, **override})
                with self.assertRaisesRegex(RuntimeError, "matching explicit Wan stop"):
                    queue.upstream_ready(self.plan)

    def test_failed_training_and_incomplete_or_wrong_split_export_are_rejected(self):
        self.write(self.source / "state.json", {"status": "failed", "error": "OOM"})
        with patch.object(queue, "process_alive", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "before successful training"):
                queue.upstream_ready(self.plan)
            path = self.completed_wan()
            self.write(path / "runtime/wm_update.json", {"step": 2000, "manifest_sha256": "other"})
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                queue.upstream_ready(self.plan)
            self.write(path / "runtime/wm_update.json", {"step": 2000, "manifest_sha256": "train"})
            (path / "runtime/model-00001.safetensors").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                queue.upstream_ready(self.plan)

    def test_native_dcp_empty_nonzero_rank_shards_are_valid_and_missing_ones_are_not(self):
        path = self.root / "iter_000000200"
        self.write(path / "wm_update_complete.json", {
            "step": 200, "world_size": 4, "identity": {"model": "cosmos", "manifest": "train"},
        })
        (path / "model").mkdir()
        (path / "model/.metadata").write_bytes(b"metadata")
        for rank in range(4):
            (path / f"model/__{rank}_0.distcp").write_bytes(b"weights" if rank == 0 else b"")
        complete, rows = queue.inspect_checkpoint(path, 200, "train")
        self.assertEqual(complete["step"], 200)
        self.assertEqual(len(rows), 5)
        packaged = queue.package_checkpoint(self.output, {"step": 200, "path": str(path)})
        manifest = json.loads((packaged.parent / "checkpoint_manifest.json").read_text())
        self.assertEqual(manifest["file_count"], 5)
        self.assertEqual((packaged / "model/__0_0.distcp").read_bytes(), b"weights")
        (path / "model/__3_0.distcp").unlink()
        with self.assertRaises(FileNotFoundError):
            queue.inspect_checkpoint(path, 200, "train")

    def metrics(self, **overrides):
        return {
            "model": "cosmos", "view": "wrist", "manifest_sha256": "heldout",
            "episodes": 100, "frames": 14288, "protocol": "variable-length",
            "metric_backbones": {"FVD": "I3D-400"}, "metric_resolution": 224,
            "native_resolution": 224, **{key: 1.0 for key in queue.METRICS}, **overrides,
        }

    def test_partial_and_complete_summaries_compare_quality_but_not_cross_node_fps(self):
        self.write(Path(self.plan["baseline_metrics"]), self.metrics(FPS=6))
        self.write(self.output / "step_000200/cosmos_metrics.json", self.metrics(FPS=8, LPIPS=0.8))
        result = queue.write_summary(self.plan)
        self.assertEqual(result["completed_checkpoints"], 1)
        self.assertFalse(result["reference_fps_comparable"])
        self.assertNotIn("FPS", result["changes_vs_baseline"]["200"])
        self.assertAlmostEqual(result["changes_vs_baseline"]["200"]["LPIPS"], -0.2)
        with self.assertRaises(ValueError):
            queue.write_summary(self.plan, final=True)
        self.write(self.output / "step_000600/cosmos_metrics.json", self.metrics())
        self.assertEqual(queue.write_summary(self.plan, final=True)["completed_checkpoints"], 2)

    def test_mixed_split_or_backbone_and_nonfinite_metrics_are_rejected(self):
        self.write(Path(self.plan["baseline_metrics"]), self.metrics())
        for override in ({"manifest_sha256": "other"}, {"metric_backbones": {}},
                         {"FID": float("nan")}, {"FPS": 0}):
            with self.subTest(override=override):
                self.write(self.output / "step_000200/cosmos_metrics.json", self.metrics(**override))
                with self.assertRaises(ValueError):
                    queue.write_summary(self.plan)

    def test_comparison_labels_historical_2000_and_does_not_treat_it_as_a_new_evaluation(self):
        self.write(Path(self.plan["baseline_metrics"]), self.metrics())
        for step in (200, 600):
            self.write(self.output / f"step_{step:06d}/cosmos_metrics.json", self.metrics(FID=0.9))
        historical = self.root / "previous_2000.json"
        self.write(historical, self.metrics(FID=0.8))
        self.plan.update(historical_metrics=[{"step": 2000, "path": str(historical)}],
                         omitted_checkpoints=[2000])
        result = queue.write_comparison(self.plan)
        self.assertEqual([row["step"] for row in result["rows"]], [0, 200, 600, 2000])
        self.assertEqual(result["rows"][-1]["source"], "historical_reference")
        self.assertEqual(result["omitted_checkpoints"], [2000])
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["completed_checkpoints"], 2)
        self.assertNotIn("FPS", result["changes_vs_baseline"]["update_2000_historical"])

    def test_adopted_evaluator_requires_process_exit_and_its_success_marker(self):
        self.plan["inflight_evaluation"] = {"pid": 300, "start_ticks": 400, "step": 1000}
        log = self.output / "logs/step_001000.log"
        log.parent.mkdir()
        log.write_text("scoring is still in progress")
        with patch.object(queue, "process_alive", return_value=True):
            self.assertFalse(queue.inflight_complete(self.plan))
        with patch.object(queue, "process_alive", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "before successful metric completion"):
                queue.inflight_complete(self.plan)
            self.write(self.output / "step_001000/cosmos_metrics.json", self.metrics())
            log.write_text(f"Complete: {self.output / 'step_001000/metrics.csv'}\n")
            self.assertTrue(queue.inflight_complete(self.plan))

    def test_finish_uses_existing_results_and_never_launches_another_gpu_evaluator(self):
        self.plan.update(finish_after_step=1000, replaces_identity={"pid": 10, "start_ticks": 20},
                         inflight_evaluation={"pid": 30, "start_ticks": 40, "step": 1000},
                         checkpoints=[{"step": step} for step in (200, 600, 1000)])
        self.write(self.output / "identity.json", self.plan["replaces_identity"])
        self.write(Path(self.plan["baseline_metrics"]), self.metrics())
        for step in (200, 600, 1000):
            self.write(self.output / f"step_{step:06d}/cosmos_metrics.json", self.metrics())
        (self.output / "logs").mkdir()
        (self.output / "logs/step_001000.log").write_text(f"Complete: {self.output / 'step_001000/metrics.csv'}\n")
        with (patch.object(queue, "verify"), patch.object(queue, "process_alive", return_value=False),
              patch.object(queue.subprocess, "Popen") as popen):
            queue.finish(self.plan)
        popen.assert_not_called()
        self.assertEqual(json.loads((self.output / "state.json").read_text())["status"], "complete")
        self.assertEqual((self.output / "exit_code").read_text(), "0\n")

    def test_command_uses_only_cosmos_on_all_eight_gpus(self):
        command = queue.evaluation_command(self.plan, self.output)
        for flag, value in (("--model", "cosmos"), ("--gpus", "0,1,2,3,4,5,6,7"),
                            ("--steps", "5"), ("--seed", "42"), ("--workers-per-gpu", "1")):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_gpu_idle_requires_no_compute_process_even_if_memory_is_low(self):
        usage = {device: 0 for device in self.plan["gpus"]}
        with (patch.object(queue, "gpu_usage", return_value=usage),
              patch.object(queue, "gpu_compute_pids", side_effect=[{"0": [123]}, {}]),
              patch.object(queue.time, "sleep") as sleep):
            queue.wait_idle(self.plan, 200)
        sleep.assert_called_once_with(2)

    def test_explicit_shared_mode_admits_other_users_only_with_enough_free_memory(self):
        self.plan.update(allow_shared_gpus=True, minimum_free_mib=32768,
                         gpu_total_mib={device: 81920 for device in self.plan["gpus"]})
        busy = {device: 60000 for device in self.plan["gpus"]}
        available = {device: 20600 for device in self.plan["gpus"]}
        with (patch.object(queue, "gpu_usage", side_effect=[busy, available]),
              patch.object(queue, "gpu_compute_pids", return_value={"0": [123]}),
              patch.object(queue.time, "sleep") as sleep):
            queue.wait_idle(self.plan, 200)
        sleep.assert_called_once_with(2)

    def test_training_failure_records_failure_without_starting_a_gpu_job(self):
        with (patch.object(queue, "verify"),
              patch.object(queue, "upstream_ready", side_effect=RuntimeError("Wan training failed")),
              patch.object(queue.subprocess, "Popen") as popen):
            with self.assertRaisesRegex(RuntimeError, "Wan training failed"):
                queue.run(self.plan)
        popen.assert_not_called()
        self.assertEqual(json.loads((self.output / "state.json").read_text())["status"], "failed")
        self.assertEqual((self.output / "exit_code").read_text(), "1\n")


if __name__ == "__main__":
    unittest.main()
