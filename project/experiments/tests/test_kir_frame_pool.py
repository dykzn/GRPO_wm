"""CPU tests for dual-view time alignment and KIR initialization boundaries."""

import io
import errno
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kir_frame_pool import FramePool, build_grasp_pool, build_pool, gripper_close_events, retry_io


def write_episode(root, *, count=26, episode=0, task=0, success=False):
    path = root / f"episode_data--suite=libero_spatial--2025--task={task}--ep={episode}--success={success}--regen_demo.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs["task_description"] = f"task {task}"
        file.attrs["success"] = success
        actions = np.repeat(np.arange(count, dtype=np.float32)[:, None], 7, axis=1)
        file["actions"] = actions
        file["proprio"] = np.zeros((count, 9), np.float32)
        for name, offset in [("primary_images_jpeg", 0), ("wrist_images_jpeg", 64)]:
            stream = file.create_dataset(name, (count,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for t in range(count):
                rgb = np.zeros((4, 4, 3), np.uint8)
                rgb[..., 0] = t + offset
                rgb[:, 0, 1] = 255
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="PNG")
                stream[t] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return path


class FramePoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "pool"

    def tearDown(self):
        self.temporary.cleanup()

    def pool(self, **kwargs):
        build_pool(self.source, self.output, workers=1, **kwargs)
        return FramePool(self.output)

    def test_frame_table_keeps_tail_but_never_invents_a_successor(self):
        write_episode(self.source)
        pool = self.pool()
        np.testing.assert_array_equal(pool.indices["frame_t"], np.arange(26))
        np.testing.assert_array_equal(pool.indices["action_t"], [*range(25), -1])
        np.testing.assert_array_equal(pool.indices["next_t"], [*range(1, 26), -1])
        np.testing.assert_array_equal(pool.indices["start_t"], [0, 8, 16])
        self.assertEqual(pool.manifest["summary"]["valid_transitions"], 25)
        self.assertEqual(pool.manifest["summary"]["unmatched_final_actions"], 1)

    def test_kir_anchors_histories_and_actions_share_exact_time(self):
        write_episode(self.source)
        sample = self.pool().read_start(2)
        self.assertEqual(sample["frame_index"], 16)
        self.assertEqual(sample["history_indices"], {"wan_primary": [13, 14, 15, 16], "cosmos_v3_wrist": [0, 14, 15, 16]})
        np.testing.assert_array_equal(sample["wan"]["memory"][:, 0, 0, 0], [13, 14, 15, 16])
        np.testing.assert_array_equal(sample["cosmos_v3"]["wrist_memory"][:, 0, 0, 0], [64, 78, 79, 80])
        self.assertEqual(sample["wan"]["reference"][0, 0, 0], 0)
        self.assertEqual(sample["cosmos_v3"]["primary_reference"][0, 0, 0], 0)
        np.testing.assert_array_equal(sample["wan"]["context_actions"][1:, 0], [12, 13, 14, 15])
        np.testing.assert_array_equal(sample["cosmos_v3"]["context_actions"][2:, 0], [13, 14, 15])
        self.assertEqual(sample["observation"]["main_images"][0, 0, 0], 16)
        self.assertEqual(sample["observation"]["wrist_images"][0, 0, 0], 80)
        # flipud source -> rot180 runtime is a horizontal flip, once per view.
        self.assertTrue((sample["observation"]["main_images"][:, -1, 1] == 255).all())
        self.assertTrue((sample["observation"]["wrist_images"][:, -1, 1] == 255).all())
        self.assertNotIn("audit_future", sample)
        self.assertNotIn("actions", sample)

    def test_initial_repeat_is_conditioning_only_future_is_real(self):
        write_episode(self.source)
        sample = self.pool().read_start(0, include_future=True)
        np.testing.assert_array_equal(sample["wan"]["memory"][:, 0, 0, 0], [0] * 4)
        np.testing.assert_array_equal(sample["wan"]["context_actions"][:, -1], [-1] * 5)
        np.testing.assert_array_equal(sample["audit_future"]["primary"][:, 0, 0, 0], range(1, 9))
        np.testing.assert_array_equal(sample["audit_future"]["actions"][:, 0], range(8))

    def test_grpo_groups_share_one_start_and_obey_explicit_kir_fraction(self):
        write_episode(self.source, task=0)
        write_episode(self.source, task=1, episode=1, success=True)
        pool = self.pool()
        for fraction in (0.0, 1.0):
            groups = pool.sample_groups(100, kir_fraction=fraction, group_size=8, seed=42)
            self.assertEqual(groups.shape, (100, 8))
            self.assertTrue((groups == groups[:, :1]).all())
            self.assertTrue((pool.indices["start_kind"][groups] == int(fraction)).all())
            tasks = {pool.episodes[int(pool.indices["start_episode"][i])]["task_id"] for i in groups[:, 0]}
            self.assertEqual(tasks, {0, 1})
        with self.assertRaises(ValueError):
            pool.sample_groups(1, kir_fraction=1.1)

    def test_duplicates_are_excluded_without_merging_distinct_trajectories(self):
        path = write_episode(self.source)
        shutil.copyfile(path, self.source / path.name.replace("ep=0", "ep=2"))
        write_episode(self.source, episode=1, task=1)
        pool = self.pool()
        self.assertEqual(pool.manifest["summary"]["episodes"], 2)
        self.assertEqual(len(pool.manifest["excluded"]), 1)

    def test_changed_sources_and_changed_inventory_are_rejected(self):
        path = write_episode(self.source)
        pool = self.pool()
        with h5py.File(path, "a") as file:
            file["actions"][0] = -42
        with self.assertRaisesRegex(ValueError, "Source changed"):
            pool.read_start(0)
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.pool()

    def test_corrupt_rgb_is_not_silently_indexed(self):
        path = write_episode(self.source)
        with h5py.File(path, "a") as file:
            file["wrist_images_jpeg"][10] = np.array([1, 2, 3], np.uint8)
        with self.assertRaisesRegex(ValueError, "failed verification"):
            self.pool()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_source_with_n_minus_one_actions_has_no_unmatched_action(self):
        path = write_episode(self.source)
        with h5py.File(path, "a") as file:
            actions = file["actions"][:-1]
            del file["actions"]
            file["actions"] = actions
        pool = self.pool()
        self.assertEqual(pool.manifest["summary"]["unmatched_final_actions"], 0)
        self.assertEqual(pool.manifest["summary"]["valid_transitions"], 25)

    def test_excluded_duplicate_changes_are_detected_on_reuse(self):
        path = write_episode(self.source)
        duplicate = self.source / path.name.replace("ep=0", "ep=2")
        shutil.copyfile(path, duplicate)
        pool = self.pool()
        with h5py.File(duplicate, "a") as file:
            file["actions"][0] = 123
        with self.assertRaisesRegex(ValueError, "Source changed"):
            pool.check_sources()

    def test_manifest_and_indices_are_checked_and_rebuild_is_idempotent(self):
        write_episode(self.source)
        pool = self.pool(stride=1)
        self.assertEqual(pool.indices["start_t"][1], 4)
        before = (self.output / "manifest.json").read_bytes()
        self.pool(stride=1)
        self.assertEqual(before, (self.output / "manifest.json").read_bytes())
        manifest = json.loads(before)
        manifest["stride"] = 2
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checksum"):
            FramePool(self.output)

    def test_storage_errors_are_retried_but_corrupt_images_fail_immediately(self):
        read = Mock(side_effect=[OSError(errno.EIO, "NFS read failed"), 42])
        with patch("kir_frame_pool.time.sleep"):
            self.assertEqual(retry_io(read), 42)
        self.assertEqual(read.call_count, 2)
        read = Mock(side_effect=OSError("cannot identify image file"))
        with self.assertRaises(OSError):
            retry_io(read)
        self.assertEqual(read.call_count, 1)

    def test_close_event_requires_stable_open_and_closed_commands(self):
        gripper = -np.ones(70)
        gripper[12] = 1  # A one-frame twitch is not the first valid grasp.
        gripper[35:48] = 1
        gripper[55:] = 1
        raw, stable = gripper_close_events(gripper)
        self.assertEqual(raw, [12, 35, 55])
        self.assertEqual(stable, [35, 55])
        self.assertEqual(gripper_close_events(np.ones(20)), ([], []))
        self.assertEqual(gripper_close_events(-np.ones(20)), ([], []))
        with self.assertRaisesRegex(ValueError, "signed"):
            gripper_close_events(np.array([-1, 2]))
        with self.assertRaises(ValueError):
            gripper_close_events(np.array([-1, np.nan, 1]))

    def test_first_close_pre16_uses_exact_non_grid_frame_and_past_only_context(self):
        path = write_episode(self.source, count=70)
        with h5py.File(path, "a") as file:
            gripper = -np.ones(70)
            gripper[12] = 1
            gripper[35:48] = 1
            gripper[55:] = 1
            file["actions"][:, 6] = gripper
        base = self.pool()
        before = (self.output / "manifest.json").read_bytes()
        destination = self.root / "grasp"
        build_grasp_pool(self.output, destination)
        pool = FramePool(destination)
        np.testing.assert_array_equal(pool.indices["start_t"], [0, 19])
        np.testing.assert_array_equal(pool.indices["start_event_t"], [-1, 35])
        self.assertEqual(pool.manifest["selection_events"][0]["stable_close_events"], [35, 55])
        self.assertEqual(pool.manifest["summary"]["kir_candidates"], 1)
        np.testing.assert_array_equal(pool.indices["frame_t"], base.indices["frame_t"])
        self.assertEqual(before, (self.output / "manifest.json").read_bytes())
        sample = pool.read_start(1)
        self.assertEqual(sample["frame_index"], 19)
        np.testing.assert_array_equal(sample["wan"]["memory"][:, 0, 0, 0], [16, 17, 18, 19])
        np.testing.assert_array_equal(sample["cosmos_v3"]["wrist_memory"][:, 0, 0, 0], [64, 81, 82, 83])
        np.testing.assert_array_equal(sample["wan"]["context_actions"][1:, 0], [15, 16, 17, 18])
        self.assertEqual(sample["observation"]["wrist_images"][0, 0, 0], 83)
        self.assertNotIn("audit_future", sample)
        self.assertNotIn("start_event_t", sample)
        self.assertNotIn("selection_events", sample)
        groups = pool.sample_groups(10, kir_fraction=1, seed=42)
        self.assertTrue((groups == 1).all())

    def test_early_first_close_is_not_clamped_or_replaced_by_a_later_attempt(self):
        path = write_episode(self.source, count=70)
        with h5py.File(path, "a") as file:
            gripper = -np.ones(70)
            gripper[18:30] = 1
            gripper[45:] = 1
            file["actions"][:, 6] = gripper
        self.pool()
        destination = self.root / "grasp"
        manifest = build_grasp_pool(self.output, destination)
        np.testing.assert_array_equal(FramePool(destination).indices["start_t"], [0])
        event = manifest["selection_events"][0]
        self.assertEqual(event["first_close_action_t"], 18)
        self.assertEqual(event["requested_start_t"], 2)
        self.assertEqual(event["status"], "insufficient_history")

    def test_missing_close_keeps_initial_and_rejects_requesting_absent_kir_starts(self):
        path = write_episode(self.source)
        with h5py.File(path, "a") as file:
            file["actions"][:, 6] = -1
        self.pool()
        destination = self.root / "grasp"
        manifest = build_grasp_pool(self.output, destination)
        self.assertEqual(manifest["selection_summary"]["skip_reasons"]["no_stable_close"], 1)
        pool = FramePool(destination)
        self.assertEqual(len(pool), 1)
        with self.assertRaisesRegex(ValueError, "No available starts"):
            pool.sample_groups(1, kir_fraction=1)

    def test_grasp_start_still_requires_eight_real_future_transitions(self):
        path = write_episode(self.source, count=55)
        with h5py.File(path, "a") as file:
            gripper = -np.ones(55)
            gripper[50:] = 1
            file["actions"][:, 6] = gripper
        self.pool()
        manifest = build_grasp_pool(self.output, self.root / "grasp", backtrack=1)
        self.assertEqual(manifest["selection_events"][0]["status"], "insufficient_future")

    def test_grasp_pool_reuse_is_immutable_and_detects_changed_sources(self):
        path = write_episode(self.source, count=55)
        with h5py.File(path, "a") as file:
            gripper = -np.ones(55)
            gripper[35:] = 1
            file["actions"][:, 6] = gripper
        self.pool()
        destination = self.root / "grasp"
        build_grasp_pool(self.output, destination)
        before = (destination / "manifest.json").read_bytes()
        build_grasp_pool(self.output, destination)
        self.assertEqual(before, (destination / "manifest.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "different settings"):
            build_grasp_pool(self.output, destination, backtrack=8)
        with self.assertRaisesRegex(ValueError, "separate output"):
            build_grasp_pool(self.output, self.output)
        with h5py.File(path, "a") as file:
            file["actions"][0, 6] = 1
        with self.assertRaisesRegex(ValueError, "Source changed"):
            build_grasp_pool(self.output, destination)


if __name__ == "__main__":
    unittest.main()
