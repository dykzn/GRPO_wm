"""CPU tests for real-video alignment, variable horizons, and metric statistics."""

import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_eval_data import FeatureMoments, frechet_distance, horizon_for, prepare_manifest, read_episode
from wm_eval_metrics import check_pair
from wm_eval_predict import DirectCosmosClient


def write_episode(root, observations, task=0, episode=1, action_count=None):
    name = f"episode_data--suite=libero_spatial--2025--task={task}--ep={episode}--success=True--regen_demo.hdf5"
    path = root / name
    with h5py.File(path, "w") as file:
        file.attrs["task_description"] = f"task {task}"
        file["actions"] = np.full((action_count or observations, 7), episode / 100, np.float32)
        file["proprio"] = np.zeros((observations, 9), np.float32)
        for key, offset in [("primary_images_jpeg", 0), ("wrist_images_jpeg", 32)]:
            stream = file.create_dataset(key, (observations,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for time in range(observations):
                image = np.zeros((4, 4, 3), np.uint8)
                image[..., 0] = time + offset
                image[:, 0, 1] = 255
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                stream[time] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return path


class EpisodeTests(unittest.TestCase):
    def test_horizon_uses_successor_observations_and_complete_chunks(self):
        self.assertEqual(horizon_for(98, 98), 96)
        self.assertEqual(horizon_for(129, 129), 128)
        self.assertEqual(horizon_for(128, 128), 120)
        self.assertEqual(horizon_for(50, 16), 16)
        self.assertEqual(horizon_for(1, 1), 0)

    def test_variable_episodes_are_not_padded_or_truncated_to_common_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, 34, episode=1)
            write_episode(root, 58, task=1, episode=2)
            manifest = prepare_manifest(root, root / "manifest.json")
            self.assertEqual([row["horizon"] for row in manifest["episodes"]], [32, 56])
            self.assertEqual(manifest["summary"]["future_frames"], 88)
            self.assertIn("unknown", manifest["split_status"])

    def test_decode_preserves_time_and_maps_both_cameras_to_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, 34)
            row = prepare_manifest(root, root / "manifest.json")["episodes"][0]
            primary, wrist, actions = read_episode(row)
            self.assertEqual(primary.shape, (33, 4, 4, 3))
            self.assertEqual(actions.shape, (32, 7))
            self.assertTrue(np.all(primary[1, :, :, 0] == 1))
            self.assertTrue(np.all(wrist[1, :, :, 0] == 33))
            self.assertTrue(np.all(primary[:, :, -1, 1] == 255))
            self.assertTrue(np.all(wrist[:, :, -1, 1] == 255))

    def test_episode_decode_avoids_a_failing_native_file_driver(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, 34)
            row = prepare_manifest(root, root / "manifest.json")["episodes"][0]
            native_open = h5py.File

            def without_disk_driver(source, *args, **kwargs):
                if not isinstance(source, io.BytesIO):
                    raise OSError(5, "simulated native HDF5 file-driver EIO")
                return native_open(source, *args, **kwargs)

            with patch("wm_eval_data.h5py.File", side_effect=without_disk_driver):
                primary, wrist, actions = read_episode(row)
            self.assertEqual(primary.shape, (33, 4, 4, 3))
            self.assertEqual(wrist.shape, primary.shape)
            self.assertEqual(actions.shape, (32, 7))

    def test_formal_episode_checksum_is_verified_before_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, 34)
            row = prepare_manifest(root, root / "manifest.json")["episodes"][0]
            row["sha256"] = "wrong-checksum"
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                read_episode(row)

    def test_duplicate_content_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_episode(root, 34)
            shutil.copyfile(path, root / path.name.replace("ep=1", "ep=2"))
            manifest = prepare_manifest(root, root / "manifest.json")
            self.assertEqual(manifest["summary"]["episodes"], 1)
            self.assertEqual(manifest["excluded"][0]["reason"], "duplicate trajectory")

    def test_smoke_selection_round_robins_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, 34, task=0, episode=1)
            write_episode(root, 34, task=0, episode=2)
            write_episode(root, 34, task=1, episode=3)
            manifest = prepare_manifest(root, root / "manifest.json", limit=2)
            self.assertEqual([row["task_id"] for row in manifest["episodes"]], [0, 1])

    def test_manifest_refuses_to_change_an_existing_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_episode(root, 34)
            output = root / "manifest.json"
            prepare_manifest(root, output)
            write_episode(root, 50, episode=2)
            with self.assertRaisesRegex(ValueError, "Existing manifest differs"):
                prepare_manifest(root, output)

    def test_prediction_pair_rejects_reference_frames_or_wrong_resolution(self):
        valid = np.zeros((32, 224, 224, 3), np.uint8)
        check_pair(valid, valid.copy(), 32)
        for bad in [valid[:-1], valid.astype(np.float32), np.zeros((32, 256, 256, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                check_pair(bad, valid, 32)


class StatisticsTests(unittest.TestCase):
    def test_streaming_statistics_equal_full_batch_covariance(self):
        values = np.random.default_rng(42).normal(size=(31, 5))
        moments = FeatureMoments()
        for part in np.array_split(values, 7):
            moments.add(part)
        mean, covariance = moments.statistics()
        np.testing.assert_allclose(mean, values.mean(0), atol=1e-12)
        np.testing.assert_allclose(covariance, np.cov(values, rowvar=False), atol=1e-12)

    def test_frechet_identity_and_translation_for_singular_small_samples(self):
        values = np.random.default_rng(123).normal(size=(3, 6))
        real, same, shifted = FeatureMoments(), FeatureMoments(), FeatureMoments()
        real.add(values)
        same.add(values.copy())
        shifted.add(values + 2)
        self.assertAlmostEqual(frechet_distance(real, same), 0, places=5)
        self.assertAlmostEqual(frechet_distance(real, shifted), 24, places=5)

    def test_frechet_matches_diagonal_gaussian_closed_form(self):
        # Four axis-aligned points give covariance diag(2/3, 8/3).
        values = np.array([[1, 0], [-1, 0], [0, 2], [0, -2]], float)
        real, pred = FeatureMoments(), FeatureMoments()
        real.add(values)
        pred.add(values * 2)
        self.assertAlmostEqual(frechet_distance(real, pred), np.trace(np.cov(values, rowvar=False)))

    def test_one_video_and_nonfinite_features_are_rejected(self):
        moments = FeatureMoments()
        moments.add(np.ones((1, 4)))
        with self.assertRaises(ValueError):
            moments.statistics()
        with self.assertRaises(ValueError):
            moments.add(np.array([[np.nan] * 4]))


class BridgeTests(unittest.TestCase):
    def test_direct_client_preserves_the_production_temporal_request(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/WorldPilot"))

        class Policy:
            def predict_action(self, **message):
                self.message = message
                return {"future_primary_clip": np.zeros((1, 8, 4, 4, 3), np.uint8)}

        policy = Policy()
        client = DirectCosmosClient(policy)
        reference = np.zeros((1, 4, 4, 3), np.uint8)
        memory = np.zeros((1, 4, 4, 4, 3), np.uint8)
        actions = np.zeros((1, 8, 7), np.float32)
        context = np.zeros((1, 5, 7), np.float32)
        context[..., -1] = -1
        result = client.query_temporal_future_batch(reference, memory, actions,
            langs=["task"], context_actions=context, guidance=1.0,
            history_images_are_model_space=np.ones((1, 5), bool))
        self.assertEqual(result["future_primary_clip"].shape, (1, 8, 4, 4, 3))
        self.assertEqual(policy.message["mode"], "temporal_future_batch")
        np.testing.assert_array_equal(policy.message["context_actions"], context)
        self.assertTrue(policy.message["history_images_are_model_space"].all())


if __name__ == "__main__":
    unittest.main()
