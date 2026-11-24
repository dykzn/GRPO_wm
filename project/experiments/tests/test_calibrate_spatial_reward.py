"""Check episode-level threshold selection and held-out scoring semantics."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_spatial_reward import (
    analyze,
    candidates,
    choose_threshold,
    confusion,
    episode_scores,
    read_primary,
    split_episodes,
)


class CalibrationTests(unittest.TestCase):
    def test_rgb_conversion_matches_training_space_and_keeps_all_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.hdf5"
            images = np.zeros((3, 256, 256, 3), np.uint8)
            images[:, 2, 3, 0] = [70, 80, 90]
            with h5py.File(path, "w") as stream:
                dataset = stream.create_dataset(
                    "primary_images_jpeg",
                    (3,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                for i, image in enumerate(images):
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG")
                    dataset[i] = np.frombuffer(buffer.getvalue(), np.uint8)
            row = {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "observations": 3,
                "source_camera_space": "cosmos_model_flipud",
            }
            np.testing.assert_array_equal(read_primary(row), images[:, :, ::-1])
            row["source_camera_space"] = "rlinf_runtime_rot180"
            np.testing.assert_array_equal(read_primary(row), images)

    def test_success_is_any_future_frame_and_kir_excludes_its_reset(self):
        self.assertEqual(
            episode_scores([1, 0.9, 0.6, 0.2, 0.4], 2),
            tuple(map(float, np.array([0.9, 0.4, 0.4], np.float32))),
        )
        self.assertEqual(
            episode_scores([1, 0.2, 0.3, 0.4], 0),
            tuple(map(float, np.array([0.4, 0.4, 0.4], np.float32))),
        )

    def test_exact_threshold_matches_torch_float32_and_is_inclusive(self):
        score = np.float32(0.7)
        result = confusion([True, False], [float(score), float(score)], 0.7)
        self.assertEqual((result["tp"], result["fp"]), (1, 1))
        above = np.nextafter(score, np.float32(np.inf))
        result = confusion([True, False], [float(score), float(score)], float(above))
        self.assertEqual((result["fn"], result["tn"]), (1, 1))
        self.assertIn(above, candidates([score]))

    def test_invalid_or_non_probability_cache_is_rejected(self):
        for values, start in [
            ([0, np.nan], 0),
            ([0, 1.1], 0),
            ([0, 1], 1),
            (np.ones((2, 2)), 0),
            ([0, -0.1], 0),
        ]:
            with self.assertRaises(ValueError):
                episode_scores(values, start)

    def test_selection_prefers_no_false_positive_then_recall(self):
        labels = [True, True, False, False]
        scores = [[0.9, 0.9], [0.8, 0.7], [0.6, 0.4], [0.2, 0.2]]
        result = choose_threshold(labels, scores, max_fpr=0)
        self.assertTrue(result["budget_feasible"])
        self.assertAlmostEqual(result["selected"]["threshold"], 0.605)
        self.assertEqual(result["selected"]["mean_recall"], 1)

    def test_saturated_false_positive_makes_budget_infeasible(self):
        result = choose_threshold([True, False], [[1, 1], [1, 1]], max_fpr=0)
        self.assertFalse(result["budget_feasible"])
        self.assertEqual(result["selected"]["worst_fpr"], 1)

    def test_split_is_stratified_repeatable_and_never_splits_frames(self):
        rows = [
            {"episode_index": t * 10 + i, "task_id": t, "success": i < 5}
            for t in range(2)
            for i in range(10)
        ]
        split = split_episodes(rows, 42)
        self.assertEqual(split, split_episodes(rows, 42))
        self.assertEqual(len(split), len(rows))
        for task in (0, 1):
            for outcome in (False, True):
                self.assertEqual(
                    sum(
                        split[r["episode_index"]] == "validation"
                        for r in rows
                        if r["task_id"] == task and r["success"] == outcome
                    ),
                    1,
                )

    def test_validation_scores_cannot_change_selected_threshold(self):
        rows = []
        for index, (success, score, split) in enumerate(
            [
                (True, 0.9, "calibration"),
                (False, 0.6, "calibration"),
                (True, 0.1, "validation"),
                (False, 1.0, "validation"),
            ]
        ):
            rows.append(
                {
                    "episode_index": index,
                    "task_id": 0,
                    "success": success,
                    "split": split,
                    "frames": 3,
                    "initial_score": score,
                    "kir_score": score,
                    "final_frame_score": score,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            before = analyze(rows, output, {}, 0)
            rows[2].update(initial_score=1.0, kir_score=1.0, final_frame_score=1.0)
            after = analyze(rows, output, {}, 0)
            self.assertEqual(before["selection"], after["selection"])
            self.assertEqual(
                before["selected_threshold_metrics"]["validation"]["initial"]["fn"], 1
            )
            self.assertEqual(
                after["selected_threshold_metrics"]["validation"]["initial"]["fn"], 0
            )
            self.assertEqual(
                json.loads((output / "report.json").read_text())["episodes"], 4
            )

    def test_zero_observed_false_positives_does_not_imply_zero_uncertainty(self):
        result = confusion([False] * 14, [0] * 14, 0.5)
        self.assertEqual(result["false_positive_rate"], 0)
        self.assertGreater(result["fpr_wilson_95_high"], 0.2)


if __name__ == "__main__":
    unittest.main()
