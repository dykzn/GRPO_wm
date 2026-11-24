"""CPU checks for colocated generation, resumption and Cosmos validation."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_eval import completed_predictions, cosmos_probe_command, prediction_commands, prediction_slots


def arguments(root, gpus="0,7", workers=2):
    return argparse.Namespace(gpus=gpus, workers_per_gpu=workers,
        output=root, manifest=root / "manifest.json", steps=5, seed=42, warmup=3)


class LaunchTests(unittest.TestCase):
    def test_colocated_slots_cover_every_episode_once(self):
        slots = prediction_slots("0,1,2,3,4,5,6,7", 2)
        self.assertEqual(len(slots), 16)
        self.assertEqual(slots, [str(gpu) for gpu in range(8) for _ in range(2)])
        shards = [list(range(500))[rank::len(slots)] for rank in range(len(slots))]
        self.assertEqual(sorted(value for shard in shards for value in shard), list(range(500)))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_duplicate_aliases_and_invalid_worker_counts_are_rejected(self):
        for gpus, workers in [("0,00", 2), ("0,1,", 2), ("0,x", 2), ("0,1", 0)]:
            with self.assertRaises(ValueError):
                prediction_slots(gpus, workers)

    def test_wan_assignments_clear_inherited_training_ranks(self):
        with patch.dict(os.environ, {"RLINF_ENV": "/actor", "RANK": "7", "WORLD_SIZE": "8", "MASTER_PORT": "29500"}):
            commands = prediction_commands(arguments(Path("/output")), "wan", Path("/logs"))
        self.assertEqual([item[1]["CUDA_VISIBLE_DEVICES"] for item in commands], ["0", "0", "7", "7"])
        for rank, (command, env, log) in enumerate(commands):
            self.assertEqual(command[command.index("--shard-index") + 1], str(rank))
            self.assertEqual(command[command.index("--num-shards") + 1], "4")
            self.assertNotIn("RANK", env)
            self.assertNotIn("MASTER_PORT", env)
            self.assertEqual(log.name, f"wan_shard{rank}.log")

    def test_cosmos_instances_get_separate_single_rank_rendezvous(self):
        with patch.dict(os.environ, {"COSMOS_PYTHON": "/cosmos/bin/python", "WORLD_SIZE": "8"}), \
                patch("wm_eval.socket.socket") as sockets:
            sockets.return_value.__enter__.return_value.getsockname.side_effect = [
                ("127.0.0.1", port) for port in range(20000, 20004)]
            commands = prediction_commands(arguments(Path("/output")), "cosmos", Path("/logs"))
        self.assertEqual(len({env["MASTER_PORT"] for _, env, _ in commands}), 4)
        for _, env, _ in commands:
            self.assertEqual((env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"]), ("0", "0", "1"))

    def test_complete_cache_requires_both_files_and_never_counts_partials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wan").mkdir()
            manifest = {"episodes": [{"trajectory_id": name} for name in ["a", "b", "c"]]}
            (root / "wan/a.json").write_text(json.dumps({"frames": 96}))
            (root / "wan/a.npz").touch()
            (root / "wan/b.json").write_text("{}")
            (root / "wan/b.npz.tmp").touch()
            (root / "wan/c.npz").touch()
            self.assertEqual(completed_predictions(manifest, root, "wan"), {"a": {"frames": 96}})

    def test_cosmos_probe_uses_first_uncached_episode_with_original_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cosmos").mkdir()
            (root / "cosmos/a.json").write_text("{}")
            (root / "cosmos/a.npz").touch()
            manifest = {"episodes": [{"trajectory_id": key} for key in ["a", "b", "c"]]}
            with patch.dict(os.environ, {"COSMOS_PYTHON": "/cosmos/bin/python"}), \
                    patch("wm_eval.socket.socket") as sockets:
                sockets.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 21000)
                command, env, log = cosmos_probe_command(arguments(root), manifest, "7", root)
            self.assertEqual(command[command.index("--shard-index") + 1], "1")
            self.assertEqual(command[command.index("--num-shards") + 1], "3")
            self.assertEqual(command[command.index("--manifest") + 1], str(root / "manifest.json"))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual(env["WM_EVAL_WORKERS_PER_GPU"], "1")
            self.assertEqual(log.name, "cosmos_first_episode.log")

    def test_probe_is_skipped_only_when_every_episode_has_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cosmos").mkdir()
            (root / "cosmos/a.json").write_text("{}")
            (root / "cosmos/a.npz").touch()
            self.assertIsNone(cosmos_probe_command(arguments(root), {"episodes": [{"trajectory_id": "a"}]}, "0", root))


if __name__ == "__main__":
    unittest.main()
