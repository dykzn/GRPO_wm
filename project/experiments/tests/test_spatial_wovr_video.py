"""CPU checks for sparse video collection and worker-importable overlays."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from spatial_wovr_launch import prepare_runtime
from spatial_wovr_video import SampledVideoMixin

NATIVE = Path("/data41/qianb2608/wan/RLinf_opd/rlinf/envs/wrappers/record_video.py")


class VideoConfig(dict):
    __getattr__ = dict.__getitem__


class CountingEnv:
    seed = 7
    num_envs = 2

    def __init__(self):
        self.metadata = {"render_fps": 10}
        self.captures = 0
        self.closes = 0

    def capture_image(self):
        self.captures += 1
        return np.zeros((2, 2, 2, 3), dtype=np.uint8)

    def reset(self):
        return {"main_images": None}, {}

    def step(self, action):
        return {"main_images": None}, 0, False, False, {}

    def chunk_step(self, action):
        return (
            [{"main_images": None}, {"main_images": None}],
            np.zeros((2, 2)),
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 2), dtype=bool),
            [{}, {}],
        )

    def close(self):
        self.closes += 1


def dependency_stubs():
    """Use real native control flow without renderer, Torch, or an encoder."""
    gym = types.ModuleType("gymnasium")
    gym.Env = type("Env", (), {})

    class Wrapper:
        def close(self):
            return self.env.close()

    gym.Wrapper = Wrapper
    imageio = types.ModuleType("imageio")
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    utils = types.ModuleType("rlinf.envs.utils")
    utils.tile_images = lambda images, nrows: images[0]
    utils.put_info_on_image = lambda image, info: image
    return {
        "gymnasium": gym,
        "imageio": imageio,
        "torch": torch,
        "rlinf.envs.utils": utils,
    }


class SparseVideoTests(unittest.TestCase):
    def setUp(self):
        if not NATIVE.is_file():
            self.skipTest("Native RecordVideo is not installed")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        spec = importlib.util.spec_from_file_location("native_video_for_test", NATIVE)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependency_stubs()):
            spec.loader.exec_module(module)
        native = module.RecordVideo
        self.video_class = type("RecordVideo", (SampledVideoMixin, native), {})
        self.native_class = native

    def wrapper(self, interval=None, native=False):
        config = VideoConfig(
            video_base_dir=self.temporary.name, info_on_video=False, fps=10
        )
        if interval is not None:
            config["record_every_n_episodes"] = interval
        env = CountingEnv()
        cls = self.native_class if native else self.video_class
        wrapper = cls(env, config)
        self.addCleanup(wrapper.close)
        wrapper.writes = []

        def submit(frames, path):
            wrapper.writes.append((frames, path))
            return types.SimpleNamespace(result=lambda: None)

        wrapper._submit_save = submit
        return wrapper

    def test_skipped_wave_never_extracts_renders_or_encodes(self):
        video = self.wrapper(interval=4)
        video.reset()
        video.chunk_step(None)
        video.flush_video()
        self.assertEqual(video.video_cnt, 1)
        self.assertEqual(len(video.writes), 1)
        captures = video.env.captures
        with (
            patch.object(
                video, "_extract_frame_batches", side_effect=AssertionError("extract")
            ),
            patch.object(video, "_append_frame", side_effect=AssertionError("overlay")),
        ):
            for _ in range(3):
                video.reset()
                video.step(None)
                video.chunk_step(None)
                video.flush_video()
                self.assertEqual(video.render_images, [])
        self.assertEqual(video.env.captures, captures)
        self.assertEqual(len(video.writes), 1)
        video.reset()
        video.step(None)
        video.flush_video("selected")
        self.assertEqual(video.video_cnt, 5)
        self.assertEqual(
            [Path(path).name for _, path in video.writes], ["0.mp4", "4.mp4"]
        )
        self.assertEqual(Path(video.writes[-1][1]).parent.name, "selected")

    def test_reset_and_duplicate_flush_do_not_advance_wave(self):
        video = self.wrapper(interval=4)
        video.flush_video()
        self.assertEqual(video.video_cnt, 0)
        video.reset()
        video.reset()
        self.assertEqual(video.video_cnt, 0)
        video.flush_video()
        video.flush_video()
        self.assertEqual(video.video_cnt, 1)
        video.reset()
        video.flush_video()
        video.flush_video()
        self.assertEqual(video.video_cnt, 2)

    def test_default_eval_and_explicit_one_match_native_lifecycle(self):
        for interval in (None, 1):
            native = self.wrapper(native=True)
            sampled = self.wrapper(interval=interval)
            for wrapper in (native, sampled):
                wrapper.flush_video()
                for _ in range(3):
                    wrapper.reset()
                    wrapper.step(None)
                    wrapper.chunk_step(None)
                    wrapper.flush_video()
                    wrapper.flush_video()
            self.assertEqual(sampled.env.captures, native.env.captures)
            self.assertEqual(sampled.video_cnt, native.video_cnt)
            self.assertEqual(
                [len(frames) for frames, _ in sampled.writes],
                [len(frames) for frames, _ in native.writes],
            )
            self.assertEqual(
                [Path(path).name for _, path in sampled.writes],
                [Path(path).name for _, path in native.writes],
            )

    def test_close_does_not_flush_partial_wave(self):
        video = self.wrapper(interval=4)
        video.reset()
        video.close()
        self.assertEqual(video.writes, [])
        self.assertEqual(video.env.closes, 1)

    def test_invalid_intervals_are_rejected(self):
        for interval in (0, -1, True, 1.5, "4", None):
            config = VideoConfig(record_every_n_episodes=interval)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.video_class(CountingEnv(), config)

    def test_prepare_runtime_installs_real_wrapper_and_fresh_worker_import(self):
        from spatial_wovr_launch import sha256

        root = Path(self.temporary.name)
        baseline, project, cosmos = (
            root / "baseline",
            root / "project",
            root / "cosmos",
        )
        runner = (
            "class EmbodiedRunner:\n    def run(self):\n        rollout_handle.wait()\n"
        )
        fixtures = {
            baseline / "rlinf/__init__.py": "",
            baseline / "rlinf/workers/env/env_worker.py": "",
            baseline / "rlinf/runners/embodied_runner.py": runner,
            baseline / "rlinf/envs/wrappers/__init__.py": (
                "from .record_video import RecordVideo\n"
            ),
            baseline / "rlinf/envs/wrappers/record_video.py": NATIVE.read_text(),
            baseline / "rlinf/envs/wrappers/other.py": "UNCHANGED = True\n",
            baseline / "rlinf/envs/utils.py": "",
            cosmos / "cosmos_policy/_src/predict2/datasets/utils.py": "",
            cosmos / "cosmos_policy/datasets/libero_dataset.py": "",
            project / "third_party/RLinf/rlinf/envs/world_model/__init__.py": "",
            project / "third_party/RLinf/rlinf/envs/__init__.py": "",
            project / "third_party/RLinf/rlinf/envs/action_utils.py": "",
        }
        for path, source in fixtures.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        benchmark = root / "benchmark"
        for suffix in (
            "assets",
            "bddl_files/libero_spatial",
            "init_files/libero_spatial",
        ):
            (benchmark / suffix).mkdir(parents=True)
        environment = {
            "PROJECT_ROOT": str(project),
            "RLINF_BASE_ROOT": str(baseline),
            "COSMOS_BASE_ROOT": str(cosmos),
            "COSMOS_WRIST_CHECKPOINT": str(root / "checkpoint"),
            "WORLDPILOT_ROOT": str(root / "worldpilot"),
            "LIBERO_ROOT": str(root / "libero"),
            "OPENPI_ROOT": str(root / "openpi"),
            "DIFFSYNTH_ROOT": str(root / "diffsynth"),
            "LIBERO_BENCHMARK_ROOT": str(benchmark),
            "LIBERO_CONFIG_PATH": str(root / "libero_config"),
        }
        run = root / "run"
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", list(sys.path)),
        ):
            prepare_runtime(run)
        metadata = json.loads((run / "source_paths.json").read_text())
        overlay = Path(metadata["record_video_overlay"])
        self.assertFalse(overlay.parent.is_symlink())
        self.assertFalse(overlay.is_symlink())
        self.assertTrue((overlay.parent / "other.py").is_symlink())
        self.assertEqual(metadata["record_video_overlay_sha256"], sha256(overlay))
        self.assertEqual(
            (baseline / "rlinf/envs/wrappers/record_video.py").read_text(),
            NATIVE.read_text(),
        )
        self.assertEqual(
            (overlay.parent / "_record_video_native.py").read_text(), NATIVE.read_text()
        )
        # Import from a fresh interpreter, using only runtime PYTHONPATH. No
        # driver monkeypatch or experiments-directory import can provide the class.
        script = (
            "import sys, types\n"
            "gym=types.ModuleType('gymnasium')\n"
            "gym.Env=type('Env',(),{})\n"
            "gym.Wrapper=type('Wrapper',(),{})\n"
            "sys.modules['gymnasium']=gym\n"
            "sys.modules['imageio']=types.ModuleType('imageio')\n"
            "torch=types.ModuleType('torch'); torch.Tensor=type('Tensor',(),{})\n"
            "sys.modules['torch']=torch\n"
            "utils=types.ModuleType('rlinf.envs.utils')\n"
            "utils.put_info_on_image=lambda image,info:image\n"
            "utils.tile_images=lambda images,nrows:images[0]\n"
            "sys.modules['rlinf.envs.utils']=utils\n"
            "from rlinf.envs.wrappers import RecordVideo\n"
            "assert RecordVideo.__module__=='rlinf.envs.wrappers.record_video'\n"
            "env=types.SimpleNamespace(seed=0,num_envs=1,metadata={})\n"
            "video=RecordVideo(env,{'record_every_n_episodes':4})\n"
            "assert video.record_every_n_episodes==4\n"
            "video._executor.shutdown(wait=True)\n"
            "print('worker overlay imported')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": str(run / "code/rlinf")},
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("worker overlay imported", result.stdout)


if __name__ == "__main__":
    unittest.main()
