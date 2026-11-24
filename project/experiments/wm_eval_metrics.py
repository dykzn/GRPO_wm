"""LPIPS, FID, variable-length I3D FVD, and official PWCNet FloLPIPS."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from wm_eval_data import FeatureMoments, atomic_json, frechet_distance

HUB_WEIGHTS = ["alexnet-owt-7be5be79.pth", "pt_inception-2015-12-05-6726825d.pth", "network-default.pytorch"]


def cuda_library_environment() -> dict[str, str]:
    """Expose this metric interpreter's existing pip CUDA libraries to CuPy."""
    import torch

    package = Path(torch.__file__).resolve().parent
    directories = [package / "lib", *sorted(package.parent.glob("nvidia/*/lib"))]
    paths = [str(directory) for directory in directories if directory.is_dir()]
    paths += [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
    return environment


def dependencies() -> dict:
    root = Path(os.environ["WM_METRIC_ASSETS"])
    missing = [f"Python package: {name}" for name in ["lpips", "pytorch_fid", "cupy"]
               if importlib.util.find_spec(name) is None]
    paths = [root / "i3d_torchscript.pt", root / "flolpips/flolpips.py",
             root / "flolpips/pwcnet.py", root / "flolpips/weights/v0.1/alex.pth"]
    paths += [Path(os.environ["TORCH_HOME"]) / "hub/checkpoints" / name for name in HUB_WEIGHTS]
    missing += [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    return {"ready": not missing, "missing": missing, "assets_root": str(root)}


def check_pair(prediction: np.ndarray, truth: np.ndarray, horizon: int) -> None:
    if prediction.dtype != np.uint8 or truth.dtype != np.uint8:
        raise ValueError("Metrics require uint8 RGB")
    if prediction.shape != truth.shape or prediction.shape != (horizon, 224, 224, 3):
        raise ValueError(f"Mismatched prediction/truth: {prediction.shape}, {truth.shape}, horizon={horizon}")


class MetricModels:
    def __init__(self):
        import torch
        import lpips
        from pytorch_fid.inception import InceptionV3

        status = dependencies()
        if not status["ready"]:
            raise RuntimeError("Metric dependencies missing: " + "; ".join(status["missing"]))
        # Every required weight must have been downloaded explicitly by setup.
        def no_download(*args, **kwargs):
            raise RuntimeError("Unexpected metric weight download; rerun scripts/setup_world_model_metrics.sh")

        torch.hub.download_url_to_file = no_download
        self.lpips = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
        root = Path(os.environ["WM_METRIC_ASSETS"])
        self.i3d = torch.jit.load(str(root / "i3d_torchscript.pt"), map_location="cuda").eval()
        sys.path.insert(0, str(root / "flolpips"))
        import cupy
        from correlation import correlation
        import flolpips

        # FloLPIPS's original CUDA source is unchanged. Adapt only the CuPy
        # compiler entry point removed by newer CuPy releases.
        if not hasattr(cupy.cuda, "compile_with_cache"):
            @cupy.memoize(for_each_device=True)
            def launch(function, source):
                return cupy.RawModule(code=source).get_function(function)

            correlation.cupy_launch = launch
        self.flo = flolpips.FloLPIPS(net="alex", version="0.1",
            model_path=str(root / "flolpips/weights/v0.1/alex.pth")).cuda().eval()
        self.flow = flolpips.PWCNet().cuda().eval()

    @staticmethod
    def tensor(frames):
        import torch

        return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).cuda().float() / 255

    def frames(self, predicted, truth):
        import torch

        with torch.inference_mode():
            a, b = self.tensor(predicted), self.tensor(truth)
            score = self.lpips(a, b, normalize=True).reshape(-1).cpu().numpy()
            fa = self.inception(a)[0].flatten(1).cpu().numpy()
            fb = self.inception(b)[0].flatten(1).cpu().numpy()
        return score, fa, fb

    def video(self, frames):
        import torch

        # StyleGAN-V exports the TF-Hub I3D used by the FVD reference, with
        # uint8 [0,255] input and its original preprocessing in the module.
        # The released TorchScript reshapes C*T with view(), so the permuted
        # BCTHW input must be contiguous before its internal resize step.
        video = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None].contiguous().cuda().float()
        with torch.inference_mode():
            features = self.i3d(video, rescale=True, resize=True, return_features=True)
        values = features.reshape(1, -1).float().cpu().numpy()
        if values.shape != (1, 400):
            raise ValueError(f"Expected TF-Hub I3D 400-d features, got {values.shape}")
        return values

    def motion(self, predicted, truth, following_predicted, following_truth):
        import torch

        with torch.inference_mode():
            a, b = self.tensor(predicted), self.tensor(truth)
            next_a, next_b = self.tensor(following_predicted), self.tensor(following_truth)
            flow_difference = self.flow(b, next_b) - self.flow(a, next_a)
            identical = (a == b).flatten(1).all(1) & (next_a == next_b).flatten(1).all(1)
            result = torch.zeros(len(a), device="cuda")
            active = ~identical
            if active.any():
                result[active] = self.flo(b[active], a[active], flow_difference[active], normalize=True)
            if not torch.isfinite(result).all():
                raise ValueError("Official FloLPIPS returned non-finite values; no replacement metric was substituted")
        return result.cpu().numpy()


def smoke(args) -> None:
    """Exercise actual pretrained backbones and CUDA flow before long generation."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Metric GPU smoke requires a visible CUDA GPU")
    models = MetricModels()
    random = np.random.default_rng(20261019)
    real = random.integers(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    predicted = np.ascontiguousarray(np.roll(real, 2, axis=2))
    same_lpips, feature_a, feature_b = models.frames(real[:2], real[:2])
    if not np.allclose(same_lpips, 0, atol=1e-6) or not np.allclose(feature_a, feature_b):
        raise ValueError("LPIPS/Inception identity sanity check failed")
    motion = models.motion(predicted[:2], real[:2], predicted[1:3], real[1:3])
    if not np.isfinite(motion).all():
        raise ValueError("FloLPIPS/PWCNet CUDA sanity check failed")
    video_feature = models.video(real)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "metric_smoke.json", {
        "status": "passed", "purpose": "synthetic backend validation only; not benchmark scores",
        "i3d_feature_shape": list(video_feature.shape), "flow_cuda": True,
    })
    print("Metric backbone and CUDA optical-flow smoke passed", flush=True)


def score(args) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; score from the GPU terminal")
    manifest = json.loads(args.manifest.read_text())
    models = MetricModels()
    all_results = []
    for model in ([args.model] if getattr(args, "model", None) else ["wan", "cosmos"]):
        directory = args.output / model
        records = []
        # Fail before scoring if any trajectory is missing: the two models
        # must be evaluated on the exact same episode IDs and horizons.
        settings = set()
        for row in manifest["episodes"]:
            path = directory / f"{row['trajectory_id']}.npz"
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise FileNotFoundError(f"Prediction is missing: {path}")
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata["episode"] != row:
                raise ValueError(f"Prediction came from another manifest: {path}")
            settings.add(metadata["settings_sha256"])
            records.append((row, path, metadata))
        if len(settings) != 1:
            raise ValueError(f"Mixed model/checkpoint/settings in {directory}")
        fid_real, fid_pred, fvd_real, fvd_pred = [FeatureMoments() for _ in range(4)]
        lpips_values, motion_values, episode_scores = [], [], []
        total_seconds, total_frames = 0.0, 0
        for index, (row, path, metadata) in enumerate(records):
            with np.load(path, allow_pickle=False) as archive:
                predicted, truth = archive["prediction"], archive["truth"]
            length = row["horizon"]
            check_pair(predicted, truth, length)
            local_lpips, local_motion = [], []
            for start in range(0, length, args.metric_batch):
                end = min(start + args.metric_batch, length)
                values, pred_features, real_features = models.frames(predicted[start:end], truth[start:end])
                local_lpips.extend(values.tolist())
                fid_pred.add(pred_features)
                fid_real.add(real_features)
            for start in range(0, length - 1, args.metric_batch):
                end = min(start + args.metric_batch, length - 1)
                values = models.motion(predicted[start:end], truth[start:end],
                                       predicted[start+1:end+1], truth[start+1:end+1])
                local_motion.extend(values.tolist())
            # Keep the actual temporal length for each matched video pair.
            # I3D pools over time; do not repeat frames or resample to a fake
            # common 128/256/512-frame horizon.
            fvd_pred.add(models.video(predicted))
            fvd_real.add(models.video(truth))
            lpips_values.extend(local_lpips)
            motion_values.extend(local_motion)
            total_frames += length
            total_seconds += metadata["seconds"]
            episode_scores.append({"trajectory_id": row["trajectory_id"], "task_id": row["task_id"],
                "frames": length, "LPIPS": float(np.mean(local_lpips)),
                "FloLPIPS": float(np.mean(local_motion)), "FPS": metadata["fps"]})
            print(f"Metrics {model}: {index+1}/{len(records)}, frames={length}", flush=True)
        result = {"model": model, "view": "primary" if model == "wan" else "wrist",
            "native_resolution": 256 if model == "wan" else 224, "metric_resolution": 224,
            "episodes": len(records), "frames": total_frames,
            "FPS": total_frames / total_seconds,
            "LPIPS": float(np.mean(lpips_values)), "FID": frechet_distance(fid_real, fid_pred),
            "FVD_variable_length": frechet_distance(fvd_real, fvd_pred),
            "FloLPIPS": float(np.mean(motion_values)), "generation_seconds": total_seconds,
            "macro_episode_LPIPS": float(np.mean([row["LPIPS"] for row in episode_scores])),
            "macro_episode_FloLPIPS": float(np.mean([row["FloLPIPS"] for row in episode_scores])),
            "horizon_min": min(row["frames"] for row in episode_scores),
            "horizon_max": max(row["frames"] for row in episode_scores),
            "manifest_sha256": manifest["manifest_sha256"],
            "protocol": manifest["protocol"], "split_status": manifest["split_status"],
            "aggregation": "FPS=sum(frames)/sum(seconds); LPIPS per-frame mean; FloLPIPS per-adjacent-pair mean; FID all frames; FVD one 400-d I3D feature per variable-length matched video",
            "metric_backbones": {"LPIPS": "AlexNet v0.1", "FID": "pytorch-fid Inception-v3 2048-d",
                                  "FVD": "StyleGAN-V TF-Hub Kinetics-400 I3D TorchScript 400-d",
                                  "FloLPIPS": "official danier97/flolpips PWCNet + AlexNet v0.1"},
        }
        atomic_json(args.output / f"{model}_episode_metrics.json", episode_scores)
        atomic_json(args.output / f"{model}_metrics.json", result)
        all_results.append(result)
    atomic_json(args.output / "metrics.json", all_results)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        columns = ["model", "view", "native_resolution", "metric_resolution", "episodes", "frames", "horizon_min", "horizon_max",
                   "FPS", "LPIPS", "FID", "FVD_variable_length", "FloLPIPS"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(json.dumps(all_results, indent=2), flush=True)
