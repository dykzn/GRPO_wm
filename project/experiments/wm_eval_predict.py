"""Autoregressive, action-conditioned video generation for the two local WMs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from wm_eval_data import atomic_json, digest_json, read_episode, resize_rgb


class WanPredictor:
    """Use the same fixed reference / rolling four-frame memory as WanEnv."""

    def __init__(self, steps: int):
        import torch
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        root = Path(os.environ["WAN_SPATIAL_CHECKPOINT"])
        self.steps = steps
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cuda",
            model_configs=[ModelConfig(path=str(root / "model-00001.safetensors"), offload_device="cpu"),
                           ModelConfig(path=str(root / "Wan2.2_VAE.pth"), offload_device="cpu")],
        )
        self.pipe.dit.to("cuda")
        self.pipe.vae.to("cuda")

    def reset(self, primary, wrist, task):
        self.reference = resize_rgb(primary[None], 256)[0]
        self.memory = np.repeat(self.reference[None], 4, axis=0)
        self.context_actions = np.zeros((5, 7), dtype=np.float32)
        self.context_actions[:, -1] = -1

    def chunk(self, actions: np.ndarray, seed: int) -> np.ndarray:
        import torch

        action_window = np.concatenate([self.context_actions, actions], axis=0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.pipe(
                seed=seed, tiled=False, input_image=[Image.fromarray(self.reference)],
                input_image4=[[Image.fromarray(frame) for frame in self.memory]],
                action=torch.from_numpy(action_window[None]).to("cuda", torch.float32),
                height=256, width=256, num_frames=13, num_inference_steps=self.steps,
                cfg_scale=1.0, progress_bar_cmd=lambda values: values, batch_size=1,
            )
        frames = np.stack([np.asarray(frame, dtype=np.uint8) for frame in result[0]])
        if frames.shape != (13, 256, 256, 3):
            raise ValueError(f"Unexpected Wan output: {frames.shape}")
        self.memory = frames[-4:].copy()
        self.context_actions[1:] = actions[-4:]
        return frames[5:]


class DirectCosmosClient:
    """Reuse the bridge request validation without WebSocket timing overhead."""

    def __init__(self, policy):
        from cosmos_bridge.cosmos_client import CosmosClient

        class Transport:
            def predict_action(self, message):
                return {"ok": True, "data": policy.predict_action(**message)}

        self.client = CosmosClient.__new__(CosmosClient)
        self.client._client = Transport()

    def query_metadata(self):
        return self.client.query_metadata()

    def query_temporal_future_batch(self, *args, **kwargs):
        return self.client.query_temporal_future_batch(*args, **kwargs)


class CosmosPredictor:
    """Run the complete custom v3 wrist checkpoint with the production bridge."""

    def __init__(self, steps: int):
        from cosmos_bridge.cosmos_server import CosmosAdvisorPolicy, CosmosCfg, _fix_degenerate_stats
        from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
        from cosmos_policy.experiments.robot import cosmos_utils
        from spatial_wovr_launch import WRIST_CONFIG
        from world_model_grpo.cosmos_wrist_adapter import CosmosWristAdapter, CosmosWristAdapterConfig

        checkpoint = os.environ["COSMOS_WRIST_CHECKPOINT"]
        model, _ = load_model_from_checkpoint(
            experiment_name=WRIST_CONFIG, s3_checkpoint_dir=checkpoint,
            config_file="cosmos_policy/config/config.py", load_ema_to_reg=False,
            experiment_opts=[f"model.config.tokenizer.vae_pth={os.environ['COSMOS_VAE']}"],
        )
        model.eval().to("cuda")
        if getattr(model, "_consolidated_checkpoint_load_audit", {}).get("verified") is not True:
            raise ValueError("Cosmos checkpoint loader did not verify the model state")
        stats = _fix_degenerate_stats(cosmos_utils.load_dataset_stats(os.environ["COSMOS_DATASET_STATS"]))
        cosmos_utils.init_t5_text_embeddings_cache(os.environ["COSMOS_T5_EMBEDDINGS"])
        cosmos_utils.ACTION_DIM = 7
        policy = CosmosAdvisorPolicy(CosmosCfg(suite="libero", config=WRIST_CONFIG,
            ckpt_path=checkpoint, chunk_size=8, action_dim=7, proprio_dim=9), model, stats)
        self.adapter = CosmosWristAdapter(
            CosmosWristAdapterConfig(primary_reference=True, future_denoising_steps=steps,
                                    guidance=1.0, gripper_action_mode="already_cosmos"),
            client=DirectCosmosClient(policy),
        )

    def reset(self, primary, wrist, task):
        self.state = self.adapter.initial_state(primary, wrist, batch_size=1)
        self.task = task

    def chunk(self, actions: np.ndarray, seed: int) -> np.ndarray:
        return self.adapter.query_chunk(self.state, actions[None], languages=[self.task], batch_seed=seed)[0]


def target_frames(model: str, primary: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Apply the model's geometric preprocessing to its corresponding truth."""
    if model == "wan":
        return resize_rgb(primary[1:], 224)
    from world_model_grpo.cosmos_wrist_adapter import _preprocess_wrist_model_space

    return np.stack([_preprocess_wrist_model_space(frame, (224, 224)) for frame in wrist[1:]])


def checkpoint_identity(model: str) -> list[dict]:
    if model == "wan":
        root = Path(os.environ["WAN_SPATIAL_CHECKPOINT"])
        files = [root / "model-00001.safetensors", root / "Wan2.2_VAE.pth"]
    else:
        root = Path(os.environ["COSMOS_WRIST_CHECKPOINT"])
        manifest = json.loads((root.parent / "checkpoint_manifest.json").read_text())
        files = [root.parent / row["path"] for row in manifest["files"] if "/model/" in row["path"]]
        for row in manifest["files"]:
            path = root.parent / row["path"]
            if not path.is_file() or path.stat().st_size != row["size"]:
                raise ValueError(f"Incomplete Cosmos checkpoint: {path}")
        files.extend(Path(os.environ[key]) for key in ["COSMOS_VAE", "COSMOS_DATASET_STATS", "COSMOS_T5_EMBEDDINGS"])
    return [{"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in files]


def predict(args) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable in this process; run prediction from the GPU terminal")
    if args.model == "cosmos" and os.environ.get("COSMOS_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["episodes"][args.shard_index::args.num_shards]
    if not rows:
        return
    settings = {"manifest": manifest["manifest_sha256"], "model": args.model, "steps": args.steps,
                "seed": args.seed, "metric_resolution": 224, "batch": 1,
                "native_resolution": 256 if args.model == "wan" else 224,
                "weights": checkpoint_identity(args.model), "protocol": manifest["protocol"],
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "warmup": args.warmup,
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "code": digest_json({name: (Path(__file__).parent / name).read_text()
                                     for name in ["wm_eval_predict.py", "wm_eval_data.py",
                                                  "world_model_grpo/cosmos_wrist_adapter.py"]})}
    capture_native = args.model == "wan" and os.environ.get("WM_EVAL_CAPTURE_NATIVE_PRIMARY") == "1"
    if capture_native:
        settings["native_primary_capture"] = True
    identity = digest_json(settings)
    directory = args.output / args.model
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows:
        output = directory / f"{row['trajectory_id']}.npz"
        metadata = output.with_suffix(".json")
        if output.is_file() and metadata.is_file():
            record = json.loads(metadata.read_text())
            if record.get("settings_sha256") != identity:
                raise ValueError(f"Existing predictions use different settings: {output}")
        else:
            pending.append(row)
    if not pending:
        print(f"{args.model} shard {args.shard_index}: all cached", flush=True)
        return
    predictor = WanPredictor(args.steps) if args.model == "wan" else CosmosPredictor(args.steps)
    primary, wrist, actions = read_episode(pending[0])
    predictor.reset(primary[0], wrist[0], pending[0]["task"])
    for index in range(args.warmup):
        predictor.chunk(actions[:8], args.seed + index)
    torch.cuda.synchronize()
    atomic_json(directory / f"shard_{args.shard_index}_runtime.json", {
        "settings": settings, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn_enabled": torch.backends.cudnn.enabled, "warmup_chunks": args.warmup,
        "fps_definition": "generated RGB frames / synchronized prediction wall time, batch=1; includes conditioning, VAE and CPU output; excludes model loading, warmup, HDF5 I/O, scoring and final metric resize",
    })
    for index, row in enumerate(pending):
        primary, wrist, actions = read_episode(row)
        predictor.reset(primary[0], wrist[0], row["task"])
        generated, times = [], []
        for start in range(0, row["horizon"], 8):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            frames = predictor.chunk(actions[start:start + 8], args.seed + start)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
            if frames.shape[0] != 8 or frames.dtype != np.uint8:
                raise ValueError("Predictor must produce exactly eight uint8 RGB frames")
            generated.append(frames)
        truth = target_frames(args.model, primary, wrist)
        prediction = resize_rgb(np.concatenate(generated), 224)
        if prediction.shape != truth.shape or prediction.shape[0] != row["horizon"]:
            raise ValueError("Prediction/truth time alignment failed")
        output = directory / f"{row['trajectory_id']}.npz"
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("wb") as stream:
            payload = {"prediction": prediction, "truth": truth}
            if capture_native:
                payload.update(native_prediction=np.concatenate(generated), native_truth=primary[1:])
            np.savez_compressed(stream, **payload)
        temporary.replace(output)
        atomic_json(output.with_suffix(".json"), {"settings_sha256": identity, "episode": row,
            "model": args.model, "view": "primary" if args.model == "wan" else "wrist",
            "frames": len(prediction), "seconds": sum(times), "chunk_seconds": times,
            "fps": len(prediction) / sum(times)})
        print(f"{args.model} shard={args.shard_index}: {index+1}/{len(pending)}, task={row['task_id']}, frames={len(prediction)}, FPS={len(prediction)/sum(times):.3f}", flush=True)
