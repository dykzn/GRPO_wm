"""Run the normal GRPO launcher with an explicitly selected updated WM pair."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import spatial_wovr_launch as native


def validate_updated_pair(cfg, pair_path: Path, baseline_wan: Path) -> dict:
    pair = json.loads(pair_path.read_text())
    if pair.get("schema") != "spatial-wm-pair-v1":
        raise ValueError("Expected a packaged paired world-model update")
    selected_wan = Path(cfg.env.train.wan_wm_hf_ckpt_path)
    source_wan = Path(pair["wan"])
    weights = selected_wan / "model-00001.safetensors"
    if Path(cfg.env.train.model_path).resolve() != weights.resolve():
        raise ValueError("Wan pipeline model_path does not select the recorded pair")
    if weights.resolve() != (source_wan / weights.name).resolve():
        raise ValueError("Wan configuration does not select the recorded pair")
    if native.sha256(weights) != pair["wan_weights_sha256"]:
        raise ValueError("Updated Wan checksum mismatch")
    for name in ("Wan2.2_VAE.pth", "resnet_rm.pth"):
        if (selected_wan / name).resolve() != (baseline_wan / name).resolve():
            raise ValueError(f"Updated GRPO must retain the frozen baseline {name}")
    checkpoint = Path(os.environ["COSMOS_WRIST_CHECKPOINT"])
    if checkpoint.resolve() != Path(pair["cosmos"]).resolve():
        raise ValueError("Cosmos configuration does not select the recorded pair")
    manifest = json.loads((checkpoint.parent / "checkpoint_manifest.json").read_text())
    records = native.wrist_records(checkpoint)
    if not records or not (checkpoint / "model/.metadata").is_file():
        raise ValueError("Updated Cosmos checkpoint is incomplete")
    for path, record in records:
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Updated Cosmos shard is incomplete: {path}")
        if native.sha256(path) != record["sha256"]:
            raise ValueError(f"Updated Cosmos checksum mismatch: {path}")
    tree = hashlib.sha256(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in manifest["files"]).encode()
    ).hexdigest()
    if tree != pair["cosmos_tree_sha256"] or tree != manifest["tree_sha256"]:
        raise ValueError("Updated Cosmos manifest differs from the packaged pair")
    return {
        "manifest": str(pair_path),
        "manifest_sha256": native.sha256(pair_path),
        "wan": str(selected_wan),
        "cosmos": str(checkpoint),
        "wan_step": pair["wan_step"],
        "cosmos_step": pair["cosmos_step"],
        "wan_weights_sha256": pair["wan_weights_sha256"],
        "cosmos_tree_sha256": tree,
        "experimental_candidate": True,
    }


def install_pair_check(
    pair_path: Path, baseline_wan: Path, baseline_cosmos: Path
) -> None:
    original = native.check_assets

    def check_assets(cfg, full: bool, run_dir: Path) -> bool:
        # Check unchanged policy/VAE/reward/KIR assets. Updated Cosmos is
        # independently verified below; its original model need not be resident.
        baseline_cfg = copy.deepcopy(cfg)
        baseline_cfg.env.train.wan_wm_hf_ckpt_path = str(baseline_wan)
        ready = original(baseline_cfg, full, run_dir, verify_wrist=False)
        updated = validate_updated_pair(cfg, pair_path, baseline_wan)
        path = run_dir / "asset_check.json"
        report = json.loads(path.read_text())
        report["updated_pair"] = updated
        report["updated_pair_full_checksums"] = True
        path.write_text(json.dumps(report, indent=2) + "\n")
        return ready

    native.check_assets = check_assets


def compose_saved_config(template: Path, overrides: list[str]):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(template)
    config_dir = Path(__file__).parent / "config"
    current_defaults = OmegaConf.load(config_dir / "spatial_wovr_pi05_8gpu.yaml")
    if cfg.env.train.get("use_kir_frame_pool", False):
        # Apply the current rollout protocol to the historical release template.
        cfg.env.train.merge_with(
            OmegaConf.masked_copy(
                current_defaults.env.train,
                ["kir_fraction", "success_horizon_grace_steps", "hallucination_trust"],
            )
        )
    if "algorithm" in cfg and "algorithm" in current_defaults:
        cfg.algorithm.merge_with(
            OmegaConf.masked_copy(
                current_defaults.algorithm,
                ["kl_penalty", "kl_beta", "hybrid_ppo_grpo"],
            )
        )
    if os.environ.get("SPATIAL_WOVR_OPENVLA_WAN_ACTOR") == "1":
        profile = config_dir / "spatial_wovr_openvla_wan_actor.yaml"
        cfg.merge_with(OmegaConf.load(profile))
    if os.environ.get("SPATIAL_WOVR_HYBRID_PPO_GRPO") == "1":
        profile = config_dir / "spatial_wovr_pi05_8gpu_hybrid_ppo_grpo.yaml"
        cfg.merge_with(OmegaConf.load(profile))
    cfg.merge_with(OmegaConf.from_dotlist(overrides))
    # Apply explicit environment overrides AFTER launcher defaults (e.g. portable.py's
    # hardcoded experiment_name / val_check_interval / save_interval) so user-selected
    # hybrid experiment settings take precedence.
    env_overrides: list[str] = []
    if (rho_max := os.environ.get("SPATIAL_WOVR_HYBRID_RHO_MAX")) not in (None, ""):
        env_overrides.append(f"algorithm.hybrid_ppo_grpo.rho_max={rho_max}")
    if (conflict_mode := os.environ.get("SPATIAL_WOVR_HYBRID_CONFLICT_MODE")) not in (None, ""):
        env_overrides.append(f"algorithm.hybrid_ppo_grpo.grad_conflict_mode={conflict_mode}")
    if extra := os.environ.get("SPATIAL_WOVR_EXTRA_OVERRIDES"):
        env_overrides.extend(part for part in extra.split() if part)
    if env_overrides:
        cfg.merge_with(OmegaConf.from_dotlist(env_overrides))
    if OmegaConf.select(cfg, "rollout.model", default=None) is not None:
        if "is_lora" not in cfg.rollout.model:
            cfg.rollout.model.is_lora = bool(
                OmegaConf.select(cfg, "actor.model.is_lora", default=False)
            )
    output = os.environ["SPATIAL_WOVR_OUTPUT"]
    cfg.runner.logger.log_path = output
    cfg.env.train.wan_wm_hf_ckpt_path = os.environ["WAN_SPATIAL_CHECKPOINT"]
    cfg.env.train.model_path = str(
        Path(os.environ["WAN_SPATIAL_CHECKPOINT"]) / "model-00001.safetensors"
    )
    cfg.env.train.cosmos_endpoints = native.make_endpoints(
        os.environ["SPATIAL_WOVR_GPUS"], int(os.environ["SPATIAL_COSMOS_PORT"])
    )
    # A resolved YAML contains copies of these dictionaries. Restore the links
    # so configure_shared_pool publishes the NEW private socket directory.
    cfg.env.train.shared_wan_pool = "${actor.shared_wan_pool}"
    cfg.env.train.memory_guard = "${runner.memory_guard}"
    cfg.env.train.video_cfg.video_base_dir = str(Path(output) / "video/train")
    cfg.env.eval.video_cfg.video_base_dir = str(Path(output) / "video/eval")
    return cfg


def main() -> int:
    original_imports = native.check_imports

    def cpu_imports(run_dir):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            return original_imports(run_dir)
        finally:
            if visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible

    native.check_imports = cpu_imports
    if template := os.environ.get("SPATIAL_WOVR_RESOLVED_TEMPLATE"):
        import hydra

        def compose(*args, overrides=None, **kwargs):
            return compose_saved_config(Path(template), overrides or [])

        hydra.compose = compose
    if pair := os.environ.get("SPATIAL_WOVR_MODEL_PAIR"):
        install_pair_check(
            Path(pair),
            Path(os.environ["SPATIAL_WOVR_BASELINE_WAN"]),
            Path(os.environ["SPATIAL_WOVR_BASELINE_COSMOS"]),
        )
    return native.main()


if __name__ == "__main__":
    raise SystemExit(main())
