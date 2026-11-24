#!/usr/bin/env python3
"""Evaluate a native LeRobot PI0.5 checkpoint on LIBERO.

This entry point keeps the LeRobot policy, processors, and environment intact.
It only supplies two compatibility fixes required by the local checkpoint:

1. Use the locally cached OpenPI PaliGemma SentencePiece model. Its token IDs
   are identical to ``google/paligemma-3b-pt-224`` but do not require gated
   Hugging Face access.
2. Load the safetensors checkpoint non-strictly because tied input/output text
   embeddings are stored only once. The alias is asserted after loading.
"""

import gc
import json
import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import numpy as np


PROJECT_ROOT = Path(
    os.environ.get(
        "LEROBOT_PI05_PROJECT_ROOT",
        os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])),
    )
).resolve()
TOKENIZER_ROOT = Path(
    os.environ.get(
        "LEROBOT_PI05_TOKENIZER",
        PROJECT_ROOT / "checkpoints/tokenizers/openpi-paligemma",
    )
).resolve()


def _install_local_tokenizer_override() -> None:
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

    original_post_init = TokenizerProcessorStep.__post_init__

    def patched_post_init(self) -> None:
        if self.tokenizer_name == "google/paligemma-3b-pt-224":
            self.tokenizer_name = str(TOKENIZER_ROOT)
        original_post_init(self)

    TokenizerProcessorStep.__post_init__ = patched_post_init


def _install_pi05_checkpoint_loader_fix() -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    original_from_pretrained = PI05Policy.from_pretrained.__func__

    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        # The checkpoint omits one side of a tied parameter pair. Strict
        # loading would raise after loading all other tensors and the upstream
        # implementation catches that exception, which makes the result look
        # suspicious even though the shared tensor is valid.
        kwargs["strict"] = False
        policy = original_from_pretrained(cls, *args, **kwargs)

        paligemma = policy.model.paligemma_with_expert.paligemma
        input_weight = paligemma.model.language_model.embed_tokens.weight
        output_weight = paligemma.lm_head.weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            paligemma.model.language_model.embed_tokens.weight = output_weight
            input_weight = paligemma.model.language_model.embed_tokens.weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise RuntimeError("PaliGemma input/output embedding weights are not tied.")

        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        print(
            "LEROBOT_PI05_CHECKPOINT_READY "
            f"parameters={parameter_count} "
            f"chunk_size={policy.config.chunk_size} "
            f"n_action_steps={policy.config.n_action_steps}",
            flush=True,
        )
        status_path = os.environ.get("LEROBOT_PI05_STATUS_FILE")
        if status_path:
            Path(status_path).write_text("RUNNING\n", encoding="utf-8")
        return policy

    PI05Policy.from_pretrained = patched_from_pretrained


def _selected_libero_task_ids(env_cfg, suite_name: str) -> list[int]:
    """Return the task ids requested for one LIBERO suite.

    The upstream LeRobot evaluator constructs every selected task environment
    before loading the policy.  For EGL LIBERO evaluation that creates one GPU
    context per environment and can exhaust VRAM.  We keep task selection
    identical while constructing only one task at a time below.
    """
    if env_cfg.task_ids is not None:
        return [int(task_id) for task_id in env_cfg.task_ids]

    from lerobot.envs.libero import _get_suite

    return list(range(len(_get_suite(suite_name).tasks)))


def _mean_or_nan(values: list[float | bool]) -> float:
    if not values:
        return float("nan")
    return float(np.asarray(values, dtype=float).mean())


def _sequential_eval_main():
    """Run LeRobot evaluation while keeping only one LIBERO task alive.

    The stock ``lerobot_eval`` path calls ``make_env`` once with all suites and
    task ids.  Its LIBERO factory eagerly creates every vector environment,
    which is particularly expensive with EGL contexts.  This entry point
    preserves the stock rollout and metric code but moves environment creation
    into the per-task loop.
    """
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts import lerobot_eval as stock_eval
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import get_safe_torch_device, init_logging
    import torch

    @parser.wrap()
    def run(cfg: EvalPipelineConfig):
        logging = __import__("logging")
        logging.info(pformat(asdict(cfg)))

        device = get_safe_torch_device(cfg.policy.device, log=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_seed(cfg.seed)

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Output dir: %s", output_dir)
        logging.info("Making policy before creating LIBERO EGL environments.")

        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )
        policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=cfg.policy,
        )

        suite_names = [suite.strip() for suite in str(cfg.env.task).split(",") if suite.strip()]
        videos_dir = output_dir / "videos"
        per_task_infos: list[dict] = []
        group_acc: dict[str, dict[str, list]] = {
            suite: {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []}
            for suite in suite_names
        }
        overall = {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []}

        autocast_context = (
            torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            for suite_name in suite_names:
                for task_id in _selected_libero_task_ids(cfg.env, suite_name):
                    logging.info(
                        "SEQUENTIAL_TASK_START suite=%s task_id=%s batch_size=%s",
                        suite_name,
                        task_id,
                        cfg.eval.batch_size,
                    )
                    task_cfg = deepcopy(cfg.env)
                    task_cfg.task = suite_name
                    task_cfg.task_ids = [task_id]
                    envs = make_env(
                        task_cfg,
                        n_envs=cfg.eval.batch_size,
                        use_async_envs=cfg.eval.use_async_envs,
                        trust_remote_code=cfg.trust_remote_code,
                    )
                    env = envs[suite_name][task_id]
                    task_videos_dir = videos_dir / f"{suite_name}_{task_id}"
                    try:
                        metrics = stock_eval.eval_one(
                            env,
                            policy=policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            n_episodes=cfg.eval.n_episodes,
                            max_episodes_rendered=10,
                            videos_dir=task_videos_dir,
                            return_episode_data=False,
                            start_seed=cfg.seed,
                        )
                    finally:
                        close_envs(envs)
                        del envs, env
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    group_acc[suite_name]["sum_rewards"].extend(metrics["sum_rewards"])
                    group_acc[suite_name]["max_rewards"].extend(metrics["max_rewards"])
                    group_acc[suite_name]["successes"].extend(metrics["successes"])
                    group_acc[suite_name]["video_paths"].extend(metrics.get("video_paths", []))
                    overall["sum_rewards"].extend(metrics["sum_rewards"])
                    overall["max_rewards"].extend(metrics["max_rewards"])
                    overall["successes"].extend(metrics["successes"])
                    overall["video_paths"].extend(metrics.get("video_paths", []))
                    per_task_infos.append(
                        {"task_group": suite_name, "task_id": task_id, "metrics": metrics}
                    )
                    task_successes = metrics["successes"]
                    logging.info(
                        "SEQUENTIAL_TASK_DONE suite=%s task_id=%s episodes=%s success=%s/%s rate=%.2f%%",
                        suite_name,
                        task_id,
                        len(task_successes),
                        int(sum(task_successes)),
                        len(task_successes),
                        100.0 * _mean_or_nan(task_successes),
                    )

        per_group = {}
        for suite_name, values in group_acc.items():
            per_group[suite_name] = {
                "avg_sum_reward": _mean_or_nan(values["sum_rewards"]),
                "avg_max_reward": _mean_or_nan(values["max_rewards"]),
                "pc_success": 100.0 * _mean_or_nan(values["successes"]),
                "n_episodes": len(values["sum_rewards"]),
                "video_paths": values["video_paths"],
            }
        overall_info = {
            "avg_sum_reward": _mean_or_nan(overall["sum_rewards"]),
            "avg_max_reward": _mean_or_nan(overall["max_rewards"]),
            "pc_success": 100.0 * _mean_or_nan(overall["successes"]),
            "n_episodes": len(overall["sum_rewards"]),
            "video_paths": overall["video_paths"],
        }
        info = {"per_task": per_task_infos, "per_group": per_group, "overall": overall_info}
        print("Overall Aggregated Metrics:", flush=True)
        print(overall_info, flush=True)
        for suite_name, metrics in per_group.items():
            print(f"Aggregated Metrics for {suite_name}:", flush=True)
            print(metrics, flush=True)
        with (output_dir / "eval_info.json").open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)
        logging.info("End of sequential eval")

    init_logging()
    register_third_party_plugins()
    run()


def main() -> None:
    if not (TOKENIZER_ROOT / "tokenizer.model").is_file():
        raise FileNotFoundError(
            f"Local PaliGemma tokenizer is incomplete: {TOKENIZER_ROOT}"
        )

    _install_local_tokenizer_override()
    _install_pi05_checkpoint_loader_fix()

    _sequential_eval_main()


if __name__ == "__main__":
    main()
