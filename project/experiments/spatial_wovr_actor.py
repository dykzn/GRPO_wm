"""Support legacy cache handoff or an actor-owned Wan sharing the allocator."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from spatial_grpo_actor import ReservedEmbodiedFSDPActor
from spatial_wovr_diagnostics import (
    RATIO_FIELDS,
    MicroBatchDiagnostics,
    count_grpo_groups,
    group_metrics,
    sum_across_ranks,
)
from spatial_wovr_hybrid import (
    compute_paired_step_schedule,
    compute_wm_rho,
    install_value_head_checkpoint_compat,
    load_state_dict_allow_missing_value_head,
    merge_real_and_wm_gradients,
    parse_hybrid_config,
    snapshot_param_grads_to_cpu,
)
from spatial_wovr_kl import (
    compute_embodied_kl_loss,
    compute_ref_logprobs,
    snapshot_trainable_weights,
    validate_kl_config,
)


class WorldModelFSDPActor(ReservedEmbodiedFSDPActor):
    """Respect the memory handoff between colocated GRPO and world models."""

    def model_provider_func(self):
        ckpt_path = self.cfg.runner.get("ckpt_path", None)
        add_value_head = bool(self.cfg.actor.model.get("add_value_head", False))
        if add_value_head:
            install_value_head_checkpoint_compat(seed=int(self.cfg.actor.get("seed", 42)))
        if not (ckpt_path and add_value_head):
            return super().model_provider_func()
        from rlinf.models import get_model

        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super(ReservedEmbodiedFSDPActor, self).model_provider_func()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        load_state_dict_allow_missing_value_head(model, state_dict)
        return model

    def _snapshot_policy_reference_weights(self) -> dict[str, torch.Tensor]:
        exclude = (
            ("value_head",)
            if bool(self.cfg.actor.model.get("add_value_head", False))
            else ()
        )
        include_names = getattr(self, "store_requires_grad_param_name", None) or None
        return snapshot_trainable_weights(
            self.model,
            include_names=include_names,
            exclude_substrings=exclude,
        )

    def init_worker(self):
        super().init_worker()
        self._kl_beta, self._kl_penalty_type = validate_kl_config(
            self.cfg.get("algorithm", None)
        )
        self._hybrid_cfg = parse_hybrid_config(self.cfg.get("algorithm", None))
        self.real_rollout_batch = None
        self._hybrid_wm_skipped = False
        self._hybrid_effective_rho = 0.0
        self._hybrid_last_real_entropy = None
        self._hybrid_post_warmup_steps = 0
        self._ref_trainable_weights = None
        if self._any_kl_enabled() and self.cfg.runner.get("resume_dir", None) is None:
            self._ref_trainable_weights = self._snapshot_policy_reference_weights()
        if self.cfg.actor.get("shared_wan_pool", {}).get("enabled", False):
            from spatial_wovr_shared_pool import start_actor_wan

            start_actor_wan(self)

    def _hybrid_enabled(self) -> bool:
        hybrid = getattr(self, "_hybrid_cfg", None)
        return bool(hybrid is not None and hybrid.enabled)

    def _real_kl_beta(self) -> float:
        if self._hybrid_enabled() and self._hybrid_cfg.real_kl_beta is not None:
            return float(self._hybrid_cfg.real_kl_beta)
        return float(getattr(self, "_kl_beta", 0.0))

    def _wm_kl_beta(self) -> float:
        if self._hybrid_enabled() and self._hybrid_cfg.wm_kl_beta is not None:
            return float(self._hybrid_cfg.wm_kl_beta)
        return float(getattr(self, "_kl_beta", 0.0))

    def _any_kl_enabled(self) -> bool:
        return (
            self._kl_enabled()
            or self._real_kl_beta() > 0.0
            or self._wm_kl_beta() > 0.0
        )

    def _kl_enabled(self) -> bool:
        return getattr(self, "_kl_beta", 0.0) > 0.0

    def _model_forward_kwargs(self, prev_logprobs: torch.Tensor | None = None) -> dict:
        from rlinf.config import SupportedModel

        model_type = SupportedModel(self.cfg.actor.model.model_type)
        kwargs = {}
        if model_type in [SupportedModel.OPENVLA, SupportedModel.OPENVLA_OFT]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif prev_logprobs is not None and model_type in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs
        return kwargs

    def _shared_actor_phase(self, name):
        owner = getattr(self, "_shared_wan_owner", None)
        return owner.actor_phase(name) if owner is not None else nullcontext()

    def _process_received_rollout_batch(self, rollout_batch: dict) -> dict:
        result = super()._process_received_rollout_batch(rollout_batch)
        from rlinf.utils.metric_utils import compute_loss_mask

        prefix_mask, _ = compute_loss_mask(result["dones"])
        self._grpo_group_counts = count_grpo_groups(
            result["rewards"],
            prefix_mask,
            result["loss_mask"],
            int(self.cfg.algorithm.group_size),
        )
        return result

    async def recv_real_rollout_trajectories(self, input_channel) -> None:
        """Receive real-simulator trajectories for the PPO stream."""
        from rlinf.data.embodied_io_struct import convert_trajectories_to_batch
        from rlinf.utils.metric_utils import compute_loss_mask, compute_split_num
        from rlinf.utils.utils import clear_memory
        from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_adv

        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        real_batch = convert_trajectories_to_batch(recv_list)
        real_cfg = self.cfg.env.real_train
        real_batch = process_nested_dict_for_adv(real_batch, int(real_cfg.rollout_epoch))
        if not real_cfg.auto_reset and not real_cfg.ignore_terminations:
            loss_mask, loss_mask_sum = compute_loss_mask(real_batch["dones"])
            real_batch["loss_mask"] = loss_mask.any(dim=-1, keepdim=True)
            real_batch["loss_mask_sum"] = loss_mask_sum[..., -1:]
        self.real_rollout_batch = real_batch

    def compute_real_advantages_and_returns(self) -> dict:
        """Compute GAE advantages & returns strictly on the real LIBERO stream."""
        from rlinf.algorithms.registry import calculate_adv_and_returns
        from rlinf.utils.metric_utils import compute_rollout_metrics

        if self.real_rollout_batch is None:
            raise RuntimeError("compute_real_advantages_and_returns called without real_rollout_batch")
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": "gae",
            "rewards": self.real_rollout_batch["rewards"],
            "dones": self.real_rollout_batch["dones"],
            "values": self.real_rollout_batch.get("prev_values", None),
            "prev_logprobs": self.real_rollout_batch.get("prev_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 0.99),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 0.95),
            "reward_type": "chunk_level",
            "loss_mask": self.real_rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.real_rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get("normalize_advantages", True),
        }
        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.real_rollout_batch.update(adv_and_ret)
        if torch.distributed.is_initialized():
            metrics = compute_rollout_metrics(self.real_rollout_batch)
        else:
            metrics = {
                "rewards_mean": float(self.real_rollout_batch["rewards"].float().mean().item()),
                "advantages_mean": float(
                    self.real_rollout_batch["advantages"].float().mean().item()
                ),
                "returns_mean": float(
                    self.real_rollout_batch["returns"].float().mean().item()
                ),
            }
        metrics["hybrid/in_critic_warmup"] = float(
            self.optimizer_steps < self.critic_warmup_steps
        )
        return metrics

    def should_run_wm_stream(self) -> bool:
        """Return False during critic warmup (or rho_max=0) to skip Wan+Cosmos."""
        if not self._hybrid_enabled():
            return True
        if self._hybrid_cfg.rho_max <= 0.0:
            return False
        if (
            self._hybrid_cfg.skip_wm_during_critic_warmup
            and self.optimizer_steps < self.critic_warmup_steps
        ):
            return False
        return True

    def skip_wm_stream(self) -> dict:
        """Record an intentionally skipped WM pass (e.g. during critic warmup)."""
        self.rollout_batch = None
        self._hybrid_wm_skipped = True
        self._hybrid_effective_rho = 0.0
        return {
            "hybrid/wm_skipped": 1.0,
            "hybrid/rho": 0.0,
            "hybrid/in_critic_warmup": float(
                self.optimizer_steps < self.critic_warmup_steps
            ),
        }

    def compute_advantages_and_returns(self) -> dict:
        metrics = super().compute_advantages_and_returns()
        metrics.update(
            group_metrics(sum_across_ranks(self._grpo_group_counts, self.device))
        )
        if self._hybrid_enabled():
            self._hybrid_wm_skipped = False
            in_warmup = self.optimizer_steps < self.critic_warmup_steps
            self._hybrid_effective_rho = compute_wm_rho(
                self._hybrid_cfg,
                in_critic_warmup=in_warmup,
                post_warmup_step=self._hybrid_post_warmup_steps,
                real_entropy=self._hybrid_last_real_entropy,
                wm_signal_groups=metrics.get("grpo_groups_signal", None),
            )
            metrics["hybrid/wm_skipped"] = 0.0
            metrics["hybrid/rho"] = float(self._hybrid_effective_rho)
            metrics["hybrid/in_critic_warmup"] = float(in_warmup)
            if self._hybrid_last_real_entropy is not None:
                metrics["hybrid/real_entropy_prev"] = float(
                    self._hybrid_last_real_entropy
                )
        return metrics

    async def sync_model_to_rollout(self):
        # Wan onloads for the next imagination phase. The actor's inactive
        # cache must not prevent another process from loading its models.
        shared = getattr(self, "_shared_wan_owner", None)
        if self._cuda_cache_reservation is not None and shared is None:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        with self._shared_actor_phase("weight_sync"):
            result = await super().sync_model_to_rollout()
        if shared is not None:
            self._cuda_cache_reservation.retain("after_weight_sync")
        if self.cfg.runner.get("memory_guard", {}).get("enabled", False):
            # Synchronization can create a new temporary state-dict cache.
            # Release it before the coordinator checks the Wan load budget;
            # otherwise it would wait on cache owned by this same job.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        return result

    def _ensure_ref_weights(self) -> None:
        if self._ref_trainable_weights is None:
            self._ref_trainable_weights = self._snapshot_policy_reference_weights()

    def _train_real_ppo_micro_batch(
        self,
        micro_batch: dict,
        metrics: dict,
        *,
        loss_scale: float,
        is_last: bool,
        in_critic_warmup: bool,
        want_entropy: bool,
        real_entropy_accumulator: list[float],
    ) -> None:
        from rlinf.algorithms.registry import policy_loss
        from rlinf.utils.metric_utils import append_to_dict
        from rlinf.utils.nested_dict_process import put_tensor_device
        from rlinf.utils.utils import masked_mean, reshape_entropy

        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        ref_logprobs = micro_batch.get("ref_logprobs", None)
        returns = micro_batch["returns"]
        prev_values = micro_batch["prev_values"]
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = self._model_forward_kwargs(prev_logprobs=prev_logprobs)
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=want_entropy,
                compute_values=True,
                use_cache=False,
                **kwargs,
            )

        hcfg = self._hybrid_cfg
        real_max_steps = int(self.cfg.env.real_train.max_episode_steps)
        single_action_dim = int(self.cfg.actor.model.get("action_dim", 7))
        loss_kwargs = {
            "loss_type": "actor_critic",
            "logprob_type": "chunk_level",
            "reward_type": "chunk_level",
            "single_action_dim": single_action_dim,
            "logprobs": output_dict["logprobs"],
            "values": output_dict["values"],
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": hcfg.real_clip_ratio_high,
            "clip_ratio_low": hcfg.real_clip_ratio_low,
            "value_clip": hcfg.real_value_clip,
            "huber_delta": hcfg.real_huber_delta,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": real_max_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": in_critic_warmup,
        }
        loss, raw_metrics = policy_loss(**loss_kwargs)
        if hcfg.real_value_loss_coef != 1.0:
            # Re-weight the critic term cleanly while keeping policy_loss contract.
            from rlinf.algorithms.losses import compute_ppo_critic_loss
            from rlinf.algorithms.utils import preprocess_loss_inputs

            prep = preprocess_loss_inputs(**loss_kwargs)
            critic_loss, _ = compute_ppo_critic_loss(**prep)
            loss = loss + (hcfg.real_value_loss_coef - 1.0) * critic_loss

        real_kl_beta = self._real_kl_beta()
        if real_kl_beta > 0.0 and ref_logprobs is not None:
            kl_penalty_loss, kl_metrics = compute_embodied_kl_loss(
                output_dict["logprobs"],
                ref_logprobs,
                kl_beta=real_kl_beta,
                kl_penalty_type=self._kl_penalty_type,
                logprob_type="chunk_level",
                reward_type="chunk_level",
                single_action_dim=single_action_dim,
                loss_mask=loss_mask,
                loss_mask_sum=loss_mask_sum,
                max_episode_steps=real_max_steps,
                critic_warmup=in_critic_warmup,
            )
            loss = loss + kl_penalty_loss
            raw_metrics.update(kl_metrics)

        if want_entropy and output_dict.get("entropy", None) is not None:
            entropy = reshape_entropy(
                output_dict["entropy"],
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=single_action_dim,
                batch_size=output_dict["logprobs"].shape[0],
            )
            ent_mask = None
            if loss_mask is not None:
                ent_mask = loss_mask.bool()
                while ent_mask.ndim < entropy.ndim:
                    ent_mask = ent_mask.unsqueeze(-1)
                ent_mask = ent_mask.expand_as(entropy)
            ent_mean = masked_mean(entropy, mask=ent_mask)
            real_entropy_accumulator.append(float(ent_mean.detach().item()))
            if self.cfg.algorithm.entropy_bonus > 0 and not in_critic_warmup:
                loss = loss - self.cfg.algorithm.entropy_bonus * ent_mean
            raw_metrics["actor/entropy_loss"] = float(ent_mean.detach().item())

        scaled_loss = loss * float(loss_scale)
        with backward_ctx:
            self.grad_scaler.scale(scaled_loss).backward()

        logged = {}
        for key, value in raw_metrics.items():
            if key.startswith("critic/"):
                logged[key] = value
            elif key.startswith("actor/"):
                logged[f"real/{key[len('actor/'):]}"] = value
            else:
                logged[key] = value
        logged["real/total_loss"] = float(scaled_loss.detach().item())
        append_to_dict(metrics, logged)

    def _train_wm_grpo_micro_batch(
        self,
        micro_batch: dict,
        metrics: dict,
        *,
        loss_scale: float,
        is_last: bool,
    ) -> None:
        from rlinf.algorithms.registry import policy_loss
        from rlinf.utils.metric_utils import append_to_dict
        from rlinf.utils.nested_dict_process import put_tensor_device
        from rlinf.utils.utils import masked_mean, reshape_entropy

        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        ref_logprobs = micro_batch.get("ref_logprobs", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        single_action_dim = int(self.cfg.actor.model.get("action_dim", 7))
        kwargs = self._model_forward_kwargs(prev_logprobs=prev_logprobs)
        # Never compute or backprop critic values on synthetic world-model data.
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=False,
                use_cache=False,
                **kwargs,
            )

        loss_kwargs = {
            "loss_type": "actor",
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": single_action_dim,
            "logprobs": output_dict["logprobs"],
            "values": None,
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": None,
            "prev_values": None,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": False,
        }
        loss, raw_metrics = policy_loss(**loss_kwargs)

        wm_kl_beta = self._wm_kl_beta()
        if wm_kl_beta > 0.0 and ref_logprobs is not None:
            kl_penalty_loss, kl_metrics = compute_embodied_kl_loss(
                output_dict["logprobs"],
                ref_logprobs,
                kl_beta=wm_kl_beta,
                kl_penalty_type=self._kl_penalty_type,
                logprob_type=self.cfg.algorithm.logprob_type,
                reward_type=self.cfg.algorithm.reward_type,
                single_action_dim=single_action_dim,
                loss_mask=loss_mask,
                loss_mask_sum=loss_mask_sum,
                max_episode_steps=self.cfg.env.train.max_episode_steps,
                critic_warmup=False,
            )
            loss = loss + kl_penalty_loss
            raw_metrics.update(kl_metrics)

        if self.cfg.algorithm.entropy_bonus > 0 and output_dict.get("entropy", None) is not None:
            entropy = reshape_entropy(
                output_dict["entropy"],
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=single_action_dim,
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss
            raw_metrics["actor/entropy_loss"] = float(entropy_loss.detach().item())

        scaled_loss = loss * float(loss_scale)
        with backward_ctx:
            self.grad_scaler.scale(scaled_loss).backward()

        logged = {}
        for key, value in raw_metrics.items():
            if key.startswith("actor/"):
                logged[f"wm/{key[len('actor/'):]}"] = value
            else:
                logged[key] = value
        logged["wm/total_loss"] = float(scaled_loss.detach().item())
        append_to_dict(metrics, logged)

        diagnostics = getattr(self, "_active_training_diagnostics", None)
        if diagnostics is not None and loss_mask is not None:
            diagnostics.record(
                loss_mask,
                {key: raw_metrics[key] for key in RATIO_FIELDS if key in raw_metrics},
            )

    def _run_hybrid_training(self) -> dict:
        """Execute WorldSample Route-A paired Real-PPO + WM-GRPO training."""
        import numpy as np
        from rlinf.utils.distributed import all_reduce_dict
        from rlinf.utils.metric_utils import (
            CRITIC_EXPLAINED_VARIANCE_KEY,
            append_to_dict,
            compute_critic_explained_variance_from_stats,
            pop_critic_explained_variance_stats,
        )
        from rlinf.utils.nested_dict_process import split_dict_to_chunk
        from rlinf.utils.utils import clear_memory, trim_host_memory
        from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_train

        if self.real_rollout_batch is None:
            raise RuntimeError("Hybrid training requires real_rollout_batch")
        if getattr(self, "is_weight_offloaded", False):
            self.load_param_and_grad(self.device)
        if getattr(self, "is_optimizer_offloaded", False):
            self.load_optimizer(self.device)

        hcfg = self._hybrid_cfg
        in_critic_warmup = self.optimizer_steps < self.critic_warmup_steps
        use_wm = (
            not in_critic_warmup
            and not self._hybrid_wm_skipped
            and self._hybrid_effective_rho > 0.0
            and self.rollout_batch is not None
        )
        effective_rho = float(self._hybrid_effective_rho) if use_wm else 0.0

        # Score reference policy logprobs once per outer step in eval/no_grad.
        if not in_critic_warmup and (
            self._real_kl_beta() > 0.0 or (use_wm and self._wm_kl_beta() > 0.0)
        ):
            self._ensure_ref_weights()
            if self._real_kl_beta() > 0.0:
                self.real_rollout_batch["ref_logprobs"] = compute_ref_logprobs(
                    self.model,
                    self.real_rollout_batch,
                    self._ref_trainable_weights,
                    device=self.device,
                    micro_batch_size=int(self.cfg.actor.micro_batch_size),
                    amp_context=self.amp_context,
                    forward_kwargs=self._model_forward_kwargs(),
                )
            if use_wm and self._wm_kl_beta() > 0.0:
                self.rollout_batch["ref_logprobs"] = compute_ref_logprobs(
                    self.model,
                    self.rollout_batch,
                    self._ref_trainable_weights,
                    device=self.device,
                    micro_batch_size=int(self.cfg.actor.micro_batch_size),
                    amp_context=self.amp_context,
                    forward_kwargs=self._model_forward_kwargs(),
                )

        self.model.train()
        mbs = int(self.cfg.actor.micro_batch_size)
        real_mbs = (
            int(hcfg.real_micro_batch_size)
            if hcfg.real_micro_batch_size is not None
            else mbs
        )
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else int(getattr(self, "_world_size", 1))
        )

        # Flatten & shuffle real stream.
        real_size = (
            self.real_rollout_batch["prev_logprobs"].shape[0]
            * self.real_rollout_batch["prev_logprobs"].shape[1]
        )
        g_real = torch.Generator()
        g_real.manual_seed(int(self.cfg.actor.seed) + int(self._rank) + 10007 * int(self.version))
        real_shuffle = torch.randperm(real_size, generator=g_real)
        with torch.no_grad():
            flat_real = process_nested_dict_for_train(self.real_rollout_batch, real_shuffle)
        if real_size % real_mbs != 0:
            raise ValueError(
                f"Real per-rank rollout size {real_size} is not divisible by real_micro_batch_size {real_mbs}"
            )
        real_micro_batches = split_dict_to_chunk(flat_real, real_size // real_mbs)

        wm_micro_batches: list[dict] = []
        if use_wm:
            wm_size = (
                self.rollout_batch["prev_logprobs"].shape[0]
                * self.rollout_batch["prev_logprobs"].shape[1]
            )
            g_wm = torch.Generator()
            g_wm.manual_seed(int(self.cfg.actor.seed) + int(self._rank))
            wm_shuffle = torch.randperm(wm_size, generator=g_wm)
            with torch.no_grad():
                flat_wm = process_nested_dict_for_train(self.rollout_batch, wm_shuffle)
            if wm_size % mbs != 0:
                raise ValueError(
                    f"WM per-rank rollout size {wm_size} is not divisible by micro_batch_size {mbs}"
                )
            wm_micro_batches = split_dict_to_chunk(flat_wm, wm_size // mbs)

        metrics: dict[str, list] = {}
        real_entropy_samples: list[float] = []
        want_real_entropy = (
            hcfg.pacing_mode == "entropy" or float(self.cfg.algorithm.entropy_bonus) > 0.0
        )

        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        if not use_wm:
            # Pure Real PPO (during critic warmup or when WM stream was gated off).
            if hcfg.real_global_batch_size is not None:
                real_per_rank_gb = int(hcfg.real_global_batch_size) // world_size
                real_ga = max(1, real_per_rank_gb // real_mbs)
            else:
                real_ga = len(real_micro_batches)
            epochs = 1 if in_critic_warmup else max(1, hcfg.real_update_epochs)
            for _ in range(epochs):
                for start in range(0, len(real_micro_batches), real_ga):
                    chunk = real_micro_batches[start : start + real_ga]
                    step_in_warmup = self.optimizer_steps < self.critic_warmup_steps
                    self.optimizer.zero_grad()
                    for idx, batch in enumerate(chunk):
                        self._train_real_ppo_micro_batch(
                            batch,
                            metrics,
                            loss_scale=1.0 / float(len(chunk)),
                            is_last=(idx + 1) == len(chunk),
                            in_critic_warmup=step_in_warmup,
                            want_entropy=want_real_entropy,
                            real_entropy_accumulator=real_entropy_samples,
                        )
                    self.torch_platform.empty_cache()
                    grad_norm, lr_list = self.optimizer_step()
                    data = {"actor/grad_norm": grad_norm, "actor/lr": lr_list[0]}
                    if len(lr_list) > 1:
                        data["critic/lr"] = lr_list[1]
                    append_to_dict(metrics, data)
        else:
            # Joint Real-PPO + rho(H_t) * WM-GRPO paired optimizer steps with
            # norm-matched and conflict-free (cosine_gate / pcgrad) gradient fusion.
            wm_per_rank_gb = int(self.cfg.actor.global_batch_size) // world_size
            default_paired_steps = max(1, (len(wm_micro_batches) * mbs) // wm_per_rank_gb)
            target_paired_steps = hcfg.paired_steps_per_epoch or default_paired_steps
            sharding_strategy = str(
                self.cfg.actor.get("fsdp_config", {}).get("sharding_strategy", "no_shard")
            )
            is_sharded_fsdp = sharding_strategy != "no_shard"
            for _ in range(update_epoch):
                real_sched, wm_sched = compute_paired_step_schedule(
                    len(real_micro_batches),
                    len(wm_micro_batches),
                    target_paired_steps=target_paired_steps,
                    real_update_epochs=hcfg.real_update_epochs,
                )
                for real_idxs, wm_idxs in zip(real_sched, wm_sched, strict=True):
                    # Phase 1: Real PPO micro-batches -> FSDP sync on last real mb -> CPU snapshot.
                    self.optimizer.zero_grad()
                    for pos, r_idx in enumerate(real_idxs):
                        self._train_real_ppo_micro_batch(
                            real_micro_batches[r_idx],
                            metrics,
                            loss_scale=1.0 / float(len(real_idxs)),
                            is_last=(pos + 1) == len(real_idxs),
                            in_critic_warmup=False,
                            want_entropy=want_real_entropy,
                            real_entropy_accumulator=real_entropy_samples,
                        )
                    real_actor_grads, real_critic_grads = snapshot_param_grads_to_cpu(
                        self.model
                    )

                    # Phase 2: WM GRPO micro-batches (unit-averaged) -> FSDP sync on last WM mb.
                    self.optimizer.zero_grad()
                    for pos, w_idx in enumerate(wm_idxs):
                        self._train_wm_grpo_micro_batch(
                            wm_micro_batches[w_idx],
                            metrics,
                            loss_scale=1.0 / float(len(wm_idxs)),
                            is_last=(pos + 1) == len(wm_idxs),
                        )

                    # Phase 3: Norm-matched, conflict-free gradient fusion + restore real critic grads.
                    grad_merge_metrics = merge_real_and_wm_gradients(
                        self.model,
                        real_actor_grads,
                        real_critic_grads,
                        rho=effective_rho,
                        grad_norm_match=hcfg.grad_norm_match,
                        grad_conflict_mode=hcfg.grad_conflict_mode,
                        grad_cos_min=hcfg.grad_cos_min,
                        is_sharded_fsdp=is_sharded_fsdp,
                    )
                    del real_actor_grads, real_critic_grads
                    append_to_dict(metrics, grad_merge_metrics)

                    self.torch_platform.empty_cache()
                    grad_norm, lr_list = self.optimizer_step()
                    data = {"actor/grad_norm": grad_norm, "actor/lr": lr_list[0]}
                    if len(lr_list) > 1:
                        data["critic/lr"] = lr_list[1]
                    append_to_dict(metrics, data)
            self._hybrid_post_warmup_steps += 1

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: float(np.mean(val)) for key, val in metrics.items()}
        mean_metric_dict["hybrid/rho"] = effective_rho
        mean_metric_dict["hybrid/in_critic_warmup"] = float(in_critic_warmup)
        if real_entropy_samples:
            mean_metric_dict["hybrid/real_entropy"] = float(np.mean(real_entropy_samples))
        if torch.distributed.is_initialized():
            mean_metric_dict = all_reduce_dict(
                mean_metric_dict, op=torch.distributed.ReduceOp.AVG
            )
        if "hybrid/real_entropy" in mean_metric_dict:
            self._hybrid_last_real_entropy = float(mean_metric_dict["hybrid/real_entropy"])
        if explained_variance_stats:
            reduced_stats = (
                all_reduce_dict(
                    explained_variance_stats, op=torch.distributed.ReduceOp.SUM
                )
                if torch.distributed.is_initialized()
                else explained_variance_stats
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        self.real_rollout_batch = None
        self.rollout_batch = None
        trim_host_memory()
        return mean_metric_dict

    def run_training(self) -> dict:
        diagnostics = MicroBatchDiagnostics()
        self._active_training_diagnostics = diagnostics
        try:
            with self._shared_actor_phase("training"):
                if self._cuda_cache_reservation is not None:
                    self._cuda_cache_reservation.retain("before_training")
                if self._hybrid_enabled():
                    metrics = self._run_hybrid_training()
                else:
                    if self._kl_enabled():
                        if getattr(self, "is_weight_offloaded", False):
                            self.load_param_and_grad(self.device)
                        self._ensure_ref_weights()
                        self.rollout_batch["ref_logprobs"] = compute_ref_logprobs(
                            self.model,
                            self.rollout_batch,
                            self._ref_trainable_weights,
                            device=self.device,
                            micro_batch_size=int(self.cfg.actor.micro_batch_size),
                            amp_context=self.amp_context,
                            forward_kwargs=self._model_forward_kwargs(),
                        )
                    metrics = super().run_training()
        finally:
            self._active_training_diagnostics = None
        if not self._hybrid_enabled() or diagnostics.micro_count > 0:
            metrics.update(diagnostics.reduce(self.device))
        return metrics

    def save_checkpoint(self, *args, **kwargs):
        with self._shared_actor_phase("save_checkpoint"):
            result = super().save_checkpoint(*args, **kwargs)
            if getattr(self, "_shared_wan_owner", None) is not None:
                self._cuda_cache_reservation.retain("after_checkpoint_save")
            return result

    def load_checkpoint(self, *args, **kwargs):
        with self._shared_actor_phase("load_checkpoint"):
            result = super().load_checkpoint(*args, **kwargs)
            if self.cfg.actor.optim.get("use_config_lr_on_resume", False):
                # Keep Adam moments and the scheduler's position when changing its base rate.
                base_lr = float(self.cfg.actor.optim.lr)
                group = self.optimizer.param_groups[0]
                schedule = self.lr_scheduler.state_dict()
                old_base_lr = schedule["base_lrs"][0]
                multiplier = group["lr"] / old_base_lr if old_base_lr else 1.0
                group.update(lr=base_lr * multiplier, initial_lr=base_lr)
                schedule["base_lrs"][0] = base_lr
                schedule["_last_lr"][0] = group["lr"]
                self.lr_scheduler.load_state_dict(schedule)
            if self._any_kl_enabled():
                # Anchor KL regularization to the resumed checkpoint (e.g. step 60).
                self._ref_trainable_weights = self._snapshot_policy_reference_weights()
            if getattr(self, "_shared_wan_owner", None) is not None:
                self._cuda_cache_reservation.retain("after_checkpoint_load")
            return result

    def _train_micro_batch_with_kl(
        self, micro_batch: dict, metrics: dict, *, is_last: bool
    ) -> None:
        from rlinf.algorithms.registry import policy_loss
        from rlinf.config import SupportedModel
        from rlinf.scheduler import Worker
        from rlinf.utils.metric_utils import append_to_dict
        from rlinf.utils.nested_dict_process import put_tensor_device
        from rlinf.utils.utils import masked_mean, reshape_entropy

        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        ref_logprobs = micro_batch["ref_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        model_type = SupportedModel(self.cfg.actor.model.model_type)
        kwargs = self._model_forward_kwargs(prev_logprobs=prev_logprobs)
        compute_values = self.cfg.algorithm.adv_type == "gae"
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if model_type in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }

        if model_type in [SupportedModel.GR00T_N1D6, SupportedModel.GR00T_N1D7]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)

        kl_penalty_loss, kl_metrics = compute_embodied_kl_loss(
            output_dict["logprobs"],
            ref_logprobs,
            kl_beta=self._kl_beta,
            kl_penalty_type=self._kl_penalty_type,
            logprob_type=self.cfg.algorithm.logprob_type,
            reward_type=self.cfg.algorithm.reward_type,
            single_action_dim=self.cfg.actor.model.get("action_dim", 7),
            loss_mask=loss_mask,
            loss_mask_sum=loss_mask_sum,
            max_episode_steps=self.cfg.env.train.max_episode_steps,
            critic_warmup=loss_kwargs["critic_warmup"],
        )
        loss = loss + kl_penalty_loss
        metrics_data.update(kl_metrics)

        entropy_loss = torch.tensor(0.0, device=output_dict["logprobs"].device)
        if self.cfg.algorithm.entropy_bonus > 0 and not loss_kwargs["critic_warmup"]:
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def train_micro_batch(self, micro_batch: dict, metrics: dict, *, is_last: bool):
        if self._kl_enabled() and "ref_logprobs" in micro_batch:
            result = self._train_micro_batch_with_kl(
                micro_batch=micro_batch, metrics=metrics, is_last=is_last
            )
            profile = getattr(self, "_active_gpu_peak_profile", None)
            if profile is not None:
                profile.after_backward()
        else:
            result = super().train_micro_batch(
                micro_batch=micro_batch, metrics=metrics, is_last=is_last
            )
        diagnostics = getattr(self, "_active_training_diagnostics", None)
        if diagnostics is not None:
            diagnostics.record(
                micro_batch["loss_mask"],
                {key: metrics[key][-1] for key in RATIO_FIELDS},
            )
        return result
