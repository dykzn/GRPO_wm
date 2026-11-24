"""Retain the warmed rollout cache alongside the actor-owned Wan pool."""

import torch
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from spatial_wovr_hybrid import (
    HybridRolloutMixin,
    install_value_head_checkpoint_compat,
)


class CachedWorldModelRollout(HybridRolloutMixin, MultiStepRolloutWorker):
    def init_worker(self):
        rollout_model = self.cfg.get("rollout", {}).get("model", {})
        if bool(rollout_model.get("add_value_head", False)):
            install_value_head_checkpoint_compat(
                seed=int(self.cfg.actor.get("seed", 42))
            )
        if self.cfg.actor.get("shared_wan_pool", {}).get("enabled", False):
            from spatial_wovr_shared_pool import retain_process_cache

            # Returning rollout cache after every weight sync lets the actor
            # take that space, leaving rollout unable to reach its next peak.
            retain_process_cache(torch)
        result = super().init_worker()
        settings = self.cfg.actor.get("shared_wan_pool", {})
        budgets = settings.get("peer_allocator_gib", {})
        if settings.get("enabled") and budgets:
            from spatial_wovr_shared_pool import reserve_peer_allocator

            reserve_peer_allocator(torch, self.device, "rollout", self._rank,
                                   budgets["rollout"], settings.log_dir)
        return result
