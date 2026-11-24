"""An RLinf actor with a persistent, reusable CUDA allocator cache."""

import torch
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from spatial_grpo_memory import CudaCacheReservation, RetainedCudaPlatform


class ReservedEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Keep update memory in the actor process between GRPO iterations."""

    def init_worker(self):
        pool = self.cfg.actor.get("memory_pool", {})
        if pool.get("enabled", False):
            if self.enable_offload or self.cfg.rollout.get("enable_offload", False):
                raise ValueError(
                    "The retained memory pool requires actor/rollout offload=false"
                )
            if self.torch_device_type != "cuda":
                raise ValueError("The retained memory pool requires CUDA")
            if torch.cuda.get_allocator_backend() != "native":
                raise ValueError(
                    "The retained memory pool requires the native CUDA allocator"
                )
        super().init_worker()
        self._cuda_cache_reservation = None
        if pool.get("enabled", False):
            self._cuda_cache_reservation = CudaCacheReservation(
                torch,
                self.device,
                float(pool.get("target_fraction", 0.80)),
                float(pool.get("device_headroom_gib", 6.0)),
                self._rank,
            )
            self.torch_platform = RetainedCudaPlatform(self.torch_platform)
            # RLinf initializes rollout and simulator workers before actors,
            # so their resident memory is already excluded by mem_get_info().
            if not self.cfg.actor.get("shared_wan_pool", {}).get("enabled", False):
                self._cuda_cache_reservation.retain("initialized")

    def run_training(self):
        profile_cfg = self.cfg.actor.get("gpu_memory_profile", {})
        profile = None
        if profile_cfg.get("enabled", False):
            from spatial_wovr_gpu_peak import ActorUpdatePeak

            profile = ActorUpdatePeak(self, torch)
            profile.start()
        self._active_gpu_peak_profile = profile
        try:
            result = super().run_training()
        except BaseException as error:
            if profile is not None:
                try:
                    profile.finish(error=error)
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception(
                        "Could not save the failed GPU peak probe"
                    )
            raise
        else:
            if profile is not None:
                profile.finish(metrics=result)
        finally:
            self._active_gpu_peak_profile = None
        if self._cuda_cache_reservation is not None:
            self._cuda_cache_reservation.retain("after_training")
        return result

    def train_micro_batch(self, *args, **kwargs):
        result = super().train_micro_batch(*args, **kwargs)
        profile = getattr(self, "_active_gpu_peak_profile", None)
        if profile is not None:
            profile.after_backward()
        return result
