"""Run the installed RLinf with the world-model memory handoff actor."""

import os
import runpy
import signal
from pathlib import Path


def main():
    import ray
    import ray._private.parameter as ray_parameter_module
    import rlinf.runners.embodied_runner as runner_module
    import rlinf.scheduler.cluster.cluster as cluster_module
    import rlinf.workers.actor.fsdp_actor_worker as actor_module
    import rlinf.workers.env.env_worker as env_module
    import rlinf.workers.rollout.hf.huggingface_worker as rollout_module
    import torch
    from spatial_wovr_actor import WorldModelFSDPActor
    from spatial_wovr_diagnostics import bind_environment_diagnostics
    from spatial_wovr_eval import SpatialEvaluationMixin
    from spatial_wovr_hybrid import HybridRunnerMixin, HybridSpatialEnvWorkerMixin
    from spatial_wovr_memory_guard import MemoryGuardRunnerMixin
    from spatial_wovr_rollout import CachedWorldModelRollout
    from spatial_wovr_runtime import (
        bind_private_ray_agent_ports,
        bind_ray_failure_cleanup,
    )

    bind_ray_failure_cleanup(cluster_module, ray)
    bind_private_ray_agent_ports(ray_parameter_module)
    bind_environment_diagnostics(runner_module)

    class SpatialWovrRunner(
        HybridRunnerMixin,
        MemoryGuardRunnerMixin,
        SpatialEvaluationMixin,
        runner_module.EmbodiedRunner,
    ):
        pass

    class HybridSpatialEnvWorker(HybridSpatialEnvWorkerMixin, env_module.EnvWorker):
        pass

    torch.cuda.memory._set_allocator_settings(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    actor_module.EmbodiedFSDPActor = WorldModelFSDPActor
    env_module.EnvWorker = HybridSpatialEnvWorker
    rollout_module.MultiStepRolloutWorker = CachedWorldModelRollout
    runner_module.EmbodiedRunner = SpatialWovrRunner

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        runpy.run_path(
            str(
                Path(os.environ["RLINF_BASE_ROOT"])
                / "examples/embodiment/train_embodied_agent.py"
            ),
            run_name="__main__",
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
