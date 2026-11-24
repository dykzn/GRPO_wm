"""Launch the RLinf entry point with the local CUDA cache actor subclass."""

import os
import runpy
from pathlib import Path


def main():
    import torch

    # Parse the allocator options even during Hydra's configuration-only mode,
    # which does not initialize CUDA and otherwise misses invalid settings.
    allocator_settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_settings:
        torch.cuda.memory._set_allocator_settings(allocator_settings)

    import rlinf.workers.actor.fsdp_actor_worker as actor_module

    from spatial_grpo_actor import ReservedEmbodiedFSDPActor

    # Ray imports the subclass from its own module in each worker process.
    # The original RLinf source tree is left unchanged.
    actor_module.EmbodiedFSDPActor = ReservedEmbodiedFSDPActor
    entrypoint = (
        Path(os.environ["RLINF_ROOT"])
        / "examples/embodiment/train_embodied_agent.py"
    )
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
