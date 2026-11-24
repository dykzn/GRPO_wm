"""Retain reusable CUDA allocator memory inside a GRPO actor process."""

import logging
import math
import os

GIB = 1024**3
ALIGNMENT = 2 * 1024**2
logger = logging.getLogger(__name__)


def reservation_target(
    reserved: int,
    free: int,
    total: int,
    target_fraction: float,
    headroom: int,
) -> int:
    """Limit new reservations to the target and globally free device memory."""
    if not math.isfinite(target_fraction) or not 0 < target_fraction < 1:
        raise ValueError("memory_pool.target_fraction must be between 0 and 1")
    if headroom < 0:
        raise ValueError("memory_pool.device_headroom_gib must be nonnegative")
    desired = int(total * target_fraction)
    available = reserved + max(0, free - headroom)
    # Do not release an existing cache when another process consumes memory.
    target = min(desired, available) // ALIGNMENT * ALIGNMENT
    return max(reserved, target)


class RetainedCudaPlatform:
    """Delegate CUDA operations while retaining the actor's unused cache.

    Only the owning actor's platform is wrapped. Rollout and simulator workers
    keep their normal allocator behavior. PyTorch can still reclaim cached
    blocks internally when an allocation needs them.
    """

    def __init__(self, platform):
        self._platform = platform

    def __getattr__(self, name):
        return getattr(self._platform, name)

    def empty_cache(self):
        """Keep cached blocks available for the next actor update."""


class CudaCacheReservation:
    """Warm PyTorch's native allocator and leave the buffers in its free list.

    No live placeholder tensor or helper process owns the retained memory.
    Training tensors reuse these blocks through the normal CUDA allocator.
    This is best-effort retention, not an exclusive GPU reservation.
    """

    def __init__(self, torch_module, device, target_fraction, headroom_gib, rank):
        if not math.isfinite(headroom_gib) or headroom_gib < 0:
            raise ValueError("memory_pool.device_headroom_gib must be finite and nonnegative")
        reservation_target(0, 0, 1, target_fraction, 0)
        self.torch = torch_module
        self.cuda = torch_module.cuda
        self.device = device
        self.target_fraction = target_fraction
        self.headroom = int(headroom_gib * GIB)
        self.rank = rank
        self._last_report = None

    def retain(self, phase):
        """Top up this actor's cache without consuming the device headroom."""
        with self.cuda.device(self.device):
            self.cuda.synchronize()
            reserved = self.cuda.memory_reserved(self.device)
            free, total = self.cuda.mem_get_info(self.device)
            target = reservation_target(
                reserved, free, total, self.target_fraction, self.headroom
            )
            if target > reserved:
                # Request the whole inactive region, not just its growth:
                # existing free blocks otherwise satisfy the allocation and
                # the allocator never reaches the desired reserved size.
                allocated = self.cuda.memory_allocated(self.device)
                try:
                    buffer = self.torch.empty(
                        target - allocated,
                        dtype=self.torch.uint8,
                        device=self.device,
                    )
                    del buffer
                    self.cuda.synchronize()
                except self.torch.OutOfMemoryError:
                    # Another process may allocate between mem_get_info and
                    # empty(). Do not turn an optional reservation into a
                    # training failure, or spin trying to take its memory.
                    logger.warning(
                        "[actor-memory-pool rank=%s] reservation raced with "
                        "another allocation; continuing with the available cache",
                        self.rank,
                    )
            reserved = self.cuda.memory_reserved(self.device)
            free, total = self.cuda.mem_get_info(self.device)
            report = (reserved // ALIGNMENT, target // ALIGNMENT)
            if phase == "initialized" or report != self._last_report:
                logger.warning(
                    "[actor-memory-pool rank=%s phase=%s] physical CUDA devices=%s "
                    "allocated=%.2f GiB reserved=%.2f GiB device_free=%.2f GiB "
                    "target=%.2f GiB headroom=%.2f GiB",
                    self.rank,
                    phase,
                    os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
                    self.cuda.memory_allocated(self.device) / GIB,
                    reserved / GIB,
                    free / GIB,
                    target / GIB,
                    self.headroom / GIB,
                )
                self._last_report = report
