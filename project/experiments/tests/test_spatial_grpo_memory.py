"""Exercise memory budgets and cache reuse without a CUDA device."""

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_grpo_memory import (  # noqa: E402
    ALIGNMENT,
    GIB,
    CudaCacheReservation,
    RetainedCudaPlatform,
    reservation_target,
)


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    def __init__(self, allocated=10 * GIB, reserved=12 * GIB, free=58 * GIB):
        self.allocated = allocated
        self.reserved = reserved
        self.free = free
        self.total = 80 * GIB
        self.empty_calls = 0
        self.requests = []
        self.race_oom = False

    def device(self, _device):
        return nullcontext()

    def synchronize(self):
        pass

    def memory_allocated(self, _device):
        return self.allocated

    def memory_reserved(self, _device):
        return self.reserved

    def mem_get_info(self, _device):
        return self.free, self.total

    def empty_cache(self):
        self.empty_calls += 1

    def empty(self, size, dtype, device):
        self.requests.append(size)
        if self.race_oom:
            raise FakeOutOfMemoryError("another process acquired memory")
        available_cache = self.reserved - self.allocated
        growth = max(0, size - available_cache)
        if growth > self.free:
            raise FakeOutOfMemoryError("insufficient device memory")
        self.reserved += growth
        self.free -= growth
        self.allocated += size
        return FakeBuffer(self, size)


class FakeBuffer:
    def __init__(self, cuda, size):
        self.cuda = cuda
        self.size = size

    def __del__(self):
        self.cuda.allocated -= self.size


class CudaCacheReservationTests(unittest.TestCase):
    def make_pool(self, cuda, fraction=0.8, headroom=6.0):
        fake_torch = SimpleNamespace(
            cuda=cuda,
            empty=cuda.empty,
            uint8="uint8",
            OutOfMemoryError=FakeOutOfMemoryError,
        )
        return CudaCacheReservation(fake_torch, 0, fraction, headroom, 0)

    def test_warms_to_target_and_keeps_no_placeholder_tensor(self):
        cuda = FakeCuda()
        pool = self.make_pool(cuda)
        pool.retain("initialized")
        self.assertEqual(cuda.reserved, 64 * GIB)
        self.assertEqual(cuda.allocated, 10 * GIB)
        self.assertEqual(cuda.free, 6 * GIB)
        # Account for cache already available, rather than only its growth.
        self.assertEqual(cuda.requests, [54 * GIB])

    def test_training_can_reuse_the_retained_blocks(self):
        cuda = FakeCuda()
        pool = self.make_pool(cuda)
        pool.retain("initialized")
        batch = cuda.empty(20 * GIB, "uint8", 0)
        self.assertEqual(cuda.free, 6 * GIB)
        del batch
        pool.retain("after_training")
        self.assertEqual(len(cuda.requests), 2)
        self.assertEqual(cuda.reserved, 64 * GIB)

    def test_other_process_memory_limits_the_target(self):
        cuda = FakeCuda(free=20 * GIB)
        self.make_pool(cuda).retain("initialized")
        self.assertEqual(cuda.reserved, 26 * GIB)
        self.assertEqual(cuda.free, 6 * GIB)

    def test_insufficient_headroom_does_not_allocate_or_release_cache(self):
        cuda = FakeCuda(free=3 * GIB)
        self.make_pool(cuda).retain("initialized")
        self.assertEqual(cuda.requests, [])
        self.assertEqual(cuda.reserved, 12 * GIB)
        self.assertEqual(cuda.empty_calls, 0)

    def test_allocation_race_is_logged_and_does_not_fail_training(self):
        cuda = FakeCuda()
        cuda.race_oom = True
        with self.assertLogs("spatial_grpo_memory", level="WARNING") as messages:
            self.make_pool(cuda).retain("initialized")
        self.assertTrue(any("raced" in message for message in messages.output))
        self.assertEqual(cuda.allocated, 10 * GIB)

    def test_actor_cache_retention_does_not_change_underlying_platform(self):
        cuda = FakeCuda()
        actor_platform = RetainedCudaPlatform(cuda)
        actor_platform.empty_cache()
        self.assertEqual(cuda.empty_calls, 0)
        self.assertEqual(actor_platform.mem_get_info(0), (cuda.free, cuda.total))
        cuda.empty_cache()
        self.assertEqual(cuda.empty_calls, 1)

    def test_invalid_settings_rejected(self):
        for fraction in [0, 1, -0.1, float("inf"), float("nan")]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.make_pool(FakeCuda(), fraction=fraction)
        for headroom in [-1, float("inf"), float("nan")]:
            with self.subTest(headroom=headroom), self.assertRaises(ValueError):
                self.make_pool(FakeCuda(), headroom=headroom)

    def test_target_rounds_down_without_exceeding_free_budget(self):
        target = reservation_target(12 * GIB, 20 * GIB + 12345, 80 * GIB, 0.8, 6 * GIB)
        self.assertEqual(target % ALIGNMENT, 0)
        self.assertLessEqual(target, 26 * GIB + 12345)


if __name__ == "__main__":
    unittest.main()
