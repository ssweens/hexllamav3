"""CPU-only checks that worker responses reuse a bounded shared-memory arena."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from exllamav3.model.hetero_rocm_worker import RocmWorker
from exllamav3.model.hetero_shared import HeteroSharedArena, TensorDescriptor


class _OutputModule:
    caps = {}

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        return x * 2 + 1


class HeterogeneousWorkerArenaTests(unittest.TestCase):
    def setUp(self):
        self.worker = RocmWorker.__new__(RocmWorker)
        self.worker.device = torch.device("cpu")
        self.worker.arena = HeteroSharedArena(1024)
        self.addCleanup(self.worker.arena.close)
        self.worker.modules = [(_OutputModule(), 0, 0)]
        self.worker.model = SimpleNamespace(last_kv_module_idx_instance=(0, 0))
        self.worker.caches = {}
        self.descriptor = TensorDescriptor(0, 1024, "torch.float32", (1, 1, 256)).as_dict()

    def request(self, step):
        x = torch.arange(256, dtype=torch.float32).reshape(1, 1, 256) + step
        # The parent's allocator is independent from the worker's. Simulate its
        # request write without changing the worker's allocation cursor.
        self.worker.arena.cpu_tensor(self.descriptor).copy_(x)
        return x, {"x": self.descriptor, "params": {}}

    def test_forward_responses_reuse_arena_after_inputs_are_consumed(self):
        with patch("torch.cuda.synchronize"):
            for step in range(16):
                x, message = self.request(step)
                response = self.worker.forward(message)
                self.assertEqual(response["x"]["offset"], 0)
                torch.testing.assert_close(self.worker.arena.get(response["x"]), x * 2 + 1)
                self.worker.prefill(self.request(step)[1])

    def test_smoke_responses_reuse_arena(self):
        def add(left, right, output):
            output.copy_(left + right)

        with patch("torch.cuda.synchronize"), patch("exllamav3.ext.exllamav3_ext.add", side_effect=add):
            for step in range(16):
                x, message = self.request(step)
                response = self.worker.smoke_add(message)
                self.assertEqual(response["x"]["offset"], 0)
                torch.testing.assert_close(self.worker.arena.get(response["x"]), x * 2)


if __name__ == "__main__":
    unittest.main()
