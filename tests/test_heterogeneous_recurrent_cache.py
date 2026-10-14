"""Heterogeneous recurrent checkpoint routing and worker cleanup."""
from types import SimpleNamespace
import unittest

from exllamav3.cache.recurrent import RecurrentCache
from exllamav3.modules.gated_delta_net import GDNState


class _HeterogeneousEngine:
    def __init__(self):
        self.deleted = []

    def delete_recurrent(self, key):
        self.deleted.append(key)


class _CheckpointState:
    checkpoint_size = 8

    def __init__(self):
        self.stashed_key = None

    def stash(self, key=None):
        self.stashed_key = key
        return {"position": 256, "checkpoint_size": self.checkpoint_size, "hetero_key": (17, key)}


class _LayerState:
    def __init__(self, device):
        self.device = device
        self.stash_calls = 0
        self.unstash_calls = 0

    def stash(self, slot):
        self.stash_calls += 1
        return f"checkpoint-{slot}"

    def unstash(self, slot, checkpoint):
        self.unstash_calls += 1
        assert checkpoint == f"checkpoint-{slot}"


class HeterogeneousRecurrentCacheTests(unittest.TestCase):
    def test_local_stash_skips_metadata_only_recurrent_layers(self):
        allocated = _LayerState("rocm:0")
        metadata_only = _LayerState(None)
        cache = SimpleNamespace(
            model=SimpleNamespace(hetero_engine=None, loaded_tp=False),
            get_all_recurrent_layers=lambda: {0: allocated, 1: metadata_only},
        )
        state = GDNState.__new__(GDNState)
        state.cache, state.position, state.checkpoint_size, state.slot = cache, 256, 8, 3

        checkpoint = state.stash()
        state.unstash(checkpoint)

        self.assertEqual(checkpoint[0], "checkpoint-3")
        self.assertNotIn(1, checkpoint)
        self.assertEqual((allocated.stash_calls, allocated.unstash_calls), (1, 1))
        self.assertEqual((metadata_only.stash_calls, metadata_only.unstash_calls), (0, 0))

    def test_recurrent_stash_receives_page_key_and_worker_checkpoint_is_deleted(self):
        engine = _HeterogeneousEngine()
        model = SimpleNamespace(hetero_engine=engine, loaded_tp=False)
        cache = RecurrentCache(model)
        state = _CheckpointState()

        cache.put("page-hash", state)

        self.assertEqual(state.stashed_key, "page-hash")
        self.assertEqual(cache["page-hash"]["hetero_key"], (17, "page-hash"))
        self.assertEqual(cache.current_size, 8)

        cache.close()
        self.assertEqual(engine.deleted, [(17, "page-hash")])


if __name__ == "__main__":
    unittest.main()
