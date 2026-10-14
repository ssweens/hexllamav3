"""Unit coverage for TabbyAPI's load_gen heterogeneous-loader entry point."""
import os
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import exllamav3.model.model as model_module
from exllamav3.model.model import Model


class HeterogeneousLoadGenTests(unittest.TestCase):
    def test_text_load_gen_routes_to_heterogeneous_loader(self):
        model = object.__new__(Model)
        model.component = "text"
        calls = {}

        def load_heterogeneous(self, **kwargs):
            calls.update(kwargs)

        model.load_heterogeneous = MethodType(load_heterogeneous, model)
        progress = []
        with (
            patch.object(model_module.torch.cuda, "device_count", return_value=3),
            patch.dict(os.environ, {"EXL3_HETERO_CUDA_LAYERS": "27"}, clear=True),
        ):
            self.assertEqual(
                list(
                    model.load_gen(
                        reserve_per_device=[3.0, 2.0, 2.0],
                        max_chunk_size=2048,
                        max_batch_size=1,
                        callback=lambda current, total: progress.append((current, total)),
                    )
                ),
                [None],
            )

        self.assertEqual(
            calls,
            {
                "cuda_layers": 27,
                "cuda_devices": [0, 1, 2],
                "reserve_per_device": [3.0, 2.0, 2.0],
                "use_per_device": None,
                "max_chunk_size": 2048,
                "max_output_size": 32,
                "max_output_factor": 1,
                "max_batch_size": 1,
                "progressbar": False,
                "verbose": False,
                "autosplit_no_forward": False,
                "arena_mb": 256,
            },
        )
        self.assertEqual(progress, [(1, 1)])

    def test_non_text_component_keeps_ordinary_loader(self):
        model = object.__new__(Model)
        model.component = "vision"
        model.config = SimpleNamespace(infer_params=SimpleNamespace())
        model.modules = [SimpleNamespace(device="cuda:0")]
        model.cache_weakrefs = {}
        model._load_single = lambda *_args: None

        def unexpected(*_args, **_kwargs):
            raise AssertionError("heterogeneous loader must only receive text components")

        model.load_heterogeneous = unexpected
        with patch.dict(os.environ, {"EXL3_HETERO_CUDA_LAYERS": "27"}, clear=False):
            self.assertEqual(list(model.load_gen(device="cuda:0")), [])


if __name__ == "__main__":
    unittest.main()
