"""Portable interpreter selection and backend-specific expert-count regressions."""
from argparse import Namespace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from exllamav3.model.hetero import HeterogeneousLayerSplit, resolve_rocm_python
from exllamav3.modules.block_sparse_mlp import _count_expert_assignments
import bench_modes


class HeterogeneousRuntimeConfigTests(unittest.TestCase):
    def executable(self, directory, name):
        path = Path(directory) / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return str(path)

    def test_rocm_interpreter_argument_precedes_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            explicit = self.executable(directory, "rocm-explicit")
            environment = self.executable(directory, "rocm-environment")
            with patch.dict(os.environ, {"EXL3_ROCM_PYTHON": environment}):
                self.assertEqual(resolve_rocm_python(explicit), explicit)
                self.assertEqual(resolve_rocm_python(), environment)

    def test_heterogeneous_benchmark_fails_before_model_construction_without_interpreter(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires a ROCm interpreter"):
                bench_modes.run_one(Namespace(mode="cuda+rocm", rocm_python=None))

    def test_transport_smoke_fails_before_arena_or_worker_without_interpreter(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("exllamav3.model.hetero.HeteroSharedArena") as arena, \
                 patch("exllamav3.model.hetero.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "requires a ROCm interpreter"):
                    from exllamav3.model.hetero import smoke_rocm_worker
                    smoke_rocm_worker()
                arena.assert_not_called()
                spawn.assert_not_called()

    def test_missing_or_invalid_interpreter_fails_before_engine_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            not_executable = Path(directory) / "not-executable"
            not_executable.write_text("not executable")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "pass rocm_python=.*EXL3_ROCM_PYTHON"):
                    resolve_rocm_python()
                with patch("torch.cuda.is_available") as cuda_available, \
                     patch("exllamav3.model.hetero.HeteroSharedArena") as arena:
                    with self.assertRaisesRegex(RuntimeError, "ROCm Python is not executable"):
                        HeterogeneousLayerSplit(object(), 1, rocm_python=str(not_executable))
                    cuda_available.assert_not_called()
                    arena.assert_not_called()

    def test_all_mode_uses_selected_interpreters_and_propagates_rocm_path(self):
        with tempfile.TemporaryDirectory() as directory:
            rocm = self.executable(directory, "rocm")
            cuda = self.executable(directory, "cuda")
            args = Namespace(
                rocm_python=rocm, cuda_python=cuda, model="model-dir", prefill_tokens=768,
                decode_tokens=128, prompt="test prompt", cuda_devices="0,1",
                cuda_reserve_gb="1,2", cuda_use_gb=None, mcl=8,
                cuda_layers=27, arena_mb=64,
            )
            with patch("bench_modes.subprocess.run") as run:
                bench_modes.run_all(args)

            self.assertEqual([call.args[0][0] for call in run.call_args_list], [cuda, rocm, cuda])
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--rocm-python") + 1], rocm)
                self.assertEqual(call.kwargs["env"]["EXL3_ROCM_PYTHON"], rocm)
            self.assertIn("--cuda-layers", run.call_args_list[2].args[0])

    def test_cuda_and_rocm_expert_counts_are_equivalent(self):
        assignments = torch.tensor([0, 3, 1, 3, 0, 2, 3, 4, 1, 4], dtype=torch.long)
        with patch("exllamav3.modules.block_sparse_mlp.torch.bincount", wraps=torch.bincount) as bincount:
            cuda = _count_expert_assignments(assignments, 4, hip=False)
            bincount.assert_called_once_with(assignments, minlength=5)
            bincount.reset_mock()
            hip = _count_expert_assignments(assignments, 4, hip=True)
            bincount.assert_not_called()
        torch.testing.assert_close(cuda, torch.tensor([2, 2, 1, 3, 2]))
        torch.testing.assert_close(hip, cuda)


if __name__ == "__main__":
    unittest.main()
