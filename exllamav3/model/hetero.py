"""CUDA-parent / ROCm-child contiguous layer-split inference engine."""
from __future__ import annotations

import atexit
import os
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
import sys
from typing import TYPE_CHECKING

import torch

from .hetero_shared import HeteroSharedArena

if TYPE_CHECKING:
    from .model import Model

DEFAULT_ROCM_PYTHON = "/home/bigkahuna/src/tabbyAPI/.venv_rocm/bin/python"


class HeterogeneousLayerSplit:
    """Own a persistent ROCm process for the trailing contiguous model layers.

    The parent owns prologue and early transformer layers on CUDA (including any
    normal CUDA layer split selected by the caller); the ROCm process owns the
    remaining transformer layers and epilogue.  Calls are serialized because a
    cache belongs to one model execution stream, exactly as it does in ordinary
    layer-split mode.
    """

    def __init__(
        self,
        model: "Model",
        cuda_layers: int,
        cuda_device: torch.device | str | int = "cuda:0",
        cuda_devices: list[int] | None = None,
        reserve_per_device: list[float] | float | None = None,
        use_per_device: list[float] | float | None = None,
        rocm_device: int = 0,
        rocm_python: str | None = None,
        arena_mb: int | None = None,
        max_chunk_size: int = 2048,
        max_output_size: int = 32,
        max_output_factor: int = 1,
        max_batch_size: int = 1,
        autosplit_no_forward: bool = False,
    ):
        if torch.version.hip:
            raise RuntimeError("load_heterogeneous must be called from the CUDA virtualenv, not the ROCm virtualenv")
        if not torch.cuda.is_available():
            raise RuntimeError("load_heterogeneous requires a CUDA-visible parent device")
        if cuda_layers <= 0 or cuda_layers >= model.config.num_hidden_layers:
            raise ValueError(f"cuda_layers must be in [1, {model.config.num_hidden_layers - 1}]")
        if reserve_per_device is not None and use_per_device is not None:
            raise ValueError("cannot specify both reserve_per_device and use_per_device")
        self.model = model
        self.cuda_layers = int(cuda_layers)
        requested_device = torch.device(cuda_device)
        if requested_device.type != "cuda":
            raise ValueError("cuda_device must name a CUDA device")
        self.cuda_devices = list(cuda_devices) if cuda_devices is not None else [requested_device.index or 0]
        if not self.cuda_devices or len(set(self.cuda_devices)) != len(self.cuda_devices):
            raise ValueError("cuda_devices must be a non-empty list of distinct CUDA device indices")
        if any(not isinstance(device, int) or device < 0 or device >= torch.cuda.device_count()
               for device in self.cuda_devices):
            raise ValueError(f"cuda_devices must be within [0, {torch.cuda.device_count() - 1}]")
        self.cuda_device = torch.device(f"cuda:{self.cuda_devices[0]}")
        self.return_device = self.cuda_device
        self.reserve_per_device = reserve_per_device
        self.use_per_device = use_per_device
        self.max_chunk_size = max_chunk_size
        self.max_output_size = max_output_size
        self.max_output_factor = max_output_factor
        self.max_batch_size = max_batch_size
        self.autosplit_no_forward = autosplit_no_forward
        self.rocm_device = int(rocm_device)
        self.rocm_python = rocm_python or os.environ.get("EXL3_ROCM_PYTHON", DEFAULT_ROCM_PYTHON)
        if not os.path.isfile(self.rocm_python) or not os.access(self.rocm_python, os.X_OK):
            raise RuntimeError(f"ROCm Python is not executable: {self.rocm_python}")
        self.arena = HeteroSharedArena((arena_mb or int(os.environ.get("EXL3_HETERO_ARENA_MB", "64"))) << 20)
        self.conn: Connection | None = None
        self.process: subprocess.Popen | None = None
        self.local_modules = self._select_local_modules()
        self.remote_enabled = bool(self._select_remote_modules())
        if not self.remote_enabled:
            self.arena.close()
            raise RuntimeError("heterogeneous split did not select any ROCm modules")
        self._closed = False

    def _select_local_modules(self) -> list:
        local, seen_block = [], False
        for entry in self.model.fwd_modules:
            module, _instance, _idx = entry
            if module.layer_idx is not None:
                seen_block = True
            remote = (module.layer_idx is not None and module.layer_idx >= self.cuda_layers) or \
                     (seen_block and module.layer_idx is None)
            if not remote:
                local.append(entry)
        return local

    def _select_remote_modules(self) -> list:
        local = set(id(m) for m, _, _ in self.local_modules)
        return [entry for entry in self.model.fwd_modules if id(entry[0]) not in local]

    @staticmethod
    def _cache_spec(parent_id: int, cache) -> dict:
        kwargs = {}
        # CacheLayer_quant's construction parameters live on every layer.
        layer = next(iter(cache.layers.values()), None)
        if layer is not None and hasattr(layer, "k_bits"):
            kwargs = {"k_bits": layer.k_bits, "v_bits": layer.v_bits, "compand_a": layer.compand_a}
        return {
            "parent_id": parent_id,
            "max_num_tokens": cache.max_num_tokens,
            "max_batch_size": cache.num_slots,
            "max_history": cache.max_history,
            "layer_type": cache.layer_type.__name__,
            "kwargs": kwargs,
        }

    def _start_worker(self) -> None:
        # Register before publishing the arena so both processes get a portable mapped alias.
        self.arena.register()
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        parent_fd, child_fd = parent_socket.detach(), child_socket.detach()
        self.conn = Connection(parent_fd)
        root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.process = subprocess.Popen(
            [self.rocm_python, "-m", "exllamav3.model.hetero_rocm_worker", "--fd", str(child_fd),
             "--arena-name", self.arena.name, "--arena-size", str(self.arena.size), "--device", str(self.rocm_device)],
            pass_fds=(child_fd,), env=env,
        )
        os.close(child_fd)
        self._request("alloc_cache", {
            "model_dir": self.model.config.directory,
            "caches": [self._cache_spec(parent_id, c) for parent_id, ref in self.model.cache_weakrefs.items()
                       if (c := ref()) is not None],
        })
        self._request("load_range", {"start_layer": self.cuda_layers})

    def _request(self, command: str, payload: dict | None = None) -> dict:
        if self.conn is None:
            raise RuntimeError("heterogeneous ROCm worker is not running")
        self.conn.send({"command": command, **(payload or {})})
        response = self.conn.recv()
        if not response.get("ok"):
            raise RuntimeError(f"ROCm worker {command} failed: {response.get('error')}\n{response.get('traceback', '')}")
        return response.get("result") or {}

    def _cuda_budgets(self) -> tuple[list[int] | None, list[int] | None]:
        """Map selected-device GiB budgets to Model_LSMixin's global device arrays."""
        device_count = torch.cuda.device_count()
        values = self.reserve_per_device if self.reserve_per_device is not None else self.use_per_device
        reserve = self.reserve_per_device is not None or values is None
        if values is None:
            selected = [0.5] * len(self.cuda_devices)
        elif isinstance(values, (float, int)):
            selected = [float(values)] * len(self.cuda_devices)
        elif isinstance(values, list):
            if len(values) == len(self.cuda_devices):
                selected = values
            elif len(values) == device_count:
                selected = [values[device] for device in self.cuda_devices]
            else:
                raise ValueError(
                    "CUDA heterogeneous budgets must have one value per cuda_devices entry "
                    "or one value per visible CUDA device"
                )
        else:
            raise ValueError("reserve_per_device and use_per_device must be a number or list")
        if reserve:
            result = [-1] * device_count
            for device, gib in zip(self.cuda_devices, selected):
                result[device] = int(float(gib) * 1024**3)
            return result, None
        result = [0] * device_count
        for device, gib in zip(self.cuda_devices, selected):
            result[device] = int(float(gib) * 1024**3)
        return None, result

    def _load_cuda_prefix(self, progressbar: bool, verbose: bool) -> None:
        modules = [module for module, _, _ in self.local_modules]
        if len(self.cuda_devices) == 1 and self.reserve_per_device is None and self.use_per_device is None:
            self.model._load_single(progressbar, self.cuda_device, self.model.config, modules, verbose)
            self.model.active_devices = self.cuda_devices
        else:
            reserve, use = self._cuda_budgets()
            loader = self.model._load_autosplit(
                progressbar, reserve, use, self.cuda_devices, self.max_chunk_size,
                self.max_output_size, self.max_output_factor, None, False, self.model.config,
                modules, verbose, self.max_batch_size, self.model.cache_weakrefs,
                self.autosplit_no_forward,
            )
            for _ in loader:
                pass
        # The activation returns from ROCm to the device holding the last CUDA-prefix block,
        # matching ordinary layer-split output placement rather than forcing a second GPU hop.
        self.return_device = modules[-1].device

    def load(self, progressbar: bool = False, verbose: bool = False) -> None:
        """Load a CUDA layer-split prefix and the remote ROCm suffix without a JIT build."""
        try:
            self._start_worker()
            self._load_cuda_prefix(progressbar, verbose)
            for host in getattr(self.model.config, "moe_cpu_hosts", {}).values():
                host.ensure_started()
            self.model.output_device = self.return_device
            for ref in self.model.cache_weakrefs.values():
                if cache := ref():
                    cache.initialized = True
            atexit.register(self.close)
        except Exception:
            self.close()
            raise

    def _pack_params(self, params: dict) -> tuple[dict, dict, int | None, list[dict] | None]:
        recurrent_states = params.get("recurrent_states")
        if recurrent_states is not None and not isinstance(recurrent_states, list):
            recurrent_states = [recurrent_states]
        recurrent = None if recurrent_states is None else [
            {
                "cache_id": id(state.cache),
                "slot": state.slot,
                "position": state.position,
            }
            for state in recurrent_states
        ]

        def export_nested(value):
            # A few architectures carry lists/dicts of residual tensors.  A tensor
            # hidden inside one must not cause Connection/torch to create an IPC FD
            # (and cannot cross virtualenv multiprocessing auth domains anyway).
            if isinstance(value, torch.Tensor):
                return {"__exl3_hetero_tensor__": self.arena.put(value)}
            if isinstance(value, list):
                return [export_nested(v) for v in value]
            if isinstance(value, tuple):
                return {"__exl3_hetero_tuple__": [export_nested(v) for v in value]}
            if isinstance(value, dict):
                return {k: export_nested(v) for k, v in value.items()}
            return value

        scalar, tensors = {}, {}
        for key, value in params.items():
            # Recurrent states own parent-side device tensors and must never be pickled or
            # copied. The worker maps the same (cache id, slot, position) to its local state.
            # It also recreates recurrent_slots on ROCm from those states.
            if key in ("cache", "recurrent_states", "recurrent_slots"):
                continue
            if isinstance(value, torch.Tensor):
                # Module chains legitimately add temporary tensor state (for example
                # hyper-connection ``residual``) after the CUDA range has run.  Export
                # every tensor by descriptor, never through Connection pickling.
                tensors[key] = self.arena.put(value)
            else:
                scalar[key] = export_nested(value)
        cache = params.get("cache")
        return scalar, tensors, id(cache) if cache is not None else None, recurrent

    def _remote(self, command: str, x: torch.Tensor, params: dict) -> torch.Tensor | None:
        # A CUDA-side mHC fold may defer the final residual injection into the next site.
        # The next site is in another process here, so materialize it into the stream tensor
        # before publishing that tensor. GatedResidual's weighted copy is only an optimization;
        # it cannot survive the tensor copy boundary and the worker will recompute it.
        from ..modules.hyperconnections import hc_flush
        hc_flush(params)
        params.pop("gr_weighted", None)
        self.arena.reset()
        x_descriptor = self.arena.put(x)
        scalar, tensor_params, cache_id, recurrent = self._pack_params(params)
        response = self._request(command, {
            "x": x_descriptor, "params": scalar, "tensor_params": tensor_params, "cache_id": cache_id,
            "recurrent_states": recurrent,
        })
        if command == "prefill":
            return None
        return self.arena.get(response["x"], self.return_device)

    def _run_local(self, x: torch.Tensor, params: dict, prefill: bool) -> tuple[torch.Tensor, bool]:
        for module, instance, idx in self.local_modules:
            params["layer_instance"] = instance
            if not prefill and module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            if prefill:
                stop = (idx, instance) == self.model.last_kv_module_idx_instance
                params["prefill"] = stop
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if prefill and stop:
                params.pop("prefill", None)
                return x, True
        params.pop("prefill", None)
        return x, False

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        x, stopped = self._run_local(x, params, prefill=False)
        assert not stopped
        return self._remote("forward", x, params)

    def prefill(self, x: torch.Tensor, params: dict) -> None:
        x, stopped = self._run_local(x, params, prefill=True)
        if not stopped:
            self._remote("prefill", x, params)
        return None

    def memory_info(self) -> dict:
        """Return the ROCm worker's live allocator and device-memory counters."""
        return self._request("memory_info")

    def reset_cache(self) -> None:
        self._request("reset_cache")

    def stash_recurrent(self, cache_id: int, slot: int, key) -> None:
        self._request("stash_recurrent", {"cache_id": cache_id, "slot": slot, "key": key})

    def restore_recurrent(self, cache_id: int, slot: int, key) -> None:
        self._request("restore_recurrent", {"cache_id": cache_id, "slot": slot, "key": key})

    def delete_recurrent(self, key) -> None:
        self._request("delete_recurrent", {"key": key})

    def copy_cache_page(self, cache_id: int, from_page: int, to_page: int, num_tokens: int) -> None:
        self._request("copy_cache_page", {"cache_id": cache_id, "from_page": from_page,
                                           "to_page": to_page, "num_tokens": num_tokens})

    def smoke_add(self, x: torch.Tensor) -> torch.Tensor:
        """Test-only CUDA → shared memory → HIP exllamav3 extension → CUDA round trip."""
        self.arena.reset()
        result = self._request("smoke_add", {"x": self.arena.put(x)})
        return self.arena.get(result["x"], self.cuda_device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.conn is not None:
            try:
                if self.process is not None and self.process.poll() is None:
                    # Keep unload a first-class control command: it releases remote weights
                    # and cache allocations before the worker tears down its HIP context.
                    self._request("unload")
                    self._request("quit")
            except Exception:
                pass
            self.conn.close()
            self.conn = None
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.process = None
        self.arena.close()


def smoke_rocm_worker(cuda_device: str | int = "cuda:0", bytes: int = 1 << 20, round_trips: int = 1) -> None:
    """Verify repeated cross-venv transport without loading model weights."""
    if round_trips < 1:
        raise ValueError("round_trips must be positive")
    # A lightweight object is sufficient for the transport test; bypass model construction.
    from types import SimpleNamespace
    fake = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2), fwd_modules=[], cache_weakrefs={})
    # Build the worker plumbing directly because a genuine Model has no weights in this smoke path.
    engine = object.__new__(HeterogeneousLayerSplit)
    engine.model, engine.cuda_layers, engine.cuda_device, engine.rocm_device = fake, 1, torch.device(cuda_device), 0
    engine.return_device = engine.cuda_device
    engine.rocm_python = os.environ.get("EXL3_ROCM_PYTHON", DEFAULT_ROCM_PYTHON)
    engine.arena = HeteroSharedArena(max(4 << 20, bytes * 3))
    engine.conn = None
    engine.process = None
    engine.local_modules, engine.remote_enabled, engine._closed = [], True, False
    try:
        engine._start_worker = lambda: None  # avoid model-specific alloc/load commands
        # Inline worker creation retaining the normal socket/connection implementation.
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        parent_fd, child_fd = parent_socket.detach(), child_socket.detach()
        engine.conn = Connection(parent_fd)
        root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy(); env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        engine.process = subprocess.Popen(
            [engine.rocm_python, "-m", "exllamav3.model.hetero_rocm_worker", "--fd", str(child_fd),
             "--arena-name", engine.arena.name, "--arena-size", str(engine.arena.size), "--device", "0"],
            pass_fds=(child_fd,), env=env,
        )
        os.close(child_fd)
        engine.arena.register()
        x = torch.arange(bytes // 4, dtype=torch.float32, device=engine.cuda_device).reshape(-1, 1)
        for step in range(round_trips):
            request = x + step
            y = engine.smoke_add(request)
            torch.testing.assert_close(y, request * 2, rtol=0, atol=0)
    finally:
        engine.close()
