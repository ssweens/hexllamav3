"""Shared-memory transport used by the CUDA/ROCm layer-split boundary.

Only compact descriptors cross the control connection.  Tensor bytes always live in
one POSIX shared-memory object, registered independently by the CUDA and HIP
runtimes.  CUDA's ``cudaHostRegister`` and HIP's compatibility implementation use
the same extension entry point, selected by the active interpreter's wheel.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory, resource_tracker
import os
import uuid

import numpy as np
import torch

from .model_tp_cuda import (
    CUDA_HOST_REGISTER_MAPPED,
    CUDA_HOST_REGISTER_PORTABLE,
    cuda_host_register,
    cuda_host_unregister,
)
from ..util.shm import check_shm_capacity

_DTYPE_BY_NAME = {str(dtype): dtype for dtype in (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.float32, torch.float64, torch.bfloat16,
)}


@dataclass(frozen=True)
class TensorDescriptor:
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    def as_dict(self) -> dict:
        return {"offset": self.offset, "nbytes": self.nbytes, "dtype": self.dtype, "shape": self.shape}

    @classmethod
    def from_dict(cls, value: dict) -> "TensorDescriptor":
        return cls(value["offset"], value["nbytes"], value["dtype"], tuple(value["shape"]))


class HeteroSharedArena:
    """A synchronized, mapped POSIX shared-memory arena.

    The protocol is intentionally request/response: descriptors remain valid until
    the receiver acknowledges the request, so offsets can be reused between passes
    without races.  It is a transport for CPU/GPU copies, not a claim of CUDA/HIP
    peer-memory interoperability (the two drivers cannot share device allocations).
    """

    def __init__(self, size: int, name: str | None = None, create: bool = True):
        if size <= 0:
            raise ValueError("shared arena size must be positive")
        self.size = int(size)
        self.owner = create
        self.name = name or f"exl3_hetero_{uuid.uuid4().hex}"
        if create:
            check_shm_capacity(self.size, "The heterogeneous CUDA/ROCm shared arena")
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            # An attaching process must not own unlinking.  Python registers every
            # SharedMemory open with its resource tracker, so explicitly remove this
            # attachment to prevent the worker from racing the CUDA parent at exit.
            resource_tracker.unregister(self.shm._name, "shared_memory")
            if self.shm.size < self.size:
                self.shm.close()
                raise RuntimeError(f"shared arena {name} is smaller than advertised")
        self._bytes = torch.as_tensor(np.ndarray((self.size,), dtype=np.uint8, buffer=self.shm.buf))
        self._next_offset = 0
        self._registered = False
        # Touch pages while the creator still has useful diagnostics if /dev/shm is exhausted.
        if create:
            self._bytes[::os.sysconf("SC_PAGESIZE")] = 0

    def export(self) -> dict:
        return {"name": self.name, "size": self.size}

    def register(self) -> None:
        """Register the arena portable + mapped in this process's CUDA or HIP runtime."""
        if self._registered:
            return
        cuda_host_register(
            self._bytes.data_ptr(), self.size,
            CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED,
        )
        self._registered = True

    def reset(self) -> None:
        self._next_offset = 0

    def _reserve(self, nbytes: int) -> int:
        # GPU copies prefer at least cache-line alignment; also preserves alignment for all torch dtypes.
        aligned = (int(nbytes) + 255) & ~255
        if self._next_offset + aligned > self.size:
            raise RuntimeError(
                f"heterogeneous shared arena exhausted: need {nbytes} bytes, "
                f"{self.size - self._next_offset} remain; increase EXL3_HETERO_ARENA_MB"
            )
        offset = self._next_offset
        self._next_offset += aligned
        return offset

    def allocate(self, tensor: torch.Tensor) -> TensorDescriptor:
        if not tensor.is_contiguous():
            raise ValueError("heterogeneous transport requires a contiguous tensor")
        nbytes = tensor.numel() * tensor.element_size()
        return TensorDescriptor(self._reserve(nbytes), nbytes, str(tensor.dtype), tuple(tensor.shape))

    def cpu_tensor(self, descriptor: TensorDescriptor | dict) -> torch.Tensor:
        d = TensorDescriptor.from_dict(descriptor) if isinstance(descriptor, dict) else descriptor
        dtype = _DTYPE_BY_NAME.get(d.dtype)
        if dtype is None:
            raise TypeError(f"unsupported heterogeneous transport dtype: {d.dtype}")
        if d.offset < 0 or d.nbytes < 0 or d.offset + d.nbytes > self.size:
            raise ValueError("shared-memory tensor descriptor is outside the arena")
        result = self._bytes.narrow(0, d.offset, d.nbytes).view(dtype).view(d.shape)
        if result.numel() * result.element_size() != d.nbytes:
            raise ValueError("shared-memory tensor descriptor has inconsistent shape")
        return result

    def put(self, tensor: torch.Tensor) -> dict:
        tensor = tensor.detach().contiguous()
        d = self.allocate(tensor)
        # ``copy_`` invokes the active CUDA/HIP runtime directly for a device source;
        # no tensor is pickled and the registered arena is the only host staging storage.
        self.cpu_tensor(d).copy_(tensor, non_blocking=False)
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)
        return d.as_dict()

    def get(self, descriptor: TensorDescriptor | dict, device: torch.device | str | None = None) -> torch.Tensor:
        value = self.cpu_tensor(descriptor)
        if device is None:
            return value
        return value.to(device, non_blocking=False, copy=True, memory_format=torch.contiguous_format)

    def close(self) -> None:
        if getattr(self, "_registered", False):
            try:
                cuda_host_unregister(self._bytes.data_ptr())
            finally:
                self._registered = False
        self._bytes = None
        self.shm.close()
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
