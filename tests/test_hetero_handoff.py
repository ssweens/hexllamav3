"""No-model-load CUDA ↔ ROCm shared-memory handoff smoke test.

Run from the CUDA venv with PYTHONPATH pointing at this checkout.  It allocates a
1 MiB CUDA tensor, invokes the ROCm wheel's real ``exllamav3_ext.add`` HIP kernel,
and requires exact CUDA-side results over 16 consecutive round trips.
"""
from exllamav3.model.hetero import smoke_rocm_worker


def test_cuda_rocm_handoff_1mb():
    smoke_rocm_worker(bytes=1 << 20, round_trips=16)


if __name__ == "__main__":
    test_cuda_rocm_handoff_1mb()
    print("CUDA -> POSIX SHM -> ROCm exllamav3_ext.add -> CUDA: 16 exact 1 MiB handoffs passed")
