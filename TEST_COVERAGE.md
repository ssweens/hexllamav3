# Test coverage

## Heterogeneous CUDA + ROCm path

`tests/test_hetero_handoff.py` is an integration smoke test, run with the CUDA virtualenv and this checkout on `PYTHONPATH`. It intentionally does not load a model. It verifies:

- CUDA and ROCm interpreters import their own active-venv `exllamav3_ext` wheel;
- the CUDA parent starts the configured ROCm subprocess using a UNIX `multiprocessing.connection.Connection`;
- a 1 MiB CUDA tensor is copied via the POSIX shared-memory arena registered portable and mapped in both runtimes;
- the ROCm process executes the real `exllamav3_ext.add` HIP kernel; and
- each result returns to CUDA with exact equality to twice that request's input; and
- 16 consecutive round trips reuse the same bounded arena without accumulating response allocations.

`bench_modes.py` additionally covers the model-level boundary when a local model is available: it runs cached prefill followed by 128 decode forwards, rejects empty generated text, and reports live accelerator-memory counters before unload. `cuda+cpu` accepts multi-CUDA `--cuda-devices` / `--cuda-{reserve,use}-gb` autosplit plus `-mcl`; `cuda+rocm` passes the same CUDA selection to the CUDA prefix and verifies the ROCm suffix. The worker creates a cache local to its HIP process and maps recurrent state by parent cache/slot identity, so KDA state and DSA/MLA cache allocations remain device-local.

GLM-5.3 real-model coverage completed with the 22-shard EXL3 download: a 767-token cached prefill plus 128 greedy decode steps passed in `cuda+cpu` (`-mcl 20`) and a 27-block CUDA / 18-block ROCm contiguous split. The generated continuation was byte-identical between modes.

## TabbyAPI / Corral live serving acceptance

On `.51`, the active Corral GLM-5.3-Flash profile loaded TabbyAPI with this checkout via `PYTHONPATH`, `EXL3_HETERO_CUDA_LAYERS=27`, the existing 3.10-bpw EXL3 path, GPU split `29.5 30.0 21.5`, and the configured 32,768-token context. Final serial OpenAI-compatible streaming requests through Corral all returned HTTP 200: a 24-token short prompt generated 24 tokens; a repeated-question long prompt used 1,483 prompt tokens and generated 128; the identical prompt was then repeated and generated 32. TabbyAPI reported prompt/decode throughput respectively as 13.56/23.10 tok/s (short), 359.95/24.77 tok/s (first long), and 317.19/24.39 tok/s (repeat long). The repeated request reused 1,280 cached prompt tokens (203 new), confirming recurrent/prefix checkpoint restoration in the serving path. `/health` remained healthy, Corral reported the model Ready, and the CUDA and ROCm workers remained alive; no worker error appeared after these accepted requests.

The long-prefill test exposed metadata-only recurrent layers in the ROCm worker's matching cache during checkpoint stash. The checkpoint path now excludes unallocated recurrent-layer states from local stash/restore and allocates worker-owned recurrent state after loading its layer range. An earlier attempt still encountered the meta-tensor failure; the final three-request acceptance above was run after the fix and passed.

These samples all ended at their token limits. They do not establish completed-answer quality or correct reasoning/content separation.

## Public fork validation

- Six focused unit tests pass in both CUDA and ROCm virtualenvs: loader opt-in, recurrent checkpoint ownership/cleanup, and worker response arena reuse. These tests do not establish single-backend model parity.
- `tests/test_heterogeneous_worker_arena.py` checks 16 forward/prefill cycles and 16 smoke responses using a 1 KiB arena and CPU tensors. Disabling `HeteroSharedArena.reset()` makes both cases fail with arena exhaustion.
- `tests/test_hetero_handoff.py` passes 16 consecutive exact 1 MiB CUDA → shared memory → ROCm extension → CUDA round trips without loading model weights.
- Python compilation and active-environment extension imports pass in both CUDA and ROCm virtualenvs. `git diff --check` passes.
- Independent read-only review found a worker response allocation cursor that accumulated across requests. Both inference and smoke responses now reset the cursor after request tensors are consumed. The focused regression tests and repeated GPU handoff verify this repair.
- Initial publication changed no native-kernel/build files or serving configuration and did not run the full upstream suite. The post-fix serving checks below supersede its model-level validation gap.

## Portable configuration and post-fix serving

- Twelve focused unit cases pass in both CUDA and ROCm virtualenvs. They cover explicit interpreter precedence, fail-before-allocation errors, `--all` child propagation, and equivalent expert counts. CUDA must call upstream `torch.bincount`; HIP must not call it.
- Heterogeneous execution requires `rocm_python` or `EXL3_ROCM_PYTHON`. The ordinary CUDA/ROCm modes do not require another backend's interpreter. No developer-local interpreter defaults remain.
- `tests/validate_tabby_heterogeneous.py` exercises serial streaming through an already-loaded OpenAI endpoint and rejects empty/incomplete generations. TabbyAPI's OpenAI usage schema does not expose cache-hit counts, so the server's metrics log is the cache evidence.
- After the worker response cursor fix and portable configuration change, Corral cold-started the configured 3.10-bpw model. Three requests generated 32, 512, and 512 tokens (1,056 total) without arena exhaustion or worker errors.
- The 1,528-token long prompt measured 384.89 tok/s prefill and 23.32 tok/s decode. The identical repeated prompt reused 1,280 tokens, processed 248 new tokens at 306.17 tok/s, and decoded at 23.76 tok/s.
- Both 512-token continuations were byte-identical. All samples ended at their token limits; completed-answer quality and reasoning/content separation are still not established.

## Native wheels and single-backend comparison

Both native wheels contain all 255 tracked Python package files from candidate `c4dcfc229ac17a9cc4f22758e8c3054f19506d4e` byte-for-byte and an ELF extension. Each was installed only in an isolated Python 3.13 environment. Package and extension imports resolve inside that environment; all 12 focused unit cases pass. Exact native `add` checks pass on both RTX 5090s, the RTX 3090, and the Radeon 8060S.

- CUDA: CUDA 12.8.1/Ubuntu 24.04 build, PyTorch 2.9.0+cu128, targets `sm_86` and `sm_120`. Host-native compilation fails against glibc 2.44; the compatible container resolves this without source/header patches.
- ROCm: HIP 7.2/clang 22 build, PyTorch 2.13.0 TheRock environment, target `gfx1151`.
- CUDA wheel SHA-256: `8cd3fe794852c695ed9fc504b947e81a656f80a6ad5c217fa4106befd8e8fd4d`.
- ROCm wheel SHA-256: `b1ca93b8ac79c1e5133fe2a0ecd3ec766e6a21da88aa5cb72b6777b143802f93`.

GLM 2.10-bpw, 767-token cached prefill, and 128 greedy decode steps produced byte-identical continuations in all four runs:

| Backend/source | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| CUDA upstream v1.6.0 | 536.12 | 46.51 |
| CUDA fork | 545.34 | 46.88 |
| ROCm upstream v1.6.0 | 278.56 | 17.71 |
| ROCm fork | 279.05 | 17.98 |

These are single warm measurements, not a statistical performance guarantee. CUDA used three GPUs and six CPU-offloaded expert layers; ROCm used device 0 with no CPU expert offload. Each baseline/fork pair used the same existing prebuilt native extension to isolate Python changes. The newly built native extensions passed kernel checks, not full weighted-model inference. Serving remains on the original venvs with the shared source checkout.

## Not established

- Full weighted-model inference with the newly built extensions, or installed-wheel-only heterogeneous worker loading.
- Concurrent heterogeneous serving, a full 32,768-token request, vision, speculative serving, or a quality benchmark.
- The full upstream test suite or hardware outside the tested NVIDIA/Radeon topology.
