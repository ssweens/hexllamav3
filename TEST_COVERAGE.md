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

- Six focused unit tests pass: loader opt-in, recurrent checkpoint ownership/cleanup, and worker response arena reuse.
- `tests/test_heterogeneous_worker_arena.py` checks 16 forward/prefill cycles and 16 smoke responses using a 1 KiB arena and CPU tensors. Disabling `HeteroSharedArena.reset()` makes both cases fail with arena exhaustion.
- `tests/test_hetero_handoff.py` passes 16 consecutive exact 1 MiB CUDA → shared memory → ROCm extension → CUDA round trips without loading model weights.
- Python compilation and active-environment extension imports pass in both CUDA and ROCm virtualenvs. `git diff --check` passes.
- Independent read-only review found a worker response allocation cursor that accumulated across requests. Both inference and smoke responses now reset the cursor after request tensors are consumed. The focused regression tests and repeated GPU handoff verify this repair.
- No native-kernel/build files or serving configuration were changed for publication. The full upstream suite and model-level inference after the response cursor repair were not run.

## Not established

- CUDA-only and ROCm-only correctness/performance parity or wheels built from this fork.
- Concurrent heterogeneous serving, a full 32,768-token request, vision, speculative serving, or a quality benchmark.
- Portable defaults for the benchmark's `--all` interpreter paths. Individual modes and `EXL3_ROCM_PYTHON` can be selected explicitly.
