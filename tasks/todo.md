# Heterogeneous CUDA + ROCm implementation

- [x] Clone `v1.6.0` into the isolated `exllamav3-hetero` fork.
- [x] Verify active-venv prebuilt extension discovery without JIT compilation.
- [x] Implement shared-memory arena, persistent ROCm worker protocol, and CUDA parent client.
- [x] Integrate contiguous CUDA/ROCm layer split with `Model`, `Cache`, and cached forwards.
- [x] Add benchmark and no-model-load smoke tests.
- [x] Run import and 1 MiB CUDA↔ROCm handoff verification; inspect diff.
- [x] Unload Corral model and verify NVIDIA VRAM is idle.
- [x] Run real cached prefill/decode benchmark for CUDA+CPU and ROCm+CPU.
- [x] Run and repair real heterogeneous CUDA+ROCm cached prefill/decode benchmark.
- [x] Trace GLM-5.3 module/cache/recurrent and multi-CUDA load paths.
- [x] Make heterogeneous execution partition recurrent state and cache ownership by split.
- [x] Add multi-CUDA CUDA-prefix placement and bench CLI support.
- [x] Run available GLM architecture/state-protocol and transport validation.
- [x] Run full GLM CUDA↔ROCm cached parity after the EXL3 shards finished downloading.
- [x] Unload .51 serving workload and confirm NVIDIA GPUs are available.
- [x] Benchmark GLM-5.3 in multi-CUDA + CPU MoE mode and repair runtime failures.
- [x] Benchmark GLM-5.3 in multi-CUDA + ROCm mode and repair runtime failures.
- [x] Record generation, throughput, and accelerator memory use.
- [x] Validate the `.51` Corral → TabbyAPI → EXL3 heterogeneous serving profile with short, long-prefill, and repeated-prefix requests.
- [x] Repair worker-local recurrent checkpoint stash/restore to skip metadata-only recurrent layer states.
- [x] Record final live serving timings, token counts, health, and worker status.

## Publish the public hexllamav3 fork

- [x] Create `ssweens/hexllamav3` as a public GitHub fork of `turboderp-org/exllamav3`.
- [x] Preserve the package/import name and local serving directory; document experimental scope.
- [x] Run focused tests, transport smoke, syntax checks, and independent read-only review.
- [x] Fix the reviewed worker response allocator leak and prove bounded reuse across repeated requests.
- [x] Commit the owned heterogeneous implementation on `hetero`, based on `v1.6.0`.
- [x] Push, set the default branch, and verify GitHub ancestry, visibility, and remote synchronization.

Publication result: https://github.com/ssweens/hexllamav3 is public, retains `turboderp-org/exllamav3` as its GitHub parent, and defaults to `hetero`. Implementation commit: `278e6469f6da1ab9aaa2a1e18c0dcd26cbdb8b96`. The local checkout remains at `exllamav3-hetero`; `origin` points to the public fork, `upstream` to GitHub upstream, and `local-source` preserves the original local remote.

## Follow-up validation (outside initial publication)

- [ ] Run CUDA-only and ROCm-only correctness/performance regressions, including the shared expert-count change.
- [ ] Build and test separate fork CUDA and ROCm wheels.
- [x] Replace developer-local interpreter defaults without changing the existing serving profile unexpectedly; `.51` explicitly configures `EXL3_ROCM_PYTHON` in TOML.
- [x] Re-run live serving after the worker arena repair: 1,056 generated tokens, 1,280 cached-prefix tokens reused, and identical 512-token continuations.

## Review

- CUDA and ROCm imports resolved their respective active-venv `exllamav3_ext.cpython-313-x86_64-linux-gnu.so` files without a JIT build.
- The 1 MiB CUDA → POSIX shared-memory → ROCm HIP `exllamav3_ext.add` → CUDA smoke test passed with exact equality.
- `py_compile` passed in both virtualenvs and `git diff --check` passed.
- `pytest` is not installed in the CUDA virtualenv, so the smoke module was executed directly instead.
- 2026-06-20 benchmark segment: Corral unload was requested; NVIDIA GPU use was 39 MiB, 56 MiB, and 35 MiB before loading a benchmark model.
- Real Qwen3-VL-30B-A3B-Instruct EXL3 cached benchmark (767-token prefill, 128-token greedy decode): CUDA+CPU 1421.44 / 192.73 tok/s; ROCm+CPU 261.69 / 48.64 tok/s; CUDA+ROCm split at 24/48 blocks 491.33 / 72.63 tok/s. All modes produced the same non-empty continuation.
- Fixed benchmark batched-decode text handling and heterogeneous nested tensor parameter export. The latter prevents PyTorch CUDA IPC/authentication failures for Qwen's nested residual state and keeps those tensors in POSIX shared memory.
- The Qwen EXL3 experts are not mul1, so the requested `--mcl 48` is explicitly skipped by the upstream loader in all modes; this is a model-format capability restriction, not a benchmark fallback.
- After every benchmark model unload, NVIDIA use returned to 39 MiB, 56 MiB, and 35 MiB. Heterogeneous worker process was not left running.
- GLM-5.3 architecture construction against the download directory confirms 45 blocks: 34 `GatedDeltaNet` KDA layers and 11 `MLAttention` sparse-DSA layers, with `ExpandStreams`/`HyperHead` mHC. Its cache is `CacheLayer_MLA_fp16`; it does not instantiate `GatedResidual` or `CacheLayer_dsa`.
- CUDA-parent/ROCm-child transport smoke passed after the change. The worker-local GLM GDN state protocol created, advanced, matched, and reset a state by parent cache id/slot without transferring state tensors.
- 2026-06-20 complete-model run: after `POST /unload`, NVIDIA use was 39 / 56 / 35 MiB. The benchmark originally dropped the recurrent state list between prefill and decode; fixed it by threading `params["recurrent_states"]` through every decode call. Mode A (`cuda+cpu`, `0,1,2`, reserves 4/3/3 GiB, `-mcl 20`) passed 767-token prefill + 128 greedy decode at 116.51 / 28.13 tok/s; live VRAM was 25.86 / 26.27 / 18.05 GiB. Mode B (`cuda+rocm`, 27 CUDA blocks, reserves 3/2/2 GiB) passed at 190.53 / 24.60 tok/s; CUDA VRAM was 26.38 / 27.46 / 20.67 GiB and ROCm GTT was 58.68 GiB (50.22 GiB allocated). The 128-token continuation was byte-identical across modes. The pinned CPU expert arena also passed but was not faster at this sequence length: 116.49 / 27.23 tok/s.
