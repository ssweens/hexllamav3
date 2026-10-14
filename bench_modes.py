#!/usr/bin/env python3
"""Benchmark EXL3 cached inference in CUDA, ROCm, and heterogeneous modes.

Examples (do not run while another workload owns the GPUs)::

  python bench_modes.py --model /path/to/exl3 --mode cuda+rocm --cuda-layers 24 --rocm-python /path/to/rocm/python
  python bench_modes.py --model /path/to/exl3 --all --mcl 8 --cuda-python /path/to/cuda/python --rocm-python /path/to/rocm/python

``--all`` launches CUDA and ROCm single-mode measurements in their matching
virtualenvs, then runs the heterogeneous parent here.  Each measurement performs
real cached prefill and 128 single-token decode steps, then prints generated text.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local EXL3 model directory")
    parser.add_argument("--mode", choices=("cuda+cpu", "rocm+cpu", "cuda+rocm"), default="cuda+rocm")
    parser.add_argument("--all", action="store_true", help="run all three modes via their selected interpreters")
    parser.add_argument("--cuda-python", default=os.environ.get("EXL3_CUDA_PYTHON", sys.executable),
                        help="CUDA interpreter for --all (default: EXL3_CUDA_PYTHON or this Python)")
    parser.add_argument("--rocm-python", default=os.environ.get("EXL3_ROCM_PYTHON"),
                        help="ROCm interpreter for heterogeneous mode and --all (or EXL3_ROCM_PYTHON)")
    parser.add_argument("--cuda-layers", type=int, default=None, help="leading transformer blocks on CUDA")
    parser.add_argument("--cuda-devices", default=None,
                        help="comma-separated CUDA devices for layer split (default: all visible CUDA GPUs)")
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--cuda-reserve-gb", default=None,
                         help="comma-separated per-selected-GPU VRAM reserves in GiB (default: 0.5)")
    budgets.add_argument("--cuda-use-gb", default=None,
                         help="comma-separated per-selected-GPU VRAM budgets in GiB")
    parser.add_argument("-mcl", "--mcl", type=int, default=0,
                        help="number of eligible mul1 MoE layers to CPU-offload (cuda+cpu / rocm+cpu only)")
    parser.add_argument("--prefill-tokens", type=int, default=768)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Explain why an inference benchmark must verify generated text.")
    parser.add_argument("--arena-mb", type=int, default=None)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def child_env(rocm_python):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["EXL3_ROCM_PYTHON"] = rocm_python
    return env


def run_all(args):
    from exllamav3.model.hetero import resolve_rocm_python

    rocm_python = resolve_rocm_python(args.rocm_python)
    cuda_python = args.cuda_python
    if not os.path.isfile(cuda_python) or not os.access(cuda_python, os.X_OK):
        raise SystemExit(f"CUDA Python is not executable: {cuda_python}")
    base = [str(ROOT / "bench_modes.py"), "--model", args.model, "--prefill-tokens", str(args.prefill_tokens),
            "--decode-tokens", str(args.decode_tokens), "--prompt", args.prompt, "--_child"]
    if args.cuda_devices:
        base += ["--cuda-devices", args.cuda_devices]
    if args.cuda_reserve_gb:
        base += ["--cuda-reserve-gb", args.cuda_reserve_gb]
    if args.cuda_use_gb:
        base += ["--cuda-use-gb", args.cuda_use_gb]
    for mode, python in (("cuda+cpu", cuda_python), ("rocm+cpu", rocm_python), ("cuda+rocm", cuda_python)):
        command = [python, *base, "--mode", mode, "--rocm-python", rocm_python]
        if mode != "cuda+rocm":
            command += ["--mcl", str(args.mcl)]
        if mode == "cuda+rocm":
            layers = args.cuda_layers
            if layers is None:
                raise SystemExit("--all requires --cuda-layers for cuda+rocm")
            command += ["--cuda-layers", str(layers)]
            if args.arena_mb:
                command += ["--arena-mb", str(args.arena_mb)]
        print("\n$", " ".join(command), flush=True)
        subprocess.run(command, env=child_env(rocm_python), check=True)


def _csv_values(value: str | None, cast, name: str) -> list:
    if value is None:
        return []
    try:
        result = [cast(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated list") from exc
    if not result or any(part < 0 for part in result):
        raise SystemExit(f"{name} must contain non-negative values")
    return result


def cuda_load_options(args, torch) -> tuple[list[int], dict]:
    devices = _csv_values(args.cuda_devices, int, "--cuda-devices")
    if not devices:
        devices = list(range(torch.cuda.device_count()))
    if not devices or len(set(devices)) != len(devices) or any(device >= torch.cuda.device_count() for device in devices):
        raise SystemExit("--cuda-devices must select distinct visible CUDA devices")
    reserve = _csv_values(args.cuda_reserve_gb, float, "--cuda-reserve-gb")
    use = _csv_values(args.cuda_use_gb, float, "--cuda-use-gb")
    if reserve and len(reserve) not in (1, len(devices)):
        raise SystemExit("--cuda-reserve-gb needs one value or one per selected CUDA device")
    if use and len(use) not in (1, len(devices)):
        raise SystemExit("--cuda-use-gb needs one value or one per selected CUDA device")
    if len(reserve) == 1:
        reserve *= len(devices)
    if len(use) == 1:
        use *= len(devices)
    options = {}
    if reserve:
        budget = [-1.0] * torch.cuda.device_count()
        for device, gib in zip(devices, reserve):
            budget[device] = gib
        options["reserve_per_device"] = budget
    elif use:
        budget = [0.0] * torch.cuda.device_count()
        for device, gib in zip(devices, use):
            budget[device] = gib
        options["use_per_device"] = budget
    else:
        # Explicitly exclude non-selected cards while retaining normal autosplit headroom.
        budget = [-1.0] * torch.cuda.device_count()
        for device in devices:
            budget[device] = 0.5
        options["reserve_per_device"] = budget
    return devices, options


def print_live_memory(model, torch) -> None:
    """Report live accelerator consumption before model.unload tears it down."""
    for device in model.active_devices:
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(f"cuda:{device} memory used={(total - free) / 2**30:.2f} GiB "
              f"free={free / 2**30:.2f} GiB allocated={allocated / 2**30:.2f} GiB "
              f"reserved={reserved / 2**30:.2f} GiB")
    engine = getattr(model, "hetero_engine", None)
    if engine is not None:
        info = engine.memory_info()
        print(f"rocm:{engine.rocm_device} GTT used={(info['total'] - info['free']) / 2**30:.2f} GiB "
              f"free={info['free'] / 2**30:.2f} GiB allocated={info['allocated'] / 2**30:.2f} GiB "
              f"reserved={info['reserved'] / 2**30:.2f} GiB")


def make_prompt(tokenizer, text: str, target: int):
    # Repeat a natural-language prompt to give all models a 512–1024-token prefill.
    ids = tokenizer.encode((text + "\n") * max(1, target), encode_special_tokens=True)
    if ids.shape[-1] < target:
        raise RuntimeError(f"prompt only tokenized to {ids.shape[-1]} tokens; provide a longer --prompt")
    return ids[:, :target]


def run_one(args):
    # Validate the cross-backend interpreter before model or cache construction.
    if args.mode == "cuda+rocm":
        from exllamav3.model.hetero import resolve_rocm_python
        args.rocm_python = resolve_rocm_python(args.rocm_python)
    # This must happen before Config construction: InferParams reads it from the environment.
    if args.mcl and args.mode == "cuda+rocm":
        raise SystemExit("--mcl is unsupported with cuda+rocm; run cuda+cpu for CPU MoE offload")
    if args.mcl:
        os.environ["EXL3_MOE_CPU_OFFLOAD"] = str(args.mcl)
    from exllamav3 import Cache, Config, Model, Tokenizer
    import torch

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache_tokens = ((args.prefill_tokens + args.decode_tokens + 256 + 255) // 256) * 256
    cache = Cache(model, max_num_tokens=cache_tokens)
    tokenizer = Tokenizer.from_config(config)

    if args.mode == "cuda+rocm":
        cuda_layers = args.cuda_layers or config.num_hidden_layers // 2
        cuda_devices, cuda_options = cuda_load_options(args, torch)
        model.load_heterogeneous(cuda_layers, rocm_python=args.rocm_python,
                                 cuda_devices=cuda_devices, arena_mb=args.arena_mb,
                                 progressbar=True, max_chunk_size=args.prefill_tokens,
                                 max_batch_size=1, **cuda_options)
    elif args.mode == "cuda+cpu":
        _cuda_devices, cuda_options = cuda_load_options(args, torch)
        model.load(progressbar=True, max_chunk_size=args.prefill_tokens, max_batch_size=1, **cuda_options)
    else:
        model.load(device="cuda:0", progressbar=True)

    try:
        input_ids = make_prompt(tokenizer, args.prompt, args.prefill_tokens)
        # Warmup prefill pass: pays one-time Triton JIT compilation and cooperative GEMM
        # autotuning costs before measuring steady-state prefill throughput.
        warmup_params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, cache_tokens)}
        model.prefill(input_ids[:, :-1], warmup_params)
        torch.cuda.synchronize()

        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, cache_tokens)}
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.prefill(input_ids[:, :-1], params)
        # Hybrid GLM KDA layers require their per-slot state on every subsequent
        # cached decode. Model.forward advances this object in place, so preserve
        # the list across the fresh per-step parameter dictionaries below.
        recurrent_states = params.get("recurrent_states")
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - start

        context = input_ids
        output_ids = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.decode_tokens):
            params = {"attn_mode": "flash_attn", "cache": cache, "past_len": context.shape[-1] - 1,
                      "batch_shape": (1, cache_tokens), "recurrent_states": recurrent_states}
            logits = model.forward(context[:, -1:], params)
            recurrent_states = params.get("recurrent_states")
            token = torch.argmax(logits[..., :config.vocab_size], dim=-1).cpu()
            output_ids.append(token.item())
            context = torch.cat((context, token), dim=-1)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - start
        text = tokenizer.decode(torch.tensor([output_ids]))
        # Tokenizer.decode returns one string per row for a batched tensor.
        if isinstance(text, list):
            text = text[0]
        if not text or not text.strip():
            raise RuntimeError("generation verification failed: decoded text is empty")
        print(f"mode={args.mode} prefill={(args.prefill_tokens - 1) / prefill_seconds:.2f} tok/s "
              f"decode={args.decode_tokens / decode_seconds:.2f} tok/s")
        print("generated:", repr(text))
        print_live_memory(model, torch)
    finally:
        model.unload()


def main():
    args = parse_args()
    if args.all and not args._child:
        run_all(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
