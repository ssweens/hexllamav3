"""Serial OpenAI/TabbyAPI serving check; load the configured model before running.

Example:
    python tests/validate_tabby_heterogeneous.py --api-base http://localhost:9999/v1 \
        --model zai-org/GLM-5.3-Flash --output serving.json

Cache-hit counts are not exposed by TabbyAPI's OpenAI usage schema. Check the
server's metrics log for the repeated-prefix request; this script records IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def generate(api_base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers=headers,
    )
    content, reasoning, response_ids = [], [], set()
    usage, finish_reason, done = None, None, False
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            packet = json.loads(payload)
            if packet.get("error"):
                raise RuntimeError(f"server stream error: {packet['error']}")
            if packet.get("id"):
                response_ids.add(packet["id"])
            if packet.get("usage"):
                usage = packet["usage"]
            for choice in packet.get("choices", []):
                delta = choice.get("delta", {})
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    text, analysis = "".join(content), "".join(reasoning)
    if not done or not usage or usage.get("completion_tokens", 0) <= 0 or not (text + analysis).strip():
        raise RuntimeError(f"incomplete or empty generation: done={done}, usage={usage}")
    if finish_reason not in {"stop", "length"}:
        raise RuntimeError(f"unexpected finish reason: {finish_reason}")
    return {
        "response_ids": sorted(response_ids), "usage": usage, "finish_reason": finish_reason,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "content": text, "reasoning": analysis,
        "continuation_sha256": hashlib.sha256(json.dumps([text, analysis]).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--long-max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.long_max_tokens <= 0 or args.timeout_seconds <= 0:
        parser.error("token and timeout limits must be positive")
    nonce = time.time_ns()
    long_prompt = f"Validation run {nonce}.\n" + (
        "Explain how shared memory avoids cross-runtime tensor ownership problems in GPU inference.\n" * 100
    )
    samples = {}
    for name, prompt, tokens in (
        ("short", "Reply with the word READY.", 32),
        ("long", long_prompt, args.long_max_tokens),
        ("repeated_prefix", long_prompt, args.long_max_tokens),
    ):
        samples[name] = generate(args.api_base, args.model, prompt, tokens, args.timeout_seconds)
        summary = {key: value for key, value in samples[name].items() if key not in {"content", "reasoning"}}
        print(name, json.dumps(summary), flush=True)
    if samples["long"]["usage"]["prompt_tokens"] < 1024:
        raise RuntimeError("long request did not reach 1024 prompt tokens")
    if samples["long"]["usage"]["prompt_tokens"] != samples["repeated_prefix"]["usage"]["prompt_tokens"]:
        raise RuntimeError("identical prompts have different token counts")
    result = {
        "model": args.model, "api_base": args.api_base, "samples": samples,
        "repeated_continuation_matches": (
            samples["long"]["continuation_sha256"] == samples["repeated_prefix"]["continuation_sha256"]
        ),
    }
    if args.output:
        with args.output.open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
