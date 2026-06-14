"""Ollama client wrapper.

Ollama's /api/generate supports `"stream": true`, returning a sequence of
newline-delimited JSON objects - one per token (roughly), each containing
that token's text plus, on the final object, the full token-count stats.

`generate_stream()` is the primitive: a generator yielding each chunk's
dict as it arrives. `generate()` is a convenience wrapper that consumes
the whole stream and returns one GenerateResult - used by
latency_report.py, which wants the final totals, not the live stream.

This means /ask (streaming, for the UI) and latency_report.py (batch
analysis) both go through the same underlying call - just consumed
differently.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


@dataclass
class GenerateResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int


def generate_stream(
    prompt: str, timeout_seconds: float = 60.0, seed: int | None = None
) -> Iterator[dict]:
    """Yield each chunk dict from Ollama's streaming /api/generate.

    Each chunk has at least a "response" key (the token text for this
    chunk) and a "done" key (bool). The final chunk (done=True) also
    includes "prompt_eval_count" and "eval_count" (token totals).

    seed: if set, Ollama's sampling becomes deterministic - same prompt +
    same seed -> same output. Useful for baseline/regression comparisons,
    where you want "did the answer change" to mean "did something we
    control change", not "did the random sampler roll differently this
    time".
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    if seed is not None:
        payload["options"] = {"seed": seed}

    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            yield json.loads(line)


def generate(
    prompt: str, timeout_seconds: float = 60.0, seed: int | None = None
) -> GenerateResult:
    """Consume the full stream and return one combined result.

    Used where we want totals rather than incremental tokens (e.g.
    latency_report.py).
    """
    answer_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in generate_stream(prompt, timeout_seconds=timeout_seconds, seed=seed):
        answer_parts.append(chunk.get("response", ""))
        if chunk.get("done"):
            prompt_tokens = chunk.get("prompt_eval_count", 0)
            completion_tokens = chunk.get("eval_count", 0)

    return GenerateResult(
        answer="".join(answer_parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
