"""Minimal Ollama client wrapper using the HTTP API directly (no extra
dependency).

Kept deliberately small in Phase 0. Phase 1 will wrap `generate` with
timing/tracing; Phase 3 will add timeout + fallback handling here.
"""

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def generate(prompt: str, timeout_seconds: float = 60.0) -> str:
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()["response"]
