"""Embedding model loading and encoding.

Kept as its own module (rather than inline) because Phase 1 will wrap
`embed_texts` with tracing/timing, and Phase 1's memory-profiling story
needs a clear, single point where the model is loaded into memory.

Device selection: tries MPS (Apple Silicon GPU) first, falls back to CPU.
This is automatic and safe - if torch/MPS isn't available (e.g. Linux CI),
it silently uses CPU, so this code runs unchanged everywhere.

Fallback: if sentence-transformers can't load a model (e.g. no network
access to huggingface.co - a real constraint in sandboxed/CI
environments), fall back to a deterministic hashing-based embedder. This
is NOT semantically meaningful, but it lets the full pipeline
(chunk -> embed -> store -> retrieve -> generate) run end-to-end for
development, testing, and CI without an internet dependency. The real
model is used automatically whenever it's available (e.g. local dev with
network, or after pre-downloading the model into the deploy image).
"""

import hashlib
import logging

import numpy as np

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model = None
_model_load_failed = False
_device = None


def get_device() -> str:
    """Pick the best available torch device: mps > cpu.

    (cuda could be added here too, for non-Apple GPU machines - same
    pattern: try importing torch, check availability, fall back safely.)
    """
    global _device
    if _device is not None:
        return _device

    try:
        import torch

        if torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    except Exception:  # noqa: BLE001 - torch not installed, or no backends attr
        _device = "cpu"

    return _device


def _try_load_model():
    global _model, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model

    try:
        from sentence_transformers import SentenceTransformer

        device = get_device()
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info(
            "Loaded sentence-transformers model: %s (device=%s)",
            EMBEDDING_MODEL,
            device,
        )
    except Exception as exc:  # noqa: BLE001 - intentionally broad: any load
        # failure (network, missing files, etc.) should trigger fallback
        logger.warning(
            "Could not load sentence-transformers model '%s' (%s). "
            "Falling back to hashing-based embedder. Retrieval results "
            "will not be semantically meaningful.",
            EMBEDDING_MODEL,
            exc,
        )
        _model_load_failed = True

    return _model


def _hashing_embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding via hashed token features.

    Maps each word to a dimension via hashing, accumulates counts, then
    L2-normalizes. Two texts sharing many words will have somewhat similar
    vectors - enough to exercise the retrieval pipeline mechanically, but
    this is a placeholder, not a real semantic embedding.
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    for word in text.lower().split():
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        vec[h % EMBEDDING_DIM] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


def using_fallback_embedder() -> bool:
    """Whether embed_texts is currently using the hashing fallback."""
    _try_load_model()
    return _model_load_failed


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _try_load_model()

    if model is not None:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return [_hashing_embed(t) for t in texts]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def warmup() -> None:
    """Eagerly trigger model loading (and a first encode, which can have
    its own one-time JIT/graph-building cost on top of weight loading).

    Call this at app startup so the ~30s first-load cost (observed on
    Apple Silicon with the real model) happens during deploy/startup
    rather than during a user's first request. Doesn't make the cost
    disappear - moves it to a less disruptive point in time.
    """
    embed_texts(["warmup"])
