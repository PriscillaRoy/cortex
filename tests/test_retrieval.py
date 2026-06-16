"""Tests for the retrieval half of the pipeline (no Ollama required).

These run with whatever embedder is available (real model or fallback),
so they pass both in local dev (with network) and in CI (without).

Phase 3: retrieve() now returns (chunks, fallback_bool). Tests updated
to unpack the tuple. A conftest.py fixture handles ingest so the DB
is populated before these tests run.
"""

import pytest
from app.rag import retrieve


def test_retrieve_returns_chunks():
    chunks, fallback = retrieve("What is point-in-time correctness?", top_k=3)

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.text
        assert chunk.source.endswith(".md")
        assert isinstance(chunk.score, float)


def test_retrieve_respects_top_k():
    chunks, _ = retrieve("debugging", top_k=2)
    assert len(chunks) == 2


def test_retrieve_covers_multiple_sources():
    """Sanity check: across varied queries, retrieval should pull from
    more than one source file."""
    queries = [
        "feature store point in time correctness",
        "MongoDB index missing guid",
        "PromQL sum or vector zero",
        "Q-learning SARSA on-policy off-policy",
        "UTMA kiddie tax 529",
    ]

    sources_seen: set[str] = set()
    for q in queries:
        chunks, _ = retrieve(q, top_k=2)
        for chunk in chunks:
            sources_seen.add(chunk.source)

    assert len(sources_seen) >= 2


def test_retrieve_fallback_flag_false_on_normal_path():
    """On a healthy vector store, fallback should be False."""
    _, fallback = retrieve("What is point-in-time correctness?", top_k=1)
    assert fallback is False
