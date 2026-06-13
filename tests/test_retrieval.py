"""Tests for the retrieval half of the pipeline (no Ollama required).

These run with whatever embedder is available (real model or fallback),
so they pass both in local dev (with network) and in CI (without).
"""

from app.rag import retrieve


def test_retrieve_returns_chunks():
    results = retrieve("What is point-in-time correctness?", top_k=3)

    assert len(results) == 3
    for chunk in results:
        assert chunk.text
        assert chunk.source.endswith(".md")
        assert isinstance(chunk.score, float)


def test_retrieve_respects_top_k():
    results = retrieve("debugging", top_k=2)
    assert len(results) == 2


def test_retrieve_covers_multiple_sources():
    """Sanity check: across a few varied queries, we should see retrieval
    pull from more than one source file (i.e. it isn't just always
    returning the same chunk regardless of query)."""
    queries = [
        "feature store point in time correctness",
        "MongoDB index missing guid",
        "PromQL sum or vector zero",
        "Q-learning SARSA on-policy off-policy",
        "UTMA kiddie tax 529",
    ]

    sources_seen: set[str] = set()
    for q in queries:
        for chunk in retrieve(q, top_k=2):
            sources_seen.add(chunk.source)

    assert len(sources_seen) >= 2
