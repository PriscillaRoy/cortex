"""Core RAG pipeline: retrieve relevant chunks, build a prompt, generate
an answer.

Phase 3 additions: per-stage timeouts + graceful degradation.

  Retrieval timeout  -> fall back to BM25 keyword search (no Milvus
                        needed). User still gets relevant chunks.
  Generation timeout -> return retrieved chunks raw with a fallback
                        flag. User gets context to read directly instead
                        of an error.

Timeouts use concurrent.futures (threading) rather than asyncio because
our pipeline is synchronous — adding async would require a larger
refactor with no meaningful benefit for a local Ollama setup.

Timeout budgets are in app/config.py so they can be tuned without
touching pipeline logic.
"""

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

from app import prompts
from app.config import TIMEOUT_GENERATE_S, TIMEOUT_RETRIEVE_S, TIMEOUT_TOTAL_S, TOP_K
from app.embeddings import embed_query
from app.llm import generate, generate_stream
from app.prompts import render_prompt
from app.timing import timed_stage
from app.vector_store import get_client, search

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


@dataclass
class AskResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    # Phase 3: degradation tracking
    retrieval_fallback: bool = False   # True if BM25 was used
    generation_fallback: bool = False  # True if generation timed out


class StreamEvent(TypedDict, total=False):
    """One event yielded by ask_stream().

    type="chunks":  retrieved_chunks populated (sent first).
    type="token":   text is one answer token.
    type="done":    prompt_tokens/completion_tokens (sent last).
    type="fallback": generation timed out; raw chunks already sent.
    """
    type: Literal["chunks", "token", "done", "fallback"]
    retrieved_chunks: list[RetrievedChunk]
    text: str
    prompt_tokens: int
    completion_tokens: int
    reason: str
    retrieval_fallback: bool


def _run_with_timeout(fn, timeout: float, *args, **kwargs):
    """Run fn(*args, **kwargs) in a thread with a timeout.

    Returns the result on success.
    Raises FuturesTimeoutError if the timeout is exceeded.
    Raises any exception fn raises (re-raised from the thread).
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)


def retrieve(query: str, top_k: int = TOP_K) -> tuple[list[RetrievedChunk], bool]:
    """Retrieve top-k chunks. Returns (chunks, used_fallback).

    Primary path: embed query -> vector search (Milvus).
    Fallback path (on timeout): BM25 keyword search over raw chunks.

    The fallback flag lets callers log/surface that degradation happened,
    without changing the interface for callers that don't care.
    """
    def _vector_retrieve():
        with timed_stage("embed_query"):
            query_embedding = embed_query(query)
        with timed_stage("vector_search"):
            client = get_client()
            hits = search(client, query_embedding, top_k)
        return [
            RetrievedChunk(
                text=hit["entity"]["text"],
                source=hit["entity"]["source"],
                chunk_index=hit["entity"]["chunk_index"],
                score=hit["distance"],
            )
            for hit in hits
        ]

    try:
        with timed_stage("retrieve_total"):
            chunks = _run_with_timeout(_vector_retrieve, TIMEOUT_RETRIEVE_S)
        return chunks, False

    except FuturesTimeoutError:
        logger.warning(
            "retrieve: vector search timed out after %.1fs — falling back to BM25",
            TIMEOUT_RETRIEVE_S,
        )
        from app.bm25 import bm25_search
        with timed_stage("retrieve_total"):
            chunks = bm25_search(query, top_k=top_k)
        return chunks, True

    except Exception as exc:
        logger.error("retrieve: vector search failed (%s) — falling back to BM25", exc)
        from app.bm25 import bm25_search
        with timed_stage("retrieve_total"):
            chunks = bm25_search(query, top_k=top_k)
        return chunks, True


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c.source}, chunk {c.chunk_index}]\n{c.text}" for c in chunks
    )


def build_prompt(query: str, chunks: list[RetrievedChunk], version: str | None = None) -> str:
    if version is None:
        version = prompts.CURRENT_VERSION
    return render_prompt(version, query=query, context=build_context(chunks))


def ask(query: str, top_k: int = TOP_K, seed: int | None = None) -> AskResult:
    """Non-streaming: wait for the full answer.

    Used by latency_report.py, baseline.py, and eval.py.
    Returns AskResult with fallback flags set if degradation occurred.
    """
    chunks, retrieval_fallback = retrieve(query, top_k=top_k)

    if retrieval_fallback:
        logger.info("ask: using BM25 fallback chunks for generation")

    prompt = build_prompt(query, chunks)

    try:
        def _generate():
            return generate(prompt, seed=seed)

        with timed_stage("generate") as extra:
            gen_result = _run_with_timeout(_generate, TIMEOUT_GENERATE_S)
            extra["prompt_tokens"] = gen_result.prompt_tokens
            extra["completion_tokens"] = gen_result.completion_tokens

        return AskResult(
            answer=gen_result.answer,
            retrieved_chunks=chunks,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            retrieval_fallback=retrieval_fallback,
            generation_fallback=False,
        )

    except FuturesTimeoutError:
        logger.warning(
            "ask: generation timed out after %.1fs — returning raw chunks",
            TIMEOUT_GENERATE_S,
        )
        return AskResult(
            answer="",
            retrieved_chunks=chunks,
            prompt_tokens=0,
            completion_tokens=0,
            retrieval_fallback=retrieval_fallback,
            generation_fallback=True,
        )


@dataclass
class CompareResult:
    retrieved_chunks: list[RetrievedChunk]
    results: dict[str, AskResult]


def ask_compare(query: str, versions: list[str], top_k: int = TOP_K) -> CompareResult:
    """Retrieve once, generate with each named prompt version."""
    chunks, _ = retrieve(query, top_k=top_k)
    results: dict[str, AskResult] = {}

    for version in versions:
        prompt = build_prompt(query, chunks, version=version)

        with timed_stage("generate") as extra:
            extra["prompt_version"] = version
            gen_result = generate(prompt)
            extra["prompt_tokens"] = gen_result.prompt_tokens
            extra["completion_tokens"] = gen_result.completion_tokens

        results[version] = AskResult(
            answer=gen_result.answer,
            retrieved_chunks=chunks,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
        )

    return CompareResult(retrieved_chunks=chunks, results=results)


def ask_stream(query: str, top_k: int = TOP_K) -> Iterator[StreamEvent]:
    """Streaming: yield chunks -> tokens -> done (or fallback event).

    On generation timeout: yields a "fallback" event after chunks are
    already sent, so the UI can show retrieved context even when
    generation fails.
    """
    chunks, retrieval_fallback = retrieve(query, top_k=top_k)

    yield {
        "type": "chunks",
        "retrieved_chunks": chunks,
        "retrieval_fallback": retrieval_fallback,
    }

    prompt = build_prompt(query, chunks)
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with timed_stage("generate") as extra:
            for piece in generate_stream(prompt):
                text = piece.get("response", "")
                if text:
                    yield {"type": "token", "text": text}
                if piece.get("done"):
                    prompt_tokens = piece.get("prompt_eval_count", 0)
                    completion_tokens = piece.get("eval_count", 0)
            extra["prompt_tokens"] = prompt_tokens
            extra["completion_tokens"] = completion_tokens

        yield {
            "type": "done",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    except Exception as exc:
        # generate_stream doesn't go through _run_with_timeout (it's a
        # generator), so we catch httpx read timeouts directly here.
        logger.warning("ask_stream: generation failed (%s) — sending fallback", exc)
        yield {
            "type": "fallback",
            "reason": f"Generation failed: {exc}",
            "retrieval_fallback": retrieval_fallback,
        }
