"""Core RAG pipeline: retrieve relevant chunks, build a prompt, generate
an answer.

Two ways to run generation:
  - ask(query)        -> waits for the full answer, returns AskResult
                          (used by latency_report.py for batch analysis)
  - ask_stream(query) -> generator yielding incremental pieces as they
                          arrive from Ollama (used by /ask for the UI)

Both go through retrieve() and build_prompt() identically - only the
generation step differs in how it's consumed. Both emit the same
stage_timing logs (embed_query, vector_search, retrieve_total, generate),
so latency_report.py's analysis works unchanged regardless of which path
produced the data.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypedDict

from app import prompts
from app.config import TOP_K
from app.embeddings import embed_query
from app.llm import generate, generate_stream
from app.prompts import render_prompt
from app.timing import timed_stage
from app.vector_store import get_client, search


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


class StreamEvent(TypedDict, total=False):
    """One event yielded by ask_stream().

    type="chunks": retrieved_chunks is populated (sent once, first).
    type="token":  text is one piece of the answer.
    type="done":   prompt_tokens/completion_tokens are populated (sent
                   last).
    """

    type: Literal["chunks", "token", "done"]
    retrieved_chunks: list[RetrievedChunk]
    text: str
    prompt_tokens: int
    completion_tokens: int


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
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


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c.source}, chunk {c.chunk_index}]\n{c.text}" for c in chunks
    )


def build_prompt(query: str, chunks: list[RetrievedChunk], version: str | None = None) -> str:
    if version is None:
        version = prompts.CURRENT_VERSION
    return render_prompt(version, query=query, context=build_context(chunks))


def ask(query: str, top_k: int = TOP_K, seed: int | None = None) -> AskResult:
    """Non-streaming: wait for the full answer. Used by latency_report.py
    and baseline.py.

    seed: passed through to generate() for reproducibility. See
    app.config.BASELINE_SEED.
    """
    with timed_stage("retrieve_total"):
        chunks = retrieve(query, top_k=top_k)

    prompt = build_prompt(query, chunks)

    with timed_stage("generate") as extra:
        gen_result = generate(prompt, seed=seed)
        extra["prompt_tokens"] = gen_result.prompt_tokens
        extra["completion_tokens"] = gen_result.completion_tokens

    return AskResult(
        answer=gen_result.answer,
        retrieved_chunks=chunks,
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
    )


@dataclass
class CompareResult:
    retrieved_chunks: list[RetrievedChunk]
    results: dict[str, AskResult]  # keyed by prompt version name


def ask_compare(query: str, versions: list[str], top_k: int = TOP_K) -> CompareResult:
    """Retrieve once, generate with each named prompt version, for
    side-by-side comparison. Each version's generation is timed/logged
    separately (stage="generate", with a "prompt_version" extra field) so
    latency differences between versions are also visible in timing logs.
    """
    with timed_stage("retrieve_total"):
        chunks = retrieve(query, top_k=top_k)

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
    """Streaming: yield retrieved chunks first, then answer tokens as
    they arrive, then a final 'done' event with token counts.

    The 'generate' stage_timing log covers the ENTIRE stream (first
    token to last) - i.e. total generation time, same definition as in
    ask(). Time-to-first-token isn't separately logged yet; that's a
    natural next addition if we want to split "latency to start
    responding" from "total generation time".
    """
    with timed_stage("retrieve_total"):
        chunks = retrieve(query, top_k=top_k)

    yield {"type": "chunks", "retrieved_chunks": chunks}

    prompt = build_prompt(query, chunks)

    prompt_tokens = 0
    completion_tokens = 0

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
