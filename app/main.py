"""Cortex API.

Run with: uvicorn app.main:app --reload

/ask streams the answer via Server-Sent Events (SSE) - the browser sees
retrieved chunks immediately, then answer tokens as they're generated,
then a final event with token counts.
"""

import json
import logging
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.embeddings import get_device, warmup
from app.prompts import PROMPT_VERSIONS
from app.rag import ask_compare, ask_stream
from app.agent import ask_agent_stream
from app.timing import _request_id, new_request_id

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cortex.startup")

app = FastAPI(title="Cortex", description="Ask questions over your notes.")


@app.on_event("startup")
def startup_warmup() -> None:
    """Load the embedding model at startup rather than on first request.

    See app.embeddings.warmup for why. Logs how long it took and which
    device (mps/cpu) is in use.
    """
    device = get_device()
    logger.info("Starting warmup (embedding device=%s)...", device)

    start = time.perf_counter()
    warmup()
    elapsed = time.perf_counter() - start

    logger.info("Warmup complete in %.1fs (device=%s)", elapsed, device)


class AskRequest(BaseModel):
    query: str
    top_k: int | None = None


class CompareRequest(BaseModel):
    query: str
    versions: list[str] | None = None  # defaults to all known versions
    top_k: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask/compare")
def compare_endpoint(request: CompareRequest) -> dict:
    """Run the same query through multiple prompt versions for
    side-by-side comparison. Retrieval happens once; each version gets
    its own generate() call against the same retrieved chunks - so any
    difference in answers/tokens/timing is attributable to the prompt,
    not to different retrieved context.
    """
    new_request_id()
    versions = request.versions or list(PROMPT_VERSIONS.keys())

    unknown = [v for v in versions if v not in PROMPT_VERSIONS]
    if unknown:
        return {"error": f"Unknown prompt version(s): {unknown}. Known: {list(PROMPT_VERSIONS.keys())}"}

    kwargs = {}
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k

    result = ask_compare(request.query, versions=versions, **kwargs)

    return {
        "retrieved_chunks": [
            {"text": c.text, "source": c.source, "chunk_index": c.chunk_index, "score": c.score}
            for c in result.retrieved_chunks
        ],
        "results": {
            version: {
                "answer": r.answer,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            for version, r in result.results.items()
        },
    }


def _sse(event: dict) -> str:
    """Format a dict as one Server-Sent Events message."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    request_id = new_request_id()
    kwargs = {}
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k

    def event_stream():
        # ContextVars don't cross into a generator run by
        # StreamingResponse the same way they do for normal request
        # handlers - re-set it explicitly so timing logs from inside
        # ask_stream carry this request_id.
        _request_id.set(request_id)

        yield _sse({"type": "request_id", "request_id": request_id})

        for event in ask_stream(request.query, **kwargs):
            if event["type"] == "chunks":
                yield _sse(
                    {
                        "type": "chunks",
                        "retrieved_chunks": [
                            {
                                "text": c.text,
                                "source": c.source,
                                "chunk_index": c.chunk_index,
                                "score": c.score,
                            }
                            for c in event["retrieved_chunks"]
                        ],
                        "retrieval_fallback": event.get("retrieval_fallback", False),
                    }
                )
            elif event["type"] == "token":
                yield _sse({"type": "token", "text": event["text"]})
            elif event["type"] == "done":
                yield _sse(
                    {
                        "type": "done",
                        "prompt_tokens": event["prompt_tokens"],
                        "completion_tokens": event["completion_tokens"],
                    }
                )
            elif event["type"] == "fallback":
                yield _sse(
                    {
                        "type": "fallback",
                        "reason": event.get("reason", "Generation timed out"),
                        "retrieval_fallback": event.get("retrieval_fallback", False),
                    }
                )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/agent")
def agent_endpoint(request: AskRequest):
    """Agentic endpoint — LLM decides which tools to call and in what order.

    Streams TraceEvents via SSE so the UI shows the agent's reasoning
    in real time: which tools it called, what they returned, and finally
    the synthesized answer.
    """
    request_id = new_request_id()

    def event_stream():
        _request_id.set(request_id)
        yield _sse({"type": "request_id", "request_id": request_id})

        for event in ask_agent_stream(request.query):
            yield _sse(event.to_dict())

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Serve the UI at / — mounted last so it doesn't shadow API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
