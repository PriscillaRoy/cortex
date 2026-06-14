"""Lightweight stage-timing instrumentation.

Phase 1 starts here deliberately simple: a context manager that times a
block and emits a structured JSON log line. This gives us real numbers
fast, without pulling in OpenTelemetry/Prometheus machinery yet. Once we
can see the data shape (which stages dominate, what the variance looks
like), we'll decide what's worth promoting to proper metrics in
`app/metrics.py`.

Each log line looks like:
    {"event": "stage_timing", "stage": "retrieve", "duration_ms": 42.3,
     "request_id": "a1b2c3d4"}

`request_id` ties multiple stage timings back to one /ask call, so we can
later reconstruct "for this request, retrieval took X, generation took Y."
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger("cortex.timing")

# Set once per request (in the FastAPI endpoint) so nested stage timings
# can tag themselves with the same request_id without threading it through
# every function signature.
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def timed_stage(stage: str):
    """Time a block and emit a structured JSON log line.

    Usage:
        with timed_stage("retrieve"):
            chunks = retrieve(query)

    The context manager yields a dict (`extra`) that the caller can write
    additional fields into - these get merged into the log line. Useful
    for attaching things measured *during* the block (e.g. token counts
    from an LLM response) without a second log line:

        with timed_stage("generate") as extra:
            result = generate(prompt)
            extra["prompt_tokens"] = result.prompt_tokens
    """
    start = time.perf_counter()
    extra: dict = {}
    try:
        yield extra
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record = {
            "event": "stage_timing",
            "stage": stage,
            "duration_ms": round(duration_ms, 2),
            "request_id": get_request_id(),
        }
        record.update(extra)
        logger.info(json.dumps(record))
