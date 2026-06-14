# Cortex Architecture

This document tracks the architecture as it evolves phase by phase. Each
phase's diagram builds on the previous one, so you can see what was added
and why.

---

## After Phase 0 — Working RAG skeleton

```
+---------------------------------------------------------------+
|                         CORTEX                                  |
|                                                                  |
|  data/notes/*.md                                                |
|       |                                                         |
|       v  app/ingest.py                                          |
|  chunk (app/chunking.py)                                        |
|       |                                                         |
|       v                                                         |
|  embed (app/embeddings.py) --[real model OR hashing fallback]   |
|       |                                                         |
|       v                                                         |
|  Milvus Lite (app/vector_store.py)                              |
|                                                                  |
|  +-------------------------------------------------------+    |
|  |  /ask  (app/main.py)                                    |    |
|  |   `- rag.ask() (app/rag.py)                             |    |
|  |       |- retrieve()                                     |    |
|  |       |   |- embed_query                                |    |
|  |       |   `- vector_search                              |    |
|  |       |- build_prompt()                                 |    |
|  |       `- generate() (Ollama, app/llm.py)                |    |
|  +-------------------------------------------------------+    |
+---------------------------------------------------------------+
```

**Files and what they do:**

| File | Role |
|---|---|
| `app/config.py` | All tunables in one place (chunk size, model names, paths) |
| `app/chunking.py` | Splits note text into overlapping chunks |
| `app/embeddings.py` | Loads embedding model, turns text into vectors (with offline fallback) |
| `app/vector_store.py` | Wraps Milvus Lite: create collection, insert, search |
| `app/llm.py` | Calls Ollama's `/api/generate` |
| `app/rag.py` | Orchestrates: retrieve -> build prompt -> generate |
| `app/ingest.py` | One-shot script: notes -> chunks -> embeddings -> Milvus |
| `app/main.py` | FastAPI app exposing `/ask` and `/health` |

---

## After Phase 1 — Latency instrumentation

```
+---------------------------------------------------------------+
|                         CORTEX                                  |
|                                                                  |
|  data/notes/*.md                                                |
|       |                                                         |
|       v  app/ingest.py                                          |
|  chunk -> embed -> Milvus Lite   (unchanged from Phase 0)        |
|                                                                  |
|  +-------------------------------------------------------+    |
|  |  /ask  (app/main.py)                                    |    |
|  |   |- new_request_id()         <-- app/timing.py         |    |
|  |   `- rag.ask()                                          |    |
|  |       |- retrieve()                                     |    |
|  |       |   |- [embed_query]    (timed)                   |    |
|  |       |   `- [vector_search]  (timed)                   |    |
|  |       |- build_prompt()                                 |    |
|  |       `- [generate]           (timed) + token counts    |    |
|  |           (returns prompt_tokens, completion_tokens)    |    |
|  +-------------------------------------------------------+    |
|                                                                  |
|  Each "timed" block emits a JSON log line:                      |
|    {"event": "stage_timing", "stage": ..., "duration_ms": ...,  |
|     "request_id": ..., [extra fields like token counts]}        |
|                                                                  |
|  app/latency_report.py:                                         |
|    runs N queries -> collects all timing log lines              |
|    -> p50/p95 per stage, per-query detail,                      |
|       token-count vs latency correlation                        |
|    -> data/latency_raw.json (every raw measurement)             |
+---------------------------------------------------------------+
```

**New/changed files:**

| File | Role |
|---|---|
| `app/timing.py` | `timed_stage()` context manager - times a block, logs JSON, supports attaching extra fields (e.g. token counts) |
| `app/latency_report.py` | Batch-runs queries, aggregates timing logs into p50/p95 + correlation analysis |
| `app/rag.py` *(changed)* | Each stage wrapped in `timed_stage(...)`; `AskResult` now includes `prompt_tokens`/`completion_tokens` |
| `app/llm.py` *(changed)* | `generate()` returns `GenerateResult` (answer + token counts) instead of a bare string |
| `app/main.py` *(changed)* | Generates a `request_id` per `/ask` call; response includes token counts |

**What this gives us:** for any request, we know how long each stage took
(`embed_query`, `vector_search`, `generate`) and how many tokens were in
the prompt/response - the foundation for the latency-budget work in Phase
3 and cost-estimation work later in Phase 1.

---

## Coming next (Phase 1 continued) — Memory profiling

Will add: RSS (memory) sampling around the embedding model load and
during concurrent requests - the piece that maps directly to the Render
OOM story.

---

## After Phase 1 — Streaming + UI

```
+---------------------------------------------------------------+
|                         CORTEX                                  |
|                                                                  |
|  data/notes/*.md -> chunk -> embed -> Milvus Lite  (unchanged)  |
|                                                                  |
|  +-------------------------------------------------------+    |
|  |  POST /ask  (app/main.py)  --> Server-Sent Events       |    |
|  |   |- new_request_id()                                   |    |
|  |   `- rag.ask_stream()  (app/rag.py)                      |    |
|  |       |- retrieve()  [embed_query] [vector_search]      |    |
|  |       |     `-- yield {"type": "chunks", ...}            |    |
|  |       |- build_prompt()                                  |    |
|  |       `- [generate]                                      |    |
|  |             for piece in generate_stream(prompt):        |    |
|  |               yield {"type": "token", "text": ...}       |    |
|  |             yield {"type": "done", prompt/completion     |    |
|  |                     tokens}                               |    |
|  +-------------------------------------------------------+    |
|                          |                                      |
|                          v  SSE: data: {...}\n\n                |
|  +-------------------------------------------------------+    |
|  |  static/index.html  (served at /)                       |    |
|  |   - pipeline status strip (embed / search / generate)   |    |
|  |   - retrieved-chunks panel (shown on "chunks" event)     |    |
|  |   - answer panel (tokens appended live, typewriter)      |    |
|  |   - time-to-first-token + total time, shown after done   |    |
|  +-------------------------------------------------------+    |
|                                                                  |
|  app/latency_report.py (unchanged interface):                   |
|    calls ask() (non-streaming) which internally consumes        |
|    the SAME generate_stream() via app/llm.py's generate()        |
|    - one underlying Ollama call, two consumption modes          |
+---------------------------------------------------------------+
```

**New/changed files:**

| File | Role |
|---|---|
| `app/llm.py` *(changed)* | `generate_stream()` - yields each token chunk from Ollama. `generate()` now consumes the stream and returns totals (used by latency_report.py) |
| `app/rag.py` *(changed)* | New `ask_stream()` generator yielding `chunks` -> `token`*N -> `done` events. `ask()` (non-streaming) kept for latency_report.py |
| `app/main.py` *(changed)* | `/ask` now returns `StreamingResponse` (SSE) instead of JSON. Serves `static/` for the UI |
| `static/index.html` | Single-file UI: pipeline status strip, retrieved-chunks panel, streaming answer panel |

**What this gives us:** the user sees retrieved chunks immediately and the
answer appears word-by-word, with the pipeline strip showing which stage
is active. Total compute time is unchanged from Phase 1's numbers - this
is purely a *perceived*-latency improvement (time-to-first-token vs.
time-to-complete-response), which the UI now measures and displays
directly.

---

## After Phase 1 — Prompt A/B + baseline comparison

```
+---------------------------------------------------------------+
|                         CORTEX                                  |
|                                                                  |
|  app/prompts.py:                                                |
|    PROMPT_VERSIONS = {                                          |
|      "v1_verbose": "...",                                       |
|      "v2_concise_2sentence": "...",                             |
|    }                                                            |
|    CURRENT_VERSION = "v2_concise_2sentence"  <- used by         |
|                                                  ask/ask_stream  |
|                                                                  |
|  +-------------------------------------------------------+    |
|  |  POST /ask/compare  (app/main.py)                       |    |
|  |   `- rag.ask_compare()  (app/rag.py)                    |    |
|  |       |- retrieve()  ONCE  [embed_query][vector_search] |    |
|  |       `- for each prompt version:                        |    |
|  |             [generate]  (tagged w/ prompt_version)       |    |
|  |   -> {retrieved_chunks, results: {v1: {...}, v2: {...}}}|    |
|  +-------------------------------------------------------+    |
|                                                                  |
|  static/index.html:                                             |
|    "compare prompts" toggle -> side-by-side v1/v2 answers       |
|    + token counts, same retrieved chunks for both                |
|                                                                  |
|  +-------------------------------------------------------+    |
|  |  app/baseline.py                                        |    |
|  |   save:  run QUERIES -> data/baselines/baseline_*.json  |    |
|  |          + data/baselines/latest.json                   |    |
|  |   check: re-run QUERIES, diff answers vs latest.json     |    |
|  |          -> "N/30 answers changed" + before/after text   |    |
|  +-------------------------------------------------------+    |
+---------------------------------------------------------------+
```

**New/changed files:**

| File | Role |
|---|---|
| `app/prompts.py` | Named prompt versions (`PROMPT_VERSIONS`) + `CURRENT_VERSION`. `render_prompt()` fills a template |
| `app/rag.py` *(changed)* | `build_prompt` now takes a `version` param via the registry. New `ask_compare()` - retrieve once, generate per version |
| `app/main.py` *(changed)* | New `POST /ask/compare` endpoint |
| `static/index.html` *(changed)* | "compare prompts" toggle showing v1 vs v2 side-by-side |
| `app/baseline.py` | `save`/`check` commands - capture answers for all `latency_report.QUERIES`, diff future runs against the saved baseline |

**What this gives us:** a way to *see* the v1-vs-v2 quality tradeoff
directly (not just token counts), and a mechanism to ask "did changing
[prompt/model/chunk size] change any answers, and which ones?" - the
foundation Phase 2's eval harness will build on by adding a *correctness*
score on top of "did it change."

**Design note - `CURRENT_VERSION` as a single switch**: `ask()`,
`ask_stream()`, `latency_report.py`, and `baseline.py` all use whichever
prompt `app/prompts.CURRENT_VERSION` points to - one flag controls the
"active" prompt for benchmarking and production. `/ask/compare` ignores
this flag entirely and always evaluates both versions against identical
retrieved context. To benchmark v1 specifically: flip `CURRENT_VERSION =
"v1_verbose"`, rerun `latency_report.py`, flip back. This is how the v1
numbers in Section 1's table were produced.

