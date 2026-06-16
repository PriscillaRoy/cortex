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

---

## After Phase 2 — Eval harness

```
+---------------------------------------------------------------+
|  data/eval_cases.json                                          |
|    30 hand-written cases, each with:                           |
|      - query                                                    |
|      - expected_sources  (which note file retrieval should hit) |
|      - expected_keywords (what key terms the answer must have)  |
|      - eval_criterion    (plain-English for LLM judge)          |
|      - unanswerable      (bool - should model decline?)         |
|                                                                  |
|  app/eval.py                                                    |
|    run_eval():                                                  |
|      for each case:                                             |
|        ask(query, seed=BASELINE_SEED)                           |
|          |- score_retrieval: chunk.source in expected_sources?  |
|          `- score_generation: all expected_keywords in answer?  |
|        if --llm-judge AND keyword fail AND eval_criterion:       |
|          judge_with_llm(query, answer, criterion)               |
|          -> YES/NO overrides keyword result                      |
|                                                                  |
|    compute_scores():                                            |
|      retrieval_hit_rate  = hits / scored_queries                |
|      generation_score    = final_pass / total  (judge-adjusted  |
|                            when judge ran, else keyword)         |
|                                                                  |
|    CI gates (--threshold, --max-regression):                    |
|      --threshold 0.80         absolute floor                    |
|      --max-regression 0.05    max drop vs previous run          |
|      both: exit 1 if either fails                               |
|                                                                  |
|  Output per-query:                                              |
|    R:+/R:-/R:~  retrieval hit/miss/N-A                         |
|    G:+/G:-      keyword pass/fail                               |
|    J:+/J:-      judge pass/fail (only when --llm-judge)         |
+---------------------------------------------------------------+
```

**New files:**

| File | Role |
|---|---|
| `app/eval.py` | Eval harness: per-query scoring, aggregate report, CI gate |
| `data/eval_cases.json` | 30 test cases with expected sources, keywords, criteria |

**What the symbols mean:**

  R:+ = Retrieval HIT  (correct source file retrieved)
  R:~ = Retrieval N/A  (unanswerable query, no expected source)
  R:- = Retrieval MISS (wrong source retrieved)
  G:+ = Generation PASS (all keyword groups satisfied)
  G:- = Generation FAIL (at least one keyword missing)
  J:+ = Judge PASS  (LLM judge overrode keyword failure as paraphrase)
  J:- = Judge FAIL  (LLM judge confirmed the failure is genuine)

**Real results (v1 verbose prompt, seed=42):**
  Retrieval:  23/23 (100%)  Generation keyword: 27/30 (90%)
  With LLM judge: 29/30 (96.7%) [2 overrides: paraphrase; 1 confirmed: hallucination]

---

## After Phase 3 — Timeouts + graceful degradation

```
+---------------------------------------------------------------+
|  app/config.py:                                                 |
|    TIMEOUT_RETRIEVE_S = 3.0   # embed + vector search          |
|    TIMEOUT_GENERATE_S = 12.0  # LLM generation                 |
|                                                                  |
|  app/rag.retrieve():                                            |
|    try:                                                          |
|      _run_with_timeout(_vector_retrieve, TIMEOUT_RETRIEVE_S)   |
|        -> embed_query + vector_search (Milvus)                  |
|    except TimeoutError:                                         |
|      bm25_search(query)  <- app/bm25.py, no Milvus needed      |
|      fallback=True                                              |
|                                                                  |
|  app/rag.ask() / ask_stream():                                  |
|    retrieve() -> (chunks, retrieval_fallback)                   |
|    try:                                                          |
|      _run_with_timeout(generate, TIMEOUT_GENERATE_S)           |
|    except TimeoutError:                                         |
|      return AskResult(answer="", generation_fallback=True)      |
|      OR yield {"type": "fallback", ...}  for streaming          |
|                                                                  |
|  static/index.html:                                             |
|    on "chunks" event + retrieval_fallback=True:                 |
|      show yellow banner "Vector search timed out - BM25 used"   |
|    on "fallback" event (generation timeout):                    |
|      show yellow banner "Generation timed out - chunks shown"   |
|      (chunks already rendered above the banner)                 |
+---------------------------------------------------------------+
```

**New/changed files:**

| File | Role |
|---|---|
| `app/bm25.py` | NEW: in-memory BM25 keyword index built from Milvus chunks at first use. Pure Python, no extra deps. Used as retrieval fallback |
| `app/rag.py` *(changed)* | `retrieve()` returns `(chunks, fallback_bool)`. Both `ask()` and `ask_stream()` have timeout + fallback paths |
| `app/config.py` *(changed)* | Adds timeout constants `TIMEOUT_RETRIEVE_S`, `TIMEOUT_GENERATE_S` |
| `app/main.py` *(changed)* | SSE stream handles `fallback` event type + `retrieval_fallback` flag |
| `static/index.html` *(changed)* | Yellow degraded-state banner for both fallback paths |
| `tests/conftest.py` | NEW: session-scoped ingest fixture so tests don't require pre-populated DB |
| `tests/test_retrieval.py` *(changed)* | Updated for tuple return; new `test_retrieve_fallback_flag_false_on_normal_path` |

**Threading vs async note**: timeouts use `concurrent.futures.ThreadPoolExecutor`
rather than `asyncio` to avoid refactoring the synchronous pipeline. This
is acceptable here because Ollama serializes inference (one request at a
time regardless) so async parallelism wouldn't improve throughput at the
real bottleneck. For a cloud LLM API with true parallel generation, the
correct production path is converting `generate()`/`generate_stream()`
to `async def` with `httpx.AsyncClient` and FastAPI async route handlers.
The migration would be: `def` -> `async def` throughout, `generate()` ->
`await generate()`, `for chunk in generate_stream()` ->
`async for chunk in generate_stream()`. The architecture is already
compatible - it's a plumbing change, not a redesign.

**To demo:**

  Retrieval fallback: set TIMEOUT_RETRIEVE_S = 0.001 in config.py,
  restart server, ask a question. Chunks appear (BM25) with yellow banner.

  Generation fallback: set TIMEOUT_GENERATE_S = 0.001, restart, ask.
  Chunks appear immediately, then yellow banner instead of answer.

  Demo via the UI at http://localhost:8000 - the banners only render there.

---

## After Phase 4 — Agentic tool-calling with reasoning trace

```
+---------------------------------------------------------------+
|  POST /agent  (app/main.py)                                    |
|       |                                                         |
|       v                                                         |
|  ask_agent_stream(query)  (app/agent.py)                       |
|       |                                                         |
|       |  messages = [system, user]                              |
|       |                                                         |
|       v  loop (max 8 steps):                                    |
|  _call_ollama_with_tools(messages)                              |
|       |                                                         |
|       +--> tool_calls? ---------> _execute_tool(name, args)    |
|       |         |                       |                       |
|       |         |   app/tools/          |                       |
|       |         |   search.py     <-----+  search_notes()       |
|       |         |   list_notes.py <-----+  list_notes()         |
|       |         |   summarize.py  <-----+  summarize_note()     |
|       |         |                       |                       |
|       |         v                       v                       |
|       |    TraceEvent              tool result                  |
|       |    (tool_call)   yield --> SSE stream                   |
|       |    (tool_result) yield --> SSE stream                   |
|       |         |                                               |
|       |    append result to messages, loop back                 |
|       |                                                         |
|       +--> content? (final answer)                              |
|                 |                                               |
|            TraceEvent(answer) yield --> SSE stream              |
|                                                                 |
|  static/index.html:                                             |
|    "agent mode" toggle -> POST /agent                           |
|    trace panel shows each step live:                            |
|      THINKING -> TOOL_CALL -> TOOL_RESULT -> ... -> answer      |
+---------------------------------------------------------------+
```

**New files:**

| File | Role |
|---|---|
| `app/agent.py` | Agent loop: `_call_ollama_with_tools()` (Ollama /api/chat with tools), `_execute_tool()` (dispatch + type coercion), `ask_agent_stream()` (yields TraceEvents), `MAX_STEPS=8` safety limit |
| `app/tools/search.py` | Wraps existing `retrieve()` as a named tool. Coerces `top_k` to int (LLMs sometimes send strings) |
| `app/tools/list_notes.py` | Reads frontmatter (tags, date, title) from `data/notes/*.md`. Returns metadata without full content — the discovery step before summarize |
| `app/tools/summarize.py` | Reads one note in full, LLM-summarizes it. Path traversal protected. Returns helpful error with available filenames if file not found |
| `app/main.py` *(changed)* | Adds `POST /agent` endpoint, streams TraceEvents as SSE |
| `static/index.html` *(changed)* | "agent mode" toggle, live trace panel (thinking/tool_call/tool_result/answer steps) |

**How tool-calling works (Ollama /api/chat):**

  We send: model + messages + tool definitions (JSON schemas)
  Ollama returns either:
    message.tool_calls -> LLM wants to call a tool (name + args)
    message.content    -> LLM has a final answer

  We loop: execute tool -> append result to messages -> send again.
  The full message history goes on every iteration so the LLM has
  complete context of what's been tried.

**Key real-world findings from running it:**

  1. llama3.2:3B sometimes skips list_notes and guesses filenames
     directly (e.g. "observability.md" instead of listing first).
     Larger models follow the system prompt more reliably. Fix options:
     force list_notes as a mandatory first step in the system prompt,
     or use a larger model.

  2. LLMs send numeric arguments as strings ("6" not 6). Fixed with
     type coercion in _execute_tool() before dispatch. Always coerce
     known numeric params defensively.

  3. On tool failure, small models sometimes return a text explanation
     of the error instead of a corrected tool_call. Detected via
     retry_signals list; loop continues with a correction hint instead
     of surfacing the internal monologue as the final answer.

  4. Error messages from tools carry real signal — summarize_note
     returns the list of available files when a filename isn't found.
     The model reads this and self-corrects. Good tool error messages
     are part of the agent's reasoning loop, not just dev debugging.
