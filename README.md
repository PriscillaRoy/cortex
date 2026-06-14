# Cortex

Ask questions over your personal notes/docs using a local RAG pipeline,
built as a vehicle for production-AI engineering practices: observability,
latency budgeting, eval-gated CI, and graceful degradation under failure.

## Status: Phase 0 (skeleton) — done

End-to-end pipeline working: notes -> chunked -> embedded -> stored in
Milvus Lite -> retrieved -> prompted to a local LLM via Ollama.

Synthetic notes in `data/notes/` are placeholders covering varied topics
(system design, debugging, observability, ML theory, personal finance) so
retrieval has something meaningful to differentiate between. Swap in your
own notes by pointing `CORTEX_NOTES_DIR` at a different folder — no code
changes needed.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need [Ollama](https://ollama.com) running locally with a model
pulled:

```bash
ollama pull llama3.2:1b
```

(Config defaults to `llama3.2:1b` for speed; change `OLLAMA_MODEL` in
`app/config.py` or via env var for a larger model.)

## Running

1. **Ingest notes** (chunk, embed, store in Milvus):

   ```bash
   python -m app.ingest
   ```

   This creates `data/cortex.db` (Milvus Lite's embedded database file —
   gitignored).

2. **Start the API**:

   ```bash
   uvicorn app.main:app --reload
   ```

3. **Ask a question**:

   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What MongoDB index was missing in the debugging notes?"}'
   ```

## A note on the embedding model

`app/embeddings.py` tries to load `all-MiniLM-L6-v2` via
sentence-transformers. If that model isn't available (no network access to
huggingface.co, or not pre-downloaded), it **falls back to a deterministic
hashing-based embedder** so the pipeline still runs end-to-end — useful for
CI and offline development.

The fallback is NOT semantically meaningful (it's word-overlap-based, not a
real embedding). If you see a warning like:

```
Could not load sentence-transformers model 'all-MiniLM-L6-v2' (...).
Falling back to hashing-based embedder.
```

...retrieval quality will be poor until either (a) you have network access
on first run (the model downloads and caches automatically), or (b) you
pre-download the model into the environment.

`app/embeddings.using_fallback_embedder()` returns whether the fallback is
active — useful for surfacing this in logs/metrics later (Phase 1).

## Architecture

```
notes/*.md -> chunk (app/chunking.py)
            -> embed (app/embeddings.py)
            -> Milvus Lite (app/vector_store.py)

query -> embed -> Milvus search -> top-k chunks
      -> build_prompt -> Ollama generate (app/llm.py)
      -> answer
```

All wired together in `app/rag.py`, exposed via FastAPI in `app/main.py`.

## Tests

```bash
python -m pytest tests/
```

Retrieval tests run regardless of which embedder is active (real model or
fallback) — they check structural correctness (right number of results,
right fields) rather than asserting specific semantic matches, since the
fallback embedder can't guarantee those. Phase 2 will add a proper
semantic-quality eval suite that requires the real embedding model.

## Roadmap

- [x] **Phase 0** — working end-to-end skeleton
- [ ] **Phase 1** — OpenTelemetry tracing, Prometheus metrics (per-stage
      latency, token/cost), Grafana dashboard, memory profiling
- [ ] **Phase 2** — eval harness (20-30 Q/A pairs), CI regression gating
- [ ] **Phase 3** — latency budgets, timeouts, graceful degradation
      (keyword-search fallback, partial-answer fallback)
- [ ] **Phase 4** — agentic multi-step queries (e.g. "summarize notes from
      this week"), full reasoning-trace logging
- [ ] **Phase 5** (stretch) — LoRA fine-tuning for structured extraction,
      DPO preference tuning

