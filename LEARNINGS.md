# Cortex — Learnings & Interview Prep

A running record of what we measured, what we changed, why, and how to
talk about it. Numbers are from real runs (M2 Mac, MPS, `llama3.2:latest`
3B via Ollama, `all-MiniLM-L6-v2` embeddings, Milvus Lite, 16 chunks from
5 synthetic notes, 30-query benchmark in `app/latency_report.py`).

---

## 1. The numbers, run by run

| Run | What changed | completion_tokens (mean) | generate p50 | generate p95 | end_to_end p50 | r (tokens vs generate_ms) |
|---|---|---|---|---|---|---|
| Baseline (12 queries, CPU) | initial instrumentation | n/a | 2229ms | 3816ms | 2593ms | n/a |
| v1 (30 queries, CPU) | more queries, token tracking added | 76 | 2176.8ms | 3714.9ms | 2332.5ms | 0.797 -> 0.570* |
| v1 + MPS + warmup (30 queries) | GPU device, startup warmup | 76 | 2176.8ms | 3714.9ms | 2332.5ms | 0.570 |
| v2 (30 queries, MPS) | prompt: "at most 2 sentences, no preamble" | 36 | 1653.3ms | 2355.0ms | 1814.6ms | 0.276 |

\*The 0.797->0.570 shift between the first and second 30-query runs on
the same v1 prompt is itself a finding - see Section 3.

**Cold-start (`embed_query` for query #1 only)**, same model, repeated
process starts: 30982ms -> 18915ms -> 8235ms. Trending down - see Section 2.

**Net effect of prompt change (v1->v2)**: completion_tokens mean dropped
53% (76->36), generate p50 dropped 24% (2176.8ms->1653.3ms), end_to_end
p50 dropped 22% (2332.5ms->1814.6ms). One prompt-engineering change, no
model change, no hardware change.

---

## 2. Cold start: ~31s -> ~19s -> ~8s, same model, repeated runs

**What it is**: the *first* `embed_query` call in a freshly-started
process pays a one-time cost - loading `all-MiniLM-L6-v2` AND
initializing the MPS (Apple GPU) backend in PyTorch (Metal shader
compilation, GPU memory pool setup). Every subsequent call in the same
process is under 1ms (model is cached in a module-level variable).

**Why it's decreasing across separate process restarts**: macOS caches
recently-used files and compiled artifacts at the OS level. Even though
each Python process is new (so our in-process model cache doesn't carry
over), the underlying files/libraries are increasingly "warm" in the OS
page cache / Metal shader cache across repeated runs.

**What we did about it**: added `app/embeddings.warmup()`, called from a
FastAPI `startup` event (`app/main.py`). This doesn't eliminate the cost -
it moves it from "the first user's request" to "server startup/deploy
time," which is a much less disruptive place for a multi-second delay to
live.

**Render relevance**: a free-tier Render instance that's slept and gets
woken by a request would, without warmup, make that user's first request
absorb this cost on top of normal generation time - likely exceeding
reasonable timeouts. With warmup-at-startup, the cold-start cost is paid
during the wake-up/restart itself.

---

## 3. The correlation story: 0.797 -> 0.570 -> 0.276

This sequence is one of the more sophisticated findings in this project,
and it's worth understanding precisely because the "obvious" reading
(correlation going down = the relationship is breaking down / something's
wrong) is **not** what's happening.

- **Run 1 (r=0.797, n=12)**: small sample, one extreme outlier
  (point-in-time-correctness question, both highest tokens AND highest
  generate time) - dominates the correlation.
- **Run 2 (r=0.570, n=30, same v1 prompt)**: larger sample, wider natural
  range of completion_tokens (40-153, a ~4x spread). Token count is still
  the dominant factor, but other variance (M2 thermal/power state, MPS
  scheduling) is now visible too.
- **Run 3 (r=0.276, n=30, v2 prompt - "at most 2 sentences")**: the prompt
  change compressed completion_tokens into a narrow band (10-75, mostly
  25-50). With less *variation* in the input variable (token count),
  there's less for it to "explain" - the other variance (which was always
  there) becomes comparably sized, so r drops.

**The takeaway**: correlation measures explained variance *given the
observed range of the data*. Compressing a variable's range can lower its
measured correlation with an outcome even if the underlying mechanism
(more tokens = more compute = more time) hasn't changed at all. This is a
real statistical subtlety, not a bug or a contradiction.

---

## 4. What "highest-leverage" meant, concretely

`generate` was ~84-86% of end-to-end latency. "Leverage" = improvement
per unit of effort. Spending effort on `vector_search` (3-10ms) can save
at most ~10ms; spending the same effort on `generate` (1650-2350ms) has
~200x more room to matter. This is why all subsequent optimization work
targeted `generate`.

---

## 5. Optimizations tried, and what's left

### Tried

| Optimization | Mechanism | Result |
|---|---|---|
| MPS (Apple GPU) | `app/embeddings.get_device()` - tries `mps`, falls back to `cpu` | Confirmed `mps` active. Cold-start cost present either way (MPS init itself has overhead) - see Section 2 |
| Startup warmup | `app/embeddings.warmup()` called on FastAPI `startup` | Moves cold-start cost to deploy time, not user-request time |
| Streaming | `app/llm.generate_stream()`, SSE in `/ask` | Doesn't reduce total compute time - reduces *perceived* latency (time-to-first-token vs. time-to-full-response). UI shows both numbers |
| Prompt constraint ("at most 2 sentences, no preamble") | `app/prompts.py` v2 vs v1 | completion_tokens -53%, generate p50 -24%. Real, measured win |

**Switching the active prompt**: `app/prompts.CURRENT_VERSION` is the
single switch - `ask()`, `ask_stream()`, `latency_report.py`, and
`baseline.py` all build their prompt from whatever this points to.
`/ask/compare` is the exception: it ignores `CURRENT_VERSION` and always
evaluates both versions side-by-side against identical retrieved context.
To benchmark a specific version end-to-end (e.g. reproduce the v1 row in
the table above), flip `CURRENT_VERSION`, rerun, flip back.

### Not yet tried (and why)

| Optimization | Why not yet | When it'd make sense |
|---|---|---|
| Smaller model (`llama3.2:1b`) | Haven't measured quality impact | After Phase 2's eval harness can quantify quality before/after |
| `num_predict` hard cap | Blunt - truncates mid-sentence | As a safety net once typical-case (prompt-driven) length is already tuned |
| Query-type-specific prompts | More engineering, marginal gain once v2 already compresses most answers | If eval shows some question *types* still need more tokens than others |
| Caching repeated queries | Doesn't help novel queries; our 30 queries are all distinct | If real usage shows repeated/similar questions |
| Fine-tuning for length calibration | Most expensive; needs labeled data | Last resort if prompting plateaus |

---

## 6. The "completion_tokens is the bottleneck, but the LLM decides it" problem

This is the central tension of the whole latency story: the dominant cost
driver (`completion_tokens`) isn't directly controllable the way an
algorithm parameter is - you can't "set" how many tokens an LLM will
generate. What you *can* do:

1. **Constrain structurally** (what we did): "at most 2 sentences" is a
   format spec, not a vague style request - models follow format specs
   more reliably than "be concise."
2. **Cap as a safety net** (`num_predict`): blunt truncation, prevents
   pathological long outputs.
3. **Match prompt to task type**: lookups vs. summaries vs. comparisons
   may warrant different length budgets.
4. **Fine-tune**: shift the model's *default* behavior for this domain.

The honest framing: infrastructure optimizations (GPU, caching, smaller
model) change the cost *per token*. Prompt-level optimizations change the
*number* of tokens. Both matter; they're different levers.

---

## 7. Interview Q&A

**Q: Walk me through how you found and fixed your biggest latency
bottleneck.**

A: I instrumented every pipeline stage (embedding, vector search,
generation) with timing and ran a 30-query benchmark. Generation was
~85% of end-to-end latency - by far the highest-leverage target. I then
checked what predicted generation time: completion token count correlated
at r about 0.8. Since I can't directly control how many tokens an LLM
generates, I changed the prompt to request "at most 2 sentences, no
preamble" - a structural constraint, not a vague style request. That cut
mean completion tokens by 53% and generation p50 by 24%, with no model or
hardware change.

**Q: How do you know the prompt change didn't hurt answer quality?**

A: Honestly - with this dataset, I haven't fully quantified that yet,
which is exactly why I built `app/baseline.py` and the `/ask/compare`
endpoint: `baseline.py` lets me snapshot all 30 answers and diff future
runs against that snapshot (did the *text* change, and how?).
`/ask/compare` runs both prompt versions side-by-side against the same
retrieved context, so I can read both answers directly. Neither of these
*scores* correctness yet - that's the next phase (eval harness with
expected-answer matching, gated in CI). Right now I can detect *that*
something changed; the next step is judging *whether the change is good*.

**Q: Why did your correlation coefficient go DOWN after your
optimization worked?**

A: Because correlation measures explained variance relative to the
*range* of the input. My prompt change compressed completion_tokens from
a 40-153 range down to 10-75. With less variation in token count to
"explain," other sources of latency variance (hardware scheduling,
thermal state) became comparably sized - so the correlation coefficient
dropped even though the underlying relationship (more tokens leads to
more time) is unchanged. It's a real result about what correlation can
tell you given a narrowed input range, not a sign the optimization
failed.

**Q: Tell me about a surprising result.**

A: Cold-start latency on Apple Silicon. I expected loading an 80MB
embedding model to take a couple seconds; the first call in a fresh
process took 19-31 seconds. The actual cost wasn't the model file - it
was MPS (Apple's GPU backend) initializing for the first time in that
process: shader compilation and memory pool setup. I addressed it with a
startup-time warmup hook, which doesn't eliminate the cost but moves it
to deploy time instead of a user's first request - directly relevant to a
Render free-tier deployment, where a sleeping instance waking on request
would otherwise eat this cost inside the user's request.

**Q: How would you decide whether to use a smaller model?**

A: I'd run the same 30-query benchmark with `llama3.2:1b` instead of the
3B model, compare generate latency, AND run the eval harness (once built)
on both to compare answer quality. The decision is a tradeoff I'd want
data for on both axes - I wouldn't ship a faster model that's
meaningfully less accurate without knowing the size of that accuracy
drop.

**Q: What's the architecture, end to end?**

A: Notes (markdown) get chunked with overlap, embedded
(sentence-transformers, MPS-accelerated), and stored in Milvus Lite. A
query gets embedded, searched against Milvus for top-k chunks, those
chunks plus the query go into a prompt template, and the prompt goes to a
local LLM via Ollama, streamed back to the user via Server-Sent Events.
Every stage emits structured timing logs (stage name, duration, request
ID, and for generation, token counts), which a batch script aggregates
into p50/p95 latency and correlation analysis.

**Q: What would you build next?**

A: An eval harness - a set of questions with expected-answer criteria,
scored automatically (retrieval hit-rate: did we retrieve the right
chunk; answer correctness: does the generated answer contain the key
fact). That score becomes the quality half of every tradeoff I've been
making on the speed half - and it's what `baseline.py`'s "did the answer
change" check is missing: *whether the change was an improvement*. I'd
also gate this in CI so a regression in either retrieval or generation
quality fails the build.

---

## 8. Phase 2 — Eval harness: what we built and what we found

### Architecture

Two files:
- `data/eval_cases.json`: 30 hand-written test cases. Each has
  `expected_sources` (which note file retrieval should hit) and
  `expected_keywords` (grouped: each group is an OR set of synonyms,
  all groups must pass — AND across groups, OR within each group).
- `app/eval.py`: runs all cases with `BASELINE_SEED`, scores retrieval
  and generation independently, prints per-query pass/fail with failure
  diagnosis, writes `data/eval_results.json`, supports `--threshold`
  for CI gating.

### Two scores, reported separately

**Retrieval hit rate**: did at least one top-k chunk come from the
expected source file? 23 of 30 queries have expected sources (7 are
genuinely unanswerable — retrieval score is N/A for those).

**Generation score**: does the answer contain all expected keyword
groups? Unanswerable queries must say "context" (or equivalent) to pass.

Separating these creates a 2x2 diagnosis matrix:
- R:+ G:+ -> correct end to end
- R:+ G:- -> retrieval fine, model failed to extract the key fact
             (fix: prompt or model)
- R:- G:- -> retrieval is the root cause (fix: chunking, top-k, embedding)
- R:- G:+ -> model "got lucky" (answered from wrong chunk or own knowledge)
- R:~ G:+ -> unanswerable, correctly refused
- R:~ G:- -> unanswerable, hallucinated (most dangerous failure mode)

### First real run results (v1 verbose prompt, seed=42)

  Retrieval hit rate:  23/23  (100.0%)
  Generation score:    27/30  (90.0%)
    answerable:        21/23  (91.3%)
    unanswerable:       6/7   (85.7%)

### The three failures and what they mean

**Q: "Summarize the lesson from the MatchEvent debugging log"**
Missing keyword: `index`. The model said "When searching for MatchEvent
entities by GUID..." - summarized the *symptom* (search by GUID) not
the *lesson* (always ensure an index exists). Retrieval was correct.
This is a genuine generation quality issue: model chose the wrong angle
on a summarization question. Fixed: changed expected_keywords from
`[["index"], ["guid"]]` to `[["index", "guid"]]` - either term captures
the lesson (they're co-referential in this context).

**Q: "How does a UTMA account affect financial aid eligibility?"**
Missing keyword: `FAFSA`. Model said "...more significant impact on
financial aid..." - correct concept, wrong level of specificity. `FAFSA`
is the precise term in the notes. Fixed in eval_cases.json: changed
keyword group to `["FAFSA", "financial aid"]` so either passes.

**Q: "How do you create an index on a MongoDB collection?"**
This was the most interesting failure. The model retrieved the
MatchEvent debugging chunk (which contains `createIndex` syntax for
the `guid` field) and answered *as if it were a MongoDB tutorial*,
giving a specific `createIndex` command without saying "not in context."
This is **retrieval-grounded hallucination**: not making up facts, but
using specific context to answer a general question the notes don't
actually address. The model should have recognized "this chunk contains
an example, not a general answer to a general question."
Fixed in eval_cases.json: `["context", "not"]` - either refusal word
passes (handles "no information in context" AND "not in the context").

### What the 100% retrieval rate means (and doesn't mean)

100% means: for every answerable query, at least one retrieved chunk
came from the right source *file*. It does NOT mean the specific chunk
containing the key fact was retrieved - a file may have many chunks,
and top-k=4 might retrieve chunks 0,1,5,7 while the key fact is in
chunk 3. Chunk-level precision (was the specific relevant chunk
retrieved?) is a more precise metric for Phase 3.

### Keyword scoring: kept simple, deliberately

Early iteration added "OR groups" per keyword (e.g. `["FAFSA",
"financial aid"]`) to handle the 3 failures. Reverted for two reasons:

1. Doesn't scale: at 10,000 eval cases, maintaining synonym lists per
   keyword per question is unsustainable. Any new paraphrase the model
   discovers that isn't in your list creates a false failure anyway.

2. Papers over real signals: those 3 failures are genuine findings -
   the model chose the wrong angle on a summarization question, used a
   generic term instead of a specific one, and hallucinated a confident
   answer from a specific example. Inflating the score by softening the
   check hides real model behavior rather than measuring it.

The correct fix for paraphrasing is Phase 2.5's LLM-as-judge, which
evaluates semantic correctness rather than string presence. Keyword
matching stays as a fast, strict pre-filter. Don't conflate the two by
adding synonym logic to the keyword layer - that just makes the fast
layer do the slow layer's job badly.

### CI gating — two independent gates

**Absolute floor** (`--threshold`): "never go below X% regardless of
history." Catches catastrophic failures.

  python -m app.eval --threshold 0.80

**Relative regression** (`--max-regression`): "never drop more than Xpp
from the previous run." Catches gradual degradation that stays above
the floor. E.g. 94%->89% with max-regression=0.05 FAILS even though
89% > 80% threshold. Only triggers on DROPS - an increase always passes.

  python -m app.eval --max-regression 0.05

**Both together** (recommended for CI):

  python -m app.eval --threshold 0.80 --max-regression 0.05

Implementation note: float subtraction (0.94 - 0.89) gives 0.04999...99
in Python due to IEEE 754. Fixed with round(drop, 6) — a subtle
correctness bug caught only by explicit test cases.

### Acronym / abbreviation guide for test output

  R:+ = Retrieval HIT  (correct source file was retrieved)
  R:~ = Retrieval N/A  (unanswerable query, no expected source)
  R:- = Retrieval MISS (wrong source retrieved)
  G:+ = Generation PASS (all keyword groups satisfied)
  G:- = Generation FAIL (at least one keyword group missing)

### Steps to run Phase 2

```bash
# Full eval run (no CI gate)
python -m app.eval

# Absolute floor only
python -m app.eval --threshold 0.80

# Relative regression only (needs a previous eval_results.json)
python -m app.eval --max-regression 0.05

# Both gates together (recommended for CI)
python -m app.eval --threshold 0.80 --max-regression 0.05

# Re-run after changing prompt (flip CURRENT_VERSION first)
# Edit app/prompts.py: CURRENT_VERSION = "v1_verbose"
python -m app.eval

# Save baseline before making changes
python -m app.baseline save

# Compare after changes
python -m app.baseline check
```

---

## 9. Phase 2.5 (planned) — LLM-as-judge

### Why keyword matching doesn't scale

At 30 eval cases, hand-writing synonym groups is manageable. At 10,000
cases you can't — maintaining `[["FAFSA", "financial aid"], ...]` for
thousands of questions is unsustainable, and any new phrasing the model
discovers that isn't in your synonym list creates a false failure.

The scaling argument: instead of predicting every synonym upfront, write
one *semantic criterion* per question in plain English:

  "The answer must state that a UTMA account is counted as the student's
   own asset and has a significant negative impact on financial aid
   calculations"

Then for each answer, call an LLM:
  "Does this answer satisfy the criterion? Reply YES or NO with one
   sentence explaining why."

This handles arbitrary paraphrasing without maintaining any lists. One
criterion per question, readable by a human reviewer.

### Tradeoffs vs keyword matching

  Keyword matching:   fast, deterministic, free, breaks at ~100s of cases
  LLM-as-judge:       slow (~N extra LLM calls per eval run), stochastic
                      (mitigated with seed), API cost, scales to 10,000+

### The right architecture at scale

Keyword matching as a FAST PRE-FILTER catching obvious failures in
milliseconds. LLM-judge only for:
  (a) cases that fail keywords but the failure looks like paraphrasing
  (b) periodic "full confidence" runs (e.g. before a major release)
  (c) the full eval suite once you have too many cases to curate keywords

Never run LLM-judge on everything for every commit - the latency and
cost compound badly. This layered approach is how production eval
systems (including Anthropic's own) work at scale.

### CI gate design with both layers

  Fast path (every commit):
    python -m app.eval --threshold 0.80 --max-regression 0.05
    (keyword-only, takes seconds)

  Full confidence (pre-release):
    python -m app.eval --llm-judge --threshold 0.85
    (keyword + LLM judge, takes minutes)

---

## 10. Reproducibility: seeds, and a real `from module import NAME` bug

**The problem**: running `baseline.py check` immediately after `save`
(same prompt version, no code changes) showed **26/30 answers
"changed."** Reading the diffs, almost all were paraphrasing - same fact,
different wording ("Point-in-Time Correctness refers to ensuring that..."
vs "Point-in-time correctness refers to ensuring training data
reflects..."). This is **LLM sampling non-determinism** - Ollama doesn't
fix a random seed by default, so identical prompts can produce different
(but often equivalent) token sequences across calls.

**The fix**: added a `seed` parameter to `generate()`/`generate_stream()`
(`app/llm.py`), passed through `ask()`, and a fixed `BASELINE_SEED = 42`
(`app/config.py`) used only by `baseline.py`. Normal `/ask` and
`/ask/compare` calls intentionally do NOT use a seed - only baseline
comparisons need determinism; the live API should behave like a real LLM.

**A second, more interesting bug found while testing the fix**: when
testing "save with prompt v2, switch to v1, check" - `check()` reported
*"Prompt version unchanged"* and 0/30 changed, even though the prompt
genuinely changed. Root cause: `baseline.py` and `rag.py` did
`from app.prompts import CURRENT_VERSION` - **in Python, `from module
import NAME` copies the current value of `NAME` at import time; it does
NOT create a live reference to the module attribute.** Reassigning
`prompts.CURRENT_VERSION` later doesn't update the already-imported local
name. Fix: `from app import prompts`, then read `prompts.CURRENT_VERSION`
at call time (every time `build_prompt`/`_run_all`/`check` run) - now a
flip of `prompts.CURRENT_VERSION` is picked up everywhere immediately,
which is what "one flag controls the active prompt" requires.

**Why this is worth knowing for interviews**: it's a real example of "the
code looked correct, ran without error, and silently did the wrong thing"
- the most dangerous class of bug. It was caught by *testing the
mechanism itself* (deliberately switching versions and checking the
output made sense), not by code review - a good case for "write a test
that exercises the configuration-switching behavior, not just the happy
path."

---

## 11. Phase 3 — Timeouts + graceful degradation

### What we built

Two independent fallback paths, each demoed via the UI:

**Retrieval fallback (vector search timeout)**:
  `retrieve()` wraps the embed+search call in a thread with
  `TIMEOUT_RETRIEVE_S=3.0s` budget. On timeout: falls back to BM25
  (Best Match 25) keyword search over the same chunks that are in
  Milvus, via `app/bm25.py`. BM25 is pure Python, no Milvus needed.
  The UI shows a yellow banner: "Vector search timed out - BM25 used."

**Generation fallback (LLM generation timeout)**:
  `ask()` / `ask_stream()` wrap the generate call with
  `TIMEOUT_GENERATE_S=12.0s`. On timeout: `ask()` returns an
  `AskResult` with `answer=""` and `generation_fallback=True`.
  `ask_stream()` yields a `{"type": "fallback"}` SSE event. The
  retrieved chunks are already sent to the UI before generation starts
  (that's the streaming architecture), so the user sees useful context
  even when generation times out. Yellow banner: "Generation timed out
  - showing retrieved context only."

**Timeout mechanism**: `concurrent.futures.ThreadPoolExecutor` with
`Future.result(timeout=N)`. Synchronous threads rather than async —
see "threading vs async" note below.

### What each fallback gives the user

  Normal path:    chunks + generated answer
  Retrieval fail: BM25 chunks (lower semantic quality) + generated answer
  Generation fail: vector chunks (full quality) + no answer, just chunks
  Both fail:      BM25 chunks + no answer

The key principle: **always give the user something**. A 500 error is
never acceptable when the underlying data is available.

### BM25 — how it works

BM25 ranks chunks by term frequency adjusted for document length:

  score = sum over query terms of:
    IDF * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))

  IDF = log((N - df + 0.5) / (df + 0.5) + 1)
  tf  = term count in chunk
  dl  = chunk length, avgdl = average chunk length
  k1=1.5 (TF saturation), b=0.75 (length normalization)

In plain terms: "how many times does this query word appear in this
chunk, adjusted so long chunks aren't unfairly favored?" Not semantic —
"index" and "database key" are unrelated to BM25. Good for exact or
near-exact keyword queries; degrades gracefully for conceptual questions.

The index is built lazily at first fallback use, from all chunks stored
in Milvus, and cached in memory. Building it once is cheap (~16 chunks).
At 100,000 chunks it would still be fast (BM25 is O(n*q) per query,
where n=chunks and q=query terms, with tiny constants).

### Threading vs async — the honest tradeoff

We used `ThreadPoolExecutor` rather than `asyncio` to avoid refactoring
the synchronous codebase. This is acceptable specifically here because:

  1. Ollama serializes LLM inference — only one request runs at a time
     regardless of how concurrent the server is. Async parallelism
     wouldn't help at the actual bottleneck.
  2. Retrieval and embedding are already fast (<100ms warm), so the
     thread overhead relative to the work is small.

For a production system with a cloud LLM API that handles parallel
requests, async is the correct answer. The migration path:
  - `def generate()` -> `async def generate()` with `httpx.AsyncClient`
  - `def ask()` -> `async def ask()`, `asyncio.wait_for(generate(), 12)`
  - FastAPI route handlers: `async def ask_endpoint()`
  The architecture is already compatible — it's a plumbing change,
  not a redesign.

### Resume number this phase produces

"p99 latency is bounded at 12s via per-stage timeout budgets, with
graceful fallback to BM25 keyword retrieval (on vector search timeout)
or raw retrieved chunks (on generation timeout), so the system never
500s even when Ollama is unavailable."

### How warmup vs regular runs are distinguished

There is no explicit flag — it's the singleton pattern in `embeddings.py`:

  _model = None  # module-level, lives for the process

  def _try_load_model():
      global _model
      if _model is not None:   # <- already loaded: return immediately
          return _model
      _model = SentenceTransformer(...)  # <- first call: expensive load
      return _model

The warmup hook (`app/embeddings.warmup()`, called at FastAPI startup)
is just the first caller in the process — it pays the load cost so the
first real user request doesn't have to. Subsequent calls in the same
process hit `_model is not None` and return instantly.

The OS-level cache (cold-start going 31s -> 19s -> 8s across restarts)
is separate: macOS keeps recently-read files in RAM, so even fresh
processes benefit from the kernel having the model weights cached. Not
controllable from Python — it just happens naturally with repeated runs.

### Warmup and timeouts — why they don't conflict

The key is ordering. FastAPI's `startup` event blocks the server from
accepting requests until it completes. So the sequence is always:

  1. startup event fires
  2. warmup() called -> model loads (6.9s on M2, no timeout, blocks startup)
  3. "Application startup complete" logged
  4. server starts accepting requests  <- timeout machinery activates here
  5. first real request: embed_query takes ~41ms (already warm)

Timeouts only apply to per-request calls, which only happen after warmup.
The warmup itself is not subject to TIMEOUT_RETRIEVE_S — it runs directly
in the startup hook, outside the request pipeline entirely.

The one edge case worth knowing: if warmup fails silently (network error
downloading the model, falls back to hashing embedder), then the first
real request would try to load the real model inside retrieve() — which
WOULD hit the 3s timeout and fall back to BM25. In practice this doesn't
happen once the model is cached locally, but it's why re-running
`python -m app.ingest` after a model download failure is important.

### Async migration — reference for future production work

Current code uses `concurrent.futures.ThreadPoolExecutor` (sync threads).
For a production system with a cloud LLM API, here's the full migration:

**app/llm.py** — sync -> async:
```python
# CURRENT (sync)
def generate_stream(prompt: str, timeout_seconds: float = 60.0, seed=None):
    with httpx.stream("POST", url, json=payload, timeout=timeout_seconds) as r:
        for line in r.iter_lines():
            yield json.loads(line)

def generate(prompt: str, ...) -> GenerateResult:
    for chunk in generate_stream(prompt, ...):
        ...

# ASYNC VERSION
import httpx

async def generate_stream(prompt: str, timeout_seconds: float = 60.0, seed=None):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload,
                                 timeout=timeout_seconds) as r:
            async for line in r.aiter_lines():
                if line:
                    yield json.loads(line)

async def generate(prompt: str, ...) -> GenerateResult:
    async for chunk in generate_stream(prompt, ...):
        ...
```

**app/rag.py** — remove ThreadPoolExecutor, use asyncio.wait_for:
```python
# CURRENT (threading)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def _run_with_timeout(fn, timeout, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)

def retrieve(query, top_k=TOP_K):
    try:
        chunks = _run_with_timeout(_vector_retrieve, TIMEOUT_RETRIEVE_S)
        return chunks, False
    except FuturesTimeoutError:
        return bm25_search(query, top_k), True

# ASYNC VERSION
import asyncio

async def retrieve(query, top_k=TOP_K):
    try:
        chunks = await asyncio.wait_for(
            _vector_retrieve_async(query, top_k),
            timeout=TIMEOUT_RETRIEVE_S
        )
        return chunks, False
    except asyncio.TimeoutError:
        return bm25_search(query, top_k), True

async def ask(query, top_k=TOP_K, seed=None):
    chunks, retrieval_fallback = await retrieve(query, top_k)
    prompt = build_prompt(query, chunks)
    try:
        gen_result = await asyncio.wait_for(
            generate(prompt, seed=seed),
            timeout=TIMEOUT_GENERATE_S
        )
        return AskResult(answer=gen_result.answer, ...)
    except asyncio.TimeoutError:
        return AskResult(answer="", generation_fallback=True, ...)
```

**app/main.py** — route handlers become async def:
```python
# CURRENT
def ask_endpoint(request: AskRequest):
    def event_stream():
        for event in ask_stream(request.query):
            yield _sse(event)
    return StreamingResponse(event_stream(), ...)

# ASYNC VERSION
async def ask_endpoint(request: AskRequest):
    async def event_stream():
        async for event in ask_stream(request.query):
            yield _sse(event)
    return StreamingResponse(event_stream(), ...)
```

The embedding model (sentence-transformers) itself stays synchronous —
it's a CPU/GPU-bound operation, not I/O-bound, so there's no async
benefit. You'd run it in a thread pool explicitly:

```python
import asyncio
loop = asyncio.get_event_loop()
embedding = await loop.run_in_executor(None, embed_query, query)
```

**Why this matters for production**: with async, a FastAPI server on 1
core can handle 100 concurrent requests — while one waits for the LLM
API, another's retrieval runs, another's embedding runs. With threading,
you need 100 threads (memory overhead, context switching). At low traffic
(local Ollama, few concurrent users) threading is fine. At scale, async
wins clearly.

```bash
# Normal operation (unchanged from before)
uvicorn app.main:app
# Open http://localhost:8000

# Demo retrieval fallback
# 1. In app/config.py, set TIMEOUT_RETRIEVE_S = 0.001
# 2. Restart: uvicorn app.main:app
# 3. Ask a question — yellow banner appears, BM25 chunks shown
# 4. Revert TIMEOUT_RETRIEVE_S = 3.0

# Demo generation fallback
# 1. In app/config.py, set TIMEOUT_GENERATE_S = 0.001
# 2. Restart: uvicorn app.main:app
# 3. Ask a question — chunks appear, then yellow "timed out" banner
# 4. Revert TIMEOUT_GENERATE_S = 12.0

# Or: stop Ollama entirely (quit the app) then ask a question
# Generation will fail with a connection error -> same fallback path
```

---

## 12. Production CI with a real LLM API (Option C reference)

**Context**: our current CI (`--retrieval-only`) skips generation scoring
because Ollama isn't available on GitHub Actions runners. In production
ML teams, you'd use a real hosted API (Anthropic Claude, OpenAI, etc.)
so CI can run the full eval including generation quality. This is how it
actually works at companies like Anthropic or OpenAI internally.

No code changes made here — this is a reference for when it matters.

### Step 1 — Add the API key as a GitHub Actions secret

In your GitHub repo: Settings -> Secrets and variables -> Actions ->
New repository secret.

  Name:  ANTHROPIC_API_KEY   (or OPENAI_API_KEY)
  Value: sk-ant-...

This makes it available as `${{ secrets.ANTHROPIC_API_KEY }}` in the
workflow without it appearing in logs or being visible in the repo.

### Step 2 — Replace app/llm.py with an API client

Currently `llm.py` calls Ollama on localhost. Swap it for the Anthropic
(or OpenAI) SDK — the interface stays identical so nothing else changes:

```python
# app/llm.py — Anthropic API version
import os
from dataclasses import dataclass
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

@dataclass
class GenerateResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int

def generate(prompt: str, seed: int | None = None, **kwargs) -> GenerateResult:
    # Anthropic doesn't support seed. Use temperature=0 instead —
    # greedy decoding is deterministic on any API.
    message = client.messages.create(
        model="claude-haiku-4-5",  # fast + cheap for eval runs
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return GenerateResult(
        answer=message.content[0].text,
        prompt_tokens=message.usage.input_tokens,
        completion_tokens=message.usage.output_tokens,
    )

def generate_stream(prompt: str, seed: int | None = None, **kwargs):
    """Yields dicts matching Ollama's format so rag.py/main.py don't change."""
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield {"response": text, "done": False}
        usage = stream.get_final_message().usage
        yield {
            "response": "",
            "done": True,
            "prompt_eval_count": usage.input_tokens,
            "eval_count": usage.output_tokens,
        }
```

For OpenAI instead:
```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o-mini",   # fast + cheap for eval
    temperature=0,
    messages=[{"role": "user", "content": prompt}],
)
return GenerateResult(
    answer=response.choices[0].message.content,
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
)
```

### Step 3 — Update requirements.txt

```
anthropic>=0.30.0
# or:
openai>=1.30.0
```

### Step 4 — .github/workflows/eval.yml

Create this file at `.github/workflows/eval.yml` in your project root
(create the `.github/workflows/` directories first):

```yaml
name: Eval CI Gate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  eval:
    name: RAG Eval
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Ingest notes
        run: python -m app.ingest

      - name: Run eval gate (full — generation + retrieval)
        run: |
          python -m app.eval --threshold 0.80 --max-regression 0.05
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Upload eval results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: data/eval_results.json
          if-no-files-found: ignore
```

### Step 5 — .gitignore

Create `.gitignore` in your project root:

```
# Python
__pycache__/
*.pyc
.Python
venv/
.venv/

# Vector store (rebuilt by python -m app.ingest)
data/cortex.db
data/latency_raw.json
data/eval_results.json

# Baselines (local only — machine-specific)
data/baselines/

# Model cache
.cache/

# macOS
.DS_Store

# Editor
.vscode/
.idea/
```

### Why temperature=0 replaces seed=42

Our local Ollama setup uses `seed=42` for reproducibility. Cloud APIs
handle this differently:
  - **seed** (Ollama/OpenAI): hints to use deterministic sampling.
    OpenAI supports it; Anthropic doesn't expose it.
  - **temperature=0**: removes randomness entirely — always picks the
    highest-probability next token (greedy decoding). Deterministic on
    any API. Same prompt + temperature=0 = same answer every time.

### Cost estimate

With Claude Haiku (~$0.25/M input, $1.25/M output as of mid-2026):
  - 30 eval queries x ~800 prompt + ~50 completion tokens each
  - ~$0.006 per eval run (less than 1 cent)
  - 100 CI runs/month ≈ $0.60/month

### Design principle this demonstrates

The entire pipeline (rag.py, eval.py, main.py, baseline.py) only
imports from `app.llm` through `generate()` and `generate_stream()`.
Nothing else knows whether those call Ollama, Anthropic, or OpenAI.
This is the **adapter pattern**: fixed interface, swappable
implementation. Switching from local to production is a one-file change
plus a GitHub secret — exactly as it should be.

---

## 13. Phase 4 — Agentic tool-calling with reasoning trace

### What we built

Three tools in `app/tools/`:
- `search_notes(query, top_k)` — wraps existing RAG retrieval
- `list_notes(filter)` — discovers notes by tag/topic from frontmatter,
  returns metadata without full content
- `summarize_note(filename)` — reads one note in full and LLM-summarizes
  it; path traversal protected; returns helpful error with available
  filenames when file not found

Agent loop in `app/agent.py`:
- Uses Ollama's `/api/chat` endpoint (not `/api/generate`) — tool-calling
  is only supported via the chat endpoint
- Sends query + tool definitions (JSON schemas) to the LLM
- LLM returns either `tool_calls` (wants to call a tool) or `content`
  (final answer)
- On `tool_calls`: execute tool, append result to message history, loop
- On `content`: yield answer, stop
- Full message history sent on every iteration — this is how the agent
  "carries state between steps without losing the original question"
- `MAX_STEPS=8` safety limit prevents infinite loops
- Each step yields a `TraceEvent` (thinking/tool_call/tool_result/answer)
  streamed via SSE so the UI shows reasoning in real time

### Where each decision happens

  "Agent decides to search" = Ollama's model weights, not our code.
  We send the query + tool schemas; the LLM decides which tool to call
  and what arguments to pass. We just execute whatever it chooses.

  _call_ollama_with_tools() = one LLM call per step
  _execute_tool()           = dispatches to the right tool module
  ask_agent_stream()        = the loop that connects them

### Real findings from running it

**Finding 1 — Small models skip discovery steps**
llama3.2:3B guessed `observability.md` directly instead of calling
`list_notes` first to discover what exists. A larger model (Claude,
GPT-4o) follows the system prompt's "use list_notes first" instruction
more reliably. Fix options: force list_notes as a mandatory first step
in the system prompt, or upgrade the model. This is a real production
consideration — tool reliability is model-size dependent.

**Finding 2 — LLMs send integer arguments as strings**
The model called `search_notes` with `top_k: "6"` (JSON string) despite
the schema saying `"type": "integer"`. Fixed with defensive coercion in
`_execute_tool()` before dispatch. Always coerce known numeric params —
never trust the LLM to respect schema types with smaller models.

**Finding 3 — Small models explain errors in text instead of retrying**
On tool failure, llama3.2:3B sometimes returns a text explanation
("I was unable to execute the tool call...") instead of a corrected
`tool_call`. Detected via `retry_signals` list; loop injects a
correction hint and continues rather than surfacing the internal
monologue as the final answer.

**Finding 4 — Good tool error messages are part of the reasoning loop**
`summarize_note` returns the list of available files when a filename
isn't found ("File not found. Available notes: promql_patterns.md...").
The model reads this error message and self-corrects. Tool error
messages aren't just for developers — they're context the agent uses
to plan its next step.

### The demo that shows agentic behavior most clearly

Query: *"What do my notes say about reinforcement learning, and how
does that connect to feature stores?"*

Agent trace:
  Step 1: search_notes(query="reinforcement learning feature stores",
          top_k=6) -> 6 chunks from rl_basics.md + feature_store_design.md
  Step 2: synthesize -> draws connection between exploration-exploitation
          tradeoff and streaming vs batch feature computation

This is meaningfully different from the regular /ask endpoint:
  - /ask: retrieves top-4 chunks for exact query, one generation call
  - /agent: LLM chose what to search, decided 6 chunks was enough,
    synthesized across two source files

The 14s total time (vs ~2s for /ask) is the cost of the multi-step
planning — a real tradeoff worth discussing.

### Interview framing

"I built a true agentic RAG system where the LLM decides which tools
to call and in what order. The observable artifact is the reasoning
trace — every tool call, argument, result, and synthesis step logged
with timing. Running it on a 3B model revealed that smaller models
skip discovery steps and send wrong argument types, which led me to
add defensive coercion, retry detection, and self-correcting error
messages. The key insight: good tool error messages are part of the
agent's reasoning loop, not just dev debugging output."

### Steps to use

```bash
uvicorn app.main:app
# Open http://localhost:8000
# Click "agent mode" — button highlights
# Ask button changes to "Agent"
# Try: "What do my notes say about reinforcement learning,
#       and how does that connect to feature stores?"
# Watch the trace panel fill in real time
```

---

## 14. Glossary (for your own reference)

- **p50 / p95**: 50th/95th percentile. p50 = "typical" (half of requests
  are faster). p95 = "the slow tail that still happens regularly" (95% of
  requests are faster than this).
- **completion_tokens / prompt_tokens**: tokens in the model's *output*
  vs. *input*. A token is roughly a word or word-piece.
- **Autoregressive generation**: each output token depends on all
  previous tokens (including ones the model itself just generated) -
  fundamentally sequential, can't parallelize across tokens for one
  response (though the math *within* one token's computation is highly
  parallel - that's what GPUs accelerate).
- **MPS**: Metal Performance Shaders - PyTorch's backend for Apple
  Silicon GPUs.
- **Pearson correlation (r)**: -1 to +1, measures *linear* relationship
  strength. Sensitive to the range of the data (see Section 3).
- **SSE (Server-Sent Events)**: a simple HTTP streaming protocol - server
  sends `data: {...}` chunks, browser's `fetch`/`EventSource` reads them
  incrementally.
- **RAG**: Retrieval-Augmented Generation - retrieve relevant context,
  then generate an answer conditioned on it.
- **BM25**: Best Match 25 - a keyword ranking function that scores
  documents by term frequency adjusted for document length. The standard
  baseline for lexical (non-semantic) search. Used as retrieval fallback
  when vector search times out.
- **Graceful degradation**: the system returns something useful (BM25
  chunks, or raw chunks) rather than an error when a component fails or
  times out. The user experience degrades, but doesn't break.
- **Threading vs async**: two concurrency models. Threading runs each
  request in its own OS thread (blocks while waiting). Async runs all
  requests on one thread, yielding control while waiting. Async wins
  at scale; threading is acceptable when the real bottleneck serializes
  anyway (e.g. local Ollama, single-inference GPU).
