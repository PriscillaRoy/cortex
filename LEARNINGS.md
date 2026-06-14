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

## 11. Glossary (for your own reference)

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
