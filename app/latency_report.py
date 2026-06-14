"""Run a batch of queries against the RAG pipeline and report p50/p95
latency per stage, plus a check for whether generation latency correlates
with token count.

Run with: python -m app.latency_report

Outputs:
  - printed summary table (p50/p95 per stage)
  - printed per-query detail table (so you can see WHICH queries were
    slow, not just aggregate stats)
  - printed correlation between completion_tokens and generate latency
  - data/latency_raw.json - every individual measurement, for further
    analysis later (Phase 2's eval work can join against this)
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from statistics import correlation, mean

from app.embeddings import get_device
from app.rag import ask
from app.timing import new_request_id


# A mix of queries across all 5 notes, plus a few repeats of similar
# topics (to see if similar questions get similar generate times - a
# hint about whether it's content-driven or just noisy) and one
# deliberately unanswerable question.
QUERIES = [
    # feature_store_design.md
    "What is point-in-time correctness in a feature store?",
    "What's the tradeoff between streaming and batch feature computation?",
    "What is the difference between an offline store and an online store?",
    "What causes training/serving skew?",
    # debug_matchevent_search.md
    "What MongoDB index was missing in the debugging notes?",
    "Summarize the lesson from the MatchEvent debugging log.",
    "Why did the search return empty results instead of an error?",
    "What was the root cause of the MatchEvent search bug?",
    # promql_patterns.md
    "How do you handle the trailing pipe bug in Grafana variable filters?",
    "Why might a Prometheus counter be missing from a dashboard entirely?",
    "What PromQL pattern avoids gaps when summing across series?",
    "What's the difference between a missing metric and a zero-value metric?",
    # rl_basics.md
    "What is the difference between Q-learning and SARSA?",
    "What is Direct Preference Optimization?",
    "What is Constitutional AI?",
    "Why does RLHF use PPO instead of Q-learning?",
    "In a grid world with a cliff, why would Q-learning and SARSA learn different paths?",
    # utma_kiddie_tax.md
    "What's the difference between a 529 plan and a UTMA account?",
    "What is the Kiddie Tax?",
    "How does a UTMA account affect financial aid eligibility?",
    # Cross-cutting / harder
    "Compare how UTMA accounts and feature stores both handle a kind of 'point in time' concept.",
    "What debugging lesson applies to both the MatchEvent bug and the missing Prometheus counter?",
    # Off-topic / unanswerable
    "What's the best pizza topping?",
    "How do you create an index on a MongoDB collection?",
    "What is the capital of France?",
    # Repeats of similar short factual questions
    "What index should be added to persistentEventMatchAttempt?",
    "What does AUTOINDEX mean in Milvus?",
    "What model is used for embeddings in this project?",
    "What is the chunk size used for splitting notes?",
    "What does top_k control in retrieval?",
]


class TimingCollector(logging.Handler):
    """Captures stage_timing log records into a list of dicts."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            return

        if data.get("event") == "stage_timing":
            self.records.append(data)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def main() -> None:
    collector = TimingCollector()
    timing_logger = logging.getLogger("cortex.timing")
    timing_logger.addHandler(collector)
    timing_logger.setLevel(logging.INFO)
    timing_logger.propagate = False

    print(f"Embedding device: {get_device()}")
    print(
        "Note: query #1 includes one-time model-load cost "
        "(observed ~30s on first run with the real model on CPU/MPS).\n"
    )
    print(f"Running {len(QUERIES)} queries...\n")

    per_query: list[dict] = []

    for i, query in enumerate(QUERIES, 1):
        request_id = new_request_id()
        wall_start = time.perf_counter()
        result = ask(query)
        wall_ms = (time.perf_counter() - wall_start) * 1000

        answer_preview = result.answer.strip().replace("\n", " ")[:60]
        print(
            f"[{i:2}/{len(QUERIES)}] {wall_ms:7.1f}ms  "
            f"tok={result.completion_tokens:>4} "
            f"{query[:50]:50} -> {answer_preview}..."
        )

        per_query.append(
            {
                "request_id": request_id,
                "query": query,
                "wall_ms": round(wall_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        )

    by_stage: dict[str, list[float]] = defaultdict(list)
    for rec in collector.records:
        by_stage[rec["stage"]].append(rec["duration_ms"])

    print("\n" + "=" * 70)
    print("SUMMARY: latency per stage")
    print("=" * 70)
    print(f"{'Stage':<20} {'count':>6} {'mean(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10}")
    print("-" * 70)

    stage_order = ["embed_query", "vector_search", "retrieve_total", "generate"]
    for stage in stage_order:
        durations = by_stage.get(stage, [])
        if not durations:
            continue
        print(
            f"{stage:<20} {len(durations):>6} "
            f"{mean(durations):>10.1f} {percentile(durations, 50):>10.1f} "
            f"{percentile(durations, 95):>10.1f}"
        )

    by_request: dict[str, dict] = defaultdict(dict)
    for rec in collector.records:
        if rec["stage"] in ("retrieve_total", "generate"):
            by_request[rec["request_id"]][rec["stage"]] = rec["duration_ms"]

    end_to_end = [
        v["retrieve_total"] + v["generate"]
        for v in by_request.values()
        if "retrieve_total" in v and "generate" in v
    ]

    print("-" * 70)
    print(
        f"{'end_to_end':<20} {len(end_to_end):>6} "
        f"{mean(end_to_end):>10.1f} {percentile(end_to_end, 50):>10.1f} "
        f"{percentile(end_to_end, 95):>10.1f}"
    )

    for row in per_query:
        gen = by_request.get(row["request_id"], {})
        row["generate_ms"] = round(gen.get("generate", 0), 1)
        row["retrieve_ms"] = round(gen.get("retrieve_total", 0), 1)

    print("\n" + "=" * 90)
    print("PER-QUERY DETAIL (sorted slowest generate() first)")
    print("=" * 90)
    print(f"{'generate(ms)':>13} {'tokens':>7} {'retrieve(ms)':>13}  query")
    print("-" * 90)
    for row in sorted(per_query, key=lambda r: -r["generate_ms"]):
        print(
            f"{row['generate_ms']:>13.1f} {row['completion_tokens']:>7} "
            f"{row['retrieve_ms']:>13.1f}  {row['query'][:55]}"
        )

    gen_times = [r["generate_ms"] for r in per_query]
    tokens = [r["completion_tokens"] for r in per_query]

    print("\n" + "=" * 70)
    print("CORRELATION: completion_tokens vs generate latency")
    print("=" * 70)
    if len(set(tokens)) > 1 and len(set(gen_times)) > 1:
        r = correlation(tokens, gen_times)
        print(f"Pearson correlation coefficient: {r:.3f}")
        print(
            "  (1.0 = perfectly correlated, 0 = no relationship, "
            "negative = inversely related)"
        )
        if r > 0.7:
            print("  -> Strong positive correlation: generate time scales with")
            print("     output length. Longer answers take proportionally longer.")
        elif r > 0.3:
            print("  -> Moderate correlation: token count partially explains")
            print("     generate time, but other factors matter too.")
        else:
            print("  -> Weak/no correlation: generate time variance is NOT")
            print("     mainly explained by output token count. Look elsewhere")
            print("     (e.g. prompt length, model warm-up, system load).")
    else:
        print("  Not enough variance to compute correlation.")

    prompt_tokens = [r["prompt_tokens"] for r in per_query]
    print(
        f"\nprompt_tokens range: {min(prompt_tokens)}-{max(prompt_tokens)} "
        f"(mean {mean(prompt_tokens):.0f})"
    )
    print(
        f"completion_tokens range: {min(tokens)}-{max(tokens)} "
        f"(mean {mean(tokens):.0f})"
    )

    out_path = Path("data/latency_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "prompt_version": "v2_concise_2sentence",
                "per_query": per_query,
                "raw_stage_records": collector.records,
            },
            f,
            indent=2,
        )
    print(f"\nRaw data written to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
