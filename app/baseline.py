"""Baseline capture and comparison.

Run with:
  python -m app.baseline save   # capture current behavior as a baseline
  python -m app.baseline check  # compare current behavior against the
                                 # most recent saved baseline

This is intentionally simple - one JSON file per saved baseline under
data/baselines/, each containing (query, prompt_version, answer,
prompt_tokens, completion_tokens, generate_ms) for the same QUERIES list
used by latency_report.py.

Phase 2's eval harness will build on this: instead of just *displaying*
diffs, it'll score answers against expected content and gate CI on the
score. This script is the "can I even compare two runs" foundation that
gating needs - get the comparison mechanics right first, scoring next.

What this catches: "I changed the prompt / model / chunk size - did
answers change, and by how much, compared to last time?" It does NOT
(yet) judge whether either answer is *correct* - that's Phase 2.
"""

import json
import sys
import time
from pathlib import Path

from app import prompts
from app.config import BASELINE_SEED
from app.embeddings import warmup
from app.latency_report import QUERIES
from app.rag import ask
from app.timing import new_request_id

BASELINE_DIR = Path("data/baselines")
LATEST_LINK = BASELINE_DIR / "latest.json"


def _run_all() -> list[dict]:
    warmup()
    current_version = prompts.CURRENT_VERSION
    results = []
    for query in QUERIES:
        new_request_id()
        result = ask(query, seed=BASELINE_SEED)
        results.append(
            {
                "query": query,
                "prompt_version": current_version,
                "answer": result.answer,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        )
    return results


def save() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    results = _run_all()

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = BASELINE_DIR / f"baseline_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "prompt_version": prompts.CURRENT_VERSION,
        "results": results,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Update "latest" pointer (write the same content under a fixed name,
    # so `check` doesn't need to find the newest timestamp).
    with LATEST_LINK.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved baseline: {out_path}")
    print(f"  {len(results)} queries, prompt_version={payload['prompt_version']}")


def check() -> None:
    if not LATEST_LINK.exists():
        print("No baseline found. Run `python -m app.baseline save` first.")
        sys.exit(1)

    with LATEST_LINK.open() as f:
        baseline = json.load(f)

    baseline_by_query = {r["query"]: r for r in baseline["results"]}

    print(f"Comparing against baseline from {baseline['timestamp']} "
          f"(prompt_version={baseline['prompt_version']})")
    print(f"Using seed={BASELINE_SEED} for both runs - with the SAME prompt "
          f"version, identical answers are expected.\n")

    current_version = prompts.CURRENT_VERSION
    version_changed = current_version != baseline["prompt_version"]

    if version_changed:
        print(f"*** PROMPT VERSION CHANGED: {baseline['prompt_version']} -> "
              f"{current_version} ***")
        print("Differences below reflect this prompt change (expected, "
              "given seed is fixed).\n")
    else:
        print(f"Prompt version unchanged ({current_version}). With seed="
              f"{BASELINE_SEED} fixed, any differences below indicate "
              f"something else changed (model, chunking, retrieved "
              f"content, Ollama version, etc.) - investigate.\n")

    current = _run_all()

    print(f"{'':3} {'tokens (base->now)':<20} {'changed?':<10} query")
    print("-" * 90)

    n_changed = 0
    n_missing = 0

    for row in current:
        query = row["query"]
        base_row = baseline_by_query.get(query)

        if base_row is None:
            print(f"NEW  {'':<20} {'(no baseline)':<10} {query[:55]}")
            n_missing += 1
            continue

        base_tok = base_row["completion_tokens"]
        now_tok = row["completion_tokens"]
        changed = base_row["answer"].strip() != row["answer"].strip()
        n_changed += int(changed)

        marker = "CHANGED" if changed else "same"
        print(f"     {base_tok:>3} -> {now_tok:<14} {marker:<10} {query[:55]}")

        if changed:
            print(f"       before: {base_row['answer'][:90]!r}")
            print(f"       now:    {row['answer'][:90]!r}")

    print("-" * 90)
    print(f"{n_changed}/{len(current)} answers changed (text differs from baseline)")
    if n_missing:
        print(f"{n_missing} queries are new (not in baseline)")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("save", "check"):
        print("Usage: python -m app.baseline [save|check]")
        sys.exit(1)

    if sys.argv[1] == "save":
        save()
    else:
        check()


if __name__ == "__main__":
    main()
