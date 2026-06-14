"""Phase 2.5: LLM-as-judge eval harness.

Run with:
  python -m app.eval                              # keyword scoring only
  python -m app.eval --llm-judge                  # keyword + LLM judge on failures
  python -m app.eval --threshold 0.80             # absolute CI gate
  python -m app.eval --max-regression 0.05        # relative regression gate
  python -m app.eval --llm-judge --threshold 0.80 --max-regression 0.05

Scoring layers:

  LAYER 1 - KEYWORD SCORING (always runs, fast, deterministic)
    Does the answer contain all expected_keywords (case-insensitive)?
    Strict: a miss is a real signal. Handles obvious pass/fail cases.

  LAYER 2 - LLM JUDGE (runs only on keyword failures, --llm-judge flag)
    For each keyword failure, asks Ollama: "Does this answer correctly
    convey the expected criterion? Reply YES or NO then one sentence."
    Handles legitimate paraphrasing that keyword matching can't.
    Only called on failing cases - keeps eval fast (3 calls not 30
    in the current baseline).

  RETRIEVAL HIT RATE (always scored, independent of generation)
    Did at least one top-k chunk come from expected_sources?

Failure diagnosis matrix (same as Phase 2):
  R:+ G:+              -> correct end to end
  R:+ G:- kw           -> keyword miss only (may pass LLM judge = paraphrase)
  R:+ G:- kw+llm       -> both scorers agree = genuine generation failure
  R:- G:-              -> retrieval is root cause
  R:~ G:+              -> unanswerable, correctly refused
  R:~ G:- kw           -> unanswerable, hallucinated (most dangerous)

All runs use BASELINE_SEED for reproducibility.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import BASELINE_SEED
from app.embeddings import warmup
from app.llm import generate
from app.rag import ask
from app.timing import new_request_id

EVAL_CASES_PATH = Path("data/eval_cases.json")
EVAL_RESULTS_PATH = Path("data/eval_results.json")

# Prompt for the LLM judge. Deliberately simple and structured so the
# model's YES/NO is easy to parse. Seed is set on the generate() call
# for reproducibility.
JUDGE_PROMPT = """You are an evaluation assistant. Determine whether a given answer correctly conveys an expected criterion.

Criterion: {criterion}

Answer to evaluate: {answer}

Does the answer correctly convey the criterion? Reply with exactly YES or NO on the first line, then one sentence explaining why."""


@dataclass
class CaseResult:
    query: str
    unanswerable: bool

    # Retrieval
    retrieved_sources: list[str]
    expected_sources: list[str]
    retrieval_hit: bool | None

    # Generation - keyword layer
    answer: str
    expected_keywords: list[str]
    missing_keywords: list[str]
    generation_pass: bool          # keyword score

    # Generation - LLM judge layer (None if not run)
    judge_criterion: str | None = None
    judge_verdict: bool | None = None    # True=YES, False=NO, None=not run
    judge_explanation: str | None = None

    # Final generation result (judge verdict when available, else keyword)
    @property
    def final_pass(self) -> bool:
        if self.judge_verdict is not None:
            return self.judge_verdict
        return self.generation_pass

    # Token / timing
    completion_tokens: int = 0
    prompt_tokens: int = 0


def score_retrieval(
    retrieved_sources: list[str], expected_sources: list[str]
) -> tuple[bool | None, list[str]]:
    """Return (hit, retrieved_sources).

    hit is True if any retrieved chunk's source file is in
    expected_sources. None if expected_sources is empty (unanswerable
    queries where we don't care what gets retrieved, only that the model
    declines).
    """
    if not expected_sources:
        return None, retrieved_sources
    hit = any(s in expected_sources for s in retrieved_sources)
    return hit, retrieved_sources


def score_generation(
    answer: str, expected_keywords: list[str]
) -> tuple[bool, list[str]]:
    """Return (pass, missing_keywords).

    Checks each keyword case-insensitively. ALL must be present to pass.

    Intentionally strict - a keyword miss is a real signal, not noise
    to paper over with synonym groups. Phase 2.5's LLM-as-judge will
    handle legitimate paraphrasing cases (e.g. "financial aid" vs
    "FAFSA") by evaluating semantic correctness rather than string
    presence. Keyword matching is the fast pre-filter; LLM-judge is the
    semantic layer. Don't conflate the two by adding synonym logic here.
    """
    answer_lower = answer.lower()
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    return len(missing) == 0, missing


def judge_with_llm(
    query: str,
    answer: str,
    criterion: str,
) -> tuple[bool, str]:
    """Ask Ollama to judge whether the answer correctly conveys the
    criterion. Returns (verdict, explanation).

    Uses BASELINE_SEED for reproducibility - same answer + same criterion
    should always produce the same verdict, so judge results are stable
    across runs just like keyword results.

    The judge prompt asks for "YES or NO on the first line" - we parse
    the first word of the response. If parsing fails (model didn't
    follow format), we default to False (fail-safe: don't silently pass
    things the judge couldn't evaluate).
    """
    prompt = JUDGE_PROMPT.format(criterion=criterion, answer=answer)
    result = generate(prompt, seed=BASELINE_SEED)
    response = result.answer.strip()

    first_line = response.split("\n")[0].strip().upper()
    verdict = first_line.startswith("YES")

    # Extract explanation (everything after first line)
    lines = response.split("\n", 1)
    explanation = lines[1].strip() if len(lines) > 1 else response

    return verdict, explanation


def run_eval(cases: list[dict], use_llm_judge: bool = False) -> list[CaseResult]:
    warmup()
    results = []

    for i, case in enumerate(cases, 1):
        new_request_id()
        result = ask(case["query"], seed=BASELINE_SEED)

        retrieved_sources = list({c.source for c in result.retrieved_chunks})

        retrieval_hit, _ = score_retrieval(
            retrieved_sources, case["expected_sources"]
        )
        generation_pass, missing_keywords = score_generation(
            result.answer, case["expected_keywords"]
        )

        case_result = CaseResult(
            query=case["query"],
            unanswerable=case.get("unanswerable", False),
            retrieved_sources=retrieved_sources,
            expected_sources=case["expected_sources"],
            retrieval_hit=retrieval_hit,
            answer=result.answer,
            expected_keywords=case["expected_keywords"],
            missing_keywords=missing_keywords,
            generation_pass=generation_pass,
            completion_tokens=result.completion_tokens,
            prompt_tokens=result.prompt_tokens,
        )

        # LLM judge: only on keyword failures, only if flag set,
        # only if the case has an eval_criterion defined
        if use_llm_judge and not generation_pass and case.get("eval_criterion"):
            verdict, explanation = judge_with_llm(
                case["query"], result.answer, case["eval_criterion"]
            )
            case_result.judge_criterion = case["eval_criterion"]
            case_result.judge_verdict = verdict
            case_result.judge_explanation = explanation

        results.append(case_result)

        # Status: keyword result + judge result when run
        r_sym = "R:+" if retrieval_hit else ("R:~" if retrieval_hit is None else "R:-")
        g_sym = "G:+" if generation_pass else "G:-"
        j_sym = (" J:+" if case_result.judge_verdict else " J:-") \
                if case_result.judge_verdict is not None else ""

        print(f"[{i:2}/{len(cases)}] {r_sym} {g_sym}{j_sym}"
              f"  tok={result.completion_tokens:>3}  {case['query'][:52]}")

        if not generation_pass:
            print(f"          missing keywords: {missing_keywords}")
            if case_result.judge_verdict is not None:
                verdict_str = "PASS" if case_result.judge_verdict else "FAIL"
                expl = (case_result.judge_explanation or "")[:80]
                print(f"          judge: {verdict_str} — {expl}")
            print(f"          answer: {result.answer[:100]!r}")

    return results


def compute_scores(results: list[CaseResult]) -> dict:
    """Compute aggregate scores, broken out by keyword and judge layers."""
    answerable = [r for r in results if not r.unanswerable]
    unanswerable = [r for r in results if r.unanswerable]
    judged = [r for r in results if r.judge_verdict is not None]

    # Retrieval
    retrieval_scored = [r for r in results if r.retrieval_hit is not None]
    retrieval_hits = sum(1 for r in retrieval_scored if r.retrieval_hit)
    retrieval_rate = retrieval_hits / len(retrieval_scored) if retrieval_scored else None

    # Keyword generation score
    kw_pass = sum(1 for r in results if r.generation_pass)
    kw_rate = kw_pass / len(results) if results else None

    # Judge-adjusted score: use judge verdict where available, else keyword
    final_pass = sum(1 for r in results if r.final_pass)
    final_rate = final_pass / len(results) if results else None

    # Answerable / unanswerable breakdown (keyword)
    ans_pass = sum(1 for r in answerable if r.generation_pass)
    ans_rate = ans_pass / len(answerable) if answerable else None
    unans_pass = sum(1 for r in unanswerable if r.generation_pass)
    unans_rate = unans_pass / len(unanswerable) if unanswerable else None

    return {
        "retrieval_hit_rate": retrieval_rate,
        "retrieval_hits": retrieval_hits,
        "retrieval_scored": len(retrieval_scored),
        # Primary score for CI gates: judge-adjusted when judge ran,
        # else keyword score
        "generation_score": final_rate,
        "generation_pass": final_pass,
        "generation_total": len(results),
        # Raw keyword score always available for comparison
        "keyword_score": kw_rate,
        "keyword_pass": kw_pass,
        # Judge stats
        "judge_run": len(judged),
        "judge_overrides": sum(
            1 for r in judged if r.judge_verdict != r.generation_pass
        ),
        # Breakdown
        "answerable_score": ans_rate,
        "answerable_pass": ans_pass,
        "answerable_total": len(answerable),
        "unanswerable_score": unans_rate,
        "unanswerable_pass": unans_pass,
        "unanswerable_total": len(unanswerable),
    }


def print_report(results: list[CaseResult], scores: dict) -> None:
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)

    rr = scores["retrieval_hit_rate"]
    gs = scores["generation_score"]
    ks = scores["keyword_score"]

    print(
        f"Retrieval hit rate:  "
        f"{scores['retrieval_hits']}/{scores['retrieval_scored']}  "
        f"({rr*100:.1f}%)" if rr is not None else "N/A"
    )

    # Show both keyword and judge-adjusted if judge ran
    if scores["judge_run"] > 0:
        print(
            f"Generation (keyword): "
            f"{scores['keyword_pass']}/{scores['generation_total']}  "
            f"({ks*100:.1f}%)"
        )
        print(
            f"Generation (+ judge): "
            f"{scores['generation_pass']}/{scores['generation_total']}  "
            f"({gs*100:.1f}%)  "
            f"[judge ran on {scores['judge_run']} failures, "
            f"overrode {scores['judge_overrides']}]"
        )
    else:
        print(
            f"Generation score:    "
            f"{scores['generation_pass']}/{scores['generation_total']}  "
            f"({gs*100:.1f}%)"
            if gs is not None else "N/A"
        )
        print(
            f"  answerable:        "
            f"{scores['answerable_pass']}/{scores['answerable_total']}  "
            f"({scores['answerable_score']*100:.1f}%)"
            if scores["answerable_score"] is not None else ""
        )
        print(
            f"  unanswerable:      "
            f"{scores['unanswerable_pass']}/{scores['unanswerable_total']}  "
            f"({scores['unanswerable_score']*100:.1f}%)"
            if scores["unanswerable_score"] is not None else ""
        )

    # Failures - show keyword failures, noting where judge overrode
    kw_failures = [r for r in results if not r.generation_pass]
    final_failures = [r for r in results if not r.final_pass]

    if kw_failures:
        print(f"\nGENERATION FAILURES — keyword ({len(kw_failures)}):")
        print("-" * 70)
        for r in kw_failures:
            retrieval_status = (
                "retrieval OK" if r.retrieval_hit
                else "retrieval also failed" if r.retrieval_hit is False
                else "retrieval N/A (unanswerable)"
            )
            judge_note = ""
            if r.judge_verdict is True:
                judge_note = "  ✓ judge OVERRODE to PASS"
            elif r.judge_verdict is False:
                judge_note = "  ✗ judge CONFIRMED FAIL"
            print(f"  Q: {r.query[:60]}")
            print(f"     missing: {r.missing_keywords}  [{retrieval_status}]{judge_note}")
            if r.judge_explanation:
                print(f"     judge:   {r.judge_explanation[:80]}")
            if not r.final_pass:
                print(f"     answer:  {r.answer[:90]!r}")

    if scores["judge_run"] > 0 and len(final_failures) < len(kw_failures):
        print(f"\nFINAL FAILURES after judge ({len(final_failures)}):")
        print("-" * 70)
        for r in final_failures:
            print(f"  {r.query[:65]}")
            print(f"    missing: {r.missing_keywords}")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cortex eval harness.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Absolute CI gate: exit 1 if EITHER retrieval_hit_rate OR "
            "generation_score falls below this value (0.0-1.0). "
            "Catches catastrophic failures regardless of prior baseline."
        ),
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
        help=(
            "For each keyword failure that has an eval_criterion, call "
            "Ollama a second time to judge semantic correctness. Handles "
            "legitimate paraphrasing that keyword matching can't. Only "
            "runs on failing cases - currently ~3 extra calls. The "
            "judge-adjusted score is used for CI gates when this flag "
            "is set."
        ),
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=None,
        help=(
            "Relative CI gate: exit 1 if EITHER score drops more than "
            "this many percentage points vs the previous eval run saved "
            "in data/eval_results.json. E.g. --max-regression 0.05 "
            "means a drop from 94%% to 89%% fails (5pp drop > 5pp max). "
            "Catches gradual degradation that stays above --threshold."
        ),
    )
    args = parser.parse_args()

    if not EVAL_CASES_PATH.exists():
        print(f"Eval cases not found at {EVAL_CASES_PATH}")
        sys.exit(1)

    with EVAL_CASES_PATH.open() as f:
        cases = json.load(f)

    # Load previous results BEFORE running (so we compare against the
    # last run, not the current one overwriting itself).
    prev_scores = None
    if args.max_regression is not None and EVAL_RESULTS_PATH.exists():
        with EVAL_RESULTS_PATH.open() as f:
            prev = json.load(f)
        prev_scores = prev.get("scores", {})
        print(
            f"Previous run: retrieval={prev_scores.get('retrieval_hit_rate', 'N/A'):.3f}  "
            f"generation={prev_scores.get('generation_score', 'N/A'):.3f}  "
            f"(from {prev.get('timestamp', 'unknown')})\n"
        )

    print(f"Running {len(cases)} eval cases (seed={BASELINE_SEED})...\n")
    print(f"{'':4} {'R:+ hit  R:~ N/A  R:- miss':28} {'G:+ pass  G:- fail':18}")
    print("-" * 70)

    results = run_eval(cases, use_llm_judge=args.llm_judge)
    scores = compute_scores(results)
    print_report(results, scores)

    # Save results for next run's regression comparison
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "seed": BASELINE_SEED,
        "scores": scores,
        "results": [asdict(r) for r in results],
    }
    with EVAL_RESULTS_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {EVAL_RESULTS_PATH}")

    # --- CI gates ---
    gate_failures = []

    # Absolute floor
    if args.threshold is not None:
        rr = scores["retrieval_hit_rate"]
        gs = scores["generation_score"]
        if rr is not None and rr < args.threshold:
            gate_failures.append(
                f"ABSOLUTE: retrieval_hit_rate={rr:.3f} < threshold={args.threshold}"
            )
        if gs is not None and gs < args.threshold:
            gate_failures.append(
                f"ABSOLUTE: generation_score={gs:.3f} < threshold={args.threshold}"
            )

    # Relative regression vs previous run
    if args.max_regression is not None:
        if prev_scores is None:
            print(
                f"\nNo previous eval_results.json found - skipping regression "
                f"check (will compare on next run)."
            )
        else:
            for metric, label in [
                ("retrieval_hit_rate", "retrieval_hit_rate"),
                ("generation_score", "generation_score"),
            ]:
                prev_val = prev_scores.get(metric)
                curr_val = scores.get(metric)
                if prev_val is None or curr_val is None:
                    continue
                drop = round(prev_val - curr_val, 6)
                if drop >= args.max_regression:
                    gate_failures.append(
                        f"REGRESSION: {label} dropped {drop*100:.1f}pp "
                        f"({prev_val:.3f} -> {curr_val:.3f}, "
                        f"max allowed={args.max_regression*100:.1f}pp)"
                    )

    if gate_failures:
        print("\nCI GATE FAILED:")
        for f in gate_failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        gates_desc = []
        if args.threshold is not None:
            gates_desc.append(f"floor={args.threshold}")
        if args.max_regression is not None:
            gates_desc.append(f"max-regression={args.max_regression*100:.1f}pp")
        if gates_desc:
            rr = scores["retrieval_hit_rate"]
            gs = scores["generation_score"]
            print(
                f"\nCI gates passed ({', '.join(gates_desc)}) — "
                f"retrieval={rr:.3f}, generation={gs:.3f}"
            )


if __name__ == "__main__":
    main()
