"""
detect_regression.py — Phase 6B: Regression Detection
=======================================================
PURPOSE:
  Compare a NEW pipeline run against the SAVED BASELINE.
  If the pass rate drops significantly → flag it as a REGRESSION.

WHY THIS MATTERS (from the speaker's framework):
  "These metrics help detect regressions."

  Real-world scenario:
    Week 1: prompt A → Llama scores 71.9%  (saved as baseline)
    Week 2: you tweak the prompt in collector.py
    Week 2: re-run collector + evaluator
    Week 2: compare new score vs 71.9%
    Drop of 5%+ → REGRESSION DETECTED ⚠️

  This is exactly how companies run CI/CD for LLM systems.
  The synthetic dataset is your automated test suite.

HOW TO TRIGGER A REGRESSION (for learning):
  1. Open src/collector.py
  2. Change COLLECTOR_PROMPT from:
       "Answer the following question clearly and concisely."
     to:
       "Answer briefly in one word only."
  3. Run: python detect_regression.py
  4. Watch the pass rate drop → regression detected!

HOW TO DETECT A GENUINE IMPROVEMENT:
  1. Improve the prompt → better answers
  2. Run: python detect_regression.py
  3. Pass rate increases → save as new baseline

THRESHOLD:
  REGRESSION if new pass rate < baseline - 5%
  IMPROVEMENT if new pass rate > baseline + 5%
  STABLE if within ±5%
"""

import json
import os
import shutil
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.collector import collect_responses
from src.evaluator import evaluate


BASELINE_PATH    = "data/evaluations/baseline_report.json"
NEW_RESPONSES    = "data/responses/regression_test_responses.json"
NEW_REPORT_PATH  = "data/evaluations/regression_report.json"
QA_PAIRS_PATH    = "data/synthetic_dataset/qa_pairs.json"
REGRESSION_THRESHOLD = 5.0   # % drop that counts as a regression


def load_baseline() -> dict:
    """Load the saved baseline report."""
    if not os.path.exists(BASELINE_PATH):
        print(f"[ERROR] No baseline found at {BASELINE_PATH}")
        print("Run python main.py first to create a baseline.")
        exit(1)

    with open(BASELINE_PATH) as f:
        report = json.load(f)

    return report["summary"]


def run_new_evaluation() -> dict:
    """
    Collect fresh responses with the CURRENT prompt settings,
    then evaluate them. This is the 'new run' we compare against baseline.
    """
    print("[regression] Collecting fresh responses with current prompt...")
    collect_responses(
        qa_pairs_path=QA_PAIRS_PATH,
        output_path=NEW_RESPONSES,
        provider="groq",
        model="llama-3.3-70b-versatile",
        max_questions=None,
        delay=None,
    )

    print()
    print("[regression] Evaluating new responses...")
    report = evaluate(
        responses_path=NEW_RESPONSES,
        output_path=NEW_REPORT_PATH,
        use_llm_judge=True,
        judge_delay=None,
    )

    return report["summary"]


def compare(baseline: dict, new_run: dict):
    """Compare baseline vs new run and print a clear verdict."""
    baseline_rate = baseline["pass_rate_pct"]
    new_rate      = new_run["pass_rate_pct"]
    delta         = round(new_rate - baseline_rate, 1)
    sign          = "+" if delta >= 0 else ""

    print()
    print("=" * 55)
    print("  REGRESSION DETECTION REPORT")
    print("=" * 55)
    print(f"  Baseline pass rate : {baseline_rate}%")
    print(f"  New run pass rate  : {new_rate}%")
    print(f"  Delta              : {sign}{delta}%")
    print()

    if delta <= -REGRESSION_THRESHOLD:
        verdict = "REGRESSION DETECTED"
        detail  = (f"Pass rate dropped {abs(delta)}% "
                   f"(threshold: {REGRESSION_THRESHOLD}%). "
                   f"Check if prompt changes broke the model.")
        symbol  = "FAIL"
    elif delta >= REGRESSION_THRESHOLD:
        verdict = "IMPROVEMENT DETECTED"
        detail  = (f"Pass rate improved {delta}%. "
                   f"Consider saving this as your new baseline.")
        symbol  = "PASS"
    else:
        verdict = "STABLE"
        detail  = f"Pass rate within ±{REGRESSION_THRESHOLD}% of baseline. No regression."
        symbol  = "OK"

    print(f"  VERDICT: [{symbol}] {verdict}")
    print(f"  {detail}")

    # Metric breakdown
    print()
    print(f"  {'Metric':<22} {'Baseline':>10} {'New Run':>10} {'Change':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    metrics = [
        ("Pass rate %",      baseline["pass_rate_pct"],      new_run["pass_rate_pct"]),
        ("Avg semantic",     baseline["avg_semantic_score"],  new_run["avg_semantic_score"]),
        ("Avg keywords",     baseline["avg_keyword_score"],   new_run["avg_keyword_score"]),
        ("Hallucinations",   baseline["hallucinations"],      new_run["hallucinations"]),
    ]

    for label, b_val, n_val in metrics:
        diff = round(n_val - b_val, 2)
        sign_d = "+" if diff >= 0 else ""
        print(f"  {label:<22} {b_val:>10} {n_val:>10} {sign_d}{diff:>9}")

    print("=" * 55)

    # Save comparison result
    result = {
        "timestamp":      datetime.now().isoformat(),
        "verdict":        verdict,
        "delta_pct":      delta,
        "threshold_pct":  REGRESSION_THRESHOLD,
        "baseline":       baseline,
        "new_run":        new_run,
    }
    output_path = "data/evaluations/regression_comparison.json"
    os.makedirs("data/evaluations", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Comparison saved -> {output_path}")

    return verdict


def save_as_new_baseline():
    """Promote the new run to become the baseline."""
    shutil.copy(NEW_REPORT_PATH, BASELINE_PATH)
    print(f"\n  New baseline saved -> {BASELINE_PATH}")


if __name__ == "__main__":
    print("=== PHASE 6B: REGRESSION DETECTION ===")
    print()
    print("  HOW TO TEST THIS:")
    print("  1. Run as-is → should show STABLE (same prompt as baseline)")
    print("  2. Edit COLLECTOR_PROMPT in src/collector.py")
    print("     Change to: 'Answer in one word only.'")
    print("  3. Re-run → watch pass rate drop → REGRESSION DETECTED")
    print()

    # ── Load baseline ─────────────────────────────────────────────────────
    print("[regression] Loading baseline...")
    baseline = load_baseline()
    print(f"[regression] Baseline: {baseline['pass_rate_pct']}% pass rate")
    print(f"             Model:    {baseline['model']}")
    print()

    # ── Run new evaluation ────────────────────────────────────────────────
    new_run = run_new_evaluation()

    # ── Compare ───────────────────────────────────────────────────────────
    verdict = compare(baseline, new_run)

    # ── If improved, offer to save as new baseline ────────────────────────
    if verdict == "IMPROVEMENT DETECTED":
        print()
        answer = input("Save new run as baseline? (y/n): ").strip().lower()
        if answer == "y":
            save_as_new_baseline()
