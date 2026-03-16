"""
compare_models.py — Phase 6A: Model Comparison
================================================
PURPOSE:
  Take the SAME 57 questions from qa_pairs.json
  → send them to 3 different free models
  → evaluate each model's responses
  → print a side-by-side comparison table

WHY THIS MATTERS (from the speaker's framework):
  "These metrics can be used to compare prompts or models,
   and benchmark system reliability over time."

  Same questions → different models → objective comparison.
  Not "which model FEELS better" but "which model SCORES higher."

MODELS COMPARED:
  1. Groq: llama-3.3-70b-versatile              (our baseline, large 70B)
  2. Groq: meta-llama/llama-4-scout-17b-16e     (Llama 4, newest architecture)
  3. Groq: llama-3.1-8b-instant                 (smallest/fastest, same used for generation)

NOTE: Originally used OpenRouter (Qwen, Mistral) but both are routed through
Venice provider which has 8 RPM limit and spend caps on free tier.
Switched to all-Groq for reliable 30 RPM with 2s delays.

RATE LIMITS:
  All Groq: 30 req/min -> 2.0s delay

OUTPUT:
  data/responses/llama4scout_responses.json
  data/responses/llama8b_responses.json
  data/evaluations/model_comparison.json
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.collector import collect_responses
from src.evaluator import evaluate


# ── Models to compare (all FREE) ─────────────────────────────────────────────
MODELS = [
    {
        "name":     "Llama 3.3 70B",
        "provider": "groq",
        "model":    "llama-3.3-70b-versatile",
        "responses_path": "data/responses/llama_responses.json",
        "report_path":    "data/evaluations/llama_report.json",
    },
    {
        "name":     "Llama 4 Scout 17B",
        "provider": "groq",
        "model":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "responses_path": "data/responses/llama4scout_responses.json",
        "report_path":    "data/evaluations/llama4scout_report.json",
    },
    {
        "name":     "Llama 3.1 8B",
        "provider": "groq",
        "model":    "llama-3.1-8b-instant",
        "responses_path": "data/responses/llama8b_responses.json",
        "report_path":    "data/evaluations/llama8b_report.json",
    },
]

QA_PAIRS_PATH = "data/synthetic_dataset/qa_pairs.json"
COMPARISON_OUTPUT = "data/evaluations/model_comparison.json"


def run_model(model_config: dict, skip_if_exists: bool = True) -> dict:
    """
    Collect responses + evaluate for one model.
    Returns the summary dict.
    """
    name     = model_config["name"]
    provider = model_config["provider"]
    model    = model_config["model"]
    resp_path   = model_config["responses_path"]
    report_path = model_config["report_path"]

    print(f"\n{'='*55}")
    print(f"  Testing: {name}")
    print(f"  Provider: {provider} | Model: {model}")
    print(f"{'='*55}")

    # ── Step 1: Collect responses (skip if already saved) ─────────────────
    if skip_if_exists and os.path.exists(resp_path):
        print(f"  [SKIP] Responses already exist at {resp_path}")
        print(f"         Delete file to re-collect.")
    else:
        collect_responses(
            qa_pairs_path=QA_PAIRS_PATH,
            output_path=resp_path,
            provider=provider,
            model=model,
            max_questions=None,
            delay=None,   # auto: groq=2s, openrouter=3s
        )

    # ── Step 2: Evaluate responses ────────────────────────────────────────
    if skip_if_exists and os.path.exists(report_path):
        print(f"  [SKIP] Report already exists at {report_path}")
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = evaluate(
            responses_path=resp_path,
            output_path=report_path,
            use_llm_judge=True,
            judge_delay=None,
        )

    return report["summary"]


def print_comparison_table(summaries: list[dict]):
    """Print a clean side-by-side comparison."""
    print("\n")
    print("=" * 65)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 65)
    print(f"  {'Model':<25} {'Pass%':>6} {'Semantic':>9} {'Keywords':>9} {'Halluc':>7} {'Speed':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")

    # Sort by pass rate descending
    summaries_sorted = sorted(summaries, key=lambda x: x["pass_rate_pct"], reverse=True)

    for s in summaries_sorted:
        winner = " <- BEST" if s == summaries_sorted[0] else ""
        print(
            f"  {s['model_name']:<25} "
            f"{s['pass_rate_pct']:>5.1f}% "
            f"{s['avg_semantic_score']:>9.2f} "
            f"{s['avg_keyword_score']:>9.2f} "
            f"{s['hallucinations']:>7} "
            f"{s['avg_response_time_s']:>6.2f}s"
            f"{winner}"
        )

    print("=" * 65)

    # Regression check vs baseline
    baseline = next((s for s in summaries if "Llama" in s["model_name"]), None)
    if baseline:
        print(f"\n  BASELINE (Llama): {baseline['pass_rate_pct']}%")
        for s in summaries:
            if s["model_name"] != baseline["model_name"]:
                diff = round(s["pass_rate_pct"] - baseline["pass_rate_pct"], 1)
                sign = "+" if diff >= 0 else ""
                print(f"  {s['model_name']}: {sign}{diff}% vs baseline")


if __name__ == "__main__":
    print("=== PHASE 6A: MODEL COMPARISON ===")
    print(f"  Questions: 57 (same set for all models)")
    print(f"  Models: {len(MODELS)}")
    print()

    all_summaries = []

    for model_config in MODELS:
        summary = run_model(model_config, skip_if_exists=True)
        summary["model_name"] = model_config["name"]
        all_summaries.append(summary)

    # ── Print comparison table ────────────────────────────────────────────
    print_comparison_table(all_summaries)

    # ── Save comparison JSON ──────────────────────────────────────────────
    os.makedirs("data/evaluations", exist_ok=True)
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": 57,
        "models": all_summaries,
    }
    with open(COMPARISON_OUTPUT, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Comparison saved -> {COMPARISON_OUTPUT}")
