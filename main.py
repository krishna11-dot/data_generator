"""
main.py — Full Pipeline Orchestrator
======================================
Runs all 4 modules in sequence with one command:

  python main.py           ← full run
  python main.py --resume  ← skip modules whose output already exists

Flow:
  Module 1: text file  → chunks.json
  Module 2: chunks     → qa_pairs.json
  Module 3: questions  → llama_responses.json
  Module 4: responses  → baseline_report.json

Each module is independent — if one fails, you know exactly where.
RESUME MODE: if a module's output file already exists, skip it and
             use the saved data. Saves API tokens when re-running.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ── Import all 4 modules ──────────────────────────────────────────────────────
from src.chunker   import chunk_text, save_chunks
from src.generator import generate_dataset
from src.collector import collect_responses
from src.evaluator import evaluate


def print_header(module_num: int, title: str):
    print()
    print("=" * 55)
    print(f"  MODULE {module_num}: {title}")
    print("=" * 55)


def skip_module(output_path: str, label: str) -> bool:
    """Return True and print a skip message if output file already exists."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else len(data.get("results", []))
        print(f"  [SKIP] {label} already exists ({count} items). Using saved data.")
        print(f"         Delete {output_path} to re-run this module.")
        return True
    return False


def run_pipeline(
    input_txt:           str   = "data/knowledge_base/python_programming.txt",
    chunks_path:         str   = "data/chunks/chunks.json",
    qa_pairs_path:       str   = "data/synthetic_dataset/qa_pairs.json",
    responses_path:      str   = "data/responses/llama_responses.json",
    report_path:         str   = "data/evaluations/baseline_report.json",
    chunk_size:          int   = 500,
    chunk_overlap:       int   = 50,
    questions_per_chunk: int   = 3,
    target_provider:     str   = "groq",
    target_model:        str   = "llama-3.3-70b-versatile",
    request_delay:       float = None,   # None = auto (groq=2s, openrouter=3s)
    resume:              bool  = True,   # skip modules whose output exists
):
    pipeline_start = time.time()

    if not os.path.exists(input_txt):
        print(f"\n[ERROR] Input file not found: {input_txt}")
        print("Save your knowledge base text file there first.\n")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 1: Chunker  (fast, no API — always runs)
    # ─────────────────────────────────────────────────────────────────────
    print_header(1, "Knowledge Base Chunker")
    t1 = time.time()
    chunks = chunk_text(input_txt, chunk_size=chunk_size, overlap=chunk_overlap)
    save_chunks(chunks, chunks_path)
    print(f"  Done in {round(time.time()-t1, 1)}s — {len(chunks)} chunks created")

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 2: Q&A Generator  (uses Groq API — skip if file exists)
    # ─────────────────────────────────────────────────────────────────────
    print_header(2, "Synthetic Q&A Generator")
    t2 = time.time()

    if resume and skip_module(qa_pairs_path, "qa_pairs.json"):
        with open(qa_pairs_path) as f:
            qa_pairs = json.load(f)
    else:
        qa_pairs = generate_dataset(
            chunks_path=chunks_path,
            output_path=qa_pairs_path,
            questions_per_chunk=questions_per_chunk,
            max_chunks=None,
            delay=request_delay,
        )
        print(f"  Done in {round(time.time()-t2, 1)}s — {len(qa_pairs)} Q&A pairs generated")

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 3: Response Collector  (uses Groq/OpenRouter — skip if exists)
    # ─────────────────────────────────────────────────────────────────────
    print_header(3, "LLM Response Collector")
    t3 = time.time()

    if resume and skip_module(responses_path, "llama_responses.json"):
        with open(responses_path) as f:
            responses = json.load(f)
    else:
        responses = collect_responses(
            qa_pairs_path=qa_pairs_path,
            output_path=responses_path,
            provider=target_provider,
            model=target_model,
            max_questions=None,
            delay=request_delay,
        )
        print(f"  Done in {round(time.time()-t3, 1)}s — {len(responses)} responses collected")

    # ─────────────────────────────────────────────────────────────────────
    # MODULE 4: Evaluator  (uses Groq for judge — skip if exists)
    # ─────────────────────────────────────────────────────────────────────
    print_header(4, "Evaluation Engine")
    t4 = time.time()

    if resume and os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        print(f"  [SKIP] baseline_report.json already exists. Using saved report.")
        print(f"         Delete {report_path} to re-run evaluation.")
    else:
        report = evaluate(
            responses_path=responses_path,
            output_path=report_path,
            use_llm_judge=True,
            judge_delay=request_delay,
        )
        print(f"  Done in {round(time.time()-t4, 1)}s")

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    total_time = round(time.time() - pipeline_start, 1)
    summary = report["summary"]

    print()
    print("=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Input file   : {input_txt}")
    print(f"  Model tested : {target_provider} / {target_model}")
    print(f"  Q&A pairs    : {len(qa_pairs)}")
    print(f"  Responses    : {len(responses)}")
    print(f"  Pass rate    : {summary['pass_rate_pct']}%  ({summary['passed']}/{summary['total_questions']})")
    print(f"  Avg semantic : {summary['avg_semantic_score']}")
    print(f"  Hallucinations: {summary['hallucinations']}")
    print(f"  Total time   : {total_time}s")
    print("=" * 55)
    print()
    print(f"  Report: {report_path}")
    print()
    print("  To force full re-run     : delete output files, then python main.py")
    print("  To compare models        : python compare_models.py")
    print("  To detect regression     : python detect_regression.py")
    print()

    return report


if __name__ == "__main__":
    # Pass --fresh to delete cached files and re-run everything
    fresh = "--fresh" in sys.argv
    run_pipeline(resume=not fresh)
