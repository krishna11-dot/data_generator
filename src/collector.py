"""
MODULE 3: LLM Response Collector (Test Runner)
================================================
PURPOSE : Load questions from Module 2 → send each question to a target LLM
          → capture the actual response + timing
          → save to data/responses/{model_name}_responses.json

WHY THIS MODULE EXISTS:
  Module 2 created "expected answers" (what the correct answer should be).
  Module 3 asks real LLMs: "What do YOU think the answer is?"
  Module 4 will then compare expected vs actual.

  This is exactly how real LLM evaluation works at companies.

SUPPORTED PROVIDERS:
  - Groq      → uses `groq` package       → model: llama-3.3-70b-versatile
  - OpenRouter → uses `openai` package    → models: qwen, mistral (all free)
    (OpenRouter speaks the same API language as OpenAI — same package, different URL)

OUTPUT per question:
  {
    "question_id"    : 0,
    "question"       : "When was Python 3.0 released?",
    "expected_answer": "2008",
    "actual_response": "Python 3.0 was released in December 2008.",
    "context"        : "...source chunk...",
    "chunk_id"       : "chunk_0000",
    "model"          : "llama-3.3-70b-versatile",
    "provider"       : "groq",
    "response_time_s": 1.23,
    "timestamp"      : "2024-01-15T10:30:00"
  }
"""

import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()


# ── The prompt sent to the TARGET model ──────────────────────────────────────
# NOTE: We do NOT give it the context/answer. We're testing if it knows.
# This is the fair test — just the question, no hints.
COLLECTOR_PROMPT = """Answer the following question clearly and concisely.

Question: {question}

Provide a direct factual answer in 1-3 sentences."""


def create_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-key-here":
        raise ValueError("GROQ_API_KEY not set in .env")
    return Groq(api_key=api_key)


def create_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your-openrouter-key-here":
        raise ValueError("OPENROUTER_API_KEY not set in .env")
    # OpenRouter uses the same API format as OpenAI — just different base_url
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def ask_groq(client: Groq, question: str, model: str) -> tuple[str, float]:
    """
    Send a question to a Groq model.
    Returns: (answer_text, response_time_in_seconds)
    """
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": COLLECTOR_PROMPT.format(question=question)}],
        temperature=0.1,   # very low = consistent, factual answers
        max_tokens=256,
    )

    elapsed = round(time.time() - start, 2)
    answer = response.choices[0].message.content.strip()
    return answer, elapsed


def ask_openrouter(client: OpenAI, question: str, model: str) -> tuple[str, float]:
    """
    Send a question to an OpenRouter model (Qwen, Mistral, etc.)
    Returns: (answer_text, response_time_in_seconds)
    """
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": COLLECTOR_PROMPT.format(question=question)}],
        temperature=0.1,
        max_tokens=256,
    )

    elapsed = round(time.time() - start, 2)
    answer = response.choices[0].message.content.strip()
    return answer, elapsed


# Rate limits per provider (enforced by sleep between calls)
PROVIDER_DELAY = {
    "groq":        2.0,   # 30 req/min → 60/30 = 2.0s per call
    "openrouter":  8.0,   # Qwen actual limit = 8 RPM under load → 60/8 = 7.5s, use 8s to be safe
}


def collect_responses(
    qa_pairs_path: str,
    output_path: str,
    provider: str,
    model: str,
    max_questions: int = None,
    delay: float = None,   # None = auto-select from PROVIDER_DELAY
) -> list[dict]:
    """
    Main function: send all questions to a model, collect responses.

    Args:
        qa_pairs_path : path to qa_pairs.json from Module 2
        output_path   : where to save responses JSON
        provider      : "groq" or "openrouter"
        model         : model name string
        max_questions : limit for testing (None = all)
        delay         : seconds between calls. None = use provider default
                        groq=2.0s (30 RPM), openrouter=3.0s (20 RPM)

    Returns:
        List of response dicts
    """
    # Auto-select correct delay for provider if not overridden
    if delay is None:
        delay = PROVIDER_DELAY.get(provider, 2.0)
    # ── Load questions from Module 2 ──────────────────────────────────────
    with open(qa_pairs_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    if max_questions:
        qa_pairs = qa_pairs[:max_questions]

    print(f"[collector] Loaded {len(qa_pairs)} questions")
    print(f"[collector] Sending to: {provider} / {model}")
    rpm = int(60 / delay)
    print(f"[collector] Delay between calls: {delay}s ({rpm} req/min limit)")
    print()

    # ── Set up the right client ───────────────────────────────────────────
    if provider == "groq":
        client = create_groq_client()
        ask_fn = lambda q: ask_groq(client, q, model)
    elif provider == "openrouter":
        client = create_openrouter_client()
        ask_fn = lambda q: ask_openrouter(client, q, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'openrouter'.")

    results = []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        print(f"  [{i+1}/{len(qa_pairs)}] Q: {question[:60]}...", end=" ")

        try:
            actual_response, response_time = ask_fn(question)

            results.append({
                "question_id":     i,
                "question":        question,
                "expected_answer": pair["expected_answer"],
                "actual_response": actual_response,
                "context":         pair["context"],
                "chunk_id":        pair["chunk_id"],
                "model":           model,
                "provider":        provider,
                "response_time_s": response_time,
                "timestamp":       datetime.now().isoformat(),
            })
            print(f"OK ({response_time}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question_id":     i,
                "question":        question,
                "expected_answer": pair["expected_answer"],
                "actual_response": None,
                "error":           str(e),
                "model":           model,
                "provider":        provider,
            })

        # ── Rate limit sleep ──────────────────────────────────────────────
        if i < len(qa_pairs) - 1:
            time.sleep(delay)

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if r.get("actual_response"))
    print()
    print(f"[collector] {success}/{len(results)} questions answered successfully")
    print(f"[collector] Saved -> {output_path}")
    return results


# ── Run directly: python src/collector.py ─────────────────────────────────────
if __name__ == "__main__":
    print("=== MODULE 3: LLM Response Collector ===")
    print()

    # ── STEP A: Test with 5 questions on Groq (fast) ─────────────────────
    print("--- STEP A: Test with 5 questions (Groq Llama 3.3) ---")
    test_results = collect_responses(
        qa_pairs_path="data/synthetic_dataset/qa_pairs.json",
        output_path="data/responses/llama_responses_test.json",
        provider="groq",
        model="llama-3.3-70b-versatile",
        max_questions=5,
        delay=None,   # auto: groq=2.0s, openrouter=3.0s
    )

    print()
    print("--- SAMPLE COMPARISON ---")
    for r in test_results[:2]:
        print(f"\nQuestion : {r['question']}")
        print(f"Expected : {r['expected_answer']}")
        print(f"Actual   : {r['actual_response']}")
        print(f"Time     : {r['response_time_s']}s")
