"""
MODULE 4: Evaluation Engine (The Decision Box)
================================================
PURPOSE : Compare expected answers (Module 2) vs actual responses (Module 3)
          → score each pair using 3 metrics
          → detect hallucinations
          → produce a final report

THE 3 METRICS (from simplest to smartest):

  1. EXACT MATCH  → function (no LLM needed)
     "2008" vs "2008"  → True
     "2008" vs "Python 3.0 was released in 2008"  → False
     Best for: dates, numbers, single-word answers

  2. KEYWORD OVERLAP  → function (no LLM needed)
     Count how many important words from expected appear in actual.
     "Guido van Rossum" vs "Van Rossum began Python in 1989"
     → "guido": missing, "van": present, "rossum": present  → 0.67 score
     Best for: named entities, technical terms

  3. SEMANTIC SIMILARITY  → LLM-as-judge (Groq Llama, free)
     Ask an LLM: "Do these two answers mean the same thing? Score 0-1."
     "2008" vs "December 3, 2008"  → 0.95 (same meaning, more detail)
     Best for: full sentences where wording differs but meaning is the same

HALLUCINATION CHECK → function
  Does the actual response contain claims NOT in the source context?
  Method: check if key words in the actual response appear in the context.
  If actual mentions things outside the context → possible hallucination.

PASS RULE:
  A question PASSES if semantic_score >= 0.6
  (we use semantic as the final judge — most accurate metric)

OUTPUT: data/evaluations/baseline_report.json
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: Exact Match  (pure function, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(expected: str, actual: str) -> bool:
    """
    True if the expected answer appears verbatim inside the actual response.
    Case-insensitive, strips punctuation.

    Example:
      expected = "2008"
      actual   = "Python 3.0 was released in 2008."
      → True  (because "2008" is inside the actual)
    """
    if not actual:
        return False
    expected_clean = expected.strip().lower()
    actual_clean = actual.strip().lower()
    return expected_clean in actual_clean


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: Keyword Overlap  (pure function, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "with",
    "by", "from", "as", "it", "its", "this", "that", "which", "what",
    "when", "how", "who", "do", "does", "did", "has", "have", "had",
}

def keyword_overlap(expected: str, actual: str) -> float:
    """
    Score 0.0 to 1.0: what fraction of important words in 'expected'
    also appear in 'actual'.

    Ignores stopwords (a, the, is, etc.) — only counts meaningful words.

    Example:
      expected = "Guido van Rossum"
      actual   = "Van Rossum created Python in 1989"
      keywords = {"guido", "van", "rossum"}
      found    = {"van", "rossum"}
      score    = 2/3 = 0.67
    """
    def extract_keywords(text: str) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 2}

    expected_keywords = extract_keywords(expected)
    if not expected_keywords:
        return 0.0

    if not actual:
        return 0.0
    actual_keywords = extract_keywords(actual)
    matched = expected_keywords & actual_keywords
    return round(len(matched) / len(expected_keywords), 2)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: Semantic Similarity  (LLM-as-judge)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an answer evaluation judge.

TASK: Does the ACTUAL answer contain the correct information from the EXPECTED answer?

EXPECTED: {expected}
ACTUAL: {actual}

IMPORTANT RULES:
- If EXPECTED is a short fact (date, name, number) and ACTUAL contains it → score 0.9-1.0
- Extra detail in ACTUAL is FINE — do not penalise for being more complete
- Only penalise if ACTUAL gives a WRONG or DIFFERENT fact

Score from 0.0 to 1.0:
- 1.0 = ACTUAL contains the expected fact (even with extra detail)
- 0.7 = mostly correct, very minor difference
- 0.5 = partially correct
- 0.2 = mostly wrong
- 0.0 = completely wrong, no answer, or contradicts expected

Reply with ONLY a JSON object, nothing else:
{{"score": 0.0, "reason": "one sentence explanation"}}"""


def semantic_similarity(
    expected: str,
    actual: str,
    client: Groq,
    model: str = "llama-3.3-70b-versatile",
) -> tuple[float, str]:
    """
    Ask Groq Llama to judge if expected and actual mean the same thing.
    Returns: (score_0_to_1, reason_string)
    """
    if not actual:
        return 0.0, "No response from model"

    prompt = JUDGE_PROMPT.format(expected=expected, actual=actual)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # deterministic — we want consistent scoring
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()

        # Parse the JSON score
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return round(score, 2), reason

    except Exception as e:
        return 0.0, f"Judge error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HALLUCINATION CHECK  (pure function, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def hallucination_check(actual: str, context: str) -> bool:
    """
    Simple check: does the actual response contain words NOT in the context?

    Method:
      Extract all meaningful words from 'actual'.
      Check what fraction appear in 'context'.
      If < 40% overlap → likely hallucination (model made stuff up).

    NOTE: This is a heuristic, not perfect. The LLM naturally uses
    different phrasing than the source. But very low overlap is a signal.

    Returns: True = possible hallucination detected
    """
    def get_words(text: str) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 3}

    actual_words = get_words(actual)
    if not actual_words:
        return False

    context_words = get_words(context)
    overlap = len(actual_words & context_words) / len(actual_words)

    # Less than 25% of actual words found in context → flag it
    return overlap < 0.25


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    responses_path: str,
    output_path: str,
    use_llm_judge: bool = True,
    judge_delay: float = None,  # None = 2.0s (Groq 30 RPM)
) -> dict:
    """
    Evaluate all responses from Module 3.

    Args:
        responses_path : path to llama_responses.json
        output_path    : where to save the report
        use_llm_judge  : whether to call Groq for semantic scoring
        judge_delay    : seconds between LLM judge calls

    Returns:
        Report dict with all metrics
    """
    if judge_delay is None:
        judge_delay = 2.0   # Groq free tier: 30 req/min

    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    print(f"[evaluator] Loaded {len(responses)} response pairs")
    print(f"[evaluator] LLM judge: {'ON' if use_llm_judge else 'OFF (using keyword only)'}")
    print()

    client = None
    if use_llm_judge:
        api_key = __import__("os").getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)

    evaluated = []
    pass_count = 0
    hallucination_count = 0

    for i, r in enumerate(responses):
        expected = r.get("expected_answer", "")
        actual = r.get("actual_response", "")
        context = r.get("context", "")

        print(f"  [{i+1}/{len(responses)}] Evaluating: {r['question'][:55]}...")

        # ── Metric 1: Exact match ─────────────────────────────────────────
        em = exact_match(expected, actual)

        # ── Metric 2: Keyword overlap ─────────────────────────────────────
        kw = keyword_overlap(expected, actual)

        # ── Metric 3: Semantic similarity (LLM judge) ─────────────────────
        if use_llm_judge and actual:
            sem_score, sem_reason = semantic_similarity(expected, actual, client)
            time.sleep(judge_delay)
        else:
            # Fallback: use keyword score if no LLM judge
            sem_score = kw
            sem_reason = "keyword overlap used (no LLM judge)"

        # ── Hallucination check ───────────────────────────────────────────
        hallucinated = hallucination_check(actual, context) if actual else False
        if hallucinated:
            hallucination_count += 1

        # ── Pass/Fail decision ────────────────────────────────────────────
        passed = sem_score >= 0.6
        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"         exact={em} | keywords={kw} | semantic={sem_score} | {status}"
              + (" [HALLUCINATION?]" if hallucinated else ""))

        evaluated.append({
            "question_id":        r.get("question_id", i),
            "question":           r["question"],
            "expected_answer":    expected,
            "actual_response":    actual,
            "chunk_id":           r.get("chunk_id", ""),
            "model":              r.get("model", ""),
            "provider":           r.get("provider", ""),
            "response_time_s":    r.get("response_time_s", 0),
            # Metrics
            "exact_match":        em,
            "keyword_score":      kw,
            "semantic_score":     sem_score,
            "semantic_reason":    sem_reason,
            "hallucination_flag": hallucinated,
            "passed":             passed,
        })

    # ── Aggregate stats ───────────────────────────────────────────────────
    total = len(evaluated)
    pass_rate = round(pass_count / total * 100, 1) if total else 0
    avg_semantic = round(sum(e["semantic_score"] for e in evaluated) / total, 2)
    avg_keyword  = round(sum(e["keyword_score"]  for e in evaluated) / total, 2)
    avg_time     = round(sum(e["response_time_s"] for e in evaluated) / total, 2)

    report = {
        "summary": {
            "total_questions":    total,
            "passed":             pass_count,
            "failed":             total - pass_count,
            "pass_rate_pct":      pass_rate,
            "hallucinations":     hallucination_count,
            "avg_semantic_score": avg_semantic,
            "avg_keyword_score":  avg_keyword,
            "avg_response_time_s": avg_time,
            "model":              evaluated[0]["model"] if evaluated else "",
            "provider":           evaluated[0]["provider"] if evaluated else "",
        },
        "results": evaluated,
    }

    # ── Save report ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 50)
    print(f"  PASS RATE      : {pass_rate}%  ({pass_count}/{total})")
    print(f"  Avg Semantic   : {avg_semantic}")
    print(f"  Avg Keywords   : {avg_keyword}")
    print(f"  Hallucinations : {hallucination_count}")
    print(f"  Avg Speed      : {avg_time}s/question")
    print("=" * 50)
    print(f"\n[evaluator] Report saved -> {output_path}")

    return report


# ── Run directly: python src/evaluator.py ────────────────────────────────────
if __name__ == "__main__":
    print("=== MODULE 4: Evaluation Engine ===")
    print()

    # Test on 5 questions first (fast, uses 5 LLM judge calls)
    print("--- STEP A: Test evaluation on 5 questions ---")

    # Load just first 5 from the full responses
    with open("data/responses/llama_responses.json", "r") as f:
        all_responses = json.load(f)

    test_path = "data/responses/llama_responses_eval_test.json"
    with open(test_path, "w") as f:
        json.dump(all_responses[:5], f, indent=2)

    report = evaluate(
        responses_path=test_path,
        output_path="data/evaluations/test_report.json",
        use_llm_judge=True,
        judge_delay=2.0,
    )
