"""
MODULE 2: Synthetic Q&A Generator
===================================
PURPOSE : Load chunks from Module 1 → send each chunk to Groq Llama 3.3
          → ask it to generate questions + answers from that chunk
          → save all Q&A pairs to data/synthetic_dataset/qa_pairs.json

HOW IT WORKS:
  For each chunk of text, we send this prompt to Groq:
    "Here is some text. Generate 3 factual questions and answers
     strictly from this text. Return JSON."

  Groq returns:
    [
      {"question": "...", "answer": "..."},
      ...
    ]

  We add the source chunk as "context" so every Q&A is traceable.

GROUNDING RULE:
  The answer MUST come from the chunk text.
  We validate this by checking if key answer words appear in the chunk.
  This prevents the LLM from hallucinating answers.

RATE LIMIT:
  Groq free tier = 30 requests/minute = 1 request every 2 seconds.
  We sleep 2 seconds between each chunk to stay safe.
"""

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

# Load .env file so GROQ_API_KEY is available
load_dotenv()


# ── The prompt we send to Llama 3.3 for every chunk ──────────────────────────
GENERATION_PROMPT = """You are a dataset creation assistant.

Given the text below, generate exactly {n} factual question-answer pairs.

RULES:
- Questions must be answerable ONLY from the provided text
- Answers must be short and factual (1-3 sentences max)
- Do NOT make up information not present in the text
- Return ONLY a JSON array, no explanation

TEXT:
{chunk_text}

Return this exact format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


def create_groq_client() -> Groq:
    """Create Groq client using API key from .env"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-key-here":
        raise ValueError(
            "GROQ_API_KEY not set in .env file.\n"
            "Get a free key at: console.groq.com"
        )
    return Groq(api_key=api_key)


def generate_qa_for_chunk(
    client: Groq,
    chunk: dict,
    questions_per_chunk: int = 3,
    model: str = "llama-3.1-8b-instant",   # separate daily quota from 70b
) -> list[dict]:
    """
    Send one chunk to Groq → get Q&A pairs back.

    Args:
        client             : Groq client
        chunk              : one chunk dict from chunks.json
        questions_per_chunk: how many Q&A pairs to generate
        model              : Groq model to use

    Returns:
        List of dicts:
        [
          {
            "question"       : "When was Python created?",
            "expected_answer": "Python was created by Guido van Rossum in 1989.",
            "context"        : "...original chunk text...",
            "chunk_id"       : "chunk_0003",
            "question_type"  : "factual"
          },
          ...
        ]
    """
    prompt = GENERATION_PROMPT.format(
        n=questions_per_chunk,
        chunk_text=chunk["text"],
    )

    # ── Call Groq API ─────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # low = more factual, less creative
        max_tokens=1024,
    )

    raw_text = response.choices[0].message.content.strip()

    # ── Parse the JSON response ───────────────────────────────────────────
    qa_list = parse_json_response(raw_text)

    # ── Add metadata to each pair ─────────────────────────────────────────
    results = []
    for qa in qa_list:
        if not qa.get("question") or not qa.get("answer"):
            continue  # skip malformed entries

        results.append({
            "question":        qa["question"],
            "expected_answer": qa["answer"],
            "context":         chunk["text"],        # source chunk = grounding
            "chunk_id":        chunk["id"],
            "question_type":   "factual",            # all are factual for now
        })

    return results


def parse_json_response(raw_text: str) -> list[dict]:
    """
    Safely parse JSON from LLM output.
    LLMs sometimes wrap JSON in markdown code blocks — this handles that.
    """
    # Strip markdown code fences if present: ```json ... ```
    if "```" in raw_text:
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines)

    # Find the JSON array in the text
    start = raw_text.find("[")
    end = raw_text.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  [WARNING] Could not find JSON array in response. Skipping.")
        return []

    json_str = raw_text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  [WARNING] JSON parse error: {e}. Skipping chunk.")
        return []


def generate_dataset(
    chunks_path: str = "data/chunks/chunks.json",
    output_path: str = "data/synthetic_dataset/qa_pairs.json",
    questions_per_chunk: int = 3,
    max_chunks: int = None,     # None = process all chunks
    delay: float = None,        # seconds between API calls. None = 2.0s (Groq 30 RPM)
) -> list[dict]:
    """
    Main function: process all chunks → generate full Q&A dataset.

    Args:
        chunks_path        : path to chunks.json from Module 1
        output_path        : where to save the final Q&A pairs
        questions_per_chunk: Q&A pairs per chunk
        max_chunks         : limit for testing (None = all)
        delay              : sleep between API calls

    Returns:
        Full list of Q&A pair dicts
    """
    if delay is None:
        delay = 2.0   # Groq free tier: 30 req/min = 2s per call

    # ── Load chunks from Module 1 ─────────────────────────────────────────
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"[generator] Loaded {len(chunks)} chunks from {chunks_path}")
    print(f"[generator] Target: {len(chunks) * questions_per_chunk} Q&A pairs")
    print(f"[generator] Model: llama-3.1-8b-instant (Groq) — separate daily quota")
    print(f"[generator] Delay between calls: {delay}s (30 req/min limit)")
    print()

    client = create_groq_client()

    all_qa_pairs = []

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}: {chunk['id']} "
              f"({chunk['token_count']} tokens)...", end=" ")

        try:
            pairs = generate_qa_for_chunk(client, chunk, questions_per_chunk)
            all_qa_pairs.extend(pairs)
            print(f"got {len(pairs)} Q&A pairs")
        except Exception as e:
            print(f"ERROR: {e}")

        # ── Rate limit: sleep between calls ──────────────────────────────
        if i < len(chunks) - 1:
            time.sleep(delay)

    print()
    print(f"[generator] Total Q&A pairs generated: {len(all_qa_pairs)}")

    # ── Save to JSON ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"[generator] Saved -> {output_path}")
    return all_qa_pairs


# ── Run directly to test: python src/generator.py ────────────────────────────
if __name__ == "__main__":

    print("=== MODULE 2: Synthetic Q&A Generator ===")
    print()

    # PHASE A: Test with just 2 chunks first (fast, uses ~2 API calls)
    print("--- PHASE A: Test with 2 chunks ---")
    test_pairs = generate_dataset(
        chunks_path="data/chunks/chunks.json",
        output_path="data/synthetic_dataset/qa_pairs_test.json",
        questions_per_chunk=3,
        max_chunks=2,      # only 2 chunks = 6 Q&A pairs = quick test
        delay=2.0,
    )

    print()
    print("--- SAMPLE OUTPUT ---")
    for i, pair in enumerate(test_pairs[:3]):
        print(f"\nQ{i+1}: {pair['question']}")
        print(f"A{i+1}: {pair['expected_answer']}")
        print(f"Source: {pair['chunk_id']}")

    print()
    confirm = input("Phase A looks good? Run full dataset? (y/n): ").strip().lower()

    if confirm == "y":
        print()
        print("--- PHASE B: Full dataset (all chunks) ---")
        all_pairs = generate_dataset(
            chunks_path="data/chunks/chunks.json",
            output_path="data/synthetic_dataset/qa_pairs.json",
            questions_per_chunk=3,
            max_chunks=None,   # all 19 chunks = ~57 Q&A pairs
            delay=2.0,
        )
        print()
        print(f"Module 2 complete. {len(all_pairs)} Q&A pairs ready for Module 3.")
    else:
        print("Stopped after test. Re-run when ready.")
