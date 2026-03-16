"""
MODULE 1: Knowledge Base Chunker
=================================
PURPOSE : Read a .txt file → split into overlapping token-based chunks
          → save to data/chunks/chunks.json

WHY TOKEN-BASED (not character-based)?
  LLMs think in tokens, not characters.
  "Python" = 1 token.  "antidisestablishmentarianism" = 6 tokens.
  If we split by characters we can accidentally cut a word in half
  mid-token, which confuses the LLM in Module 2.

OVERLAP?
  Chunk 1: tokens 0-500
  Chunk 2: tokens 450-950   ← 50 token overlap
  This ensures a sentence that spans a boundary appears in BOTH chunks,
  so neither chunk loses context at the edges.
"""

import json
import os
import tiktoken


def chunk_text(filepath: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split a text file into overlapping token-based chunks.

    Args:
        filepath   : path to the .txt input file
        chunk_size : number of tokens per chunk (default 500)
        overlap    : tokens shared between consecutive chunks (default 50)

    Returns:
        List of dicts, each dict = one chunk:
        {
            "id"          : "chunk_0000",
            "text"        : "actual text of the chunk...",
            "start_token" : 0,
            "end_token"   : 500,
            "token_count" : 500
        }
    """
    # ── 1. Read the source text ──────────────────────────────────────────
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[chunker] Loaded file: {filepath}")
    print(f"[chunker] Total characters: {len(text):,}")

    # ── 2. Tokenize with tiktoken (same tokenizer GPT-4 / Llama uses) ────
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    print(f"[chunker] Total tokens: {len(tokens):,}")

    # ── 3. Slide a window across the token list ───────────────────────────
    chunks = []
    start = 0
    chunk_id = 0
    step = chunk_size - overlap  # how far we move forward each time

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))

        chunk_tokens = tokens[start:end]
        chunk_text_str = encoding.decode(chunk_tokens)

        chunks.append({
            "id": f"chunk_{chunk_id:04d}",      # e.g. "chunk_0007"
            "text": chunk_text_str,
            "start_token": start,
            "end_token": end,
            "token_count": len(chunk_tokens),
        })

        chunk_id += 1
        start += step

    print(f"[chunker] Created {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")
    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """
    Save chunks list to a JSON file.
    Creates parent directories automatically.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[chunker] Saved chunks -> {output_path}")


def load_chunks(input_path: str) -> list[dict]:
    """Load chunks from a previously saved JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Run directly to test: python src/chunker.py ──────────────────────────────
if __name__ == "__main__":
    INPUT_FILE = "data/knowledge_base/python_programming.txt"
    OUTPUT_FILE = "data/chunks/chunks.json"

    # Safety check — tell the user clearly if file is missing
    if not os.path.exists(INPUT_FILE):
        print(f"\n[ERROR] Input file not found: {INPUT_FILE}")
        print("Please save your Wikipedia text to that path first.\n")
        exit(1)

    chunks = chunk_text(INPUT_FILE)
    save_chunks(chunks, OUTPUT_FILE)

    # ── Quick sanity check ──────────────────────────────────────────────
    print("\n--- FIRST CHUNK PREVIEW ---")
    print(f"ID         : {chunks[0]['id']}")
    print(f"Token count: {chunks[0]['token_count']}")
    print(f"Text (first 200 chars):\n{chunks[0]['text'][:200]}")
    print("\n--- LAST CHUNK PREVIEW ---")
    print(f"ID         : {chunks[-1]['id']}")
    print(f"Token count: {chunks[-1]['token_count']}")
    print("\nModule 1 complete. chunks.json is ready for Module 2.")
