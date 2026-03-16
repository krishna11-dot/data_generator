# Synthetic Data Generation Pipeline

A complete end-to-end pipeline to automatically generate synthetic datasets from source text, evaluate LLM performance, and detect regressions. 

This project implements a structured framework to generate artificial test data from a local knowledge base (`.txt`), query various Large Language Models (LLMs) with that data, and score their answers using three distinct metrics (Exact Match, Keyword Match, Semantic Similarity via an LLM judge).

For full details on the architecture, design decisions, and exact workflow, please read the comprehensive **[HOW IT WORKS guide](docs/HOW_IT_WORKS.md)**.

## The Big Picture

You give the system a text file. It automatically:
1. **Chunks** the text into readable pieces.
2. **Generates** questions + correct answers strictly grounded in those pieces.
3. **Collects** responses from target LLMs when tested on those questions (without context).
4. **Evaluates** how well they answered using deterministic functions and an LLM-as-a-judge.
5. **Compares** multiple models side-by-side.
6. **Detects** if future prompt or model changes break performance (Regressions).

## Quick Start

### 1. Installation

Requires Python 3.10+. Create a virtual environment and install the required packages:

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory and add your API keys:

```ini
GROQ_API_KEY=your_groq_api_key_here
# Optional: for OpenRouter models
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Run the Full Pipeline

Place your knowledge base text file in `data/knowledge_base/` (e.g., `python_programming.txt`), then run the orchestrator:

```bash
python main.py
```
*Note: The pipeline uses a caching mechanism. If a module's output already exists, it skips execution to save API tokens. To force a full re-run, use `python main.py --fresh`.*

## Commands Reference

| Command | Description |
|---|---|
| `python main.py` | Run the full 4-module pipeline (uses cached files if present) |
| `python main.py --fresh` | Force a complete re-run from scratch (ignores cache) |
| `python compare_models.py` | Phase 6A: Compare 3 Llama models on the same questions |
| `python detect_regression.py` | Phase 6B: Detect if prompt/model changes broke performance |

You can also run individual modules directly:
```bash
python -m src.chunker
python -m src.generator
python -m src.collector
python -m src.evaluator
```

## Directory Structure

```text
data_generator/
├── data/
│   ├── knowledge_base/      # Input: your source text (.txt)
│   ├── chunks/              # Module 1 Output: chunked text
│   ├── synthetic_dataset/   # Module 2 Output: generated Q&A pairs
│   ├── responses/           # Module 3 Output: LLM test answers
│   └── evaluations/         # Module 4 Output: scores and reports
├── docs/
│   └── HOW_IT_WORKS.md      # Detailed documentation
├── src/
│   ├── chunker.py           # Module 1: text -> chunks
│   ├── generator.py         # Module 2: chunks -> Q&A pairs
│   ├── collector.py         # Module 3: questions -> LLM responses
│   └── evaluator.py         # Module 4: responses -> scores report
├── config/                  # Settings (chunk size, model names)
├── requirements.txt         # Dependencies
├── main.py                  # Full pipeline orchestrator
├── compare_models.py        # Model comparisons
└── detect_regression.py     # Regression detection CI/CD
```

## Learn More

To understand the core methodology, how API rate limits are handled, and why specific models were chosen for different steps (like `llama-3.1-8b` for generation and `llama-3.3-70b` for the evaluation judge), read the detailed **[HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**.
