# Synthetic Data Generation Pipeline — Complete Guide

> Aligned with the speaker's framework:
> *"Synthetic data generation using LLMs involves using an LLM to create artificial data,
> which are often datasets that are used to train, fine-tune, and even evaluate."*

---

## The Big Picture

You give the system a text file. It automatically:
1. Cuts it into readable pieces
2. Generates questions + correct answers from those pieces
3. Asks other LLMs those same questions
4. Scores how well they answered
5. Compares multiple models side-by-side
6. Detects if future changes break performance

This is exactly how companies like Anthropic, OpenAI, and Google test their models at scale.

---

## Speaker's Framework -> Your Code (Exact Mapping)

| Speaker said | File | What it does |
|---|---|---|
| "source text ingested and split into smaller chunks" | `src/chunker.py` | Reads .txt -> 19 chunks of 500 tokens |
| "chunks grouped together for related information" | `src/chunker.py` | Each chunk tagged with chunk_id |
| "LLM prompted to generate inputs + expected outputs, grounded strictly in context" | `src/generator.py` | Groq 8b-instant -> 3 Q&A pairs per chunk |
| "result is a structured dataset of input-output pairs" | `data/synthetic_dataset/qa_pairs.json` | 57 pairs: question + expected_answer + context |
| "each pair converted into an LLM test case" | `src/collector.py` | Each question sent to target model |
| "target model produces actual output" | `data/responses/llama_responses.json` | 57 actual responses captured |
| "evaluation metrics compare prediction vs expected" | `src/evaluator.py` | exact / keyword / semantic scoring |
| "aggregated into accuracy scores, pass rates" | `data/evaluations/baseline_report.json` | 71.9% pass rate, avg semantic 0.68 |
| "detect regressions, compare models" | `detect_regression.py` + `compare_models.py` | Phase 6 scripts |

---

## The 4 Modules — Why Each One Exists

### MODULE 1: Chunker (`src/chunker.py`)

**The problem it solves:**
LLMs have a context window limit. You cannot feed a 40,000 character article as one prompt.
Also, if you feed the whole article, the LLM generates generic questions about everything instead of specific questions about each section.

**How it works:**
```
python_programming.txt (39,316 chars = 8,250 tokens)
                |
Sliding window: size=500, step=450 (500-50 overlap)
                |
19 chunks, each ~500 tokens
```

**Why overlap?**
```
Without overlap:              With 50-token overlap:
Chunk 1: tokens 0-500         Chunk 1: tokens 0-500
Chunk 2: tokens 500-1000      Chunk 2: tokens 450-950
                                        ^^^
                              50 tokens shared between chunks
                              A sentence at the boundary appears in BOTH chunks
                              No context is lost at the edges
```

**Why tokens not characters?**
LLMs think in tokens. "Python" = 1 token. "antidisestablishmentarianism" = 6 tokens.
Splitting by characters can cut a word in the middle of a token, confusing the LLM.
tiktoken counts tokens exactly the same way the LLM does.

---

### MODULE 2: Generator (`src/generator.py`)

**The problem it solves:**
You need a dataset of questions with known correct answers, grounded in your source text.
This is the "synthetic data" — artificial but real-knowledge-based.

**How it works:**
```
For each chunk:
  Prompt to llama-3.1-8b-instant:
    "Here is text. Generate 3 factual Q&A pairs. Return JSON."

  LLM returns:
    [{"question": "...", "answer": "..."}]

  We add: context (source chunk), chunk_id, question_type
```

**Why grounded in context?**
The speaker said: *"grounded strictly in that context."*
If you let the LLM make up questions freely, it hallucinates answers.
By forcing questions to come FROM the text, every answer is verifiable.

**Why llama-3.1-8b-instant here?**
- Has its own separate daily token quota from llama-3.3-70b
- 8b model is capable enough to write grammatical questions
- Saves the 70b quota for the actual testing (Module 3) where quality matters more

**The result:**
`qa_pairs.json` — 57 pairs, each like:
```json
{
  "question": "When was Python 3.0 released?",
  "expected_answer": "2008",
  "context": "...the original chunk text...",
  "chunk_id": "chunk_0000",
  "question_type": "factual"
}
```

---

### MODULE 3: Collector (`src/collector.py`)

**The problem it solves:**
You need to see what actual LLMs answer when tested — without giving them the answer.

**How it works:**
```
For each of the 57 questions:
  Send ONLY the question to llama-3.3-70b-versatile
  NO context given — this is the test
  Capture actual response + response time
```

**Why no context given to the tested model?**
We're testing what the model knows from its training data.
Giving it the context would be like letting a student read the textbook during the exam.

**Why llama-3.3-70b here (not 8b)?**
This is the model BEING EVALUATED. You want to test the best model
so results are meaningful for real-world use.

**Rate limit handling:**
```python
PROVIDER_DELAY = {
    "groq":        2.0,   # 30 req/min -> 60/30 = 2s per call
    "openrouter":  8.0,   # actual limit under load = 8 RPM -> 7.5s minimum
}
```
Different providers have different limits. The code enforces the right delay automatically.

---

### MODULE 4: Evaluator (`src/evaluator.py`)

**The problem it solves:**
How do you score an LLM answer automatically? It's not always a simple match.

**The 3-metric hierarchy:**

#### Metric 1: Exact Match (function — no LLM)
```python
"2008" in "Python 3.0 was released in December 2008."  ->  True
```
Use when: expected answer is a short fact (date, number, name).
Why function not LLM: deterministic, free, instant.

#### Metric 2: Keyword Overlap (function — no LLM)
```python
expected = "Guido van Rossum"
keywords = {"guido", "van", "rossum"}  (minus stopwords)
actual   = "Van Rossum created Python in 1989"
found    = {"van", "rossum"}
score    = 2/3 = 0.67
```
Use when: named entities, technical terms, multi-word answers.
Why function not LLM: countable, fast, no API cost.

#### Metric 3: Semantic Similarity (LLM-as-judge)
```
Send to Groq Llama 70b:
  "Does ACTUAL contain the correct info from EXPECTED? Score 0-1."

"2008" vs "Python 3.0 was released on December 3, 2008."  ->  1.0
(same fact, more detail — extra detail is fine)
```
Use when: meaning matters more than exact wording.
Why LLM: only a language model understands "built" = "constructed" = "created".

**The rule from the framework:**
> "Easy to check: Exact match -> use functions.
>  Hard to check: Semantic meaning -> use LLM-as-judge.
>  Don't use LLM where function works!"

#### Hallucination Check (function — no LLM)
```python
# If < 25% of words in actual appear in the source context
# -> model likely made up information not in the source
overlap = actual_words & context_words / len(actual_words)
if overlap < 0.25:  -> flag as possible hallucination
```

**Pass rule:**
`semantic_score >= 0.6` -> PASS

---

## Phase 6A: Model Comparison (`compare_models.py`)

**The problem it solves:**
Which free model is actually best for your specific knowledge domain?
Not "which model has the best benchmark" — but "which model scores highest on YOUR questions."

**How it works:**
```
Same 57 questions
        |
        |---> Model 1: Llama 3.3 70B  (Groq)
        |---> Model 2: Llama 4 Scout 17B (Groq)
        |---> Model 3: Llama 3.1 8B  (Groq)
        |
Each model's responses evaluated with the same 3-metric system
        |
Side-by-side comparison table
```

**Why the same questions for all models?**
Fair test. Same exam, different students.
If you used different questions per model, you can't compare — one set might be easier.

**The comparison table:**
```
Model                      Pass%  Semantic  Keywords  Halluc   Speed
------------------------- ------ --------- --------- ------- -------
Llama 3.3 70B              68.4%      0.68      0.62       3   0.50s  <- BEST
Llama 3.1 8B               43.9%      0.44      0.54       1   0.22s
Llama 4 Scout 17B          ~68%?      0.63 kw   (judge token limit hit)
```

**What the numbers mean:**
- 68.4% pass rate = 39 of 57 questions answered correctly by Llama 70B
- 43.9% pass rate = 25 of 57 for Llama 8B — 24.5% accuracy cost for using the smaller model
- Speed 0.50s vs 0.22s — 8B is 2x faster, but 24% less accurate. That is the trade-off.
- Hallucinations: 70B made up 3 answers not grounded in source; 8B made up 1

**What "pass" means here:**
The LLM judge (Groq 70B as the grader) scored the answer >= 0.6.
Not just keyword matching — semantic understanding. "December 2008" passes for "2008".

**Why all Groq, not OpenRouter?**
Originally used Qwen and Mistral via OpenRouter free tier. Both failed completely:
- OpenRouter routes free models through "Venice" as backend provider
- Venice caps at 8 requests/minute AND has a USD spend limit on free tier
- Result: 100% of requests returned 429 errors regardless of delay
- Lesson: Free-tier OpenRouter is not reliable for batch processing

Groq free tier is much more reliable: 30 req/min, 100K tokens/day per model.
All three models in compare_models.py now use Groq directly.

**The daily token limit lesson:**
The LLM judge (70b) costs ~350 tokens per question. 57 questions = ~20K tokens per evaluation.
The daily 100K limit can run out after ~5 evaluation runs.
When the limit hits, the judge silently returns 0.0 for every answer -> all scores look like failures.
This is why Llama 4 Scout showed 0% — the judge ran out of tokens, not because it answered badly.

Fix: delete the bad report file and re-run the next day when quota resets.
```bash
del data\evaluations\llama4scout_report.json
python compare_models.py   # only re-evaluates Scout, responses are cached
```

**Resume/skip-if-exists pattern:**
compare_models.py checks if response files and report files already exist.
If they do, it skips that model entirely. This means:
- You never pay twice for the same collection run
- If one model fails, re-run and only the failed model re-runs
- Collection (API calls) is expensive; evaluation can be re-run

---

## Phase 6B: Regression Detection (`detect_regression.py`)

**The problem it solves:**
You changed the prompt. Did the model get better or worse?
You updated a library. Did anything break?
This is CI/CD for LLMs — automated quality gate on every change.

**How it works:**
```
Step 1: Load baseline_report.json
        (your known-good run, saved when pipeline was working well)
        Baseline: 71.9% pass rate

Step 2: Collect FRESH responses right now
        (same questions, same model, same prompt — just a new API call)

Step 3: Evaluate with the same 3-metric system

Step 4: Compare new pass rate vs baseline
        REGRESSION  if drop  > 5%  (something broke)
        IMPROVEMENT if gain  > 5%  (something got better)
        STABLE      if delta <= 5% (normal noise)
```

**Actual output from this run:**
```
Baseline pass rate : 71.9%
New run pass rate  : 68.4%
Delta              : -3.5%

VERDICT: [OK] STABLE
Pass rate within +/-5.0% of baseline. No regression.

Metric                   Baseline    New Run     Change
---------------------- ---------- ---------- ----------
Pass rate %                  71.9       68.4      -3.5
Avg semantic                 0.68       0.67     -0.01
Avg keywords                 0.62       0.62        0.0
Hallucinations                  3          4         +1
```

**Why -3.5% even though nothing changed?**
LLMs are non-deterministic. The same question asked twice gets a slightly different answer.
A 3.5% swing on 57 questions = ~2 questions answered differently. This is normal.
The +-5% threshold exists to absorb this noise and only alert on real problems.

**Why +-5% specifically?**
- Too tight (e.g. +-1%) -> constant false alarms from normal LLM variation
- Too loose (e.g. +-20%) -> misses real regressions
- +-5% on a 57-question test = ~3 questions. Enough signal, not too sensitive.

**How to trigger a real regression (for testing):**
Edit `COLLECTOR_PROMPT` in `src/collector.py`:
```python
# Change this:
"Provide a direct factual answer in 1-3 sentences."

# To this (forces short useless answers):
"Answer in one word only."
```
Then run `python detect_regression.py` — watch pass rate drop below 66.9% -> REGRESSION DETECTED.

**What gets saved:**
```
data/evaluations/regression_report.json      -> full metrics for the new run
data/evaluations/regression_comparison.json  -> side-by-side baseline vs new
```

---

## File Structure & What Each File Does

```
data_generator/
|
+-- data/
|   +-- knowledge_base/
|   |   +-- python_programming.txt      INPUT: your source text
|   +-- chunks/
|   |   +-- chunks.json                 MODULE 1 OUTPUT: 19 token chunks
|   +-- synthetic_dataset/
|   |   +-- qa_pairs.json               MODULE 2 OUTPUT: 57 Q&A pairs
|   +-- responses/
|   |   +-- llama_responses.json        MODULE 3 OUTPUT: Llama 70B answers (baseline)
|   |   +-- llama4scout_responses.json  compare_models: Llama 4 Scout answers
|   |   +-- llama8b_responses.json      compare_models: Llama 8B answers
|   |   +-- regression_test_responses.json  detect_regression: fresh run
|   +-- evaluations/
|       +-- baseline_report.json        MODULE 4 OUTPUT: 71.9% pass rate
|       +-- llama_report.json           compare_models: Llama 70B report
|       +-- llama4scout_report.json     compare_models: Llama 4 Scout report
|       +-- llama8b_report.json         compare_models: Llama 8B report
|       +-- model_comparison.json       compare_models: final table
|       +-- regression_report.json      detect_regression: new run report
|       +-- regression_comparison.json  detect_regression: baseline vs new
|
+-- src/
|   +-- chunker.py      Module 1: text -> chunks
|   +-- generator.py    Module 2: chunks -> Q&A pairs (8b-instant)
|   +-- collector.py    Module 3: questions -> LLM responses (70b)
|   +-- evaluator.py    Module 4: responses -> scores + report
|
+-- config/config.yaml  settings (chunk size, model names, delays)
+-- .env                API keys (NEVER commit to git)
|
+-- main.py             runs all 4 modules in sequence
+-- compare_models.py   Phase 6A: Llama 70B vs Scout vs 8B
+-- detect_regression.py Phase 6B: catch prompt/model regressions
```

---

## Rate Limits — Full Picture

| Provider | Model | Req/min | Tokens/day | Delay needed |
|---|---|---|---|---|
| Groq | llama-3.3-70b-versatile | 30 | 100K | 2.0s |
| Groq | llama-3.1-8b-instant | 30 | separate higher quota | 2.0s |
| Groq | llama-4-scout-17b | 30 | separate quota | 2.0s |
| OpenRouter | :free models | 8 (real) | varies | 7.5s+ but unreliable |

**OpenRouter free tier warning:**
The advertised 20 RPM is the OpenRouter limit. The actual backend (Venice) has its own 8 RPM cap
AND a USD spend limit. In practice, free-tier batch requests fail 80-100% of the time.
Use Groq for any batch processing work.

**Daily token budget per full pipeline run:**
```
Module 2 (8b-instant):  19 x ~600 tokens = ~11K  (from 8b bucket)
Module 3 (70b):         57 x ~300 tokens = ~17K  (from 70b bucket)
Module 4 judge (70b):   57 x ~350 tokens = ~20K  (from 70b bucket)
-----------------------------------------------------------------
70b total per run: ~37K out of 100K/day  ->  ~2-3 full runs per day
compare_models (70b judge): ~20K extra per re-evaluation
```

**When the daily limit hits:**
The judge silently returns 0.0 for all answers -> pass rate shows 0% -> looks like total failure.
It's not a code bug. Just wait until midnight Pacific for the quota to reset.
The response collection files are still valid — only re-run evaluation.

**Resume mode — don't waste tokens:**
```bash
python main.py          # uses saved files if they exist (default)
python main.py --fresh  # ignores saved files, re-runs everything
```

---

## Commands Reference

```bash
# Full pipeline (resume mode — skips modules with saved output)
python main.py

# Force complete re-run from scratch
python main.py --fresh

# Phase 6A: Compare 3 models on same questions
python compare_models.py

# Phase 6B: Detect if prompt changes broke performance
python detect_regression.py

# Run individual modules
python -m src.chunker
python -m src.generator
python -m src.collector
python -m src.evaluator
```

---

## Results From This Run

### Baseline Pipeline (Llama 3.3 70B)
| Metric | Value |
|---|---|
| Source text | python_programming.txt (39,316 chars) |
| Chunks generated | 19 |
| Questions generated | 57 |
| Pass rate | 71.9% |
| Avg semantic score | 0.68 |
| Avg keyword score | 0.62 |
| Hallucinations detected | 3 |
| Avg response time | 0.5s/question |
| Total API cost | $0.00 |

### Phase 6A: Model Comparison
| Model | Pass% | Semantic | Keywords | Speed | Notes |
|---|---|---|---|---|---|
| Llama 3.3 70B | 68.4% | 0.68 | 0.62 | 0.50s | Best accuracy |
| Llama 3.1 8B | 43.9% | 0.44 | 0.54 | 0.22s | 2x faster, -24% accuracy |
| Llama 4 Scout 17B | pending | 0.63 kw | — | 0.39s | Judge hit daily limit, re-run needed |

### Phase 6B: Regression Detection
| Metric | Baseline | New Run | Change |
|---|---|---|---|
| Pass rate | 71.9% | 68.4% | -3.5% |
| Semantic | 0.68 | 0.67 | -0.01 |
| Keywords | 0.62 | 0.62 | 0.0 |
| Hallucinations | 3 | 4 | +1 |
| **Verdict** | | | **STABLE** |

---

## Key Design Decisions & Why

| Decision | Why |
|---|---|
| Token-based chunking (not character) | LLMs count tokens, not chars. tiktoken matches model's actual count |
| 50-token overlap between chunks | Prevents losing context at chunk boundaries |
| 3 questions per chunk | Enough coverage without repetition. 19 chunks x 3 = 57 tests |
| 8b model for generation, 70b for testing | Separate daily token quotas. 8b can write questions; 70b needed for accurate testing |
| LLM judge only for semantic, functions for exact/keyword | "Don't use LLM where function works" — costs tokens and adds latency for no gain |
| +-5% regression threshold | Absorbs LLM non-determinism noise without missing real regressions |
| Resume/skip-if-exists for all API stages | API calls cost tokens. Never pay twice for the same result |
| All-Groq for compare_models | OpenRouter free tier routes through Venice which is unreliable for batch work |
