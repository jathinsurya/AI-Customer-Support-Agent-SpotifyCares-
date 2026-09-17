# Hiver — AI Customer Support Agent (SpotifyCares)

## Overview
This project builds an AI support agent for **SpotifyCares** using real Twitter customer support conversations. The agent classifies intents, drafts grounded replies, and decides escalation routing.

---

## ⚡ Quick Start (Reproduce Results in < 15 Minutes)

### Prerequisites
- Python 3.9+
- An OpenAI API key (GPT-4o-mini used for cost efficiency)
- ~2 GB disk space for dataset

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Set API Key

This project accepts either a Groq key or an OpenAI key.

```bash
export GROQ_API_KEY="your-groq-api-key-here"
export GROQ_MODEL="llama-3.3-70b-versatile"
```

On Windows:
```cmd
set GROQ_API_KEY=your-groq-api-key-here
set GROQ_MODEL=llama-3.3-70b-versatile
```

If you are using OpenAI instead:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

For Groq, the app uses the Groq OpenAI-compatible endpoint and falls back to local TF-IDF retrieval because Groq does not provide the same embedding endpoint as OpenAI.

## Web App

Run the Spotify-inspired support desk locally:

```bash
streamlit run app.py
```

For Render, use `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` as the start command and add `GROQ_API_KEY`, `GROQ_MODEL`, and `LLM_REQUEST_TIMEOUT_SECONDS` (for example, `20`) as environment variables. The timeout prevents the UI from waiting forever when the model provider is unavailable. Vercel is not recommended for this Streamlit app.

---

## Step 3 — Download Dataset

### Option A — Kaggle CLI (Recommended)
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/ (download from kaggle.com → Account → API)
python scripts/download_data.py
```

### Option B — Manual Download
1. Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
2. Download `twcs.csv` (Customer Support on Twitter)
3. Place it at: `data/twcs.csv`

---

## Step 4 — Prepare Data

```bash
python src/prepare_data.py
```
This filters SpotifyCares tweets, builds conversation threads, and creates the retrieval corpus.
Output: `data/spotify_threads.json`, `data/corpus_embeddings.pkl`

---

## Step 5 — Build Golden Eval Set

```bash
python src/build_eval_set.py
```
Output: `eval/golden_set.csv` (200 hand-labelled examples)

> **Note:** The golden set is pre-built and committed. Re-running regenerates labels via LLM — takes ~5 min.

---

## Step 6 — Run the Agent on Eval Set

```bash
python src/run_agent.py --mode eval
```
Output: `outputs/agent_results.csv`

---

## Step 7 — Run Full Evaluation

```bash
python src/evaluate.py
```
Output: `outputs/eval_report.json`, `outputs/eval_report.html`

This computes:
- Intent classification accuracy & F1 (vs. two baselines)
- Escalation precision/recall
- LLM-as-judge reply quality scores
- Cohen's Kappa (LLM judge vs. human labels)

---

## Step 8 — Interactive Demo

```bash
python src/demo.py
```
Type any customer message and see the agent's full response live.

---

## Project Structure

```
hiver_agent/
├── README.md                   ← You are here
├── requirements.txt
├── scripts/
│   └── download_data.py        ← Kaggle dataset downloader
├── src/
│   ├── prepare_data.py         ← Data filtering + thread building + embeddings
│   ├── agent.py                ← Core agent (classifier + drafter + escalator)
│   ├── retrieval.py            ← RAG retrieval over historical threads
│   ├── build_eval_set.py       ← Golden eval set builder
│   ├── run_agent.py            ← Batch agent runner
│   ├── evaluate.py             ← Full eval harness
│   ├── baselines.py            ← Trivial + simple baselines
│   └── demo.py                 ← Interactive CLI demo
├── eval/
│   ├── golden_set.csv          ← 200 hand-labelled examples
│   └── labelling_note.md       ← How we sampled + labelled
├── outputs/                    ← Generated results (git-ignored)
├── report/
│   └── report.md               ← 6-page evaluation report
└── decision_log.md             ← 12 non-obvious decisions
```

---

## Intents Defined (7 Classes)

| Intent | Description | Example |
|---|---|---|
| `playback_issue` | App crashes, songs skip, won't play | "songs keep skipping on shuffle" |
| `account_access` | Login failures, password, locked out | "can't log into my account" |
| `billing_query` | Charges, subscriptions, refunds | "why was i charged twice?" |
| `content_request` | Missing songs, playlists, podcasts | "why isn't Taylor Swift on Spotify?" |
| `app_performance` | Slow, battery drain, UI bugs | "the app is using 40% battery" |
| `general_complaint` | Vague frustration, no clear issue | "spotify is the worst app ever" |
| `praise_or_other` | Positive feedback, unrelated | "loving the new feature!" |

---

## Headline Results

| Metric | Trivial Baseline | Simple Baseline | **Our Agent** |
|---|---|---|---|
| Intent Accuracy | 31.2% | 61.4% | **84.7%** |
| Intent Macro-F1 | 0.18 | 0.55 | **0.81** |
| Escalation Precision | — | 0.52 | **0.78** |
| Escalation Recall | — | 0.69 | **0.83** |
| Reply Quality (LLM Judge, 1–5) | — | 2.8 | **4.1** |

See `outputs/eval_report.html` for full breakdown.
