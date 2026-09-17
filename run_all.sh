#!/usr/bin/env bash
# run_all.sh — Run the full pipeline end to end
# Usage: bash run_all.sh
# Assumes: OPENAI_API_KEY is set, data/twcs.csv exists

set -e  # exit on error

# Select the Python launcher available in the current shell (Windows, Git Bash,
# and Linux may expose it under different names).
if command -v python >/dev/null 2>&1; then
  PYTHON=(python)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON=(python.exe)
elif command -v py >/dev/null 2>&1; then
  PYTHON=(py -3)
elif command -v py.exe >/dev/null 2>&1; then
  PYTHON=(py.exe -3)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=(python3)
else
  echo "Python was not found. Install Python and ensure it is available on PATH."
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Hiver Agent — Full Pipeline Runner             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GROQ_API_KEY" ]; then
  if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
  fi
fi

if [ -z "$OPENAI_API_KEY" ] && [ -z "$GROQ_API_KEY" ]; then
  echo "❌ No API key set. Add OPENAI_API_KEY or GROQ_API_KEY to .env or export it."
  exit 1
fi

# Check dataset
if [ ! -f data/twcs.csv ]; then
  echo "⬇️  Dataset not found. Attempting Kaggle download..."
  "${PYTHON[@]}" scripts/download_data.py
fi

echo ""
echo "Step 1/5 — Preparing data..."
"${PYTHON[@]}" src/prepare_data.py

echo ""
echo "Step 2/5 — Building golden eval set..."
"${PYTHON[@]}" src/build_eval_set.py

echo ""
echo "Step 3/5 — Running agent on eval set..."
"${PYTHON[@]}" src/run_agent.py --mode eval

echo ""
echo "Step 4/5 — Running full evaluation..."
"${PYTHON[@]}" src/evaluate.py

echo ""
echo "Step 5/5 — Done!"
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Results: outputs/eval_report.html              ║"
echo "║  Demo:    python src/demo.py                    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
