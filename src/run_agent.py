#!/usr/bin/env python3
"""
run_agent.py
------------
Run the agent (and baselines) over the golden eval set.
Outputs: outputs/agent_results.csv

Usage:
  python src/run_agent.py --mode eval      # run on golden set
  python src/run_agent.py --mode demo      # single interactive message
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "eval"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

GOLDEN_PATH = EVAL_DIR / "golden_set.csv"
RESULTS_PATH = OUTPUTS_DIR / "agent_results.csv"

sys.path.insert(0, str(Path(__file__).parent))


def run_eval():
    if not GOLDEN_PATH.exists():
        print("❌ eval/golden_set.csv not found.")
        print("   Run: python src/build_eval_set.py")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No API key set. Add OPENAI_API_KEY or GROQ_API_KEY to your environment/.env")
        sys.exit(1)

    from agent import classify_intent, draft_reply, decide_escalation
    from retrieval import retrieve
    from baselines import trivial_classify, zeroshot_classify

    # Load golden set
    with open(GOLDEN_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        examples = list(reader)

    if len(examples) < 150:
        print(f"❌ Golden set has {len(examples)} examples; at least 150 are required.")
        print("   Run: python src/build_eval_set.py")
        sys.exit(1)

    print(f"📊 Running agent on {len(examples)} golden examples...")
    print("   (This calls OpenAI API — ~$0.30-0.50 total)\n")

    fieldnames = [
        "example_id", "customer_text",
        "gold_intent", "gold_escalate", "gold_reply_quality",
        # Agent predictions
        "agent_intent", "agent_confidence", "agent_escalate", "agent_escalation_reason",
        "agent_reply",
        # Baselines
        "trivial_intent", "zeroshot_intent",
    ]

    results = []
    for ex in tqdm(examples, desc="Agent"):
        ctext = ex["customer_text"]
        ex_id = ex["example_id"]

        try:
            # Agent
            intent, conf = classify_intent(ctext)
            retrieved = retrieve(ctext, k=3)
            reply = draft_reply(ctext, intent, retrieved)
            escalate, reason = decide_escalation(ctext, intent, conf, retrieved)

            # Baselines
            trivial = trivial_classify(ctext)
            zeroshot = zeroshot_classify(ctext)

            row = {
                "example_id": ex_id,
                "customer_text": ctext,
                "gold_intent": ex["gold_intent"],
                "gold_escalate": ex["gold_escalate"],
                "gold_reply_quality": ex["gold_reply_quality"],
                "agent_intent": intent,
                "agent_confidence": round(conf, 3),
                "agent_escalate": escalate,
                "agent_escalation_reason": reason,
                "agent_reply": reply,
                "trivial_intent": trivial,
                "zeroshot_intent": zeroshot,
            }
            results.append(row)

            # Small delay to respect rate limits
            time.sleep(0.3)

        except Exception as e:
            print(f"\n⚠️  Error on example {ex_id}: {e}")
            results.append({
                "example_id": ex_id,
                "customer_text": ctext,
                "gold_intent": ex["gold_intent"],
                "gold_escalate": ex["gold_escalate"],
                "gold_reply_quality": ex["gold_reply_quality"],
                "agent_intent": "ERROR",
                "agent_confidence": 0.0,
                "agent_escalate": False,
                "agent_escalation_reason": str(e),
                "agent_reply": "",
                "trivial_intent": trivial_classify(ctext),
                "zeroshot_intent": "",
            })

    # Write results
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved → outputs/agent_results.csv")
    print("   Next: python src/evaluate.py")


def run_demo():
    from agent import run_agent

    print("\n" + "="*60)
    print("  SpotifyCares AI Agent — Interactive Demo")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        try:
            text = input("📩 Customer message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        print("\n⏳ Processing...\n")
        result = run_agent(text)
        print(f"🏷️  Intent:     {result.intent} (confidence: {result.intent_confidence:.0%})")
        print(f"💬 Draft reply: {result.draft_reply}")
        print(f"🚨 Escalate:   {'YES — ' + result.escalation_reason if result.escalate else 'No (auto-handle)'}")
        if result.retrieved_examples:
            print(f"\n📚 Top similar past case:")
            ex = result.retrieved_examples[0]
            print(f"   Customer: {ex['customer_text'][:80]}...")
            print(f"   Spotify:  {ex['brand_reply'][:80]}...")
            print(f"   Similarity: {ex['score']:.3f}")
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "demo"], default="eval")
    args = parser.parse_args()

    if args.mode == "eval":
        run_eval()
    else:
        run_demo()
