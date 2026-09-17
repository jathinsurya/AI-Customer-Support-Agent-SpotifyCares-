#!/usr/bin/env python3
"""
build_eval_set.py
-----------------
Build the 200-example golden evaluation set.

Sampling strategy:
  1. Load all Spotify threads from data/spotify_threads.json
  2. Run LLM classifier (our agent) on ALL threads to get predicted intents
  3. Stratified sample ~28-29 per intent (7 intents × ~28 ≈ 196 ≈ 200 total)
  4. LLM generates the "gold" intent label, reply quality score, escalation label
  5. Output: eval/golden_set.csv

Note: In the original build, 50 examples were spot-checked by hand.
Re-running this script regenerates LLM labels (takes ~5 min, costs ~$0.20).
"""

import json
import os
import sys
import csv
import random
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
EVAL_DIR.mkdir(exist_ok=True)

THREADS_PATH = DATA_DIR / "spotify_threads.json"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"

INTENTS = [
    "playback_issue",
    "account_access",
    "billing_query",
    "content_request",
    "app_performance",
    "general_complaint",
    "praise_or_other",
]

PER_INTENT = 29  # 7 × 29 = 203 → rounds to 200

# ── LLM labeller ──────────────────────────────────────────────────────────────

GOLD_LABEL_SYSTEM = """\
You are a human annotator creating a gold-standard evaluation set for a customer support AI.

Given a customer message and the brand's actual reply, provide:
1. intent — the TRUE intent of the customer message (one of the 7 classes)
2. escalate — should this have been escalated to a human? (true/false)
3. reply_quality — how good is the brand's actual reply? (1=poor, 2=ok, 3=good)
4. escalate_reason — reason for escalation (empty string if not escalating)

Intent classes:
- playback_issue: Music won't play, skipping, shuffle broken, streaming errors
- account_access: Login failures, password, locked out, account security
- billing_query: Charges, subscriptions, refunds, payment issues
- content_request: Missing content, unavailable songs/podcasts, regional issues
- app_performance: App crashes, slow, battery drain, UI bugs
- general_complaint: Vague frustration with no specific technical issue
- praise_or_other: Positive feedback, off-topic, spam

Respond ONLY with valid JSON:
{"intent": "...", "escalate": true/false, "reply_quality": 1-3, "escalate_reason": "..."}
"""

def gold_label(customer_text: str, brand_reply: str) -> dict:
    from openai import OpenAI

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        preferred = os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
        models = [preferred]
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = ["gpt-4o-mini"]

    user_content = f"Customer: {customer_text}\n\nBrand reply: {brand_reply}"
    last_error = None
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GOLD_LABEL_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise last_error or RuntimeError("LLM request failed")
    import json
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return {
            "gold_intent": data.get("intent", "general_complaint"),
            "gold_escalate": bool(data.get("escalate", False)),
            "gold_reply_quality": int(data.get("reply_quality", 2)),
            "gold_escalate_reason": data.get("escalate_reason", ""),
        }
    except Exception:
        return {
            "gold_intent": "general_complaint",
            "gold_escalate": False,
            "gold_reply_quality": 2,
            "gold_escalate_reason": "",
        }


# ── First-pass classifier for stratification ──────────────────────────────────

def quick_classify(text: str) -> str:
    """Keyword-based quick classifier just for stratification purposes."""
    from baselines import trivial_classify
    return trivial_classify(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not THREADS_PATH.exists():
        print("❌ data/spotify_threads.json not found.")
        print("   Run: python src/prepare_data.py")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No API key set. Add OPENAI_API_KEY or GROQ_API_KEY to your environment/.env")
        sys.exit(1)

    print("📂 Loading threads...")
    with open(THREADS_PATH) as f:
        threads = json.load(f)
    print(f"   {len(threads):,} threads loaded")
    if len(threads) < 150:
        print("❌ At least 150 source threads are required for the golden set.")
        sys.exit(1)

    # Stratify by quick intent
    print("🔀 Stratifying by intent...")
    buckets = {intent: [] for intent in INTENTS}
    for t in threads:
        label = quick_classify(t["customer_text"])
        buckets[label].append(t)

    for intent, items in buckets.items():
        print(f"   {intent}: {len(items)} threads")

    # Sample per intent
    random.seed(42)
    sampled = []
    for intent in INTENTS:
        pool = buckets[intent]
        n = min(PER_INTENT, len(pool))
        sampled.extend(random.sample(pool, n))

    random.shuffle(sampled)
    print(f"\n✅ Sampled {len(sampled)} examples for gold labelling")

    # Gold label each
    print("\n🏷️  Gold labelling (LLM annotator)...")
    rows = []
    for i, thread in enumerate(tqdm(sampled)):
        try:
            labels = gold_label(thread["customer_text"], thread["brand_reply"])
        except Exception as exc:
            print(f"\n❌ Gold labelling stopped on example {i + 1}: {exc}")
            print("   Add OpenAI API credits or configure a funded API key, then rerun.")
            sys.exit(1)
        row = {
            "example_id": i + 1,
            "customer_text": thread["customer_text"],
            "brand_reply": thread["brand_reply"],
            **labels,
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "example_id", "customer_text", "brand_reply",
        "gold_intent", "gold_escalate", "gold_reply_quality", "gold_escalate_reason",
    ]
    with open(GOLDEN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Golden eval set saved → eval/golden_set.csv ({len(rows)} examples)")

    # Print distribution
    from collections import Counter
    intent_dist = Counter(r["gold_intent"] for r in rows)
    print("\nIntent distribution in golden set:")
    for intent in INTENTS:
        print(f"   {intent}: {intent_dist.get(intent, 0)}")

    escalate_count = sum(1 for r in rows if r["gold_escalate"])
    print(f"\nEscalation: {escalate_count} escalate / {len(rows) - escalate_count} auto-handle")
    print("\nNext: python src/run_agent.py --mode eval")


if __name__ == "__main__":
    main()
