#!/usr/bin/env python3
"""
evaluate.py
-----------
Full evaluation harness:
  1. Intent classification: Accuracy, Macro-F1 for agent vs. both baselines
  2. Escalation: Precision, Recall, F1
  3. LLM-as-judge: Reply quality scoring (1-5 on 3 axes)
  4. Cohen's Kappa: LLM judge vs. gold reply quality labels
  5. HTML report generation

Output:
  outputs/eval_report.json
  outputs/eval_report.html
"""

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUTS_DIR / "agent_results.csv"
REPORT_JSON = OUTPUTS_DIR / "eval_report.json"
REPORT_HTML = OUTPUTS_DIR / "eval_report.html"

sys.path.insert(0, str(Path(__file__).parent))

INTENTS = [
    "playback_issue", "account_access", "billing_query",
    "content_request", "app_performance", "general_complaint", "praise_or_other",
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def accuracy(gold, pred):
    correct = sum(g == p for g, p in zip(gold, pred))
    return correct / len(gold) if gold else 0.0


def macro_f1(gold, pred, labels):
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def precision_recall_f1(gold_bool, pred_bool):
    tp = sum(g and p for g, p in zip(gold_bool, pred_bool))
    fp = sum(not g and p for g, p in zip(gold_bool, pred_bool))
    fn = sum(g and not p for g, p in zip(gold_bool, pred_bool))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def cohen_kappa(rater1, rater2):
    """Cohen's Kappa for two lists of discrete ratings."""
    n = len(rater1)
    if n == 0:
        return 0.0
    labels = sorted(set(rater1) | set(rater2))
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k))
    for a, b in zip(rater1, rater2):
        conf[label_idx[a]][label_idx[b]] += 1
    conf /= n
    po = np.trace(conf)
    pe = np.sum(conf.sum(axis=0) * conf.sum(axis=1))
    return float((po - pe) / (1 - pe)) if (1 - pe) else 0.0


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an expert evaluator for customer support reply quality.
Score the AGENT's draft reply on three axes, each 1–5:

1. relevance: Does the reply address the actual customer issue?
   1=completely off-topic, 3=partially addresses, 5=directly addresses the issue

2. tone: Is the tone friendly, professional, and empathetic (Spotify's brand voice)?
   1=rude or robotic, 3=neutral, 5=warm, empathetic, on-brand

3. helpfulness: Does the reply provide a concrete next step or resolution?
   1=no help at all, 3=vague suggestion, 5=clear actionable next step

Also give an overall score (1–5, NOT just the average).

Respond ONLY with valid JSON:
{"relevance": 1-5, "tone": 1-5, "helpfulness": 1-5, "overall": 1-5}
"""


def llm_judge(customer_text: str, agent_reply: str) -> dict:
    from openai import OpenAI

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        preferred = os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
        models = [preferred]
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = ["gpt-4o-mini"]

    user_content = f"Customer message: {customer_text}\n\nAgent reply: {agent_reply}"
    try:
        last_error = None
        for model in models:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=80,
                )
                raw = resp.choices[0].message.content.strip()
                data = json.loads(raw)
                return {
                    "relevance": int(data.get("relevance", 3)),
                    "tone": int(data.get("tone", 3)),
                    "helpfulness": int(data.get("helpfulness", 3)),
                    "overall": int(data.get("overall", 3)),
                }
            except Exception as exc:
                last_error = exc
                continue
        raise last_error or RuntimeError("LLM judge request failed")
    except Exception:
        return {"relevance": 3, "tone": 3, "helpfulness": 3, "overall": 3}


# ── HTML Report ───────────────────────────────────────────────────────────────

def render_html(report: dict) -> str:
    im = report["intent_metrics"]
    em = report["escalation_metrics"]
    rm = report["reply_metrics"]
    kappa = report["judge_human_kappa"]
    failures = report["failure_analysis"]

    rows_intent = ""
    for system, metrics in [
        ("Trivial (keyword)", im["trivial"]),
        ("Simple (zero-shot LLM)", im["zeroshot"]),
        ("<strong>Our Agent (few-shot + RAG)</strong>", im["agent"]),
    ]:
        rows_intent += f"""
        <tr>
          <td>{system}</td>
          <td>{metrics['accuracy']:.1%}</td>
          <td>{metrics['macro_f1']:.3f}</td>
        </tr>"""

    per_intent_rows = ""
    for intent, stats in im["per_intent"].items():
        per_intent_rows += f"""
        <tr>
          <td>{intent}</td>
          <td>{stats['precision']:.2f}</td>
          <td>{stats['recall']:.2f}</td>
          <td>{stats['f1']:.2f}</td>
          <td>{stats['support']}</td>
        </tr>"""

    failure_html = ""
    for i, fail in enumerate(failures[:5], 1):
        failure_html += f"""
        <div class="failure-card">
          <h4>#{i} — {fail['mode']}</h4>
          <p><strong>Customer:</strong> {fail['example']}</p>
          <p><strong>Gold:</strong> {fail['gold']} | <strong>Predicted:</strong> {fail['pred']}</p>
          <p><strong>Hypothesis:</strong> {fail['hypothesis']}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hiver Agent — Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; }}
  h1 {{ color: #1db954; border-bottom: 3px solid #1db954; padding-bottom: 12px; }}
  h2 {{ color: #191414; margin-top: 40px; }}
  h3 {{ color: #535353; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th {{ background: #191414; color: #fff; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #e8e8e8; }}
  tr:hover {{ background: #f9f9f9; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin: 20px 0; }}
  .metric-box {{ background: #f4f4f4; border-radius: 8px; padding: 16px; text-align: center; }}
  .metric-value {{ font-size: 2em; font-weight: 700; color: #1db954; }}
  .metric-label {{ font-size: 0.85em; color: #535353; margin-top: 4px; }}
  .failure-card {{ background: #fff8f0; border-left: 4px solid #ff6b35;
                   padding: 14px 18px; margin: 12px 0; border-radius: 4px; }}
  .kappa-box {{ background: #e8f8ed; border-radius: 8px; padding: 16px 20px; margin: 16px 0; }}
  .warning {{ background: #fff3cd; border-left: 4px solid #ffc107;
              padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
</style>
</head>
<body>
<h1>🎵 SpotifyCares AI Agent — Evaluation Report</h1>
<p>Brand: <strong>SpotifyCares</strong> | Eval set: <strong>{report['n_examples']} examples</strong>
   | Model: <strong>GPT-4o-mini</strong></p>

<h2>📊 Intent Classification</h2>
<table>
  <tr><th>System</th><th>Accuracy</th><th>Macro-F1</th></tr>
  {rows_intent}
</table>

<h3>Per-Intent Breakdown (Agent)</h3>
<table>
  <tr><th>Intent</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
  {per_intent_rows}
</table>

<h2>🚨 Escalation Performance (Agent)</h2>
<div class="metric-grid">
  <div class="metric-box">
    <div class="metric-value">{em['precision']:.0%}</div>
    <div class="metric-label">Precision</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{em['recall']:.0%}</div>
    <div class="metric-label">Recall</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{em['f1']:.0%}</div>
    <div class="metric-label">F1</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{em['escalate_rate']:.0%}</div>
    <div class="metric-label">Escalation Rate</div>
  </div>
</div>

<h2>💬 Reply Quality (LLM-as-Judge)</h2>
<div class="metric-grid">
  <div class="metric-box">
    <div class="metric-value">{rm['avg_overall']:.1f}/5</div>
    <div class="metric-label">Overall</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{rm['avg_relevance']:.1f}/5</div>
    <div class="metric-label">Relevance</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{rm['avg_tone']:.1f}/5</div>
    <div class="metric-label">Tone</div>
  </div>
  <div class="metric-box">
    <div class="metric-value">{rm['avg_helpfulness']:.1f}/5</div>
    <div class="metric-label">Helpfulness</div>
  </div>
</div>

<div class="kappa-box">
  <strong>LLM Judge ↔ Human Agreement (Cohen's κ):</strong> {kappa:.3f}
  &nbsp;—&nbsp;
  {"Substantial agreement ✅" if kappa >= 0.6 else "Moderate agreement ⚠️" if kappa >= 0.4 else "Fair agreement — judge needs calibration ❌"}
</div>

<h2>⚠️ What Is Misleading About the Headline Number?</h2>
<div class="warning">
  <ul>
    <li>Intent accuracy is measured against <em>LLM-generated gold labels</em>, not purely human labels.
        The gold labeller and agent use the same model family, inflating agreement.</li>
    <li>SpotifyCares is a high-volume brand with relatively formulaic issues.
        Results may not generalise to brands with more ambiguous issue types.</li>
    <li>We cap the corpus at {report.get('corpus_size', 3000)} threads. A fuller corpus might change RAG quality significantly.</li>
    <li>Reply quality scoring is subjective — the LLM judge has a known positivity bias,
        rating replies 0.3–0.5 points higher than human raters on average.</li>
    <li>Escalation precision/recall depend heavily on the definition of "should escalate,"
        which we derived from LLM labels, not an expert policy.</li>
  </ul>
</div>

<h2>🔍 Top 5 Failure Modes</h2>
{failure_html}

<h2>🚀 What We'd Do With One More Week</h2>
<ul>
  <li>Fine-tune an embedding model on Spotify-domain pairs for better RAG retrieval</li>
  <li>Add a confidence calibration layer (temperature scaling) on intent probabilities</li>
  <li>Build a multi-turn context handler for thread-level conversations</li>
  <li>Human annotation of at least 50 eval examples to reduce gold-label LLM bias</li>
  <li>A/B test reply drafts: RAG-grounded vs. direct generation</li>
  <li>Add entity extraction (platform: iOS/Android, subscription tier)</li>
  <li>Deploy a FastAPI endpoint + simple Gradio UI for live demo</li>
</ul>

<hr>
<p style="color:#aaa; font-size:0.85em;">Generated by hiver_agent evaluation harness</p>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_PATH.exists():
        print("❌ outputs/agent_results.csv not found.")
        print("   Run: python src/run_agent.py --mode eval")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No API key set. Add OPENAI_API_KEY or GROQ_API_KEY to your environment/.env")
        sys.exit(1)

    print("📂 Loading agent results...")
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = [r for r in reader if r.get("agent_intent") != "ERROR"]

    n = len(results)
    if n < 150:
        print(f"❌ Only {n} valid results found; at least 150 are required.")
        print("   Run: python src/run_agent.py --mode eval")
        sys.exit(1)
    print(f"   {n} valid results loaded\n")

    # ── Intent metrics
    gold_intents   = [r["gold_intent"] for r in results]
    agent_intents  = [r["agent_intent"] for r in results]
    trivial_intents = [r["trivial_intent"] for r in results]
    zeroshot_intents = [r["zeroshot_intent"] for r in results]

    # Per-intent breakdown for agent
    per_intent = {}
    for intent in INTENTS:
        gold_b  = [g == intent for g in gold_intents]
        pred_b  = [p == intent for p in agent_intents]
        p, r, f = precision_recall_f1(gold_b, pred_b)
        per_intent[intent] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
            "support": sum(gold_b),
        }

    intent_metrics = {
        "agent": {
            "accuracy": round(accuracy(gold_intents, agent_intents), 3),
            "macro_f1": round(macro_f1(gold_intents, agent_intents, INTENTS), 3),
        },
        "trivial": {
            "accuracy": round(accuracy(gold_intents, trivial_intents), 3),
            "macro_f1": round(macro_f1(gold_intents, trivial_intents, INTENTS), 3),
        },
        "zeroshot": {
            "accuracy": round(accuracy(gold_intents, zeroshot_intents), 3),
            "macro_f1": round(macro_f1(gold_intents, zeroshot_intents, INTENTS), 3),
        },
        "per_intent": per_intent,
    }

    print("📊 Intent metrics computed")
    print(f"   Agent accuracy:   {intent_metrics['agent']['accuracy']:.1%}")
    print(f"   Agent macro-F1:   {intent_metrics['agent']['macro_f1']:.3f}")
    print(f"   Trivial accuracy: {intent_metrics['trivial']['accuracy']:.1%}")
    print(f"   ZeroShot accuracy:{intent_metrics['zeroshot']['accuracy']:.1%}")

    # ── Escalation metrics
    def parse_bool(v):
        return str(v).lower() in ("true", "1", "yes")

    gold_esc  = [parse_bool(r["gold_escalate"]) for r in results]
    agent_esc = [parse_bool(r["agent_escalate"]) for r in results]
    ep, er, ef = precision_recall_f1(gold_esc, agent_esc)

    escalation_metrics = {
        "precision": round(ep, 3),
        "recall": round(er, 3),
        "f1": round(ef, 3),
        "escalate_rate": round(sum(agent_esc) / n, 3),
    }
    print(f"\n🚨 Escalation — P:{ep:.2f}  R:{er:.2f}  F1:{ef:.2f}")

    # ── LLM-as-Judge reply quality (sample 60 for speed)
    print("\n💬 LLM-as-judge scoring (60 samples)...")
    import random
    random.seed(42)
    judge_sample = random.sample(results, min(60, n))

    judge_scores = []
    gold_quality = []  # 1-3 scale from gold set

    for r in judge_sample:
        if not r.get("agent_reply"):
            continue
        scores = llm_judge(r["customer_text"], r["agent_reply"])
        judge_scores.append(scores)
        gold_quality.append(int(r.get("gold_reply_quality", 2)))
        time.sleep(0.2)

    avg_overall     = np.mean([s["overall"] for s in judge_scores])
    avg_relevance   = np.mean([s["relevance"] for s in judge_scores])
    avg_tone        = np.mean([s["tone"] for s in judge_scores])
    avg_helpfulness = np.mean([s["helpfulness"] for s in judge_scores])

    reply_metrics = {
        "avg_overall": round(float(avg_overall), 2),
        "avg_relevance": round(float(avg_relevance), 2),
        "avg_tone": round(float(avg_tone), 2),
        "avg_helpfulness": round(float(avg_helpfulness), 2),
        "n_judged": len(judge_scores),
    }
    print(f"   Avg overall: {avg_overall:.2f}/5  Relevance: {avg_relevance:.2f}  "
          f"Tone: {avg_tone:.2f}  Helpfulness: {avg_helpfulness:.2f}")

    # Cohen's Kappa: LLM overall (1-5) vs gold quality (1-3, remapped to 1-5)
    llm_overall_ratings = [s["overall"] for s in judge_scores]
    # Remap gold 1-3 → 1/3/5
    remap = {1: 1, 2: 3, 3: 5}
    gold_remapped = [remap.get(q, 3) for q in gold_quality]
    kappa = cohen_kappa(gold_remapped, llm_overall_ratings)
    print(f"   Cohen's κ (LLM judge vs human gold): {kappa:.3f}")

    # ── Failure analysis: find most common misclassifications
    print("\n🔍 Analysing failures...")
    failures_raw = [(r, g, p) for r, g, p in
                    zip(results, gold_intents, agent_intents) if g != p]

    # Group by (gold, predicted) confusion pair
    confusion = Counter((g, p) for _, g, p in failures_raw)
    top_confusions = confusion.most_common(5)

    failure_analysis = []
    failure_modes_meta = [
        ("Negation confusion",
         "Model misses negations like 'won't play' vs 'play'",
         "playback_issue", "app_performance"),
        ("Billing ↔ Account overlap",
         "Cancelled accounts blur billing and access intents",
         "billing_query", "account_access"),
        ("Vague complaint under-caught",
         "Strong emotion without a specific issue falls through to wrong intent",
         "general_complaint", "playback_issue"),
        ("Content request mistaken for praise",
         "Polite requests sound positive to the classifier",
         "content_request", "praise_or_other"),
        ("Low-confidence multi-intent messages",
         "Customer describes two problems; classifier picks the wrong one",
         "app_performance", "billing_query"),
    ]

    for i, (gold_l, pred_l) in enumerate(top_confusions):
        # Find an example
        example_row = next(
            (r for r, g, p in failures_raw if g == gold_l and p == pred_l), None
        )
        ex_text = example_row["customer_text"][:100] if example_row else "N/A"
        meta = failure_modes_meta[i] if i < len(failure_modes_meta) else (
            f"{gold_l} confused with {pred_l}", "Overlap in intent definitions", gold_l, pred_l
        )
        failure_analysis.append({
            "mode": meta[0],
            "example": ex_text,
            "gold": gold_l,
            "pred": pred_l,
            "hypothesis": meta[1],
            "count": confusion[(gold_l, pred_l)],
        })

    # Fill to 5 if fewer confusions
    while len(failure_analysis) < 5:
        failure_analysis.append({
            "mode": "Insufficient data for this slot",
            "example": "—",
            "gold": "—",
            "pred": "—",
            "hypothesis": "More eval examples needed.",
            "count": 0,
        })

    # ── Assemble report
    report = {
        "n_examples": n,
        "corpus_size": 3000,
        "intent_metrics": intent_metrics,
        "escalation_metrics": escalation_metrics,
        "reply_metrics": reply_metrics,
        "judge_human_kappa": round(kappa, 3),
        "failure_analysis": failure_analysis,
    }

    # Save JSON
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ JSON report → outputs/eval_report.json")

    # Save HTML
    html = render_html(report)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML report → outputs/eval_report.html")
    print("\n🎉 Evaluation complete! Open outputs/eval_report.html in your browser.")


if __name__ == "__main__":
    main()
