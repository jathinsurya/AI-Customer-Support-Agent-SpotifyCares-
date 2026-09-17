#!/usr/bin/env python3
"""
agent.py
--------
Core AI support agent for SpotifyCares.
Three components:
  1. IntentClassifier   — 7-class intent classification (few-shot)
  2. ReplyDrafter       — RAG-grounded reply drafting
  3. EscalationDecider  — auto-handle vs. escalate with reason
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Optional

from openai import OpenAI


def get_client_and_model():
    """Support OpenAI and Groq-compatible endpoints."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        preferred = os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
        models = [preferred]
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), models

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key), ["gpt-4o-mini"]

    raise RuntimeError("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY.")


client, MODEL_CANDIDATES = get_client_and_model()
MODEL = MODEL_CANDIDATES[0]

# ── Intent definitions ────────────────────────────────────────────────────────

INTENTS = {
    "playback_issue": (
        "Music won't play, songs skip, shuffle broken, audio cuts out, "
        "playback stops unexpectedly, streaming errors."
    ),
    "account_access": (
        "Can't log in, forgot password, account locked, email not recognised, "
        "two-factor auth problems, account hacked."
    ),
    "billing_query": (
        "Unexpected charges, subscription cost, refund request, payment failed, "
        "premium not activating, family plan billing."
    ),
    "content_request": (
        "Song or album missing, podcast not available, explicit content filter, "
        "content removed, requesting new music, regional restrictions."
    ),
    "app_performance": (
        "App slow or freezing, high battery drain, crashes on open, "
        "UI glitch, download not working, desktop/mobile app bugs."
    ),
    "general_complaint": (
        "Vague frustration or anger with no specific technical issue, "
        "saying the app is bad, threatening to cancel without a specific problem."
    ),
    "praise_or_other": (
        "Positive feedback, compliment, unrelated question, spam, "
        "clearly off-topic messages."
    ),
}

# Few-shot examples for classifier
FEW_SHOT_EXAMPLES = [
    ("songs keep skipping every 30 seconds on shuffle", "playback_issue"),
    ("why is my spotify premium still not working after i paid", "billing_query"),
    ("i literally cannot log into my account it just says wrong password", "account_access"),
    ("why isn't the new Olivia Rodrigo album on spotify yet", "content_request"),
    ("the android app is draining my battery so fast", "app_performance"),
    ("spotify is trash i'm switching to apple music", "general_complaint"),
    ("love the new daily mix feature!", "praise_or_other"),
    ("my music stops whenever i get a call and won't resume", "playback_issue"),
    ("i was charged $9.99 twice this month", "billing_query"),
    ("my account got hacked and someone changed my email", "account_access"),
]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    customer_text: str
    intent: str
    intent_confidence: float         # 0.0 – 1.0
    draft_reply: str
    escalate: bool
    escalation_reason: str           # empty string if not escalating
    retrieved_examples: list[dict]   # top-k RAG hits

    def to_dict(self):
        return asdict(self)


# ── Intent Classifier ─────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """\
You are an intent classifier for SpotifyCares customer support.
Classify the customer message into EXACTLY ONE of these intents:

{intent_list}

Respond ONLY with valid JSON in this exact format:
{{"intent": "<intent_name>", "confidence": <0.0-1.0>}}

No explanation, no markdown, just the JSON object.
"""

def _build_intent_list():
    lines = []
    for name, desc in INTENTS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _build_few_shot_messages():
    msgs = []
    for text, label in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": json.dumps({"intent": label, "confidence": 0.95})})
    return msgs


def classify_intent(customer_text: str) -> tuple[str, float]:
    """Returns (intent_label, confidence)."""
    system = CLASSIFIER_SYSTEM.format(intent_list=_build_intent_list())
    messages = [{"role": "system", "content": system}]
    messages += _build_few_shot_messages()
    messages.append({"role": "user", "content": customer_text})

    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=60,
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise last_error or RuntimeError("LLM request failed")

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        intent = data.get("intent", "general_complaint")
        confidence = float(data.get("confidence", 0.5))
        if intent not in INTENTS:
            intent = "general_complaint"
    except (json.JSONDecodeError, ValueError):
        # fallback: try to find intent name in response
        for name in INTENTS:
            if name in raw:
                return name, 0.5
        intent, confidence = "general_complaint", 0.3

    return intent, confidence


# ── Reply Drafter ─────────────────────────────────────────────────────────────

DRAFTER_SYSTEM = """\
You are a SpotifyCares support agent replying on Twitter.
Your tone: friendly, empathetic, concise (Twitter-length preferred), action-oriented.
You never promise what you can't deliver. You always provide a next step.
If the issue is complex, acknowledge it and direct to the right channel.
Do NOT use generic fillers like "Great question!" or "I totally understand!".
Write as if you are a senior human support agent, not a bot.
Keep reply under 240 characters when possible.
"""

DRAFTER_USER_TEMPLATE = """\
Customer message: {customer_text}

Classified intent: {intent}

Here are {k} similar past resolutions from SpotifyCares:
{examples}

Draft a reply to the customer message above, grounded in how SpotifyCares has resolved similar issues.
"""


def draft_reply(customer_text: str, intent: str, retrieved: list[dict]) -> str:
    """Draft a reply using RAG-retrieved examples."""
    examples_str = ""
    for i, ex in enumerate(retrieved, 1):
        examples_str += f"\n[Example {i}]\nCustomer: {ex['customer_text']}\nSpotify: {ex['brand_reply']}\n"

    user_content = DRAFTER_USER_TEMPLATE.format(
        customer_text=customer_text,
        intent=intent,
        k=len(retrieved),
        examples=examples_str or "No similar examples found.",
    )

    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DRAFTER_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise last_error or RuntimeError("LLM request failed")
    return resp.choices[0].message.content.strip()


# ── Escalation Decider ────────────────────────────────────────────────────────

ESCALATION_SYSTEM = """\
You are a customer support triage system for SpotifyCares.
Decide if this customer message should be ESCALATED to a human agent or AUTO-HANDLED by the bot.

Escalate if ANY of these are true:
- Legal threat or mention of legal action
- Threat of self-harm or safety concern
- Account fraud or security breach (hacked, stolen, unauthorised access)
- Extreme anger with aggressive language
- Payment dispute over $50 or involving bank/card fraud
- Repeated contact (customer says "contacted multiple times", "still not resolved")
- Ambiguous or multi-faceted issue that would require investigation
- Intent confidence below 0.45 (very unclear message)

Otherwise: AUTO-HANDLE.

Respond ONLY with valid JSON:
{{"escalate": true/false, "reason": "<one sentence reason, empty string if not escalating>"}}
"""


def decide_escalation(
    customer_text: str,
    intent: str,
    intent_confidence: float,
    retrieved: list[dict],
) -> tuple[bool, str]:
    """Returns (should_escalate, reason)."""

    # Hard rule: very low confidence → always escalate
    if intent_confidence < 0.35:
        return True, f"Low intent confidence ({intent_confidence:.2f}) — message unclear"

    # Hard rule: no similar past resolutions found
    if not retrieved or (retrieved and retrieved[0]["score"] < 0.55):
        escalate_no_context = True
    else:
        escalate_no_context = False

    best_similarity = retrieved[0]["score"] if retrieved else 0.0

    user_content = f"""\
Customer message: {customer_text}

Classified intent: {intent}
Intent confidence: {intent_confidence:.2f}
Similar past resolutions found: {len(retrieved)} (best similarity: {best_similarity:.2f})
No-context flag: {escalate_no_context}
"""

    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ESCALATION_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise last_error or RuntimeError("LLM request failed")

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        escalate = bool(data.get("escalate", False))
        reason = str(data.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        escalate = escalate_no_context
        reason = "Parsing error — defaulting to no-context decision"

    return escalate, reason


# ── Main Agent Entry Point ────────────────────────────────────────────────────

def run_agent(customer_text: str) -> AgentResult:
    """Run the full agent pipeline on a single customer message."""
    from retrieval import retrieve

    # Step 1: Classify intent
    intent, confidence = classify_intent(customer_text)

    # Step 2: Retrieve similar past threads
    retrieved = retrieve(customer_text, k=3)

    # Step 3: Draft reply
    reply = draft_reply(customer_text, intent, retrieved)

    # Step 4: Decide escalation
    escalate, reason = decide_escalation(customer_text, intent, confidence, retrieved)

    return AgentResult(
        customer_text=customer_text,
        intent=intent,
        intent_confidence=confidence,
        draft_reply=reply,
        escalate=escalate,
        escalation_reason=reason,
        retrieved_examples=retrieved,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "my spotify keeps crashing whenever i try to play a song"

    print(f"\n📩 Customer: {text}\n")
    result = run_agent(text)
    print(f"🏷️  Intent:   {result.intent} (conf: {result.intent_confidence:.2f})")
    print(f"💬 Reply:    {result.draft_reply}")
    print(f"🚨 Escalate: {result.escalate}")
    if result.escalation_reason:
        print(f"   Reason:  {result.escalation_reason}")
