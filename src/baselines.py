#!/usr/bin/env python3
"""
baselines.py
------------
Two baselines for intent classification comparison:
  1. Trivial: keyword matching (no LLM)
  2. Simple: zero-shot LLM with no examples or definitions
"""

import json
import os
import re
from openai import OpenAI


def get_llm_client():
    """Return an OpenAI-compatible client using the active provider."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        preferred = os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
        models = [preferred]
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), models

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key), ["gpt-4o-mini"]

    raise RuntimeError("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY.")

INTENTS = [
    "playback_issue",
    "account_access",
    "billing_query",
    "content_request",
    "app_performance",
    "general_complaint",
    "praise_or_other",
]

# ── Baseline 1: Keyword Matching ──────────────────────────────────────────────

KEYWORD_MAP = {
    "playback_issue": [
        "skip", "won't play", "wont play", "not playing", "stops", "pauses",
        "shuffle", "buffering", "audio", "song stops", "music stops", "no sound",
        "keeps stopping", "freezing", "play button", "queue",
    ],
    "account_access": [
        "log in", "login", "password", "can't access", "locked", "sign in",
        "signin", "forgot", "email", "username", "two factor", "2fa", "hacked",
        "account", "verify",
    ],
    "billing_query": [
        "charged", "charge", "payment", "refund", "subscription", "premium",
        "billing", "invoice", "price", "cost", "money", "card", "bank",
        "free trial", "cancel",
    ],
    "content_request": [
        "not available", "missing", "where is", "why isn't", "can't find",
        "add", "request", "podcast", "album", "song not", "playlist",
        "region", "country",
    ],
    "app_performance": [
        "crash", "slow", "battery", "lag", "freeze", "bug", "glitch",
        "update", "install", "download", "android", "iphone", "ios", "mac",
        "windows", "desktop",
    ],
    "praise_or_other": [
        "love", "great", "amazing", "thanks", "thank you", "awesome",
        "best app", "well done",
    ],
}


def trivial_classify(text: str) -> str:
    """Keyword-matching baseline. Returns the intent with most keyword hits."""
    text_lower = text.lower()
    scores = {intent: 0 for intent in INTENTS}

    for intent, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1

    best = max(scores, key=scores.get)
    # If no keywords matched, default to general_complaint
    if scores[best] == 0:
        return "general_complaint"
    return best


# ── Baseline 2: Zero-Shot LLM ─────────────────────────────────────────────────

ZERO_SHOT_SYSTEM = """\
Classify the customer message into one of these categories:
playback_issue, account_access, billing_query, content_request,
app_performance, general_complaint, praise_or_other.

Reply ONLY with the category name and nothing else.
"""


def zeroshot_classify(text: str) -> str:
    """Zero-shot LLM baseline — no examples, no definitions."""
    client, model_candidates = get_llm_client()
    last_error = None
    for model in model_candidates:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ZERO_SHOT_SYSTEM},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=20,
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise last_error or RuntimeError("LLM request failed")
    raw = resp.choices[0].message.content.strip().lower().replace(" ", "_")
    # Find closest match
    for intent in INTENTS:
        if intent in raw:
            return intent
    return "general_complaint"


if __name__ == "__main__":
    examples = [
        "songs keep skipping every 30 seconds on shuffle",
        "why was i charged twice this month",
        "the app crashes every time i open it",
        "can't log into my account",
        "spotify is trash i hate it",
    ]
    print(f"{'Text':<45} {'Trivial':<20} {'ZeroShot':<20}")
    print("-" * 85)
    for ex in examples:
        t = trivial_classify(ex)
        z = zeroshot_classify(ex)
        print(f"{ex[:44]:<45} {t:<20} {z:<20}")
