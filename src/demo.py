#!/usr/bin/env python3
"""
demo.py — Interactive SpotifyCares AI Agent demo
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))


def main():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No API key set. Add OPENAI_API_KEY or GROQ_API_KEY to your environment/.env")
        sys.exit(1)

    print("""
╔══════════════════════════════════════════════════════════════╗
║       🎵  SpotifyCares AI Support Agent — Live Demo         ║
║                                                              ║
║  Type a customer tweet/message and see the agent respond.   ║
║  Type 'quit' to exit.                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    sample_messages = [
        "my spotify keeps crashing every time i open it on iphone",
        "why was i charged $9.99 twice this month??",
        "I cant log into my account it says my password is wrong but i just changed it",
        "why isnt the new drake album on spotify",
        "spotify is absolute trash im switching to apple music",
    ]

    print("💡 Sample messages to try:")
    for i, msg in enumerate(sample_messages, 1):
        print(f"   {i}. {msg}")
    print()

    from agent import run_agent

    while True:
        try:
            raw = input("📩 Your message (or 1-5 for sample): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye! 🎵")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("\nBye! 🎵")
            break

        # Allow numeric shortcut
        if raw in ("1", "2", "3", "4", "5"):
            text = sample_messages[int(raw) - 1]
            print(f"   → {text}")
        else:
            text = raw

        print("\n⏳ Thinking...\n")

        try:
            result = run_agent(text)
        except Exception as e:
            print(f"❌ Agent error: {e}\n")
            continue

        # Format escalation badge
        if result.escalate:
            esc_display = f"🔴 ESCALATE — {result.escalation_reason}"
        else:
            esc_display = "🟢 AUTO-HANDLE"

        # Confidence bar
        conf_pct = int(result.intent_confidence * 20)
        conf_bar = "█" * conf_pct + "░" * (20 - conf_pct)

        print("┌─────────────────────────────────────────────────────────┐")
        print(f"│ 🏷️  INTENT:  {result.intent:<44} │")
        print(f"│    Conf:    [{conf_bar}] {result.intent_confidence:.0%}    │")
        print("├─────────────────────────────────────────────────────────┤")
        print(f"│ {esc_display:<59} │")
        print("├─────────────────────────────────────────────────────────┤")

        # Wrap reply at 55 chars
        reply = result.draft_reply
        words = reply.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= 55:
                current = (current + " " + word).strip()
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        print(f"│ 💬 DRAFT REPLY:                                         │")
        for line in lines:
            print(f"│   {line:<57} │")

        if result.retrieved_examples:
            print("├─────────────────────────────────────────────────────────┤")
            ex = result.retrieved_examples[0]
            sim_pct = int(ex['score'] * 100)
            similar_short = ex['customer_text'][:50]
            print(f"│ 📚 Similar case ({sim_pct}% match):                          │")
            print(f"│   \"{similar_short}...\"   │")

        print("└─────────────────────────────────────────────────────────┘\n")


if __name__ == "__main__":
    main()
