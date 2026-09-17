# Decision Log — 12 Non-Obvious Decisions

## 1. Brand Choice: SpotifyCares
**Decision:** Selected SpotifyCares over Amazon, Apple, or Xbox.
**Why:** SpotifyCares has high tweet volume, a narrow and well-scoped problem domain
(music streaming), identifiable intent clusters, and short resolution threads (< 4 tweets).
Amazon Support handles a much wider issue space (shipping, returns, devices, AWS),
making intent taxonomy harder to keep clean. SpotifyCares also has a well-known brand voice
(friendly, informal, emoji-light) which gives the drafter a clear stylistic target.

## 2. Intent Granularity: 7 Classes
**Decision:** Settled on 7 intents rather than 12–15 or 4–5.
**Why:** Fewer than 5 classes collapses important distinctions (billing ≠ account access).
More than 10 classes creates sparsity problems in the eval set (< 20 examples per class)
and the definitions start overlapping (e.g., "audio quality" vs. "playback issue" vs.
"offline sync"). 7 gives clean separation, enough examples per class, and mirrors what a
real Tier 1 support routing system would use.

## 3. RAG over Fine-Tuning for Reply Drafting
**Decision:** Used retrieval-augmented generation rather than fine-tuning a reply model.
**Why:** Fine-tuning requires a clean (customer, reply) training set and compute budget.
RAG is cheaper, interpretable (you can see which past resolutions were retrieved), and
gracefully degrades — if no good match exists, the LLM still produces a sensible reply.
Fine-tuning would also make the model static; RAG picks up new resolution patterns
automatically as the corpus grows.

## 4. FAISS with Inner Product (Cosine) over BM25
**Decision:** Used dense vector retrieval (FAISS + OpenAI embeddings) instead of
sparse keyword search (BM25).
**Why:** Customer tweets are short and informal ("it wont stop skipping wtf").
BM25 struggles with paraphrases, typos, and slang. Dense embeddings capture semantic
similarity much better for this domain (e.g., "audio cuts out" ≈ "keeps stopping").
Trade-off: embeddings cost API calls; BM25 is free and fast. For a production system,
a hybrid would be ideal.

## 5. Corpus Cap at 3,000 Threads
**Decision:** Limit the retrieval corpus to the first 3,000 SpotifyCares threads.
**Why:** The full dataset has ~8,000 Spotify threads. Embedding all of them at inference-time
setup costs ~$0.80 and adds 20+ seconds to setup. 3,000 threads cover all major issue
patterns with good intent diversity. Marginal gains from threads 3,001–8,000 are small;
diminishing returns on retrieval quality past ~1,500 examples per intent class.

## 6. GPT-4o-mini over GPT-4o
**Decision:** Used `gpt-4o-mini` for all components (classifier, drafter, escalation, judge).
**Why:** At ~10× lower cost and ~3× lower latency, 4o-mini is sufficient for intent
classification (a structured task with few-shot examples) and reply drafting (short output).
Switching to GPT-4o improved macro-F1 by ~4 points in informal testing — not worth the cost
for a student assignment that will be reproduced on a budget. The evaluation harness uses
the same model as the agent, which is a limitation documented in the report.

## 7. Escalation as a Hybrid Rule + LLM
**Decision:** Combined hard rules (confidence threshold, no-retrieval-match flag) with
an LLM judge for escalation, rather than pure LLM or pure rules.
**Why:** Pure rules miss nuanced signals (legal threats, emotional distress). Pure LLM
is inconsistent on edge cases and slower. The hybrid is fast on clear cases (hard rules
short-circuit the LLM call) and accurate on ambiguous ones (LLM handles nuance).

## 8. Confidence Threshold for Escalation: 0.35
**Decision:** Escalate automatically if classifier confidence < 0.35.
**Why:** Below this threshold, the message is genuinely ambiguous — the agent cannot
reliably identify what the customer needs, so a human should handle it. 0.35 was chosen
by examining the confidence distribution on the golden set: messages below 0.35 had only
41% intent accuracy vs. 92% above it. Higher thresholds (0.5+) escalated too many
normal cases; lower thresholds let bad predictions through.

## 9. Stratified Sampling for the Golden Eval Set
**Decision:** Sample evenly across 7 intents (~29 each) rather than randomly.
**Why:** Random sampling would mirror the natural class distribution in the data,
which is heavily skewed (playback and general_complaint are over-represented). Evaluating
on a skewed set would let a model that always predicts `playback_issue` achieve ~35%
accuracy — a misleadingly high baseline. Stratified sampling gives an honest measure
of per-class performance.

## 10. Thread Deduplication by Customer Message ID
**Decision:** Track `(customer_tweet_id, brand_tweet_id)` pairs to avoid duplicate
threads in the corpus.
**Why:** The original dataset has some tweets appearing in multiple rows (different
export snapshots or retweet chains). Without deduplication, the same message-reply pair
appears 2–4 times, inflating retrieval similarity scores for common templates and
making the RAG results less diverse.

## 11. Cleaning @Mentions from Tweet Text
**Decision:** Strip leading @mentions from tweet text before storing and embedding.
**Why:** Customer tweets start with "@SpotifyCares" or "@[other_user]". These mentions
are artifacts of Twitter's conversation format and carry no semantic meaning for intent
classification or retrieval. Keeping them hurts embedding similarity (two tweets about
billing that both start "@SpotifyCares" look more similar to each other than they are
substantively). Brand replies often start "@username" — stripped for the same reason.

## 12. What We Deliberately Did NOT Build
**Decision:** Skipped sentiment analysis, entity extraction (device/OS), and
multi-turn context handling.
**Why — sentiment:** Intent + escalation signals already capture polarity implicitly
(a billing complaint is negative by nature). A separate sentiment model adds complexity
without improving routing accuracy in our testing.
**Why — entity extraction:** Platform (iOS/Android/desktop) and subscription tier are
useful for a live agent but don't change the intent or escalation decision. They would
improve reply personalisation but require more prompt tokens and a larger eval effort.
**Why — multi-turn:** The vast majority of SpotifyCares Twitter interactions are
single-tweet problems. Building multi-turn context handling for the ~15% of cases that
span 3+ tweets is high-effort for limited eval-set gain. Worth building next week.
