# Golden Evaluation Set — Labelling Note

## Set Size
**203 examples** (29 per intent × 7 intents = 203; final CSV may be ~200 after deduplication).

## Sampling Strategy

### Step 1 — Source Pool
All threads were drawn from `data/spotify_threads.json` — filtered SpotifyCares conversations
from the Kaggle Twitter Customer Support dataset. Total pool: ~3,000 threads.

### Step 2 — Stratification
A lightweight keyword classifier (`baselines.trivial_classify`) was run over all threads to
produce a rough intent bucket for each. This enabled **stratified sampling**: 29 examples were
drawn randomly (seed=42) from each of the 7 intent buckets, ensuring even coverage of all
intent classes. Without stratification, rare classes (e.g. `praise_or_other`) would be
under-represented.

### Step 3 — LLM Gold Labelling
Each sampled example (customer text + brand's actual reply) was sent to `gpt-4o-mini` acting
as an expert annotator. The annotator was given:
- The 7 intent definitions
- The actual brand reply (used to infer gold intent from context)
- Instructions to label: `gold_intent`, `gold_escalate`, `gold_reply_quality` (1–3)

### Step 4 — Human Spot-Check (50 examples)
50 of the 203 examples were manually reviewed by the author. Agreement with LLM labels:

| Label Type         | Human-LLM Agreement |
|--------------------|---------------------|
| Intent             | ~87%                |
| Escalation (Y/N)   | ~82%                |
| Reply quality      | ~76%                |

Disagreements were mostly on borderline cases (e.g. billing vs. account for "premium not
showing after payment"). The LLM labels were **not corrected** in these cases to avoid
introducing reviewer bias into the gold set — instead, this is captured as a limitation in
the report.

## Known Biases in This Eval Set

1. **LLM-generated gold labels**: the same model family (GPT-4o-mini) generates both gold
   labels and agent predictions. This inflates agreement metrics. A fully human-labelled set
   would produce lower (but more honest) accuracy numbers.

2. **Keyword-stratified sampling**: the stratification used keyword matching, not true intent.
   Some examples may be misplaced into wrong buckets during sampling, meaning the final
   per-class distribution is approximate.

3. **No multi-turn context**: each example is a single customer tweet. Real conversations often
   span multiple turns; this eval set underrepresents that complexity.

4. **Historical brand replies as context**: using the actual brand reply to determine gold
   intent is a mild form of label leakage. A cleaner approach would label intent from the
   customer tweet alone. We use the reply as a helpful disambiguation signal, which mirrors
   real annotator practice.

5. **English-only**: the dataset is filtered to English tweets only. Spotify has a global
   user base; non-English issues are entirely unrepresented.

## File Schema

| Column | Description |
|---|---|
| `example_id` | Integer ID (1–203) |
| `customer_text` | Cleaned customer tweet text |
| `brand_reply` | SpotifyCares' actual reply |
| `gold_intent` | Gold intent label (one of 7 classes) |
| `gold_escalate` | Should this be escalated? (True/False) |
| `gold_reply_quality` | Quality of brand's actual reply (1=poor, 2=ok, 3=good) |
| `gold_escalate_reason` | Reason for escalation (empty if not escalating) |
