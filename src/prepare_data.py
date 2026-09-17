#!/usr/bin/env python3
"""
prepare_data.py
---------------
1. Load twcs.csv
2. Filter to SpotifyCares conversations
3. Build complete conversation threads (customer msg + brand reply pairs)
4. Embed all customer messages for RAG retrieval
5. Save spotify_threads.json and corpus_embeddings.pkl
"""

import os
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TWCS_PATH = DATA_DIR / "twcs.csv"
THREADS_PATH = DATA_DIR / "spotify_threads.json"
EMBEDDINGS_PATH = DATA_DIR / "corpus_embeddings.pkl"

BRAND_HANDLE = "SpotifyCares"
MAX_THREADS = 3000  # cap for speed; still very representative


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    print(f"📂 Loading {path.name} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"   {len(df):,} total rows, columns: {list(df.columns)}")
    return df


def build_threads(df: pd.DataFrame) -> list[dict]:
    """
    The dataset schema:
      tweet_id, author_id, inbound (True=customer), created_at, text,
      response_tweet_id, in_response_to_tweet_id
    """
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "yes"])

    # Spotify support tweets (outbound from brand)
    brand_tweets = df[
        (df["author_id"].fillna("").str.casefold() == BRAND_HANDLE.casefold())
        & (~df["inbound"])
    ].copy()

    # Map tweet_id → text for quick lookup
    id_to_row = df.set_index("tweet_id")

    threads = []
    seen_pairs = set()

    for _, brand_row in tqdm(brand_tweets.iterrows(), total=len(brand_tweets),
                              desc="Building threads"):
        parent_id = str(brand_row.get("in_response_to_tweet_id", ""))
        if not parent_id or parent_id == "nan":
            continue
        if parent_id not in id_to_row.index:
            continue

        customer_row = id_to_row.loc[parent_id]
        if not customer_row.get("inbound", False):
            continue  # parent wasn't a customer tweet

        pair_key = (parent_id, brand_row["tweet_id"])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        customer_text = str(customer_row.get("text", "")).strip()
        brand_reply = str(brand_row.get("text", "")).strip()

        # Skip very short or empty
        if len(customer_text) < 10 or len(brand_reply) < 10:
            continue

        # Clean @mentions at start
        def clean(t):
            import re
            t = re.sub(r"^(@\w+\s*)+", "", t).strip()
            return t

        threads.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": brand_row["tweet_id"],
            "customer_text": clean(customer_text),
            "brand_reply": clean(brand_reply),
            "created_at": str(brand_row.get("created_at", "")),
        })

        if len(threads) >= MAX_THREADS:
            break

    print(f"   Built {len(threads):,} conversation threads")
    return threads


def embed_texts(texts: list[str]) -> tuple[np.ndarray, object, str]:
    """Embed a list of texts using OpenAI's small embedding model."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        raise RuntimeError("Groq does not provide the OpenAI embedding model used by this project; falling back to TF-IDF.")

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    BATCH = 100
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i: i + BATCH]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        vecs = [item.embedding for item in resp.data]
        all_embeddings.extend(vecs)

    return np.array(all_embeddings, dtype=np.float32), None, "openai"


def tfidf_texts(texts: list[str]) -> tuple[np.ndarray, object, str]:
    """Create a local retrieval representation when embeddings are unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return matrix, vectorizer, "tfidf"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not TWCS_PATH.exists():
        print("❌ data/twcs.csv not found.")
        print("   Run: python scripts/download_data.py")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No API key set. Add OPENAI_API_KEY or GROQ_API_KEY to your environment/.env")
        sys.exit(1)

    # 1. Load
    df = load_raw(TWCS_PATH)

    # 2. Build threads
    threads = build_threads(df)
    if not threads:
        print(
            f"❌ No {BRAND_HANDLE} threads were found. "
            "Check the brand handle and dataset schema."
        )
        sys.exit(1)

    # 3. Save threads
    with open(THREADS_PATH, "w") as f:
        json.dump(threads, f, indent=2)
    print(f"✅ Saved {len(threads)} threads → data/spotify_threads.json")

    # 4. Embed customer texts
    print("\n🔢 Creating embeddings (this hits OpenAI API ~30 batches)...")
    customer_texts = [t["customer_text"] for t in threads]
    try:
        embeddings, vectorizer, backend = embed_texts(customer_texts)
    except Exception as exc:
        print(f"⚠️  OpenAI embeddings unavailable ({exc})")
        print("   Falling back to local TF-IDF retrieval.")
        embeddings, vectorizer, backend = tfidf_texts(customer_texts)

    # 5. Save embeddings
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "threads": threads,
            "backend": backend,
            "vectorizer": vectorizer,
        }, f)
    print(f"✅ Saved {backend} retrieval matrix ({embeddings.shape}) → data/corpus_embeddings.pkl")

    print("\n✅ Data preparation complete!")
    print("   Next: python src/build_eval_set.py")


if __name__ == "__main__":
    main()
