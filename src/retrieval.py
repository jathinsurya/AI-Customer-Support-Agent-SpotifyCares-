#!/usr/bin/env python3
"""
retrieval.py
------------
Nearest-neighbour retrieval over historical Spotify support threads.
Used by the reply drafter to ground responses in real past resolutions.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = ROOT / "data" / "corpus_embeddings.pkl"

_index = None
_threads = None
_embeddings = None
_normed_embeddings = None
_backend = None
_vectorizer = None


def _load():
    global _index, _threads, _embeddings, _normed_embeddings, _backend, _vectorizer
    if _normed_embeddings is not None:
        return

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "data/corpus_embeddings.pkl not found. "
            "Run: python src/prepare_data.py"
        )

    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    _embeddings = data["embeddings"]  # shape (N, dim)
    _threads = data["threads"]
    _backend = data.get("backend", "openai")
    _vectorizer = data.get("vectorizer")

    if _embeddings.ndim != 2 or _embeddings.shape[0] == 0:
        raise ValueError("Retrieval corpus is empty. Run python src/prepare_data.py")

    # Normalise for cosine similarity
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    _normed_embeddings = (_embeddings / (norms + 1e-9)).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    if _backend == "tfidf":
        if _vectorizer is None:
            raise ValueError("TF-IDF vectorizer is missing from retrieval corpus")
        vec = _vectorizer.transform([text]).toarray().astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    from openai import OpenAI

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        raise RuntimeError("Groq does not provide the embedding model used by this project; use the local TF-IDF retrieval corpus.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.reshape(1, -1)


def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    Return the k most similar historical threads to the query.
    Each result: {"customer_text": ..., "brand_reply": ..., "score": float}
    """
    _load()
    vec = embed_query(query)
    scores = (_normed_embeddings @ vec.T).ravel()
    k = min(k, len(scores))
    indices = np.argpartition(scores, -k)[-k:]
    indices = indices[np.argsort(scores[indices])[::-1]]

    results = []
    for idx in indices:
        thread = _threads[idx]
        results.append({
            "customer_text": thread["customer_text"],
            "brand_reply": thread["brand_reply"],
            "score": float(scores[idx]),
        })
    return results


def corpus_size() -> Optional[int]:
    try:
        _load()
        return len(_threads)
    except FileNotFoundError:
        return None
