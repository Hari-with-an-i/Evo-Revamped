"""
Lazy-loaded SBERT wrapper using all-MiniLM-L6-v2 (~80MB).

Model is downloaded on first call to get_sbert() — not at import time.
"""
from __future__ import annotations

_model = None


def get_sbert():
    """Return the cached SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cosine_similarities(query: str, texts: list[str]) -> list[float]:
    """Return cosine similarity between query and each text (embeddings L2-normalised)."""
    if not texts:
        return []
    model = get_sbert()
    q_emb = model.encode([query], normalize_embeddings=True)
    t_emb = model.encode(texts, normalize_embeddings=True)
    return (q_emb @ t_emb.T)[0].tolist()
