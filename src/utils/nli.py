"""
Lazy-loaded NLI wrapper using cross-encoder/nli-deberta-v3-small (~550MB).

Model is downloaded on first call to get_nli() — not at import time.
Score order from the model is always [contradiction, entailment, neutral].
"""
from __future__ import annotations

_model = None
LABELS = ["contradiction", "entailment", "neutral"]


def get_nli():
    """Return the cached CrossEncoder model, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # type: ignore[import]
        _model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _model


def score_pairs(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """
    Score NLI pairs and return per-pair label probabilities.

    Args:
        pairs: list of (premise, hypothesis) string tuples.

    Returns:
        List of dicts with keys "contradiction", "entailment", "neutral".
    """
    if not pairs:
        return []
    scores = get_nli().predict(pairs)  # shape: (N, 3)
    return [dict(zip(LABELS, row.tolist())) for row in scores]
