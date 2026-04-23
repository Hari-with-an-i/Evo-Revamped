"""
Lazy-loaded RoBERTa sentiment wrapper using cardiffnlp/twitter-roberta-base-sentiment (~480MB).

Model is downloaded on first call — not at import time.
Label mapping: LABEL_0=negative (-score), LABEL_1=neutral (0.0), LABEL_2=positive (+score).
"""
from __future__ import annotations

_pipeline = None


def get_sentiment_pipeline():
    """Return the cached sentiment pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline  # type: ignore[import]
        _pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
            truncation=True,
            max_length=512,
        )
    return _pipeline


def score_sentiment(texts: list[str]) -> list[float]:
    """
    Score a list of text strings. Returns floats in [-1.0, +1.0].
    Empty input returns []. Handles tokenizer truncation internally.
    """
    if not texts:
        return []
    pipe = get_sentiment_pipeline()
    results = pipe(texts, batch_size=16)
    scores: list[float] = []
    for r in results:
        label: str = r["label"]
        conf: float = r["score"]
        if label == "LABEL_2":    # positive
            scores.append(conf)
        elif label == "LABEL_0":  # negative
            scores.append(-conf)
        else:                     # LABEL_1 neutral
            scores.append(0.0)
    return scores
