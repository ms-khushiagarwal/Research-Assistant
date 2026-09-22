"""Query-passage cross-encoder reranking after hybrid candidate retrieval.

Settings: RERANKER_MODEL (Hub ID or local directory), RERANKER_DEVICE
(default cpu), and RERANKER_BATCH_SIZE (default 8). Model inputs are truncated
to 512 tokens per pair; original passage content is preserved for generation.
Scores are ranking signals, not calibrated probabilities.
"""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def load_reranker(model_name, device):
    # Lazy import/loading: empty searches and ingestion do not load another model.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name, device=device, max_length=512,
        cache_folder=str(Path(__file__).resolve().parents[1] / "models" / "rerankers"),
        trust_remote_code=False,
    )


def rerank_results(query, candidates, top_k=5, model=None, batch_size=None):
    """Score all candidates before truncation; preserve source IDs and metadata.

    Ties retain hybrid retrieval order. Invalid model output fails explicitly so
    the caller can display a warning and use the original hybrid ranking.
    An injected model must implement predict(pairs, **kwargs).
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not query.strip() or not candidates:
        return []
    if batch_size is None:
        batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if model is None:
        model = load_reranker(os.getenv("RERANKER_MODEL", DEFAULT_MODEL),
                              os.getenv("RERANKER_DEVICE", "cpu"))
    pairs = [(query, candidate["content"]) for candidate in candidates]
    scores = np.asarray(model.predict(
        pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
    ), dtype=float)
    if scores.shape == (len(candidates), 1):
        scores = scores[:, 0]
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise ValueError("Reranker must return one finite relevance score per candidate.")
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:top_k]
    return [{**candidates[i], "rerank_score": float(scores[i]), "hybrid_rank": i + 1}
            for i in order]
