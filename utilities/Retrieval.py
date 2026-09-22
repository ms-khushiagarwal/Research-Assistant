"""Exact cosine + BM25 retrieval for a local, in-memory snapshot of Chroma.

Both rankings cover the same corpus. Reciprocal rank fusion (RRF) combines
their ranks without pretending cosine and BM25 scores share a scale.
The snapshot is refreshed on each query so uploads are immediately searchable.
"""
from collections import Counter
import math
import re

import numpy as np


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


def bm25_scores(query, documents, k1=1.5, b=0.75):
    terms = [Counter(tokenize(document)) for document in documents]
    lengths = np.array([sum(row.values()) for row in terms], dtype=float)
    scores = np.zeros(len(documents), dtype=float)
    if not len(documents) or not lengths.sum():
        return scores
    average_length = lengths.mean()
    for term in set(tokenize(query)):
        frequency = np.array([row[term] for row in terms], dtype=float)
        document_frequency = np.count_nonzero(frequency)
        idf = math.log1p((len(documents) - document_frequency + 0.5)
                        / (document_frequency + 0.5))
        scores += idf * frequency * (k1 + 1) / (
            frequency + k1 * (1 - b + b * lengths / average_length)
        )
    return scores


def hybrid_retrieve(collection, query, embed_query, top_k=5, candidate_k=30,
                    dense_weight=0.5, rrf_k=60):
    """Return unique ranked chunks, with source metadata and component scores.

    Exact cosine is computed on stored vectors, including collections originally
    created with Chroma's default L2 metric. No destructive reindex is required.
    This scans all vectors and text; use an indexed lexical backend for large corpora.
    """
    if top_k < 1 or candidate_k < 1 or rrf_k < 0 or not 0 <= dense_weight <= 1:
        raise ValueError("Invalid hybrid retrieval settings.")
    if not query.strip():
        return []
    snapshot = collection.get(include=["documents", "metadatas", "embeddings"])
    rows = []
    seen = set()
    for index, chunk_id in enumerate(snapshot["ids"]):
        content = snapshot["documents"][index]
        metadata = (snapshot.get("metadatas") or [None] * len(snapshot["ids"]))[index] or {}
        if not content or not content.strip():
            continue
        key = (metadata.get("paper_path"), metadata.get("page_number"), content)
        if key in seen:
            continue
        seen.add(key)
        rows.append((chunk_id, content, metadata, snapshot["embeddings"][index]))
    if not rows:
        return []
    vectors = np.asarray([row[3] for row in rows], dtype=float)
    query_vector = np.asarray(embed_query(query), dtype=float)
    if vectors.ndim != 2 or query_vector.shape != (vectors.shape[1],):
        raise ValueError("Embedding dimensions differ. Re-ingest PDFs with the current text model.")
    if not np.isfinite(vectors).all() or not np.isfinite(query_vector).all():
        raise ValueError("Embeddings contain nonfinite values.")
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    cosine = np.divide(vectors @ query_vector, denominator,
                       out=np.zeros(len(rows)), where=denominator > 0)
    lexical = bm25_scores(query, [row[1] for row in rows])
    limit = min(len(rows), max(top_k, candidate_k))
    dense_order = sorted(range(len(rows)), key=lambda i: (-cosine[i], rows[i][0]))[:limit]
    lexical_order = sorted((i for i in range(len(rows)) if lexical[i] > 0),
                           key=lambda i: (-lexical[i], rows[i][0]))[:limit]
    fused = {}
    for weight, ranking in ((dense_weight, dense_order), (1 - dense_weight, lexical_order)):
        if weight == 0:
            continue
        for rank, index in enumerate(ranking, 1):
            fused[index] = fused.get(index, 0.0) + weight / (rrf_k + rank)
    selected = sorted(fused, key=lambda i: (-fused[i], -cosine[i], rows[i][0]))[:top_k]
    return [{"id": rows[i][0], "content": rows[i][1], "metadata": rows[i][2],
             "score": fused[i], "cosine_similarity": float(cosine[i]),
             "bm25_score": float(lexical[i])} for i in selected]
