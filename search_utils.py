"""In-memory cosine similarity search over document embeddings."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


def build_search_index(chunks: Sequence[str], vectors: Sequence[Sequence[float]]):
    """Build a FAISS index from embeddings aligned with the source chunks."""
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must have the same nonzero length.")
    if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        raise ValueError("Each chunk must contain nonempty text.")

    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError("Embeddings must contain numeric vectors.") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite, nonempty vectors of equal dimension.")

    lengths = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError("Embedding vectors must have nonzero, finite magnitude.")
    normalized = np.ascontiguousarray(matrix / lengths[:, None])

    import faiss

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(normalized)
    return index


def search_chunks(index, chunks: Sequence[str], question: str,
                  embedder: QueryEmbedder, k: int = 3) -> list[tuple[str, float]]:
    """Return the most relevant chunks with their cosine similarity scores."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Search question must not be blank.")
    if k < 1:
        raise ValueError("k must be positive.")
    if index.ntotal != len(chunks):
        raise ValueError("Index size does not match the number of chunks.")
    if not chunks:
        return []

    try:
        vector = np.asarray(embedder.embed_query(question), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Query embedding must be numeric.") from exc
    if vector.ndim != 1 or vector.shape[0] != index.d or not np.isfinite(vector).all():
        raise ValueError("Query embedding has invalid dimensions or values.")
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length == 0:
        raise ValueError("Query embedding must have nonzero, finite magnitude.")

    query = np.ascontiguousarray((vector / length).reshape(1, -1))
    scores, positions = index.search(query, min(k, len(chunks)))
    return [(chunks[int(i)], float(score)) for i, score in zip(positions[0], scores[0]) if i >= 0]
