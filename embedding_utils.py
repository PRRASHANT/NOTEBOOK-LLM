"""Generate document embeddings for semantic retrieval."""

from collections.abc import Sequence
from typing import Protocol


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def embed_chunks(chunks: Sequence[str], embedder: DocumentEmbedder) -> list[list[float]]:
    """Convert nonempty text chunks to equally sized embedding vectors."""
    if not chunks:
        return []
    if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        raise ValueError("Each chunk must contain nonempty text.")

    vectors = embedder.embed_documents(list(chunks))
    if len(vectors) != len(chunks):
        raise ValueError("The embedding model returned an unexpected number of vectors.")
    if not vectors[0]:
        raise ValueError("The embedding model returned an empty vector.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding vectors must all have the same dimension.")
    return vectors
