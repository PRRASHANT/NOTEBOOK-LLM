"""Build a searchable index from extracted document text."""

from embedding_utils import embed_chunks
from search_utils import build_search_index
from text_utils import chunk_text


def index_document(text: str, embedder, *, chunk_size: int = 1000,
                   chunk_overlap: int = 200):
    """Return aligned text chunks, a FAISS index, and embedding dimensions."""
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Cannot index a document without selectable text.")

    vectors = embed_chunks(chunks, embedder)
    index = build_search_index(chunks, vectors)
    return chunks, index, len(vectors[0])
