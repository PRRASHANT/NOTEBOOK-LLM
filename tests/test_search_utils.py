"""Tests for vector indexing and semantic retrieval."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from search_utils import build_search_index, search_chunks


class FakeIndexFlatIP:
    """Small in-memory stand-in for the FAISS inner-product index API."""

    def __init__(self, dimension):
        self.d = dimension
        self.ntotal = 0
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors):
        assert vectors.dtype == np.float32 and vectors.flags.c_contiguous
        self.vectors = np.concatenate((self.vectors, vectors))
        self.ntotal = len(self.vectors)

    def search(self, queries, k):
        similarities = queries @ self.vectors.T
        indices = np.argsort(-similarities, axis=1)[:, :k]
        return np.take_along_axis(similarities, indices, axis=1), indices


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.queries = []

    def embed_query(self, question):
        self.queries.append(question)
        return self.vector


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.faiss_patch = patch.dict(sys.modules, {"faiss": types.SimpleNamespace(IndexFlatIP=FakeIndexFlatIP)})
        self.faiss_patch.start()
        self.addCleanup(self.faiss_patch.stop)
        self.chunks = ["cats", "dogs", "birds"]
        self.vectors = [[5, 0], [0, 3], [-1, 0]]

    def test_builds_normalized_index(self):
        index = build_search_index(self.chunks, self.vectors)
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(index.d, 2)
        np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), [1, 1, 1])

    def test_returns_most_similar_chunk_first(self):
        index = build_search_index(self.chunks, self.vectors)
        model = FakeEmbedder([0, 10])
        results = search_chunks(index, self.chunks, "pet question", model, k=2)
        self.assertEqual(results[0][0], "dogs")
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertEqual(model.queries, ["pet question"])
        self.assertEqual(len(results), 2)

    def test_k_larger_than_index_returns_available_chunks(self):
        index = build_search_index(self.chunks, self.vectors)
        self.assertEqual(len(search_chunks(index, self.chunks, "q", FakeEmbedder([1, 0]), k=99)), 3)

    def test_rejects_invalid_index_inputs(self):
        cases = [([], []), (["a"], []), ([""], [[1]]),
                 (["a"], [[0, 0]]), (["a"], [[float("nan"), 1]]),
                 (["a", "b"], [[1, 2], [3]]), (["a"], [["no"]])]
        for chunks, vectors in cases:
            with self.subTest(chunks=chunks, vectors=vectors):
                with self.assertRaises(ValueError):
                    build_search_index(chunks, vectors)

    def test_rejects_bad_questions_and_query_vectors(self):
        index = build_search_index(self.chunks, self.vectors)
        for question, vector, k in [("", [1, 0], 3), ("q", [1, 0], 0),
                                    ("q", [1], 3), ("q", [0, 0], 3),
                                    ("q", [float("inf"), 0], 3)]:
            with self.subTest(question=question, vector=vector, k=k):
                with self.assertRaises(ValueError):
                    search_chunks(index, self.chunks, question, FakeEmbedder(vector), k=k)

    def test_rejects_mismatched_index_and_chunks(self):
        index = build_search_index(self.chunks, self.vectors)
        with self.assertRaises(ValueError):
            search_chunks(index, ["only one"], "q", FakeEmbedder([1, 0]))


if __name__ == "__main__":
    unittest.main()
