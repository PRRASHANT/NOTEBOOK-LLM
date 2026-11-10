"""Integration checks against the installed native FAISS implementation."""

import importlib.util
import unittest

import numpy as np

from search_utils import build_search_index, search_chunks


class FixedQueryEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, text):
        return self.vector


@unittest.skipUnless(importlib.util.find_spec("faiss") is not None, "faiss-cpu is not installed")
class NativeFaissIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = ["cats", "dogs", "birds"]
        self.vectors = [[4.0, 0.0], [0.0, 7.0], [-2.0, 0.0]]

    def test_builds_native_inner_product_index(self):
        import faiss

        index = build_search_index(self.chunks, self.vectors)
        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertEqual(index.ntotal, len(self.chunks))
        self.assertEqual(index.d, 2)
        np.testing.assert_allclose(index.reconstruct(0), [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(index.reconstruct(1), [0.0, 1.0], atol=1e-6)

    def test_search_ranks_by_cosine_similarity(self):
        index = build_search_index(self.chunks, self.vectors)
        results = search_chunks(index, self.chunks, "pet", FixedQueryEmbedder([0.0, 5.0]), k=3)
        self.assertEqual([text for text, _ in results][0], "dogs")
        self.assertAlmostEqual(results[0][1], 1.0, places=6)
        self.assertAlmostEqual(results[1][1], 0.0, places=6)
        self.assertAlmostEqual(results[2][1], 0.0, places=6)

    def test_requested_results_are_limited_to_index_size(self):
        index = build_search_index(self.chunks, self.vectors)
        results = search_chunks(index, self.chunks, "pet", FixedQueryEmbedder([0.0, 5.0]), k=100)
        self.assertEqual(len(results), 3)
        self.assertEqual({text for text, _ in results}, set(self.chunks))


if __name__ == "__main__":
    unittest.main()
