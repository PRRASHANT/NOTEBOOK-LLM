"""Tests for document embedding preparation."""

import unittest

from embedding_utils import embed_chunks


class FakeEmbedder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(texts)
        if self.result is not None:
            return self.result
        return [[float(len(text)), 1.0] for text in texts]


class EmbedChunksTests(unittest.TestCase):
    def test_embeds_each_chunk_in_order(self):
        model = FakeEmbedder()
        result = embed_chunks(["one", "longer"], model)
        self.assertEqual(result, [[3.0, 1.0], [6.0, 1.0]])
        self.assertEqual(model.calls, [["one", "longer"]])

    def test_empty_input_skips_model(self):
        model = FakeEmbedder()
        self.assertEqual(embed_chunks([], model), [])
        self.assertEqual(model.calls, [])

    def test_rejects_blank_or_nonstring_chunks(self):
        for chunks in (["ok", ""], ["  "], ["ok", None]):
            with self.subTest(chunks=chunks):
                with self.assertRaises(ValueError):
                    embed_chunks(chunks, FakeEmbedder())

    def test_rejects_missing_vectors(self):
        with self.assertRaisesRegex(ValueError, "unexpected number"):
            embed_chunks(["first", "second"], FakeEmbedder([[1.0]]))

    def test_rejects_empty_vectors(self):
        with self.assertRaisesRegex(ValueError, "empty vector"):
            embed_chunks(["hello"], FakeEmbedder([[]]))

    def test_rejects_inconsistent_dimensions(self):
        with self.assertRaisesRegex(ValueError, "same dimension"):
            embed_chunks(["one", "two"], FakeEmbedder([[1.0, 2.0], [3.0]]))


if __name__ == "__main__":
    unittest.main()
