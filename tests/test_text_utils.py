"""Tests for document text chunking."""

import unittest

from text_utils import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_empty_or_whitespace_only_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("  \n  "), [])

    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("hello", chunk_size=10, chunk_overlap=2), ["hello"])

    def test_overlapping_chunks_preserve_text(self):
        text = "abcdefghijklmnop"
        chunks = chunk_text(text, chunk_size=7, chunk_overlap=2)
        self.assertEqual(chunks, ["abcdefg", "fghijkl", "klmnop"])
        self.assertEqual(chunks[0] + "".join(c[2:] for c in chunks[1:]), text)

    def test_no_overlap(self):
        self.assertEqual(chunk_text("abcdefghij", chunk_size=4, chunk_overlap=0),
                         ["abcd", "efgh", "ij"])

    def test_chunk_length_is_bounded(self):
        chunks = chunk_text("a" * 2401)
        self.assertTrue(all(0 < len(chunk) <= 1000 for chunk in chunks))
        self.assertEqual(len(chunks), 3)

    def test_invalid_configuration(self):
        for size, overlap in [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_text("some text", size, overlap)


if __name__ == "__main__":
    unittest.main()
