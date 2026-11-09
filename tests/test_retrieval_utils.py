"""Tests for context-aware document retrieval queries."""

import unittest

from retrieval_utils import build_retrieval_query


class RetrievalQueryTests(unittest.TestCase):
    def test_first_question_is_unchanged(self):
        self.assertEqual(build_retrieval_query("  What is RAG?  "), "What is RAG?")

    def test_standalone_question_does_not_use_old_topic(self):
        self.assertEqual(
            build_retrieval_query("What is vector search?", ["What is RAG?"]),
            "What is vector search?",
        )

    def test_pronoun_followup_includes_previous_question(self):
        self.assertEqual(
            build_retrieval_query("What are its limitations?", ["What is RAG?"]),
            "What is RAG?\nFollow-up question: What are its limitations?",
        )

    def test_short_followup_includes_previous_question(self):
        self.assertEqual(
            build_retrieval_query("And the benefits?", ["Explain vector databases."]),
            "Explain vector databases.\nFollow-up question: And the benefits?",
        )

    def test_chained_followup_uses_standalone_anchor(self):
        self.assertEqual(
            build_retrieval_query("And the drawbacks?", ["What is RAG?", "What are its benefits?"]),
            "What is RAG?\nFollow-up question: And the drawbacks?",
        )

    def test_context_is_limited_to_recent_turns(self):
        self.assertEqual(
            build_retrieval_query("What about its costs?", ["Old topic", "Older topic", "Another topic", "Recent topic"]),
            "Recent topic\nFollow-up question: What about its costs?",
        )

    def test_anchor_is_bounded(self):
        self.assertEqual(
            build_retrieval_query("What about it?", ["A" * 900]),
            "A" * 500 + "\nFollow-up question: What about it?",
        )

    def test_invalid_current_question_is_rejected(self):
        for value in (None, "", " \n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_retrieval_query(value)

    def test_invalid_relevant_history_is_rejected(self):
        for history in ([None], ["  "], ["Topic", None]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                build_retrieval_query("What about it?", history)

    def test_followup_without_history_remains_unchanged(self):
        self.assertEqual(build_retrieval_query("And its limitations?"), "And its limitations?")


if __name__ == "__main__":
    unittest.main()
