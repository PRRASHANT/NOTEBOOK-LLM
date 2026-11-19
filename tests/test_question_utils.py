"""Tests for question validation boundaries."""

import unittest

from question_utils import MAX_QUESTION_CHARACTERS, validate_question


class QuestionValidationTests(unittest.TestCase):
    def test_trims_question(self):
        self.assertEqual(validate_question("  What is FAISS?  "), "What is FAISS?")

    def test_rejects_blank_question(self):
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            validate_question(" \n\t ")

    def test_accepts_exact_limit(self):
        question = "x" * MAX_QUESTION_CHARACTERS
        self.assertEqual(validate_question(question), question)

    def test_rejects_over_limit(self):
        with self.assertRaisesRegex(ValueError, "2000-character limit"):
            validate_question("x" * (MAX_QUESTION_CHARACTERS + 1))

    def test_trims_before_checking_limit(self):
        question = " " + "x" * MAX_QUESTION_CHARACTERS + " "
        self.assertEqual(len(validate_question(question)), MAX_QUESTION_CHARACTERS)


if __name__ == "__main__":
    unittest.main()
