"""Tests for document-grounded answer generation."""

import unittest
from types import SimpleNamespace

from answer_utils import NO_ANSWER, answer_question


class FakeChatModel:
    def __init__(self, content="The answer is in the document. [1]"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class AnswerQuestionTests(unittest.TestCase):
    def test_builds_numbered_context_and_returns_answer(self):
        model = FakeChatModel()
        answer = answer_question("What is RAG?", [("RAG uses retrieval.", 0.9),
                                                   ("It provides context.", 0.8)], model)
        self.assertEqual(answer, "The answer is in the document. [1]")
        messages = model.calls[0]
        self.assertEqual([role for role, _ in messages], ["system", "human"])
        self.assertIn("[1] RAG uses retrieval.", messages[1][1])
        self.assertIn("[2] It provides context.", messages[1][1])
        self.assertIn("What is RAG?", messages[1][1])

    def test_prompts_for_grounded_answers_and_source_citations(self):
        model = FakeChatModel()
        answer_question("question", [("source", 0.7)], model)
        system = model.calls[0][0][1]
        self.assertIn("ONLY", system)
        self.assertIn(NO_ANSWER, system)
        self.assertIn("cite", system)
        self.assertIn("untrusted data", system)

    def test_no_passages_skips_model(self):
        model = FakeChatModel()
        self.assertEqual(answer_question("question", [], model), NO_ANSWER)
        self.assertEqual(model.calls, [])

    def test_rejects_blank_question(self):
        model = FakeChatModel()
        for question in ("", " \n ", None):
            with self.subTest(question=question), self.assertRaises(ValueError):
                answer_question(question, [("source", 0.5)], model)
        self.assertEqual(model.calls, [])

    def test_rejects_blank_passages(self):
        model = FakeChatModel()
        for text in ("", " \n ", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                answer_question("question", [(text, 0.5)], model)
        self.assertEqual(model.calls, [])

    def test_rejects_empty_model_output(self):
        for output in ("", "  ", None, ["unexpected"]):
            with self.subTest(output=output), self.assertRaises(ValueError):
                answer_question("question", [("source", 0.5)], FakeChatModel(output))

    def test_strips_answer_whitespace(self):
        self.assertEqual(answer_question("q", [("source", 0.5)],
                                         FakeChatModel("  Answer [1]\n")), "Answer [1]")


if __name__ == "__main__":
    unittest.main()
