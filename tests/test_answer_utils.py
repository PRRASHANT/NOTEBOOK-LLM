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


    def test_recent_history_is_included_but_not_treated_as_evidence(self):
        model = FakeChatModel()
        answer_question("What about its limitations?", [("The approach has limits.", 0.8)],
                        model, history=[("What is RAG?", "It combines retrieval and generation.")])
        system, user = (message for _, message in model.calls[0])
        self.assertIn("NOT evidence", system)
        self.assertIn("Previous question: What is RAG?", user)
        self.assertIn("Previous answer: It combines retrieval", user)
        self.assertIn("Document excerpts:\\n[1] The approach has limits.", user)

    def test_only_last_three_history_turns_are_sent(self):
        model = FakeChatModel()
        history = [(f"Question {i}", f"Answer {i}") for i in range(5)]
        answer_question("Follow-up", [("source", 0.5)], model, history=history)
        prompt = model.calls[0][1][1]
        self.assertNotIn("Question 0", prompt)
        self.assertNotIn("Question 1", prompt)
        for i in (2, 3, 4):
            self.assertIn(f"Question {i}", prompt)

    def test_rejects_invalid_history(self):
        for history in ([('valid', '')], [('valid', None)], [('missing answer',)],
                        [(' ', 'answer')]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                answer_question("Follow-up", [("source", 0.5)], FakeChatModel(), history)

    def test_history_is_truncated_to_bounded_size(self):
        model = FakeChatModel()
        answer_question("Follow-up", [("source", 0.5)], model,
                        history=[("Q" * 700, "A" * 2000)])
        prompt = model.calls[0][1][1]
        self.assertIn("Q" * 500, prompt)
        self.assertNotIn("Q" * 501, prompt)
        self.assertIn("A" * 1500, prompt)
        self.assertNotIn("A" * 1501, prompt)



if __name__ == "__main__":
    unittest.main()
