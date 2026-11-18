"""Tests for bounded document conversation state."""

import unittest

from conversation_utils import MAX_STORED_TURNS, append_conversation_turn


class ConversationUtilsTests(unittest.TestCase):
    def test_append_preserves_turns_below_limit(self):
        turns = []
        append_conversation_turn(turns, {"question": "first"})
        self.assertEqual(turns, [{"question": "first"}])

    def test_at_limit_drops_only_oldest(self):
        turns = [{"question": str(i)} for i in range(MAX_STORED_TURNS)]
        append_conversation_turn(turns, {"question": "new"})
        self.assertEqual(len(turns), MAX_STORED_TURNS)
        self.assertEqual(turns[0]["question"], "1")
        self.assertEqual(turns[-1]["question"], "new")

    def test_large_existing_history_is_trimmed(self):
        turns = [{"question": str(i)} for i in range(MAX_STORED_TURNS + 10)]
        append_conversation_turn(turns, {"question": "latest"})
        self.assertEqual(len(turns), MAX_STORED_TURNS)
        self.assertEqual(turns[0]["question"], "11")
        self.assertEqual(turns[-1]["question"], "latest")

    def test_mutates_original_list(self):
        turns = []
        original = turns
        append_conversation_turn(turns, {"answer": "ok"})
        self.assertIs(turns, original)


if __name__ == "__main__":
    unittest.main()
