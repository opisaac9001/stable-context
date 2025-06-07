# yawl/tests/test_retriever.py
import unittest
from yawl.retriever.chat_history import ChatHistoryRetriever
# For a simple tokenizer if needed. ChatHistoryRetriever uses its own SimpleCharTokenizer by default.
# from yawl.context.context_manager import MockTokenizer
from yawl.retriever.chat_history import SimpleCharTokenizer # Explicitly import if used

class TestChatHistoryRetriever(unittest.TestCase):

    def setUp(self):
        # Using the default SimpleCharTokenizer internal to ChatHistoryRetriever for these tests
        self.tokenizer = SimpleCharTokenizer()
        self.retriever = ChatHistoryRetriever(tokenizer=self.tokenizer, recall_budget_tokens=100)

    def test_instantiation(self):
        self.assertIsNotNone(self.retriever)
        self.assertEqual(self.retriever.recall_budget_tokens, 100)
        self.assertIsInstance(self.retriever.tokenizer, SimpleCharTokenizer)

    def test_add_message(self):
        initial_count = len(self.retriever.stored_messages)
        msg_id = self.retriever.add_message("Hello there", role="user")
        self.assertIsNotNone(msg_id)
        self.assertEqual(len(self.retriever.stored_messages), initial_count + 1)
        self.assertEqual(self.retriever.stored_messages[-1]['text'], "Hello there")
        self.assertEqual(self.retriever.stored_messages[-1]['role'], "user")
        self.assertEqual(self.retriever.stored_messages[-1]['id'], msg_id)

    def test_retrieve_empty(self):
        snippets = self.retriever.retrieve("any query")
        self.assertEqual(snippets, [])

    def test_retrieve_with_data(self):
        self.retriever.add_message("First message content", role="user")
        self.retriever.add_message("Second message for context", role="ai")
        snippets = self.retriever.retrieve("query about first")
        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0]['r'], "retrieved_context")
        # Placeholder formats as: f"...'{role}: {text[:50]}...'"
        # Ensure the assertion correctly reflects this including the role.
        original_text_obj = self.retriever.stored_messages[0] # The first message added
        expected_text_part = f"{original_text_obj['role']}: {original_text_obj['text'][:50]}"
        self.assertIn(expected_text_part, snippets[0]['c'])

    def test_retrieve_fixed_internal_truncation(self):
        """Tests the retriever's internal [:50] truncation of the original message text,
           when the overall budget is permissive."""
        self.retriever.recall_budget_tokens = 200 # Ensure overall budget doesn't interfere
        long_message = "This is a very long message that is well over fifty characters to test the internal text truncation feature of the placeholder."
        role = "user"
        self.retriever.add_message(long_message, role=role)

        snippets = self.retriever.retrieve("query for long message")
        self.assertEqual(len(snippets), 1)

        # Check that the original message text is truncated to 50 chars within the snippet
        expected_text_in_snippet = long_message[:50]
        full_expected_snippet_part = f"{role}: {expected_text_in_snippet}"
        self.assertIn(full_expected_snippet_part, snippets[0]['c'])

        # Check that the part of the message beyond 50 chars is NOT in the snippet
        # (unless the snippet formatting itself adds it, which it shouldn't for this part)
        self.assertNotIn(long_message[50:], snippets[0]['c'].replace(f"{role}: {expected_text_in_snippet}", "").replace("...", "")) # Be careful with "..."

        # Ensure the "(truncated)" suffix (from overall budget logic) is NOT present
        self.assertNotIn("(truncated)", snippets[0]['c'])

    def test_retrieve_overall_snippet_budget_truncation(self):
        """Tests the overall snippet truncation based on recall_budget_tokens."""
        short_message = "This is a test message, not too long." # len 36
        role = "user"
        self.retriever.add_message(short_message, role=role)

        # Calculate length of the prefix used by retriever for budgeting available_chars
        # Prefix in retriever: f"Retrieved (placeholder): First stored message was '{role}: ...'"
        # For role="user", this is "Retrieved (placeholder): First stored message was 'user: ...'" (len 55)
        fixed_prefix_len_for_calc = len(f"Retrieved (placeholder): First stored message was '{role}: ...'")

        # Set budget to force truncation of the *overall* snippet
        # Full snippet without outer truncation: "Retrieved (placeholder): First stored message was 'user: This is a test message, not too long....'"
        # Approx len: 51 (prefix part) + 36 (message) + 3 (...) = 90
        # Let's set budget to 70.
        self.retriever.recall_budget_tokens = 70

        snippets = self.retriever.retrieve("query for budget test")
        self.assertEqual(len(snippets), 1)

        available_chars_for_text = self.retriever.recall_budget_tokens - fixed_prefix_len_for_calc

        # This assertion helps debug if the test setup for budget is wrong
        self.assertTrue(available_chars_for_text > 0 and available_chars_for_text < len(short_message),
                        f"Test setup error: available_chars_for_text ({available_chars_for_text}) "
                        f"is not ideal for testing truncation. Message len: {len(short_message)}, "
                        f"Budget: {self.retriever.recall_budget_tokens}, Prefix len for calc: {fixed_prefix_len_for_calc}")

        expected_text_in_snippet = short_message[:available_chars_for_text]

        self.assertIn(f"{role}: {expected_text_in_snippet}", snippets[0]['c'])
        self.assertIn("(truncated)", snippets[0]['c'])

        # Check overall length is somewhat respected (it won't be exact due to "..." and "(truncated)")
        # The snippet text is f"Retrieved (placeholder): First stored message was '{role}: {text_slice}...' (truncated)"
        # Its length will be approx len_of_prefix_before_text_slice + len(text_slice) + len("...' (truncated)")
        expected_max_len = len(f"Retrieved (placeholder): First stored message was '{role}: ") + \
                           available_chars_for_text + \
                           len("...' (truncated)")
        self.assertLessEqual(len(snippets[0]['c']), expected_max_len + 2) # Allow a little slack
        self.assertGreater(len(snippets[0]['c']), fixed_prefix_len_for_calc)


if __name__ == '__main__':
    unittest.main()
