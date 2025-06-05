# llm_context_os/tests/test_chat_history_retriever.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import time
import os
import shutil # For cleaning up any potential dummy db paths if needed for a specific test

from llm_context_os.retriever.chat_history import (
    ChatHistoryRetriever,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    CHROMADB_AVAILABLE,
    TIKTOKEN_AVAILABLE, # Import to allow tests to know if tiktoken would be default
    BasicCharTokenizer # In case we want to test this fallback specifically
)

# Mock tiktoken globally for tests if it's part of the fallback logic we want to control
# This is only needed if we want to test the fallback logic within ChatHistoryRetriever's __init__
# when TIKTOKEN_AVAILABLE is True but we want to simulate its absence for a specific test case.
# For most tests, we'll provide a mock_tokenizer directly.
# For testing the __init__ fallback:
# @patch('llm_context_os.retriever.chat_history.TIKTOKEN_AVAILABLE', False) # Example for one test

@unittest.skipUnless(SENTENCE_TRANSFORMERS_AVAILABLE, "SentenceTransformers library not available, skipping ChatHistoryRetriever tests.")
@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB library not available, skipping ChatHistoryRetriever tests.")
@patch('llm_context_os.retriever.chat_history.SentenceTransformer')
@patch('llm_context_os.retriever.chat_history.chromadb.PersistentClient')
class TestChatHistoryRetriever(unittest.TestCase):

    def setUp(self, MockChromaDBClient, MockSentenceTransformer): # Mocks passed by class decorators
        # Configure the mock instances returned by the patched constructors
        self.mock_sentence_transformer_instance = MockSentenceTransformer.return_value
        self.mock_sentence_transformer_instance.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3] # Default dummy embedding

        self.mock_chromadb_client_instance = MockChromaDBClient.return_value
        self.mock_collection = MagicMock()
        self.mock_collection.count.return_value = 0 # Default: empty collection
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        self.mock_tokenizer = MagicMock()
        # Define side effect for tokenizer.encode to simulate token counting via len(encode())
        def mock_encode_for_len(text_input):
            return [1] * len(text_input.split()) # Returns list of N items, N=word count
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)

        # Path for PersistentClient, ensure it's cleaned up if it were real
        self.test_db_path = "data/test_vector_dbs/test_chat_history_db"
        # For mocked tests, actual creation isn't essential, but good practice if any part isn't mocked
        # os.makedirs(self.test_db_path, exist_ok=True)


        self.retriever = ChatHistoryRetriever(
            recall_budget_tokens=50,
            tokenizer=self.mock_tokenizer,
            vector_db_path=os.path.dirname(self.test_db_path), # Pass the base 'data/test_vector_dbs'
            embedding_model_name='fake_embedding_model',
            collection_name='test_chat_collection'
        )
        # Ensure mocks are reset for each test method if they are class members being modified
        self.mock_sentence_transformer_instance.reset_mock()
        self.mock_chromadb_client_instance.reset_mock()
        self.mock_collection.reset_mock()
        self.mock_tokenizer.reset_mock()

        # Re-apply default return values that might be cleared by reset_mock if instance is shared across tests
        # (though setUp runs for each test, so instances are new, but their .return_value might need resetting if complex)
        MockSentenceTransformer.return_value = self.mock_sentence_transformer_instance
        MockChromaDBClient.return_value = self.mock_chromadb_client_instance
        self.mock_sentence_transformer_instance.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)


    def test_initialization_successful(self, MockChromaDBClient, MockSentenceTransformer):
        # Retriever is initialized in setUp. We check if mocks were called as expected.
        MockSentenceTransformer.assert_called_with('fake_embedding_model')

        # The path passed to PersistentClient is os.path.join('data/test_vector_dbs', "chat_history_db")
        # So, check that the base path 'data/test_vector_dbs/chat_history_db' was used.
        expected_db_full_path = os.path.join(os.path.dirname(self.test_db_path), ChatHistoryRetriever.DEFAULT_DB_SUBDIR)
        MockChromaDBClient.assert_called_with(path=expected_db_full_path)

        self.mock_chromadb_client_instance.get_or_create_collection.assert_called_with(name='test_chat_collection')
        self.assertIsNotNone(self.retriever.embedding_model)
        self.assertIsNotNone(self.retriever.collection)
        self.assertIs(self.retriever.tokenizer, self.mock_tokenizer)

    @patch('llm_context_os.retriever.chat_history.TIKTOKEN_AVAILABLE', False) # Force tiktoken to be unavailable
    def test_initialization_fallback_tokenizer(self, MockChromaDBClient, MockSentenceTransformer):
        # Test retriever's __init__ when no tokenizer is passed and tiktoken is unavailable
        retriever_fallback = ChatHistoryRetriever(
            vector_db_path=os.path.dirname(self.test_db_path),
            embedding_model_name='another_fake_model',
            tokenizer=None # Explicitly pass None
        )
        self.assertIsInstance(retriever_fallback.tokenizer, BasicCharTokenizer)

    def test_add_message_success(self, MockChromaDBClient, MockSentenceTransformer):
        msg_text = "A test message for adding."
        msg_role = "user"
        msg_id = str(uuid.uuid4())

        returned_id = self.retriever.add_message(msg_text, msg_role, message_id=msg_id)
        self.assertEqual(returned_id, msg_id)
        self.mock_sentence_transformer_instance.encode.assert_called_once_with(msg_text)
        self.mock_collection.add.assert_called_once_with(
            ids=[msg_id],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=[msg_text],
            metadatas=[{
                'role': msg_role,
                'timestamp': ANY, # time.time() is called, so use ANY
                'text_length_chars': len(msg_text),
                'source': 'chat_history'
            }]
        )

    def test_add_message_no_id_provided(self, MockChromaDBClient, MockSentenceTransformer):
        self.retriever.add_message("Another message", "assistant")
        self.mock_collection.add.assert_called_once()
        call_args = self.mock_collection.add.call_args
        self.assertTrue(uuid.UUID(call_args[1]['ids'][0])) # Check if the first ID is a valid UUID string

    def test_retrieve_simple_success(self, MockChromaDBClient, MockSentenceTransformer):
        mock_query_results = {
            'ids': [['id1', 'id2']],
            'documents': [['Document one text.', 'Document two text.']],
            'metadatas': [[{'role': 'user', 'timestamp': time.time() - 100},
                           {'role': 'assistant', 'timestamp': time.time() - 50}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 2 # Ensure collection is not empty

        query_text = "Relevant query"
        snippets = self.retriever.retrieve(query_text, n_results=2)

        self.mock_sentence_transformer_instance.encode.assert_called_with(query_text)
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )
        self.assertEqual(len(snippets), 2)
        self.assertIn("Document one text.", snippets[0]['c'])
        self.assertIn("Document two text.", snippets[1]['c'])
        self.assertEqual(snippets[0]['r'], 'retrieved_context')

    def test_retrieve_token_budget_respected(self, MockChromaDBClient, MockSentenceTransformer):
        self.retriever.recall_budget_tokens = 10 # Small budget: "Previously, role said (at X): "Y Z"" (approx 7 + Y + Z)
                                                 # if Y, Z are one word each.

        # Configure tokenizer: "Previously, user said: \"Doc one.\"" -> 6 words.
        # "Previously, assistant said: \"Doc two here.\"" -> 7 words.
        # "Previously, user said: \"Third doc is long.\"" -> 7 words.

        doc1_text = "Doc one." # formatted with prefix: ~6 tokens by word count
        doc2_text = "Doc two here." # ~7 tokens
        doc3_text = "Third doc is long." # ~7 tokens

        mock_query_results = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [[doc1_text, doc2_text, doc3_text]],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()},
                           {'role': 'assistant', 'timestamp': time.time()},
                           {'role': 'user', 'timestamp': time.time()}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 3

        # Define what mock_tokenizer.encode().len() returns for each formatted snippet
        # Snippet format: f"Previously, {role} said{timestamp_str}: \"{doc_text}\""
        # Words: "Previously," "role" "said" "timestamp_str:" "doc_text_words..."
        # Roughly 4 + len(doc_text.split())
        def custom_token_count_for_budget_test(text_input):
            if doc1_text in text_input: return 4 + len(doc1_text.split()) # 4 + 2 = 6
            if doc2_text in text_input: return 4 + len(doc2_text.split()) # 4 + 3 = 7
            if doc3_text in text_input: return 4 + len(doc3_text.split()) # 4 + 4 = 8
            return len(text_input.split()) # Fallback
        self.mock_tokenizer.encode = MagicMock(side_effect=lambda t: [1] * custom_token_count_for_budget_test(t))

        snippets = self.retriever.retrieve("Query for budgeting")

        self.assertEqual(len(snippets), 1) # Only first snippet (6 tokens) should fit budget of 10
        self.assertIn(doc1_text, snippets[0]['c'])


    def test_retrieve_no_results_from_db(self, MockChromaDBClient, MockSentenceTransformer):
        self.mock_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        self.mock_collection.count.return_value = 0 # Or >0 but query returns empty

        snippets = self.retriever.retrieve("Any query")
        self.assertEqual(len(snippets), 0)

    def test_retrieve_skips_identical_and_current_history(self, MockChromaDBClient, MockSentenceTransformer):
        query_text = "This is the query text."
        doc1_text_identical_to_query = query_text
        doc2_text_in_current_history = "This doc is in current history."
        doc3_text_unique = "This is a unique and relevant document."

        mock_query_results = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [[doc1_text_identical_to_query, doc2_text_in_current_history, doc3_text_unique]],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()}] * 3],
            'distances': [[0.1, 0.15, 0.2]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 3

        current_history = [{'r': 'user', 'c': doc2_text_in_current_history}]

        snippets = self.retriever.retrieve(query_text, current_chat_history=current_history)

        self.assertEqual(len(snippets), 1)
        self.assertIn(doc3_text_unique, snippets[0]['c'])

    def tearDown(self):
        # Clean up any directories created for tests if they were real
        if os.path.exists(os.path.dirname(self.test_db_path)):
             # This would remove the parent 'data/test_vector_dbs' if used directly.
             # Be cautious if self.test_db_path is not specific enough.
             # For fully mocked tests, this might not be strictly necessary unless some part of init bypasses mock.
             # For ChatHistoryRetriever, it creates vector_db_full_path = os.path.join(vector_db_path, self.DEFAULT_DB_SUBDIR)
             # So, self.test_db_path should be 'data/test_vector_dbs/test_chat_history_db'
             # and the base passed to retriever is 'data/test_vector_dbs'
             # The retriever will create 'data/test_vector_dbs/chat_history_db'
             # So, we should clean 'data/test_vector_dbs/chat_history_db' (self.retriever.vector_db_full_path)
             # or its parent if it's unique to this test run.
             # For now, assuming mocks prevent actual disk writes for these unit tests.
             pass


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
