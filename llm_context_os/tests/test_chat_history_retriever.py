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
@patch('llm_context_os.retriever.chat_history.SentenceTransformer') # Patches ST for both bi-encoder and cross-encoder
@patch('llm_context_os.retriever.chat_history.chromadb.PersistentClient')
class TestChatHistoryRetriever(unittest.TestCase):

    def setUp(self, MockChromaDBClient, MockSentenceTransformer):
        self.MockSentenceTransformer = MockSentenceTransformer # Store the class mock
        self.MockChromaDBClient = MockChromaDBClient

        # Mock for the bi-encoder (embedding_model)
        self.mock_bi_encoder_instance = MagicMock()
        mock_bi_embedding_array = MagicMock()
        mock_bi_embedding_array.tolist.return_value = [0.1, 0.2, 0.3] # Default dummy embedding
        self.mock_bi_encoder_instance.encode.return_value = mock_bi_embedding_array

        # Mock for the cross-encoder (if initialized)
        self.mock_cross_encoder_instance = MagicMock()

        # SentenceTransformer class mock will provide these instances based on model name
        def sentence_transformer_side_effect(model_name_or_path):
            if model_name_or_path == 'fake_embedding_model':
                return self.mock_bi_encoder_instance
            elif model_name_or_path == 'fake-cross-encoder-model':
                return self.mock_cross_encoder_instance
            # Allow for None or other model names to pass through if not these specific ones,
            # though tests should specify one of these.
            # Or raise error for unexpected model name:
            # raise ValueError(f"Unexpected model name for SentenceTransformer mock in CHR test: {model_name_or_path}")
            # For now, let's assume tests will use these names or None for cross_encoder.
            # If a test uses a different name, it might get a simple MagicMock() from ST(), not configured.
            # This is fine as long as tests explicitly set up mocks for specific model names they use.
            return MagicMock() # Default mock for other names
        self.MockSentenceTransformer.side_effect = sentence_transformer_side_effect

        self.mock_chromadb_client_instance = self.MockChromaDBClient.return_value
        self.mock_collection = MagicMock()
        self.mock_collection.count.return_value = 0
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        self.mock_tokenizer = MagicMock()
        def mock_encode_for_len(text_input):
            return [1] * len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)
        self.mock_tokenizer.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))


        self.test_db_base_path = "data/test_vector_dbs" # Base path, retriever adds its own subdir
        self._reinit_retriever() # Initialize self.retriever

    def _reinit_retriever(self,
                          recall_budget=50,
                          cross_encoder_name='fake-cross-encoder-model',
                          rerank_top_n=5,
                          embedding_model_name='fake_embedding_model',
                          collection_name='test_chat_collection'):
        self.retriever = ChatHistoryRetriever(
            recall_budget_tokens=recall_budget,
            tokenizer=self.mock_tokenizer,
            vector_db_path=self.test_db_base_path,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            cross_encoder_model_name=cross_encoder_name,
            rerank_top_n_candidates=rerank_top_n
        )
        # Reset instance mocks that might be called by __init__ or other methods
        self.mock_bi_encoder_instance.reset_mock()
        self.mock_cross_encoder_instance.reset_mock()
        self.mock_chromadb_client_instance.reset_mock()
        self.mock_collection.reset_mock()
        self.mock_tokenizer.reset_mock()

        # Re-apply default return values for mocks
        # self.MockSentenceTransformer.side_effect is already set up
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        self.MockChromaDBClient.return_value = self.mock_chromadb_client_instance
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        def mock_encode_for_len(text_input): return [1] * len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)
        self.mock_tokenizer.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))


    def test_initialization_successful(self, MockChromaDBClient, MockSentenceTransformer):
        # Retriever is initialized in _reinit_retriever called by setUp.
        # Check if SentenceTransformer was called for bi-encoder and cross-encoder
        MockSentenceTransformer.assert_any_call('fake_embedding_model')
        MockSentenceTransformer.assert_any_call('fake-cross-encoder-model')

        self.assertIsNotNone(self.retriever.embedding_model)
        self.assertIsNotNone(self.retriever.cross_encoder)

        expected_db_full_path = os.path.join(self.test_db_base_path, ChatHistoryRetriever.DEFAULT_DB_SUBDIR)
        expected_db_full_path = os.path.join(os.path.dirname(self.test_db_path), ChatHistoryRetriever.DEFAULT_DB_SUBDIR)
        MockChromaDBClient.assert_called_with(path=expected_db_full_path)

        self.mock_chromadb_client_instance.get_or_create_collection.assert_called_with(name='test_chat_collection')
        self.assertIsNotNone(self.retriever.embedding_model)
        self.assertIsNotNone(self.retriever.collection)
        self.assertIs(self.retriever.tokenizer, self.mock_tokenizer)

    @patch('llm_context_os.retriever.chat_history.TIKTOKEN_AVAILABLE', False)
    def test_initialization_fallback_tokenizer(self, MockChromaDBClient, MockSentenceTransformer):
        # For this test, we need to re-init the retriever with tokenizer=None
        # and ensure the SentenceTransformer mock doesn't rely on specific names not used here.
        # The class-level MockSentenceTransformer will be used.
        # We need to ensure that the side_effect can handle 'another_fake_model' or we adjust.
        # Let's make the side_effect more robust or use specific names.

        # Temporarily adjust side_effect for this test if needed, or ensure it's general
        original_side_effect = self.MockSentenceTransformer.side_effect
        def specific_side_effect(model_name):
            if model_name == 'another_fake_model': return MagicMock() # Just need an object
            return original_side_effect(model_name) # Fallback to original complex mock
        self.MockSentenceTransformer.side_effect = specific_side_effect

        retriever_fallback = ChatHistoryRetriever(
            vector_db_path=self.test_db_base_path, # Use base path
            embedding_model_name='another_fake_model', # Different name to avoid conflict with setUp's specific mock logic
            tokenizer=None, # Explicitly pass None
            cross_encoder_model_name=None # Disable cross-encoder for this specific init test
        )
        self.assertIsInstance(retriever_fallback.tokenizer, BasicCharTokenizer)
        self.MockSentenceTransformer.side_effect = original_side_effect # Restore


    def test_add_message_success(self, MockChromaDBClient, MockSentenceTransformer):
        msg_text = "A test message for adding."
        msg_role = "user"
        msg_id = str(uuid.uuid4())

        returned_id = self.retriever.add_message(msg_text, msg_role, message_id=msg_id)
        self.assertEqual(returned_id, msg_id)
        self.mock_bi_encoder_instance.encode.assert_called_once_with(msg_text)
        self.mock_collection.add.assert_called_once_with(
            ids=[msg_id],
            embeddings=[[0.1, 0.2, 0.3]], # From mock_bi_encoder_instance
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
        self.mock_collection.count.return_value = 2

        query_text = "Relevant query"
        # Use the default retriever from setUp which has cross-encoder enabled
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=2)

        # Mock cross-encoder to return scores that don't change the order for simplicity here
        self.mock_cross_encoder_instance.predict.return_value = [0.9, 0.8] # Higher for doc1

        snippets = self.retriever.retrieve(query_text, n_results=2)

        self.mock_bi_encoder_instance.encode.assert_called_with(query_text)
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2, # rerank_top_n_candidates from _reinit_retriever
            include=['documents', 'metadatas', 'distances']
        )
        self.mock_cross_encoder_instance.predict.assert_called_once()
        self.assertEqual(len(snippets), 2)
        self.assertIn("Document one text.", snippets[0]['c'])
        self.assertIn("~rerank_score 0.9000", snippets[0]['c'])
        self.assertIn("Document two text.", snippets[1]['c'])
        self.assertIn("~rerank_score 0.8000", snippets[1]['c'])
        self.assertEqual(snippets[0]['r'], 'retrieved_context')


    def test_retrieve_with_reranking_logic(self, MockChromaDBClient, MockSentenceTransformer):
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=3)
        self.assertIsNotNone(self.retriever.cross_encoder)

        # Initial ChromaDB results (Doc B is best by distance)
        initial_candidates_data = {
            'ids': [['idA', 'idB', 'idC']],
            'documents': [['Doc A: low relevance', 'Doc B: high relevance by vector', 'Doc C: medium relevance by vector']],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()}] * 3],
            'distances': [[0.8, 0.1, 0.5]] # B, C, A
        }
        self.mock_collection.query.return_value = initial_candidates_data
        self.mock_collection.count.return_value = 3

        query = "find relevant doc"
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.5]*3] # Query embedding

        # Cross-encoder scores: C is best, then A, then B (opposite of distance)
        self.mock_cross_encoder_instance.predict.return_value = [0.7, 0.2, 0.9] # Scores for A, B, C

        snippets = self.retriever.retrieve(query, n_results=2)

        self.mock_cross_encoder_instance.predict.assert_called_with([
            (query, 'Doc A: low relevance'),
            (query, 'Doc B: high relevance by vector'),
            (query, 'Doc C: medium relevance by vector')
        ])
        self.assertEqual(len(snippets), 2)
        self.assertIn("Doc C: medium relevance by vector", snippets[0]['c']) # Highest cross_score 0.9
        self.assertIn("~rerank_score 0.9000", snippets[0]['c'])
        self.assertIn("Doc A: low relevance", snippets[1]['c'])      # Second highest cross_score 0.7
        self.assertIn("~rerank_score 0.7000", snippets[1]['c'])


    def test_retrieve_reranking_no_cross_encoder(self, MockChromaDBClient, MockSentenceTransformer):
        self._reinit_retriever(cross_encoder_name=None, rerank_top_n=3) # Disable cross-encoder
        self.assertIsNone(self.retriever.cross_encoder)

        initial_candidates_data = { # Sorted by distance: B, C, A
            'ids': [['idB', 'idC', 'idA']],
            'documents': [['Doc B', 'Doc C', 'Doc A']],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()}] * 3],
            'distances': [[0.1, 0.5, 0.8]]
        }
        self.mock_collection.query.return_value = initial_candidates_data
        self.mock_collection.count.return_value = 3
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.5]*3]

        snippets = self.retriever.retrieve("query", n_results=2)

        self.mock_cross_encoder_instance.predict.assert_not_called()
        self.assertEqual(len(snippets), 2)
        self.assertIn("Doc B", snippets[0]['c']) # Original Chroma order
        self.assertIn("~rerank_score N/A", snippets[0]['c'])
        self.assertIn("Doc C", snippets[1]['c'])
        self.assertIn("~rerank_score N/A", snippets[1]['c'])
        # Check n_results for query call (should be n_results if no cross-encoder)
        self.mock_collection.query.assert_called_with(
            query_embeddings=ANY, n_results=2, include=ANY # Uses n_results directly
        )


    def test_retrieve_reranking_cross_encoder_error(self, MockChromaDBClient, MockSentenceTransformer):
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=3)
        self.assertIsNotNone(self.retriever.cross_encoder)
        self.mock_cross_encoder_instance.predict.side_effect = Exception("CrossEncoder predict failed")

        initial_candidates_data = { # Sorted by distance: B, C, A
            'ids': [['idB', 'idC', 'idA']],
            'documents': [['Doc B', 'Doc C', 'Doc A']],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()}] * 3],
            'distances': [[0.1, 0.5, 0.8]]
        }
        self.mock_collection.query.return_value = initial_candidates_data
        self.mock_collection.count.return_value = 3
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.5]*3]

        with self.assertLogs(level='WARNING') as log: # Check for warning log
            snippets = self.retriever.retrieve("query", n_results=2)
            self.assertTrue(any("Error during CrossEncoder prediction/re-ranking" in record.getMessage() for record in log.records))

        self.assertEqual(len(snippets), 2)
        self.assertIn("Doc B", snippets[0]['c']) # Falls back to Chroma order
        self.assertIn("~rerank_score N/A", snippets[0]['c'])
        self.assertIn("Doc C", snippets[1]['c'])
        self.assertIn("~rerank_score N/A", snippets[1]['c'])


    def test_retrieve_token_budget_respected(self, MockChromaDBClient, MockSentenceTransformer):
        self._reinit_retriever(recall_budget=10, cross_encoder_name=None) # Small budget, no rerank for simplicity

        doc1_text = "Doc one."
        doc2_text = "Doc two here."
        doc3_text = "Third doc is long."

        mock_query_results = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [[doc1_text, doc2_text, doc3_text]],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()}] * 3],
            'distances': [[0.1, 0.2, 0.3]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 3

        def custom_token_count_for_budget_test(text_input):
            if doc1_text in text_input: return 6 # Approx token count for formatted snippet
            if doc2_text in text_input: return 7
            if doc3_text in text_input: return 8
            return len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=lambda t: [1] * custom_token_count_for_budget_test(t))

        snippets = self.retriever.retrieve("Query for budgeting")

        self.assertEqual(len(snippets), 1)
        self.assertIn(doc1_text, snippets[0]['c'])


    def test_retrieve_no_results_from_db(self, MockChromaDBClient, MockSentenceTransformer):
        self.mock_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        self.mock_collection.count.return_value = 0

        snippets = self.retriever.retrieve("Any query")
        self.assertEqual(len(snippets), 0)

    def test_retrieve_skips_identical_and_current_history(self, MockChromaDBClient, MockSentenceTransformer):
        self._reinit_retriever(cross_encoder_name=None) # No rerank for simplicity
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
        # Base path for retriever DBs during tests
        test_db_parent_dir = Path(self.test_db_base_path)
        # Construct the specific path the retriever would have used
        retriever_specific_db_path = test_db_parent_dir / ChatHistoryRetriever.DEFAULT_DB_SUBDIR

        if retriever_specific_db_path.exists():
            try:
                # Ensure client/collection are not locking files
                if hasattr(self.retriever, 'collection') and self.retriever.collection: del self.retriever.collection
                if hasattr(self.retriever, 'db_client') and self.retriever.db_client: del self.retriever.db_client
                time.sleep(0.1) # Brief pause
                shutil.rmtree(retriever_specific_db_path)
                # print(f"Cleaned up test DB: {retriever_specific_db_path}")
            except Exception as e:
                print(f"Warning: Could not cleanup test DB directory {retriever_specific_db_path}: {e}")

        # Clean up the base test directory if it's empty and was potentially created by tests
        if test_db_parent_dir.exists() and not any(test_db_parent_dir.iterdir()):
            try:
                shutil.rmtree(test_db_parent_dir)
                # print(f"Cleaned up base test DB directory: {test_db_parent_dir}")
            except Exception as e:
                print(f"Warning: Could not cleanup base test DB directory {test_db_parent_dir}: {e}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
