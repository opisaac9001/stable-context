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
from llm_context_os.retriever.chat_history import BM25Okapi # Import for patching target


@unittest.skipUnless(SENTENCE_TRANSFORMERS_AVAILABLE, "SentenceTransformers library not available, skipping ChatHistoryRetriever tests.")
@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB library not available, skipping ChatHistoryRetriever tests.")
@patch('llm_context_os.retriever.chat_history.BM25Okapi') # Added BM25Okapi patch
@patch('llm_context_os.retriever.chat_history.SentenceTransformer')
@patch('llm_context_os.retriever.chat_history.chromadb.PersistentClient')
class TestChatHistoryRetriever(unittest.TestCase):

    def setUp(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        self.MockSentenceTransformer = MockSentenceTransformer
        self.MockChromaDBClient = MockChromaDBClient
        self.MockBM25Okapi = MockBM25Okapi


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

        self.mock_bm25_index_instance = self.MockBM25Okapi.return_value
        self.mock_bm25_index_instance.get_scores.return_value = []


        self.test_db_base_path = "data/test_vector_dbs"
        self._reinit_retriever()

    def _reinit_retriever(self,
                          recall_budget=50,
                          cross_encoder_name='fake-cross-encoder-model',
                          rerank_top_n=5,
                          embedding_model_name='fake_embedding_model',
                          collection_name='test_chat_collection',
                          enable_hybrid_search=True, # New param
                          rrf_k=60):                 # New param
        self.retriever = ChatHistoryRetriever(
            recall_budget_tokens=recall_budget,
            tokenizer=self.mock_tokenizer,
            vector_db_path=self.test_db_base_path,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            cross_encoder_model_name=cross_encoder_name,
            rerank_top_n_candidates=rerank_top_n,
            enable_hybrid_search=enable_hybrid_search, # Pass to constructor
            rrf_k_constant=rrf_k                     # Pass to constructor
        )
        # Reset instance mocks
        self.mock_bi_encoder_instance.reset_mock()
        self.mock_cross_encoder_instance.reset_mock()
        self.mock_chromadb_client_instance.reset_mock()
        self.mock_collection.reset_mock()
        self.mock_tokenizer.reset_mock()
        self.mock_bm25_index_instance.reset_mock()


        # Re-apply default return values
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        self.MockChromaDBClient.return_value = self.mock_chromadb_client_instance
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        def mock_encode_for_len(text_input): return [1] * len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)
        self.mock_tokenizer.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))
        self.MockBM25Okapi.return_value = self.mock_bm25_index_instance


    def test_initialization_successful(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        # Retriever is initialized in _reinit_retriever called by setUp.
        MockSentenceTransformer.assert_any_call('fake_embedding_model')
        MockSentenceTransformer.assert_any_call('fake-cross-encoder-model')

        self.assertIsNotNone(self.retriever.embedding_model)
        self.assertIsNotNone(self.retriever.cross_encoder) # Assuming fake-cross-encoder-model is not None

        expected_db_full_path = os.path.join(self.test_db_base_path, ChatHistoryRetriever.DEFAULT_DB_SUBDIR)
        MockChromaDBClient.assert_called_with(path=expected_db_full_path)

        self.mock_chromadb_client_instance.get_or_create_collection.assert_called_with(name='test_chat_collection')
        self.assertIsNotNone(self.retriever.embedding_model)
        self.assertIsNotNone(self.retriever.collection)
        self.assertIs(self.retriever.tokenizer, self.mock_tokenizer)

    @patch('llm_context_os.retriever.chat_history.TIKTOKEN_AVAILABLE', False)
    def test_initialization_fallback_tokenizer(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        original_side_effect = self.MockSentenceTransformer.side_effect
        def specific_side_effect(model_name):
            if model_name == 'another_fake_model': return MagicMock()
            return original_side_effect(model_name)
        self.MockSentenceTransformer.side_effect = specific_side_effect

        retriever_fallback = ChatHistoryRetriever(
            vector_db_path=self.test_db_base_path,
            embedding_model_name='another_fake_model',
            tokenizer=None,
            cross_encoder_model_name=None
        )
        self.assertIsInstance(retriever_fallback.tokenizer, BasicCharTokenizer)
        self.MockSentenceTransformer.side_effect = original_side_effect


    def test_add_message_success(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        msg_text = "A test message for adding."
        msg_role = "user"
        msg_id = str(uuid.uuid4())

        returned_id = self.retriever.add_message(msg_text, msg_role, message_id=msg_id)
        self.assertEqual(returned_id, msg_id)
        self.mock_bi_encoder_instance.encode.assert_called_once_with(msg_text)
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
        self.assertIn("cross: 0.9000", snippets[0]['c']) # Updated to new format
        self.assertIn("Document two text.", snippets[1]['c'])
        self.assertIn("cross: 0.8000", snippets[1]['c']) # Updated to new format
        self.assertEqual(snippets[0]['r'], 'retrieved_context')


    def test_retrieve_with_reranking_logic(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        # This test focuses on the cross-encoder re-ranking after results (dense-only path for this test)
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=3, enable_hybrid_search=False)
        self.assertIsNotNone(self.retriever.cross_encoder)
        self.assertFalse(self.retriever.enable_hybrid_search)

        # Initial ChromaDB results (Doc B is best by distance, then C, then A)
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
            # Order based on dense results as hybrid is false
            (query, 'Doc B: high relevance by vector'),
            (query, 'Doc C: medium relevance by vector'),
            (query, 'Doc A: low relevance')
        ])
        self.assertEqual(len(snippets), 2)
        self.assertIn("Doc C: medium relevance by vector", snippets[0]['c'])
        self.assertIn("cross: 0.9000", snippets[0]['c']) # Updated to new format
        self.assertIn("Doc A: low relevance", snippets[1]['c'])
        self.assertIn("cross: 0.7000", snippets[1]['c']) # Updated to new format


    def test_retrieve_reranking_no_cross_encoder(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        self._reinit_retriever(cross_encoder_name=None, rerank_top_n=3, enable_hybrid_search=False)
        self.assertIsNone(self.retriever.cross_encoder)
        self.assertFalse(self.retriever.enable_hybrid_search)


        initial_candidates_data = {
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
        self.assertIn("Doc B", snippets[0]['c'])
        self.assertIn("cross: N/A", snippets[0]['c']) # Updated to new format
        self.assertIn("Doc C", snippets[1]['c'])
        self.assertIn("cross: N/A", snippets[1]['c']) # Updated to new format

        # Rerank_top_n is 3, n_results is 2. Query should ask for rerank_top_n (3) if cross_encoder or hybrid is on.
        # If both are off, it should use n_results (2) * 3 (the old factor) or just n_results if that logic is simplified.
        # Current PdfRetriever uses `top_k * 3` if no cross_encoder. ChatHistoryRetriever uses `n_results`
        # Let's assume ChatHistoryRetriever uses `n_results` if cross_encoder is None and hybrid is off.
        # The test was `n_results=2` for dense query.
        self.mock_collection.query.assert_called_with(
            query_embeddings=ANY, n_results=3, include=ANY # Corrected: rerank_top_n is 3 for this test.
        )


    def test_retrieve_reranking_cross_encoder_error(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=3, enable_hybrid_search=False)
        self.assertIsNotNone(self.retriever.cross_encoder)
        self.mock_cross_encoder_instance.predict.side_effect = Exception("CrossEncoder predict failed")

        initial_candidates_data = {
            'ids': [['idB', 'idC', 'idA']],
            'documents': [['Doc B', 'Doc C', 'Doc A']],
            'metadatas': [[{'role': 'user', 'timestamp': time.time()}] * 3],
            'distances': [[0.1, 0.5, 0.8]]
        }
        self.mock_collection.query.return_value = initial_candidates_data
        self.mock_collection.count.return_value = 3
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.5]*3]

        with self.assertLogs(level='WARNING') as log:
            snippets = self.retriever.retrieve("query", n_results=2)
            self.assertTrue(any("Error during CrossEncoder prediction/re-ranking" in record.getMessage() for record in log.records))

        self.assertEqual(len(snippets), 2)
        self.assertIn("Doc B", snippets[0]['c'])
        self.assertIn("cross: N/A", snippets[0]['c']) # Updated
        self.assertIn("Doc C", snippets[1]['c'])
        self.assertIn("cross: N/A", snippets[1]['c']) # Updated


    def test_retrieve_token_budget_respected(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        self._reinit_retriever(recall_budget=10, cross_encoder_name=None, enable_hybrid_search=False)

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
            if doc1_text in text_input: return 6
            if doc2_text in text_input: return 7
            if doc3_text in text_input: return 8
            return len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=lambda t: [1] * custom_token_count_for_budget_test(t))

        snippets = self.retriever.retrieve("Query for budgeting")

        self.assertEqual(len(snippets), 1)
        self.assertIn(doc1_text, snippets[0]['c'])


    def test_retrieve_no_results_from_db(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        self._reinit_retriever(enable_hybrid_search=True) # Enable to test BM25 path too
        self.mock_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        self.mock_collection.count.return_value = 0
        self.mock_bm25_index_instance.get_scores.return_value = []


        snippets = self.retriever.retrieve("Any query")
        self.assertEqual(len(snippets), 0)

    def test_retrieve_skips_identical_and_current_history(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi): # Added MockBM25Okapi
        self._reinit_retriever(cross_encoder_name=None, enable_hybrid_search=False)
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

    # --- New Hybrid Search Tests ---

    def test_retrieve_hybrid_search_enabled_and_bm25_works(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi):
        self._reinit_retriever(enable_hybrid_search=True, rerank_top_n=3, cross_encoder_name='fake-cross-encoder-model')
        self.assertTrue(self.retriever.enable_hybrid_search)
        self.assertIsNotNone(self.retriever.bm25_index)

        # Setup BM25 corpus
        self.retriever.bm25_corpus_texts = ["BM25 msg 1", "BM25 msg 2: better match", "Common message text"]
        self.retriever.bm25_message_ids = ["bm25_id1", "bm25_id2", "common_id1_bm25"] # Use a distinct ID for BM25 common message
        self.retriever.is_bm25_index_built = False
        self.mock_bm25_index_instance.get_scores.return_value = [0.2, 0.9, 0.5] # Scores for bm25_id1, bm25_id2, common_id1_bm25

        # Mock Dense (ChromaDB) results
        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = {
            'ids': [['common_id1_chroma', 'dense_only_id1']], # Use a distinct ID for Chroma common message
            'documents': [['Common message text', 'Dense only message']],
            'metadatas': [[{'role': 'user'}], [{'role': 'assistant'}]],
            'distances': [[0.1, 0.2]] # common_id1_chroma (dist 0.1), dense_only_id1 (dist 0.2)
        }
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.7]*3]

        # Mock Cross-encoder
        self.mock_cross_encoder_instance.predict.side_effect = lambda pairs: [
            0.95 if "BM25 msg 2" in p[1] else
            0.85 if "Common message" in p[1] else
            0.75 if "Dense only" in p[1] else 0.1 for p in pairs
        ]

        query = "match"
        # Manually add common_id1_from_bm25 text to doc_details_cache to simulate it being findable for BM25 part
        # This is a bit of a hack due to how doc_details_cache is populated.
        # A cleaner way would be to ensure bm25_message_ids and chroma IDs align if they refer to same content.
        # For this test, we'll assume common_id1_bm25 and common_id1_chroma are the same underlying document.
        # The RRF logic should handle distinct IDs that might point to the same content if their text matches.
        # Let's assume for this test that "common_id1_from_bm25" is the ID used in bm25_message_ids,
        # and "common_id1_chroma" is the ID from Chroma for the same text.
        # The doc_details_cache will initially get common_id1_chroma. BM25 will add common_id1_bm25.
        # RRF will treat them as separate items if IDs are different.
        # For simplicity, let's use the same ID "common_id" for the shared document.
        self.retriever.bm25_message_ids = ["bm25_id1", "bm25_id2", "common_id"]
        self.mock_bm25_index_instance.get_scores.return_value = [0.2, 0.9, 0.5] # for bm25_id1, bm25_id2, common_id
        self.mock_collection.query.return_value['ids'] = [['common_id', 'dense_only_id1']]


        snippets = self.retriever.retrieve(query, n_results=2)

        self.MockBM25Okapi.assert_called_with([['bm25', 'msg', '1'], ['bm25', 'msg', '2:', 'better', 'match'], ['common', 'message', 'text']])
        self.mock_bm25_index_instance.get_scores.assert_called_once()
        self.mock_collection.query.assert_called_once()
        self.mock_cross_encoder_instance.predict.assert_called_once()

        self.assertEqual(len(snippets), 2)
        # Expected RRF order (approx): bm25_id2 (BM25 high), common_id (Dense high, BM25 mid), dense_only_id1 (Dense mid)
        # Cross-encoder re-ranks these.
        self.assertIn("BM25 msg 2: better match", snippets[0]['c'])
        self.assertIn("cross: 0.9500", snippets[0]['c'])
        self.assertIn("Common message text", snippets[1]['c'])
        self.assertIn("cross: 0.8500", snippets[1]['c'])


    def test_retrieve_hybrid_search_disabled(self, MockChromaDBClient, MockSentenceTransformer, MockBM25Okapi):
        self._reinit_retriever(enable_hybrid_search=False, rerank_top_n=2, cross_encoder_name='fake-cross-encoder-model')
        self.assertFalse(self.retriever.enable_hybrid_search)

        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = {
            'ids': [['id1', 'id2']], 'documents': [['Dense Doc 1', 'Dense Doc 2']],
            'metadatas': [[{}, {}]], 'distances': [[0.1, 0.2]]
        }
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.1]*3]
        self.mock_cross_encoder_instance.predict.return_value = [0.9, 0.8]

        snippets = self.retriever.retrieve("query", n_results=1)

        self.mock_bm25_index_instance.get_scores.assert_not_called()
        self.mock_collection.query.assert_called_once()
        self.mock_cross_encoder_instance.predict.assert_called_once()

        self.assertEqual(len(snippets), 1)
        self.assertIn("Dense Doc 1", snippets[0]['c'])
        self.assertIn("cross: 0.9000", snippets[0]['c'])
        self.assertIn("sparse: N/A", snippets[0]['c'])
        self.assertIn("rrf: N/A", snippets[0]['c'])


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
