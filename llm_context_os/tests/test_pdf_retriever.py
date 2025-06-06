# llm_context_os/tests/test_pdf_retriever.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import time
from pathlib import Path
import os # For dummy file creation in tests if needed
import shutil # For cleanup

from llm_context_os.retriever.pdf_retriever import (
    PdfRetriever,
    LANGCHAIN_TEXT_SPLITTERS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    CHROMADB_AVAILABLE
)
# PyPDF2 is imported directly in the module, so we'll patch its PdfReader
# tiktoken availability is handled by PdfRetriever's internal tokenizer fallback

# --- Conditional import for PyPDF2 to check its availability for skipping tests ---
PYPDF2_AVAILABLE = False
try:
    from PyPDF2 import PdfReader # Check if it's importable for the test suite itself
    PYPDF2_AVAILABLE = True # Keep for conditional skipping if some tests remain for it, though likely remove
except ImportError:
    PYPDF2_AVAILABLE = False # Ensure it's False if not found
    PdfReader = None # type: ignore

FITZ_AVAILABLE = False
try:
    import fitz # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None # type: ignore
    print("PyMuPDF (fitz) not installed. Some PdfRetriever tests might be skipped or fail.")


@unittest.skipUnless(
    FITZ_AVAILABLE and LANGCHAIN_TEXT_SPLITTERS_AVAILABLE and \
    SENTENCE_TRANSFORMERS_AVAILABLE and CHROMADB_AVAILABLE,
    "One or more core dependencies (PyMuPDF, langchain-text-splitters, sentence-transformers, chromadb) not available. Skipping PdfRetriever tests."
)
# Remove PdfReader patch, add fitz.open patch
@patch('llm_context_os.retriever.pdf_retriever.fitz.open')
@patch('llm_context_os.retriever.pdf_retriever.RecursiveCharacterTextSplitter')
@patch('llm_context_os.retriever.pdf_retriever.SentenceTransformer') # This patches ST for both bi-encoder and cross-encoder
@patch('llm_context_os.retriever.pdf_retriever.chromadb.PersistentClient')
class TestPdfRetriever(unittest.TestCase):

    def setUp(self, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen): # Order matters
        # Store mock classes passed by decorators
        self.MockFitzOpen = MockFitzOpen
        self.MockSplitter = MockSplitter
        self.MockSentenceTransformer = MockSentenceTransformer # This is the class
        self.MockChromaDBClient = MockChromaDBClient

        # Configure the mock instances that will be returned by the patched constructors/functions
        self.mock_fitz_doc_instance = self.MockFitzOpen.return_value
        self.mock_fitz_page_instance = MagicMock()
        self.mock_fitz_page_instance.get_text.return_value = "Page 1 text from PyMuPDF."
        self.mock_fitz_doc_instance.load_page.return_value = self.mock_fitz_page_instance
        self.mock_fitz_doc_instance.__len__.return_value = 1 # Simulate one page

        self.mock_splitter_instance = self.MockSplitter.return_value
        self.mock_splitter_instance.split_text.return_value = ["PyMuPDF chunk 1.", "PyMuPDF chunk 2."]

        # Mock for the bi-encoder (embedding_model)
        self.mock_bi_encoder_instance = MagicMock()
        mock_bi_embedding_array = MagicMock()
        mock_bi_embedding_array.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] # Embeddings for two chunks
        self.mock_bi_encoder_instance.encode.return_value = mock_bi_embedding_array

        # Mock for the cross-encoder (if initialized)
        self.mock_cross_encoder_instance = MagicMock()
        # self.mock_cross_encoder_instance.predict.return_value = [0.9, 0.1] # Example scores

        # SentenceTransformer class mock will provide these instances based on model name
        def sentence_transformer_side_effect(model_name_or_path):
            if model_name_or_path == 'fake-embedding-model':
                return self.mock_bi_encoder_instance
            elif model_name_or_path == 'fake-cross-encoder-model':
                return self.mock_cross_encoder_instance
            raise ValueError(f"Unexpected model name for SentenceTransformer mock: {model_name_or_path}")
        self.MockSentenceTransformer.side_effect = sentence_transformer_side_effect

        self.mock_chromadb_client_instance = self.MockChromaDBClient.return_value
        self.mock_collection = MagicMock()
        self.mock_collection.count.return_value = 0
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        self.mock_tokenizer = MagicMock()
        def mock_encode_for_len(text_input):
            # Simple mock: token count is number of words
            return [1] * len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)
        self.mock_tokenizer.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))


        self.test_db_base_path = "temp_test_pdf_retriever_db_unit"
        # Initialize PdfRetriever for each test
        self._reinit_retriever()


    def _reinit_retriever(self, cross_encoder_name='fake-cross-encoder-model', rerank_top_n=5):
        # Helper to re-initialize retriever, useful if constructor logic changes or needs testing with different params
        self.retriever = PdfRetriever(
            vector_db_path=self.test_db_base_path,
            embedding_model_name='fake-embedding-model',
            tokenizer=self.mock_tokenizer,
            recall_budget_tokens=100, # Small for testing budget
            chunk_size=10, # Small for testing
            chunk_overlap=2,
            cross_encoder_model_name=cross_encoder_name,
            rerank_top_n_candidates=rerank_top_n
        )
        # Reset instance mocks that might be called by __init__ or other methods
        self.mock_fitz_doc_instance.reset_mock()
        self.mock_fitz_page_instance.reset_mock()
        self.mock_splitter_instance.reset_mock()
        self.mock_bi_encoder_instance.reset_mock()
        self.mock_cross_encoder_instance.reset_mock() # if it was created
        self.mock_chromadb_client_instance.reset_mock()
        self.mock_collection.reset_mock()

        # Re-apply default return values for mocks that might be altered by specific tests
        # These are mocks of the *classes* or factory functions
        self.MockFitzOpen.return_value = self.mock_fitz_doc_instance
        self.mock_fitz_doc_instance.load_page.return_value = self.mock_fitz_page_instance
        self.mock_fitz_doc_instance.__len__.return_value = 1
        self.mock_fitz_page_instance.get_text.return_value = "Page 1 text from PyMuPDF."


        self.MockSplitter.return_value = self.mock_splitter_instance
        self.mock_splitter_instance.split_text.return_value = ["PyMuPDF chunk 1.", "PyMuPDF chunk 2."]

        # self.MockSentenceTransformer.return_value is handled by side_effect now
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.1,0.2,0.3],[0.4,0.5,0.6]]

        self.MockChromaDBClient.return_value = self.mock_chromadb_client_instance
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection


    def tearDown(self):
        if hasattr(self.retriever, 'vector_db_path') and self.retriever.vector_db_path and self.retriever.vector_db_path.exists():
            # Ensure client/collection are not locking files
            if hasattr(self.retriever, 'collection') and self.retriever.collection: del self.retriever.collection
            if hasattr(self.retriever, 'db_client') and self.retriever.db_client: del self.retriever.db_client
            # Attempt cleanup, but don't fail test if it has issues (common on Windows with Chroma/SQLite)
            try:
                shutil.rmtree(self.retriever.vector_db_path)
            except Exception as e:
                print(f"Warning: Could not cleanup test DB directory {self.retriever.vector_db_path}: {e}")
        # If base path was different and also needs cleanup
        elif Path(self.test_db_base_path).exists() and self.test_db_base_path != str(self.retriever.vector_db_path):
             try:
                shutil.rmtree(self.test_db_base_path)
             except Exception as e:
                print(f"Warning: Could not cleanup base test DB directory {self.test_db_base_path}: {e}")


    def test_initialization_successful(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.MockSentenceTransformer.assert_called_with('fake-embedding-model')
        expected_persistent_client_path = str(Path(self.test_db_base_path).resolve())
        self.MockChromaDBClient.assert_called_with(path=expected_persistent_client_path)
        self.mock_chromadb_client_instance.get_or_create_collection.assert_called_with(name=PdfRetriever.DEFAULT_COLLECTION_NAME)
        self.MockSplitter.assert_called_with(
            chunk_size=10, chunk_overlap=2, length_function=len, is_separator_regex=False
        )
        self.assertIsNotNone(self.retriever.embedding_model)
        self.assertIsNotNone(self.retriever.collection)
        self.assertIsNotNone(self.retriever.text_splitter)
        self.assertIs(self.retriever.tokenizer, self.mock_tokenizer)

    @patch('llm_context_os.retriever.pdf_retriever.LANGCHAIN_TEXT_SPLITTERS_AVAILABLE', False)
    def test_initialization_no_splitter_lib(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        retriever_no_splitter = PdfRetriever(vector_db_path=self.test_db_base_path, tokenizer=self.mock_tokenizer)
        self.assertIsNone(retriever_no_splitter.text_splitter)

    @patch('llm_context_os.retriever.pdf_retriever.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_initialization_no_embedding_lib(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        retriever_no_embed = PdfRetriever(vector_db_path=self.test_db_base_path, tokenizer=self.mock_tokenizer)
        self.assertIsNone(retriever_no_embed.embedding_model)

    @patch('llm_context_os.retriever.pdf_retriever.CHROMADB_AVAILABLE', False)
    def test_initialization_no_chromadb_lib(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        retriever_no_db = PdfRetriever(vector_db_path=self.test_db_base_path, tokenizer=self.mock_tokenizer)
        self.assertIsNone(retriever_no_db.db_client)
        self.assertIsNone(retriever_no_db.collection)

    def test_upload_document_success(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        fake_pdf_path_str = "dummy_document.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "dummy_document"
            mock_path_constructor.return_value = mock_path_instance

            success, msg, doc_id, num_chunks = self.retriever.upload_document(fake_pdf_path_str)

        self.assertTrue(success)
        self.assertEqual(doc_id, "dummy_document")
        self.assertEqual(num_chunks, 2)
        self.assertIn("Successfully extracted, embedded, and stored", msg)

        self.MockFitzOpen.assert_called_with(fake_pdf_path_str)
        self.mock_fitz_doc_instance.load_page.assert_called_with(0) # Page number 0 for first page
        self.mock_fitz_page_instance.get_text.assert_called_with("text")
        self.mock_splitter_instance.split_text.assert_called_with("Page 1 text from PyMuPDF.")
        self.mock_bi_encoder_instance.encode.assert_called_with(["PyMuPDF chunk 1.", "PyMuPDF chunk 2."])

        self.mock_collection.add.assert_called_once()
        add_args = self.mock_collection.add.call_args[1]
        self.assertEqual(len(add_args['ids']), 2)
        self.assertEqual(add_args['embeddings'], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(add_args['documents'], ["PyMuPDF chunk 1.", "PyMuPDF chunk 2."])
        self.assertEqual(add_args['metadatas'][0]['doc_id'], "dummy_document")
        self.assertEqual(add_args['metadatas'][0]['pdf_path'], fake_pdf_path_str)

    def test_upload_document_pdf_read_error(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.MockFitzOpen.side_effect = Exception("PyMuPDF Read Error") # Simulate fitz.open failing
        fake_pdf_path_str = "error.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "error" # doc_id comes from here
            mock_path_constructor.return_value = mock_path_instance
            success, msg, doc_id, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error parsing PDF file error.pdf with PyMuPDF: PyMuPDF Read Error", msg)
        self.assertEqual(doc_id, "error") # doc_id should still be derived if path was valid initially

    def test_upload_document_no_text_extracted(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_fitz_page_instance.get_text.return_value = "" # Simulate no text from page
        fake_pdf_path_str = "empty_text.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "empty_text"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, num_chunks = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("No text could be extracted from PDF using PyMuPDF", msg)
        self.assertEqual(num_chunks, 0)

    def test_upload_document_embedding_error(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_bi_encoder_instance.encode.side_effect = Exception("Embedding Error")
        fake_pdf_path_str = "embed_error.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "embed_error"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error during embedding or storing chunks", msg)

    def test_upload_document_db_add_error(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_collection.add.side_effect = Exception("DB Add Error")
        fake_pdf_path_str = "db_add_error.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "db_add_error"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error during embedding or storing chunks", msg)

    def test_retrieve_from_pdf_success_no_rerank(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        # Ensure no cross-encoder for this test
        self._reinit_retriever(cross_encoder_name=None)
        self.assertIsNone(self.retriever.cross_encoder)

        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = {
            'ids': [['id1', 'id2']],
            'documents': [['Retrieved Doc 1 text.', 'Retrieved Doc 2 text.']],
            'metadatas': [[{'doc_id': 'pdf1', 'page_number': 1}, {'doc_id': 'pdf1', 'page_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        query_text = "Relevant query"
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.1,0.2,0.3]]

        snippets = self.retriever.retrieve_from_pdf(query_text, doc_ids=["pdf1"], top_k=1)

        self.mock_bi_encoder_instance.encode.assert_called_with(query_text)
        # n_results should be top_k * 3 when no cross-encoder
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1 * 3,
            where={"doc_id": "pdf1"},
            include=['documents', 'metadatas', 'distances']
        )
        self.assertEqual(len(snippets), 1)
        self.assertIn("Retrieved Doc 1 text.", snippets[0]['c'])
        self.assertIn("~rerank_score N/A", snippets[0]['c'])
        self.assertEqual(snippets[0]['r'], 'retrieved_pdf_chunk')


    def test_retrieve_with_reranking_success(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=3) # Rerank top 3
        self.assertIsNotNone(self.retriever.cross_encoder)

        self.mock_collection.count.return_value = 3
        # Initial Chroma results (lower distance is better)
        initial_results = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [['Doc A text.', 'Doc B text.', 'Doc C text.']],
            'metadatas': [[{'doc_id': 'pdf1', 'page_number': 1}],
                          [{'doc_id': 'pdf1', 'page_number': 2}],
                          [{'doc_id': 'pdf1', 'page_number': 3}]],
            'distances': [[0.3, 0.1, 0.2]] # Doc B is best, then C, then A by distance
        }
        self.mock_collection.query.return_value = initial_results

        query_text = "Find B"
        # Mock bi-encoder for query
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.5,0.5,0.5]]

        # Mock cross-encoder scores (higher is better). Let's make Doc C best, then A, then B
        self.mock_cross_encoder_instance.predict.return_value = [0.7, 0.1, 0.9] # Scores for A, B, C
                                                                            # Expected order: C, A, B

        snippets = self.retriever.retrieve_from_pdf(query_text, top_k=2)

        self.mock_bi_encoder_instance.encode.assert_called_with(query_text)
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.5,0.5,0.5]],
            n_results=3, # self.retriever.rerank_top_n_candidates
            where=None,
            include=['documents', 'metadatas', 'distances']
        )
        self.mock_cross_encoder_instance.predict.assert_called_once_with([
            (query_text, 'Doc A text.'), (query_text, 'Doc B text.'), (query_text, 'Doc C text.')
        ])

        self.assertEqual(len(snippets), 2)
        self.assertIn("Doc C text.", snippets[0]['c']) # Doc C had score 0.9
        self.assertIn("~rerank_score 0.9000", snippets[0]['c'])
        self.assertIn("Doc A text.", snippets[1]['c']) # Doc A had score 0.7
        self.assertIn("~rerank_score 0.7000", snippets[1]['c'])


    def test_retrieve_reranking_cross_encoder_error(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model')
        self.assertIsNotNone(self.retriever.cross_encoder)
        self.mock_cross_encoder_instance.predict.side_effect = Exception("CrossEncoder failed")

        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = { # Sorted by distance: B then A
            'ids': [['id_b', 'id_a']],
            'documents': [['Doc B text.', 'Doc A text.']],
            'metadatas': [[{'doc_id': 'pdf1', 'page_number': 1}], [{'doc_id': 'pdf1', 'page_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        query_text = "Find A or B"
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.6,0.6,0.6]]

        snippets = self.retriever.retrieve_from_pdf(query_text, top_k=1)

        # Should fall back to distance-based ranking
        self.assertEqual(len(snippets), 1)
        self.assertIn("Doc B text.", snippets[0]['c']) # Doc B has lower distance (0.1)
        self.assertIn("~rerank_score N/A", snippets[0]['c']) # Rerank score should be N/A


    def test_retrieve_from_pdf_token_budget(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self._reinit_retriever(cross_encoder_name=None) # No reranking for this specific budget test simplicity
        self.retriever.recall_budget_tokens = 25 # Small budget

        # Mock tokenizer to count "words" (space separated strings)
        # Snippet format: "Retrieved from '{doc_id_meta}' (Page {page_num_meta}, ~distance {res_item['distance']:.4f}, ~rerank_score {cross_score_str}): \"{res_item['text']}\""
        # Example: "Retrieved from 'd1' (Page 1, ~distance 0.1000, ~rerank_score N/A): "Chunk one."" has 14 "words"
        # Example: "Retrieved from 'd1' (Page 2, ~distance 0.2000, ~rerank_score N/A): "Chunk two is longer."" has 16 "words"

        doc1_text = "Chunk one."
        doc2_text = "Chunk two is longer."

        mock_query_results = {
            'ids': [['id1', 'id2']],
            'documents': [[doc1_text, doc2_text]],
            'metadatas': [[{'doc_id': 'd1', 'page_number': 1}],
                          [{'doc_id': 'd1', 'page_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 2

        snippets = self.retriever.retrieve_from_pdf("Query for budgeting", top_k=2)

        self.assertEqual(len(snippets), 1) # Only first snippet should fit
        self.assertIn(doc1_text, snippets[0]['c'])

    def test_retrieve_from_pdf_no_results(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        self.mock_collection.count.return_value = 0 # Or just no results from query
        snippets = self.retriever.retrieve_from_pdf("Query for no results")
        self.assertEqual(len(snippets), 0)

    def test_retrieve_unavailable_components(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        original_embed_model = self.retriever.embedding_model
        self.retriever.embedding_model = None
        snippets = self.retriever.retrieve_from_pdf("Query")
        self.assertEqual(snippets, [])
        self.retriever.embedding_model = original_embed_model # Restore

        original_collection = self.retriever.collection
        self.retriever.collection = None
        snippets = self.retriever.retrieve_from_pdf("Query")
        self.assertEqual(snippets, [])
        self.retriever.collection = original_collection


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# End of llm_context_os/tests/test_pdf_retriever.py
