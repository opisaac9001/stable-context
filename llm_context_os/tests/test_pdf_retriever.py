# yawl/tests/test_pdf_retriever.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import time
from pathlib import Path
import os # For dummy file creation in tests if needed
import shutil # For cleanup

from yawl.retriever.pdf_retriever import (
    PdfRetriever,
    LANGCHAIN_TEXT_SPLITTERS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    CHROMADB_AVAILABLE,
    # PdfRetriever now imports nltk and numpy directly
)
import nltk # For mocking nltk.sent_tokenize and nltk.download
import numpy as np # For creating dummy embeddings

# PyPDF2 related imports are no longer needed as PyMuPDF (fitz) is the primary PDF parser.

FITZ_AVAILABLE = False
try:
    import fitz # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None # type: ignore
    print("PyMuPDF (fitz) not installed. Some PdfRetriever tests might be skipped or fail.")

# Mock NLTK data find and download for tests
@patch('nltk.data.find')
@patch('nltk.download')


@unittest.skipUnless(
    FITZ_AVAILABLE and LANGCHAIN_TEXT_SPLITTERS_AVAILABLE and \
    SENTENCE_TRANSFORMERS_AVAILABLE and CHROMADB_AVAILABLE,
    "One or more core dependencies (PyMuPDF, langchain-text-splitters, sentence-transformers, chromadb) not available. Skipping PdfRetriever tests."
)
# Remove PdfReader patch, add fitz.open patch
@patch('yawl.retriever.pdf_retriever.BM25Okapi') # Add BM25Okapi patch
@patch('yawl.retriever.pdf_retriever.fitz.open')
@patch('yawl.retriever.pdf_retriever.RecursiveCharacterTextSplitter')
@patch('yawl.retriever.pdf_retriever.SentenceTransformer')
@patch('yawl.retriever.pdf_retriever.chromadb.PersistentClient')
class TestPdfRetriever(unittest.TestCase):

    def setUp(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi): # Order of mocks matters
        # Store mock classes/functions passed by decorators
        self.MockNltkDownload = MockNltkDownload
        self.MockNltkDataFind = MockNltkDataFind
        self.MockFitzOpen = MockFitzOpen
        self.MockSplitter = MockSplitter
        self.MockSentenceTransformer = MockSentenceTransformer
        self.MockChromaDBClient = MockChromaDBClient
        self.MockBM25Okapi = MockBM25Okapi

        # Default behavior for nltk.data.find (e.g., 'punkt' is found)
        self.MockNltkDataFind.return_value = True

        # Configure the mock instances that will be returned by the patched constructors/functions
        self.mock_fitz_doc_instance = self.MockFitzOpen.return_value
        self.mock_fitz_page_instance = MagicMock()
        self.mock_fitz_page_instance.get_text.return_value = "Page 1 text from PyMuPDF. Sentence two. Sentence three."
        self.mock_fitz_doc_instance.load_page.return_value = self.mock_fitz_page_instance
        self.mock_fitz_doc_instance.__len__.return_value = 1

        self.mock_splitter_instance = self.MockSplitter.return_value
        self.mock_splitter_instance.split_text.return_value = ["Recursive chunk 1.", "Recursive chunk 2."]

        # Mocks for SentenceTransformer instances
        self.mock_main_embedding_model_instance = MagicMock(name="MainEmbeddingModel")
        # self.mock_main_embedding_model_instance.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        # Configure .encode(...).tolist() in two steps for clarity if needed, or direct for simple cases.
        # For now, encode will return a mock that has a tolist method.
        mock_main_encode_output = MagicMock()
        mock_main_encode_output.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.mock_main_embedding_model_instance.encode.return_value = mock_main_encode_output


        self.mock_semantic_embedder_instance = MagicMock(name="SemanticChunkerEmbeddingModel")
        # Default for semantic embedder (can be overridden in tests)
        # This should return a direct numpy array as per SentenceTransformer.encode()
        self.mock_semantic_embedder_instance.encode.return_value = np.array([
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9]
        ])


        self.mock_cross_encoder_instance = MagicMock(name="CrossEncoderModel")

        # SentenceTransformer class mock will provide these instances based on model name
        def sentence_transformer_side_effect(model_name_or_path, **kwargs): # Added **kwargs
            if model_name_or_path == 'fake-embedding-model':
                return self.mock_main_embedding_model_instance
            elif model_name_or_path == 'fake-semantic-chunker-model' or \
                 (self.retriever and model_name_or_path == self.retriever.embedding_model_name and self.retriever.chunking_strategy == "semantic" and not self.retriever.semantic_chunker_embedding_model_name): # if semantic model is None and strategy is semantic
                return self.mock_semantic_embedder_instance
            elif model_name_or_path == 'fake-cross-encoder-model':
                return self.mock_cross_encoder_instance
            # Fallback for unexpected model names, useful for debugging tests
            print(f"Warning: MockSentenceTransformer called with unhandled model name: {model_name_or_path}")
            return MagicMock()
        self.MockSentenceTransformer.side_effect = sentence_transformer_side_effect

        self.mock_chromadb_client_instance = self.MockChromaDBClient.return_value
        self.mock_collection = MagicMock()
        self.mock_collection.count.return_value = 0
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode = MagicMock(side_effect=lambda t: [1] * len(t.split()))
        self.mock_tokenizer.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))

        self.mock_bm25_index_instance = self.MockBM25Okapi.return_value
        self.mock_bm25_index_instance.get_scores.return_value = []

        self.test_db_base_path = "temp_test_pdf_retriever_db_unit"
        self._reinit_retriever()


    def _reinit_retriever(self,
                          cross_encoder_name='fake-cross-encoder-model',
                          rerank_top_n=5,
                          enable_hybrid_search=True,
                          rrf_k=60,
                          chunking_strategy="recursive", # New param for chunking
                          semantic_chunker_model_name=None, # New
                          semantic_breakpoint_type="percentile", # New
                          semantic_breakpoint_amount=5.0, # New
                          semantic_min_sentences=2 # New
                          ):
        # Helper to re-initialize retriever
        self.retriever = PdfRetriever(
            vector_db_path=self.test_db_base_path,
            embedding_model_name='fake-embedding-model',
            tokenizer=self.mock_tokenizer,
            recall_budget_tokens=100,
            chunk_size=10,
            chunk_overlap=2,
            cross_encoder_model_name=cross_encoder_name,
            rerank_top_n_candidates=rerank_top_n,
            enable_hybrid_search=enable_hybrid_search,
            rrf_k_constant=rrf_k,
            # Semantic chunking params
            chunking_strategy=chunking_strategy,
            semantic_chunker_embedding_model=semantic_chunker_model_name,
            semantic_chunker_breakpoint_threshold_type=semantic_breakpoint_type,
            semantic_chunker_breakpoint_threshold_amount=semantic_breakpoint_amount,
            semantic_chunker_min_chunk_sentences=semantic_min_sentences
        )
        # Reset instance mocks
        self.mock_fitz_doc_instance.reset_mock()
        self.mock_fitz_page_instance.reset_mock()
        self.mock_splitter_instance.reset_mock()
        self.mock_main_embedding_model_instance.reset_mock() # Changed from mock_bi_encoder_instance
        self.mock_semantic_embedder_instance.reset_mock() # Added
        self.mock_cross_encoder_instance.reset_mock()
        self.mock_chromadb_client_instance.reset_mock()
        self.mock_collection.reset_mock()
        self.mock_bm25_index_instance.reset_mock()

        # Re-apply default return values and behaviors
        self.MockFitzOpen.return_value = self.mock_fitz_doc_instance
        self.mock_fitz_doc_instance.load_page.return_value = self.mock_fitz_page_instance
        self.mock_fitz_doc_instance.__len__.return_value = 1
        self.mock_fitz_page_instance.get_text.return_value = "Page 1 text from PyMuPDF. Sentence two. Sentence three."

        self.MockSplitter.return_value = self.mock_splitter_instance
        self.mock_splitter_instance.split_text.return_value = ["Recursive chunk 1.", "Recursive chunk 2."]

        # Main embedding model encode output
        mock_main_encode_output = MagicMock()
        mock_main_encode_output.tolist.return_value = [[0.1,0.2,0.3],[0.4,0.5,0.6]]
        self.mock_main_embedding_model_instance.encode.return_value = mock_main_encode_output

        # Semantic embedder encode output (direct numpy array)
        self.mock_semantic_embedder_instance.encode.return_value = np.array([
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9]
        ])

        self.MockChromaDBClient.return_value = self.mock_chromadb_client_instance
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection
        self.MockBM25Okapi.return_value = self.mock_bm25_index_instance


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

    @patch('yawl.retriever.pdf_retriever.LANGCHAIN_TEXT_SPLITTERS_AVAILABLE', False)
    def test_initialization_no_splitter_lib(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        # Ensure that when re-initing, chunking_strategy is recursive if splitter is not available.
        # The PdfRetriever init logic itself doesn't set strategy based on LANGCHAIN_TEXT_SPLITTERS_AVAILABLE,
        # but rather if self.text_splitter ends up being None.
        with patch('yawl.retriever.pdf_retriever.RecursiveCharacterTextSplitter', None): # Make it None after import
             retriever_no_splitter = PdfRetriever(vector_db_path=self.test_db_base_path, tokenizer=self.mock_tokenizer, chunking_strategy="recursive")
             self.assertIsNone(retriever_no_splitter.text_splitter)


    @patch('yawl.retriever.pdf_retriever.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_initialization_no_embedding_lib(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        # Side effect for SentenceTransformer should be None or raise error if SENTENCE_TRANSFORMERS_AVAILABLE is False
        original_st_side_effect = self.MockSentenceTransformer.side_effect
        self.MockSentenceTransformer.side_effect = Exception("ST not available")
        retriever_no_embed = PdfRetriever(vector_db_path=self.test_db_base_path, tokenizer=self.mock_tokenizer, chunking_strategy="semantic")
        self.assertIsNone(retriever_no_embed.embedding_model)
        self.assertIsNone(retriever_no_embed.semantic_sentence_embedder)
        self.assertEqual(retriever_no_embed.chunking_strategy, "recursive") # Should fallback
        self.MockSentenceTransformer.side_effect = original_st_side_effect # Restore

    @patch('yawl.retriever.pdf_retriever.CHROMADB_AVAILABLE', False)
    def test_initialization_no_chromadb_lib(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        retriever_no_db = PdfRetriever(vector_db_path=self.test_db_base_path, tokenizer=self.mock_tokenizer)
        self.assertIsNone(retriever_no_db.db_client)
        self.assertIsNone(retriever_no_db.collection)

    def test_upload_document_success_recursive_strategy(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self._reinit_retriever(chunking_strategy="recursive")
        fake_pdf_path_str = "dummy_document_recursive.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
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
        self.mock_fitz_doc_instance.load_page.assert_called_with(0)
        self.mock_fitz_page_instance.get_text.assert_called_with("text")
        self.mock_splitter_instance.split_text.assert_called_with("Page 1 text from PyMuPDF. Sentence two. Sentence three.")
        self.mock_main_embedding_model_instance.encode.assert_called_with(["Recursive chunk 1.", "Recursive chunk 2."]) # Changed from mock_bi_encoder_instance

        self.mock_collection.add.assert_called_once()
        add_args = self.mock_collection.add.call_args[1]
        self.assertEqual(len(add_args['ids']), 2)
        self.assertEqual(add_args['embeddings'], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(add_args['documents'], ["Recursive chunk 1.", "Recursive chunk 2."])
        self.assertEqual(add_args['metadatas'][0]['doc_id'], "dummy_document_recursive")
        self.assertEqual(add_args['metadatas'][0]['pdf_path'], fake_pdf_path_str)

    def test_upload_document_pdf_read_error(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self.MockFitzOpen.side_effect = Exception("PyMuPDF Read Error")
        fake_pdf_path_str = "error.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "error" # doc_id comes from here
            mock_path_constructor.return_value = mock_path_instance
            success, msg, doc_id, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error parsing PDF file error.pdf with PyMuPDF: PyMuPDF Read Error", msg)
        self.assertEqual(doc_id, "error")

    def test_upload_document_no_text_extracted(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self.mock_fitz_page_instance.get_text.return_value = ""
        fake_pdf_path_str = "empty_text.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "empty_text"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, num_chunks = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("No text could be extracted from PDF using PyMuPDF", msg)
        self.assertEqual(num_chunks, 0)

    def test_upload_document_embedding_error(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self.mock_main_embedding_model_instance.encode.side_effect = Exception("Embedding Error") # Changed from mock_bi_encoder_instance
        fake_pdf_path_str = "embed_error.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "embed_error"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error during embedding or storing chunks", msg)

    def test_upload_document_db_add_error(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self.mock_collection.add.side_effect = Exception("DB Add Error")
        fake_pdf_path_str = "db_add_error.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "db_add_error"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error during embedding or storing chunks", msg)

    # Test for NLTK punkt download process
    def test_nltk_punkt_download_flow(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        # First call to find() raises DownloadError, second call (after download) succeeds
        MockNltkDataFind.side_effect = [nltk.downloader.DownloadError, True]
        MockNltkDownload.return_value = True # Simulate successful download

        self._reinit_retriever(chunking_strategy="semantic")

        MockNltkDataFind.assert_any_call('tokenizers/punkt')
        MockNltkDownload.assert_called_once_with('punkt', quiet=True)
        self.assertIsNotNone(self.retriever.semantic_sentence_embedder) # Should still init if download "succeeds"

    # --- Semantic Chunking Specific Tests ---
    @patch('nltk.sent_tokenize')
    def test_chunk_semantically_logic(self, mock_sent_tokenize, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self._reinit_retriever(
            chunking_strategy="semantic",
            semantic_chunker_model_name='fake-semantic-chunker-model', # Ensure semantic_sentence_embedder is set
            semantic_breakpoint_type="percentile",
            semantic_breakpoint_amount=25, # Split if similarity < 25th percentile
            semantic_min_sentences=1
        )
        self.assertIsNotNone(self.retriever.semantic_sentence_embedder)

        sample_text = "Sentence one. Sentence two is related. Sentence three is very different. Sentence four is a bit related to three."
        mock_sentences = ["Sentence one.", "Sentence two is related.", "Sentence three is very different.", "Sentence four is a bit related to three."]
        mock_sent_tokenize.return_value = mock_sentences

        # Mock embeddings for clearer splits: S1, S2 close. S3 far. S4 far.
        clear_mock_embeddings = np.array([
            [1.0, 0.0, 0.0], # S1
            [0.9, 0.1, 0.0], # S2 (similar to S1)
            [0.0, 1.0, 0.0], # S3 (different from S1/S2)
            [0.0, 0.9, 0.1]  # S4 (similar to S3, different from S1/S2)
        ])
        # Expected similarities: sim(S1,S2) ~0.9. sim(S2,S3) ~0.01. sim(S3,S4) ~0.9.
        # If percentile amount is 25, threshold will be around 0.01. So split S2-S3.
        self.mock_semantic_embedder_instance.encode.return_value = clear_mock_embeddings # Use the correct mock instance

        expected_chunks = [
            "Sentence one. Sentence two is related.",
            "Sentence three is very different. Sentence four is a bit related to three."
        ]

        # Mock numpy.percentile to return a value that causes the desired split
        # Given similarities [~0.9, ~0.01, ~0.9], the 25th percentile would be ~0.01
        # If threshold_val is 0.1, then sim(S2,S3) < 0.1, so it splits.
        with patch('numpy.percentile', return_value=0.1) as mock_np_percentile:
            chunks = self.retriever._chunk_semantically(sample_text)

        mock_sent_tokenize.assert_called_once_with(sample_text)
        self.mock_semantic_embedder_instance.encode.assert_called_once_with(mock_sentences)
        mock_np_percentile.assert_called_once()
        self.assertEqual(chunks, expected_chunks)


    @patch('yawl.retriever.pdf_retriever.PdfRetriever._chunk_semantically')
    def test_upload_document_with_semantic_chunking_strategy(self, mock_chunk_semantically, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        self._reinit_retriever(chunking_strategy="semantic", semantic_chunker_model_name='fake-semantic-chunker-model')
        self.assertTrue(self.retriever.chunking_strategy == "semantic")
        self.assertIsNotNone(self.retriever.semantic_sentence_embedder)

        predefined_semantic_chunks = ["Semantic chunk A.", "Semantic chunk B is longer."]
        mock_chunk_semantically.return_value = predefined_semantic_chunks

        fake_pdf_path = "semantic_test.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "semantic_test"
            mock_path_constructor.return_value = mock_path_instance

            self.retriever.upload_document(fake_pdf_path)

        mock_chunk_semantically.assert_called_once_with("Page 1 text from PyMuPDF. Sentence two. Sentence three.")
        self.mock_splitter_instance.split_text.assert_not_called()

        self.mock_main_embedding_model_instance.encode.assert_called_with(predefined_semantic_chunks)
        self.mock_collection.add.assert_called_once()
        added_docs = self.mock_collection.add.call_args[1]['documents']
        self.assertEqual(added_docs, predefined_semantic_chunks)


    def test_upload_document_semantic_chunking_fallback_to_recursive(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
        # Simulate SentenceTransformer failing to load semantic_sentence_embedder
        # Store original side_effect
        original_st_side_effect = self.MockSentenceTransformer.side_effect

        def sentence_transformer_error_side_effect(model_name_or_path, **kwargs):
            if model_name_or_path == 'fake-embedding-model': # Main model loads fine
                return self.mock_main_embedding_model_instance
            elif model_name_or_path == 'fake-semantic-chunker-model-intended-to-fail':
                print(f"Simulating load failure for: {model_name_or_path}") # Debug print
                raise ValueError("Simulated failure to load semantic model")
            elif model_name_or_path == 'fake-cross-encoder-model':
                return self.mock_cross_encoder_instance
            print(f"Mock ST called with unhandled model: {model_name_or_path}")
            return MagicMock() # Fallback for other unexpected calls
        self.MockSentenceTransformer.side_effect = sentence_transformer_error_side_effect

        # Reinitialize retriever, expecting it to fallback
        self._reinit_retriever(
            chunking_strategy="semantic",
            semantic_chunker_model_name='fake-semantic-chunker-model-intended-to-fail'
        )

        # The __init__ logic in PdfRetriever should catch the error and set strategy to "recursive"
        self.assertEqual(self.retriever.chunking_strategy, "recursive", "Chunking strategy should fallback to recursive")
        self.assertIsNone(self.retriever.semantic_sentence_embedder, "Semantic embedder should be None after fallback")

        fake_pdf_path = "semantic_fallback.pdf"
        with patch('yawl.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "semantic_fallback"
            mock_path_constructor.return_value = mock_path_instance

            self.retriever.upload_document(fake_pdf_path)

        self.mock_splitter_instance.split_text.assert_called_once()
        self.mock_main_embedding_model_instance.encode.assert_called_with(["Recursive chunk 1.", "Recursive chunk 2."])

        # Restore original side_effect for other tests
        self.MockSentenceTransformer.side_effect = original_st_side_effect


    def test_retrieve_from_pdf_success_no_rerank(self, MockNltkDownload, MockNltkDataFind, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockFitzOpen, MockBM25Okapi):
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
        # n_results for dense query when no cross-encoder and no hybrid search:
        # num_initial_candidates = self.rerank_top_n_candidates if self.cross_encoder else top_k * 3
        # If cross-encoder is also None (as per this test's reinit), then it's top_k * 3.
        expected_n_results = 1 * 3
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=expected_n_results,
            where={"doc_id": "pdf1"},
            include=['documents', 'metadatas', 'distances']
        )
        self.assertEqual(len(snippets), 1)
        self.assertIn("Retrieved Doc 1 text.", snippets[0]['c'])
        self.assertIn("~rerank_score N/A", snippets[0]['c']) # Cross-encoder score
        self.assertIn("rrf: N/A", snippets[0]['c']) # RRF score
        self.assertEqual(snippets[0]['r'], 'retrieved_pdf_chunk')


    def test_retrieve_with_reranking_success(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        # Hybrid search should be enabled by default by _reinit_retriever call in setUp
        # This test focuses on the cross-encoder part after RRF (or dense if hybrid is off)
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', rerank_top_n=3, enable_hybrid_search=False)
        self.assertIsNotNone(self.retriever.cross_encoder)
        self.assertFalse(self.retriever.enable_hybrid_search) # Explicitly testing non-hybrid path to cross-encoder

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
            n_results=3, # This is rerank_top_n_candidates from _reinit_retriever
            where=None,
            include=['documents', 'metadatas', 'distances']
        )
        # cross_encoder receives the top `rerank_top_n` from dense results because hybrid is False
        self.mock_cross_encoder_instance.predict.assert_called_once_with([
            (query_text, 'Doc B text.'), (query_text, 'Doc C text.'), (query_text, 'Doc A text.') # Order from dense
        ])

        self.assertEqual(len(snippets), 2)
        # Based on cross_encoder scores [0.7 for A, 0.1 for B, 0.9 for C], after re-sorting
        self.assertIn("Doc C text.", snippets[0]['c'])
        self.assertIn("cross: 0.9000", snippets[0]['c'])
        self.assertIn("Doc A text.", snippets[1]['c'])
        self.assertIn("cross: 0.7000", snippets[1]['c'])


    def test_retrieve_reranking_cross_encoder_error(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        self._reinit_retriever(cross_encoder_name='fake-cross-encoder-model', enable_hybrid_search=False)
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

        # Should fall back to pre-cross-encoder order (dense only in this test setup)
        self.assertEqual(len(snippets), 1)
        self.assertIn("Doc B text.", snippets[0]['c'])
        self.assertIn("cross: N/A", snippets[0]['c'])


    def test_retrieve_hybrid_search_enabled_and_bm25_works(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        self._reinit_retriever(enable_hybrid_search=True, rerank_top_n=3) # Ensure hybrid is on
        self.assertTrue(self.retriever.enable_hybrid_search)
        self.assertIsNotNone(self.retriever.bm25_index, "BM25 index should be mocked via _reinit_retriever if BM25Okapi is patched")

        # Setup BM25 corpus (as if documents were uploaded)
        self.retriever.bm25_corpus_texts = ["BM25 doc 1", "BM25 doc 2: better match", "Common doc also in dense"]
        self.retriever.bm25_chunk_ids = ["bm25_id1", "bm25_id2", "common_id1"]
        self.retriever.is_bm25_index_built = False # Force rebuild

        # Mock BM25 scores (higher is better)
        # Query: "match" -> bm25_doc2 (0.8), common_doc (0.5), bm25_doc1 (0.2)
        self.mock_bm25_index_instance.get_scores.return_value = [0.2, 0.8, 0.5]

        # Mock Dense (ChromaDB) results (lower distance is better)
        # common_doc (0.1), dense_doc_only (0.2)
        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = {
            'ids': [['common_id1', 'dense_only_id1']],
            'documents': [['Common doc also in dense', 'Dense only doc text']],
            'metadatas': [[{'doc_id': 'pdf1', 'page_number': 1}], [{'doc_id': 'pdf2', 'page_number': 1}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.7]*3] # Query embedding

        # Mock Cross-encoder (higher is better)
        # Let's say cross-encoder boosts "BM25 doc 2: better match"
        def cross_encoder_predict_side_effect(pairs):
            scores = []
            for q, doc_text in pairs:
                if "BM25 doc 2" in doc_text: scores.append(0.95)
                elif "Common doc" in doc_text: scores.append(0.85)
                elif "Dense only" in doc_text: scores.append(0.75)
                else: scores.append(0.1)
            return scores
        self.mock_cross_encoder_instance.predict.side_effect = cross_encoder_predict_side_effect

        snippets = self.retriever.retrieve_from_pdf("match", top_k=2)

        self.MockBM25Okapi.assert_called_with([['bm25', 'doc', '1'], ['bm25', 'doc', '2:', 'better', 'match'], ['common', 'doc', 'also', 'in', 'dense']])
        self.mock_bm25_index_instance.get_scores.assert_called_once()
        self.mock_collection.query.assert_called_once() # Dense query
        self.mock_cross_encoder_instance.predict.assert_called_once() # Cross-encoder on RRF results

        # RRF expected order (approx): bm25_id2 (BM25 high), common_id1 (Dense high, BM25 mid), dense_only_id1 (Dense mid)
        # Cross-encoder re-ranks these.
        self.assertEqual(len(snippets), 2)
        self.assertIn("BM25 doc 2: better match", snippets[0]['c']) # Expected top due to cross-encoder
        self.assertIn("cross: 0.9500", snippets[0]['c'])
        self.assertIn("Common doc also in dense", snippets[1]['c'])
        self.assertIn("cross: 0.8500", snippets[1]['c'])


    def test_retrieve_hybrid_search_disabled(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        self._reinit_retriever(enable_hybrid_search=False, rerank_top_n=2)
        self.assertFalse(self.retriever.enable_hybrid_search)

        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = {
            'ids': [['id1', 'id2']], 'documents': [['Dense Doc 1', 'Dense Doc 2']],
            'metadatas': [[{}, {}]], 'distances': [[0.1, 0.2]]
        }
        self.mock_bi_encoder_instance.encode.return_value.tolist.return_value = [[0.1]*3]
        self.mock_cross_encoder_instance.predict.return_value = [0.9, 0.8] # Scores for Doc1, Doc2

        snippets = self.retriever.retrieve_from_pdf("query", top_k=1)

        self.mock_bm25_index_instance.get_scores.assert_not_called() # BM25 should not be called
        self.mock_collection.query.assert_called_once() # Dense query
        self.mock_cross_encoder_instance.predict.assert_called_once() # Cross-encoder still runs on dense

        self.assertEqual(len(snippets), 1)
        self.assertIn("Dense Doc 1", snippets[0]['c']) # Based on cross-encoder score
        self.assertIn("cross: 0.9000", snippets[0]['c'])
        self.assertIn("sparse: N/A", snippets[0]['c']) # No sparse score
        self.assertIn("rrf: N/A", snippets[0]['c'])   # No RRF score


    def test_retrieve_from_pdf_token_budget(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        self._reinit_retriever(cross_encoder_name=None, enable_hybrid_search=False) # No rerank/hybrid for budget test simplicity
        self.retriever.recall_budget_tokens = 25

        doc1_text = "Chunk one."
        doc2_text = "Chunk two is longer."
        mock_query_results = {
            'ids': [['id1', 'id2']], 'documents': [[doc1_text, doc2_text]],
            'metadatas': [[{'doc_id': 'd1', 'page_number': 1}], [{'doc_id': 'd1', 'page_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 2

        snippets = self.retriever.retrieve_from_pdf("Query for budgeting", top_k=2)

        self.assertEqual(len(snippets), 1)
        self.assertIn(doc1_text, snippets[0]['c'])

    def test_retrieve_from_pdf_no_results(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        self.mock_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        self.mock_collection.count.return_value = 0
        self.mock_bm25_index_instance.get_scores.return_value = [] # BM25 also returns no results

        snippets = self.retriever.retrieve_from_pdf("Query for no results")
        self.assertEqual(len(snippets), 0)

    def test_retrieve_unavailable_components(self, MockFitzOpen, MockSplitter, MockSentenceTransformer, MockChromaDBClient, MockBM25Okapi):
        original_embed_model = self.retriever.embedding_model
        self.retriever.embedding_model = None
        snippets = self.retriever.retrieve_from_pdf("Query")
        self.assertEqual(snippets, [])
        self.retriever.embedding_model = original_embed_model

        original_collection = self.retriever.collection
        self.retriever.collection = None
        snippets = self.retriever.retrieve_from_pdf("Query")
        self.assertEqual(snippets, [])
        self.retriever.collection = original_collection


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# End of yawl/tests/test_pdf_retriever.py
