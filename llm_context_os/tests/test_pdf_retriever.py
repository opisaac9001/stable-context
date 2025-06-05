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
    PYPDF2_AVAILABLE = True
except ImportError:
    PdfReader = None # type: ignore # Make it a placeholder if not available

@unittest.skipUnless(
    PYPDF2_AVAILABLE and LANGCHAIN_TEXT_SPLITTERS_AVAILABLE and \
    SENTENCE_TRANSFORMERS_AVAILABLE and CHROMADB_AVAILABLE,
    "One or more core dependencies (PyPDF2, langchain-text-splitters, sentence-transformers, chromadb) not available. Skipping PdfRetriever tests."
)
@patch('llm_context_os.retriever.pdf_retriever.PdfReader')
@patch('llm_context_os.retriever.pdf_retriever.RecursiveCharacterTextSplitter')
@patch('llm_context_os.retriever.pdf_retriever.SentenceTransformer')
@patch('llm_context_os.retriever.pdf_retriever.chromadb.PersistentClient')
class TestPdfRetriever(unittest.TestCase):

    def setUp(self, MockChromaDBClient, MockSentenceTransformer, MockSplitter, MockPdfReader): # Order matters, must match decorators
        # Store mock classes passed by decorators
        self.MockPdfReader = MockPdfReader
        self.MockSplitter = MockSplitter
        self.MockSentenceTransformer = MockSentenceTransformer
        self.MockChromaDBClient = MockChromaDBClient

        # Configure the mock instances that will be returned by the patched constructors
        self.mock_pdf_reader_instance = self.MockPdfReader.return_value
        self.mock_pdf_reader_instance.pages = [MagicMock(extract_text=MagicMock(return_value="Page 1 text."))]

        self.mock_splitter_instance = self.MockSplitter.return_value
        self.mock_splitter_instance.split_text.return_value = ["Text chunk 1.", "Text chunk 2."]

        self.mock_sentence_transformer_instance = self.MockSentenceTransformer.return_value
        # Ensure .tolist() is also part of the mock structure
        mock_embedding_array = MagicMock()
        mock_embedding_array.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] # Two embeddings for two chunks
        self.mock_sentence_transformer_instance.encode.return_value = mock_embedding_array


        self.mock_chromadb_client_instance = self.MockChromaDBClient.return_value
        self.mock_collection = MagicMock()
        self.mock_collection.count.return_value = 0 # Default: empty collection
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection

        self.mock_tokenizer = MagicMock()
        def mock_encode_for_len(text_input):
            return [1] * len(text_input.split())
        self.mock_tokenizer.encode = MagicMock(side_effect=mock_encode_for_len)

        self.test_db_base_path = "temp_test_pdf_retriever_db_unit"

        self.retriever = PdfRetriever(
            vector_db_path=self.test_db_base_path,
            embedding_model_name='fake-embedding-model',
            tokenizer=self.mock_tokenizer,
            recall_budget_tokens=100,
            chunk_size=10,
            chunk_overlap=2
        )
        # Reset mocks created by @patch at class level if they are to be configured per test
        # However, their return_value (the instance) is what we mostly configure here.
        # For clarity, ensure a clean state for instance method calls on these mocks.
        self.mock_pdf_reader_instance.reset_mock()
        self.mock_splitter_instance.reset_mock()
        self.mock_sentence_transformer_instance.reset_mock()
        self.mock_chromadb_client_instance.reset_mock()
        self.mock_collection.reset_mock()

        # Re-apply default return values for mocks that might be altered by specific tests
        self.MockPdfReader.return_value = self.mock_pdf_reader_instance
        self.MockSplitter.return_value = self.mock_splitter_instance
        self.MockSentenceTransformer.return_value = self.mock_sentence_transformer_instance
        self.MockChromaDBClient.return_value = self.mock_chromadb_client_instance
        self.mock_chromadb_client_instance.get_or_create_collection.return_value = self.mock_collection
        self.mock_sentence_transformer_instance.encode.return_value.tolist.return_value = [[0.1,0.2,0.3],[0.4,0.5,0.6]]


    def tearDown(self):
        # This path is constructed inside PdfRetriever: Path(vector_db_path) / DEFAULT_DB_SUBDIR (if used) or just vector_db_path
        # self.retriever.vector_db_path is the one to check and clean.
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

        self.MockPdfReader.assert_called_with(fake_pdf_path_str)
        self.mock_pdf_reader_instance.pages[0].extract_text.assert_called()
        self.mock_splitter_instance.split_text.assert_called_with("Page 1 text.")
        self.mock_sentence_transformer_instance.encode.assert_called_with(["Text chunk 1.", "Text chunk 2."])

        self.mock_collection.add.assert_called_once()
        add_args = self.mock_collection.add.call_args[1]
        self.assertEqual(len(add_args['ids']), 2)
        self.assertEqual(add_args['embeddings'], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(add_args['documents'], ["Text chunk 1.", "Text chunk 2."])
        self.assertEqual(add_args['metadatas'][0]['doc_id'], "dummy_document")
        self.assertEqual(add_args['metadatas'][0]['pdf_path'], fake_pdf_path_str)

    def test_upload_document_pdf_read_error(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.MockPdfReader.side_effect = Exception("PDF Read Error")
        fake_pdf_path_str = "error.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "error"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, _ = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("Error parsing PDF file", msg)

    def test_upload_document_no_text_extracted(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_pdf_reader_instance.pages = [MagicMock(extract_text=MagicMock(return_value=""))]
        fake_pdf_path_str = "empty_text.pdf"
        with patch('llm_context_os.retriever.pdf_retriever.Path') as mock_path_constructor:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.stem = "empty_text"
            mock_path_constructor.return_value = mock_path_instance
            success, msg, _, num_chunks = self.retriever.upload_document(fake_pdf_path_str)

        self.assertFalse(success)
        self.assertIn("No text could be extracted", msg)
        self.assertEqual(num_chunks, 0)

    def test_upload_document_embedding_error(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_sentence_transformer_instance.encode.side_effect = Exception("Embedding Error")
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

    def test_upload_document_db_add_error(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
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

    def test_retrieve_from_pdf_success(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_collection.count.return_value = 2
        self.mock_collection.query.return_value = {
            'ids': [['id1', 'id2']],
            'documents': [['Retrieved Doc 1 text.', 'Retrieved Doc 2 text.']],
            'metadatas': [[{'doc_id': 'pdf1', 'page_number': 1}, {'doc_id': 'pdf1', 'page_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        query_text = "Relevant query"
        # Default query embedding from setUp is [0.1,0.2,0.3] (single list for single query text)
        # So encode().tolist() should return that.
        self.mock_sentence_transformer_instance.encode.return_value.tolist.return_value = [[0.1,0.2,0.3]]


        snippets = self.retriever.retrieve_from_pdf(query_text, doc_ids=["pdf1"], top_k=1)

        self.mock_sentence_transformer_instance.encode.assert_called_with(query_text)
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1 * 2,
            where={"doc_id": "pdf1"},
            include=['documents', 'metadatas', 'distances']
        )
        self.assertEqual(len(snippets), 1)
        self.assertIn("Retrieved Doc 1 text.", snippets[0]['c'])
        self.assertEqual(snippets[0]['r'], 'retrieved_pdf_chunk')

    def test_retrieve_from_pdf_token_budget(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.retriever.recall_budget_tokens = 11

        doc1_text = "Chunk one." # Formatted: "Retrieved from 'd1' (Page 1, ~distance 0.1000): "Chunk one."" -> 11 words
        doc2_text = "Chunk two is longer." # Formatted: ~13 words

        def budget_test_tokenizer_encode(text_input): return [1] * len(text_input.split())
        self.retriever.tokenizer.encode = MagicMock(side_effect=budget_test_tokenizer_encode)

        mock_query_results = {
            'ids': [['id1', 'id2']],
            'documents': [[doc1_text, doc2_text]],
            'metadatas': [[{'doc_id': 'd1', 'page_number': 1}],
                          [{'doc_id': 'd1', 'page_number': 2}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_collection.query.return_value = mock_query_results
        self.mock_collection.count.return_value = 2

        snippets = self.retriever.retrieve_from_pdf("Query for budgeting")

        self.assertEqual(len(snippets), 1)
        self.assertIn(doc1_text, snippets[0]['c'])

    def test_retrieve_from_pdf_no_results(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
        self.mock_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        self.mock_collection.count.return_value = 0
        snippets = self.retriever.retrieve_from_pdf("Query for no results")
        self.assertEqual(len(snippets), 0)

    def test_retrieve_unavailable_components(self, MockPdfReader, MockSplitter, MockSentenceTransformer, MockChromaDBClient):
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

# End of llm_context_os/tests/test_pdf_retriever.py
