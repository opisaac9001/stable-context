# llm_context_os/tests/test_pdf_retriever.py
import unittest
import shutil
from pathlib import Path
import typing as t # Ensure typing is imported

from llm_context_os.retriever.pdf_retriever import PdfRetriever
from llm_context_os.retriever.pdf_retriever import MockTextSplitter # If needed for type hints or direct use

class TestPdfRetriever(unittest.TestCase):
    def setUp(self):
        self.test_data_dir_root = Path("data_test_pdf_retriever_module") # Unique root for this test class
        self.test_pdf_db_path = self.test_data_dir_root / "pdf_vector_db"
        self.dummy_pdf_dir = self.test_data_dir_root / "dummy_pdfs"

        # Clean up before each test method to ensure isolation
        if self.test_data_dir_root.exists():
            shutil.rmtree(self.test_data_dir_root)
        self.dummy_pdf_dir.mkdir(parents=True, exist_ok=True)
        # PdfRetriever's __init__ will create self.test_pdf_db_path

        self.retriever = PdfRetriever(
            vector_db_path=str(self.test_pdf_db_path),
            chunk_size=100, # Use smaller chunks for more granular testing
            chunk_overlap=15
        )

        # Create some dummy PDF files for testing
        self.pdf_path1 = self.dummy_pdf_dir / "doc1_apples.pdf"
        self.pdf_path2 = self.dummy_pdf_dir / "doc2_bananas_and_oranges.pdf"
        with open(self.pdf_path1, 'w') as f:
            f.write("This document is all about apples. Red apples, green apples. Apples are healthy.")
        with open(self.pdf_path2, 'w') as f:
            f.write("Information on bananas. And also some details about oranges. Bananas are yellow.")

    def tearDown(self):
        if self.test_data_dir_root.exists():
            shutil.rmtree(self.test_data_dir_root)

    def test_instantiation(self):
        self.assertIsNotNone(self.retriever)
        self.assertTrue(self.test_pdf_db_path.exists(), "Vector DB path should be created by PdfRetriever constructor.")
        self.assertIsInstance(self.retriever.text_splitter, MockTextSplitter) # Check if it uses the mock

    def test_upload_document_success(self):
        success, msg, doc_id, num_chunks = self.retriever.upload_document(str(self.pdf_path1))
        self.assertTrue(success, f"Upload failed: {msg}")
        self.assertEqual(doc_id, "doc1_apples")
        self.assertIsNotNone(num_chunks)
        self.assertGreater(num_chunks, 0, "Number of chunks should be positive for non-empty simulated text.")
        self.assertIn(doc_id, self.retriever.document_chunks)
        self.assertEqual(len(self.retriever.document_chunks[doc_id]), num_chunks)

    def test_upload_document_file_not_found(self):
        success, msg, doc_id, num_chunks = self.retriever.upload_document("non_existent_document.pdf")
        self.assertFalse(success)
        self.assertIsNone(doc_id)
        self.assertIsNone(num_chunks)
        self.assertIn("Error: PDF file not found", msg)

    def test_retrieve_from_pdf_specific_doc_found(self):
        self.retriever.upload_document(str(self.pdf_path1)) # doc1_apples: apples
        self.retriever.upload_document(str(self.pdf_path2)) # doc2_bananas_and_oranges: bananas, oranges

        # Query for "apples", targeting only doc1_apples
        snippets = self.retriever.retrieve_from_pdf("apples", doc_ids=["doc1_apples"])
        self.assertGreater(len(snippets), 0, "Should retrieve snippets for 'apples' from doc1_apples.")
        for snippet in snippets:
            self.assertIn("apples", snippet['c'].lower())
            self.assertIn("Source: doc1_apples", snippet['c'])
            self.assertNotIn("bananas", snippet['c'].lower()) # Should not have content from doc2

    def test_retrieve_from_pdf_specific_doc_not_found_in_doc(self):
        self.retriever.upload_document(str(self.pdf_path1)) # doc1_apples: apples
        self.retriever.upload_document(str(self.pdf_path2)) # doc2_bananas_and_oranges: bananas, oranges

        # Query for "bananas", but targeting only doc1_apples (which has apples)
        snippets = self.retriever.retrieve_from_pdf("bananas", doc_ids=["doc1_apples"])
        self.assertEqual(len(snippets), 0, "Should not retrieve 'bananas' snippets from doc1_apples.")

    def test_retrieve_from_pdf_all_docs(self):
        self.retriever.upload_document(str(self.pdf_path1))
        self.retriever.upload_document(str(self.pdf_path2))

        # Query for "bananas", should find in doc2_bananas_and_oranges
        snippets_bananas = self.retriever.retrieve_from_pdf("bananas", top_k=1)
        self.assertEqual(len(snippets_bananas), 1, "Should find one snippet for 'bananas'.")
        if snippets_bananas: # Check content if found
            self.assertIn("bananas", snippets_bananas[0]['c'].lower())
            self.assertIn("Source: doc2_bananas_and_oranges", snippets_bananas[0]['c'])

        # Query for "apples", should find in doc1_apples
        snippets_apples = self.retriever.retrieve_from_pdf("apples", top_k=1)
        self.assertEqual(len(snippets_apples), 1, "Should find one snippet for 'apples'.")
        if snippets_apples: # Check content if found
            self.assertIn("apples", snippets_apples[0]['c'].lower())
            self.assertIn("Source: doc1_apples", snippets_apples[0]['c'])

    def test_retrieve_no_match_in_any_doc(self):
        self.retriever.upload_document(str(self.pdf_path1))
        self.retriever.upload_document(str(self.pdf_path2))
        snippets = self.retriever.retrieve_from_pdf("exotic_fruits_like_durian")
        self.assertEqual(len(snippets), 0, "Should return no snippets for a query with no matches.")

    def test_retrieve_top_k_limit(self):
        # Upload doc1 which will result in multiple chunks due to short chunk_size
        # PdfRetriever's simulated text is long enough for this.
        _, _, _, num_chunks_doc1 = self.retriever.upload_document(str(self.pdf_path1))
        self.assertGreater(num_chunks_doc1, 2, "Need at least 3 chunks in dummy doc1 for this test.")

        snippets = self.retriever.retrieve_from_pdf("apples", top_k=2) # Query for content in doc1
        self.assertEqual(len(snippets), 2, "Should retrieve exactly top_k=2 snippets.")

        snippets_more_than_available = self.retriever.retrieve_from_pdf("apples", top_k=num_chunks_doc1 + 5)
        self.assertEqual(len(snippets_more_than_available), num_chunks_doc1,
                         "Should retrieve all available matching chunks if top_k is larger.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
