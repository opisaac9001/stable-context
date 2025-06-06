# llm_context_os/retriever/pdf_retriever.py
import typing as t
from pathlib import Path
import uuid
import os
import fitz  # PyMuPDF
import shutil # For demo cleanup
import time # For timestamp in metadata and demo cleanup delay

# Attempt to import RecursiveCharacterTextSplitter
LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = True
    print("Successfully imported RecursiveCharacterTextSplitter.")
except ImportError:
    print("Warning: langchain-text-splitters not found. PDF text splitting will be basic or disabled.")
    RecursiveCharacterTextSplitter = None # Placeholder

# Conditional imports for SentenceTransformer and ChromaDB
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("Successfully imported SentenceTransformer.")
except ImportError:
    print("Warning: sentence-transformers library not found. PDF embedding will not function.")
    SentenceTransformer = None

CHROMADB_AVAILABLE = False
try:
    import chromadb
    CHROMADB_AVAILABLE = True
    print("Successfully imported chromadb.")
except ImportError:
    print("Warning: chromadb library not found. PDF vector storage will not function.")
    chromadb = None

class PdfRetriever:
    DEFAULT_COLLECTION_NAME = "pdf_document_chunks_v1"
    def __init__(self,
                 vector_db_path: str = "data/vector_dbs/pdf_rag_db", # More specific default path
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 tokenizer = None,
                 recall_budget_tokens: int = 1024,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 collection_name: t.Optional[str] = None,
                 cross_encoder_model_name: t.Optional[str] = "ms-marco-MiniLM-L-6-v2", # Default from plan
                 rerank_top_n_candidates: int = 20): # Default from plan
        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        self.tokenizer = tokenizer
        self.recall_budget_tokens = recall_budget_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_collection_name = collection_name or self.DEFAULT_COLLECTION_NAME

        self.cross_encoder_model_name = cross_encoder_model_name
        self.rerank_top_n_candidates = rerank_top_n_candidates
        self.cross_encoder = None

        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"[PdfRetriever] Initialized SentenceTransformer (bi-encoder) model: {self.embedding_model_name}")
            except Exception as e:
                print(f"[PdfRetriever] Error initializing SentenceTransformer (bi-encoder) model '{self.embedding_model_name}': {e}")

            if self.cross_encoder_model_name:
                try:
                    self.cross_encoder = SentenceTransformer(self.cross_encoder_model_name)
                    print(f"[PdfRetriever] Initialized CrossEncoder model: {self.cross_encoder_model_name}")
                except Exception as e:
                    print(f"[PdfRetriever] Error initializing CrossEncoder model '{self.cross_encoder_model_name}': {e}")
                    self.cross_encoder = None # Ensure it's None if init fails
        else:
            print("[PdfRetriever] SentenceTransformers library not available. Embedding and CrossEncoder models not loaded.")

        self.db_client = None
        self.collection = None
        if CHROMADB_AVAILABLE and chromadb:
            try:
                self.vector_db_path.mkdir(parents=True, exist_ok=True)
                resolved_db_path = str(self.vector_db_path.resolve())
                self.db_client = chromadb.PersistentClient(path=resolved_db_path)
                self.collection = self.db_client.get_or_create_collection(
                    name=self.db_collection_name
                    # metadata={"hnsw:space": "cosine"} # Example if needed
                )
                print(f"[PdfRetriever] Initialized ChromaDB client at '{resolved_db_path}' and collection '{self.db_collection_name}'. Count: {self.collection.count()}")
            except Exception as e:
                print(f"[PdfRetriever] Error initializing ChromaDB client or collection at '{resolved_db_path}': {e}")
        else:
            print("[PdfRetriever] ChromaDB library not available. Vector storage not initialized.")

        if LANGCHAIN_TEXT_SPLITTERS_AVAILABLE and RecursiveCharacterTextSplitter is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            print(f"[PdfRetriever] Initialized RecursiveCharacterTextSplitter with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        else:
            print("[PdfRetriever] Warning: langchain-text-splitters not found or import failed. Text splitting will not be effective.")
            self.text_splitter = None

        print(f"[PdfRetriever] Initialized. Embedding: '{self.embedding_model_name if self.embedding_model else 'N/A'}'")


    def upload_document(self, pdf_path: str) -> t.Tuple[bool, str, t.Optional[str], t.Optional[int]]:
        print(f"[PdfRetriever] upload_document called for PDF: '{pdf_path}'")
        pdf_file = Path(pdf_path)
        if not pdf_file.exists() or not pdf_file.is_file():
            msg = f"Error: PDF file not found or is not a file: {pdf_path}"
            print(f"[PdfRetriever] {msg}")
            return False, msg, None, None

        doc_id = pdf_file.stem # Use filename stem as document ID
        print(f"[PdfRetriever] Processing document ID: {doc_id} from path: {pdf_path}")

        try:
            doc = fitz.open(pdf_path) # Open PDF with PyMuPDF
            extracted_text_parts = []
            num_pages_processed = 0
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                try:
                    page_text = page.get_text("text") # Extract plain text
                    if page_text:
                        extracted_text_parts.append(page_text)
                    num_pages_processed +=1
                except Exception as e_page:
                    print(f"[PdfRetriever] Warning: Could not extract text from page {page_num + 1} of {doc_id} using PyMuPDF: {e_page}")

            doc.close() # Close the document

            if not extracted_text_parts:
                msg = f"Error: No text could be extracted from PDF using PyMuPDF: {pdf_path}"
                print(f"[PdfRetriever] {msg}")
                return False, msg, doc_id, 0

            full_text_content = "\n\n".join(extracted_text_parts) # Join pages with double newlines for better separation
            print(f"[PdfRetriever] Extracted ~{len(full_text_content)} characters from {num_pages_processed} pages in {doc_id} using PyMuPDF.")

        except Exception as e:
            # This will catch fitz.errors.FitzError for invalid PDFs too
            msg = f"Error parsing PDF file {pdf_path} with PyMuPDF: {e}"
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0 # Return doc_id if available, else it might be None if error was early

        if not self.text_splitter:
            msg = "Error: Text splitter not available (e.g., langchain-text-splitters not installed). Cannot process document."
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0

        print(f"[PdfRetriever] Splitting text for document {doc_id}...")
        try:
            # RecursiveCharacterTextSplitter's split_text method returns a list of strings.
            raw_chunks = self.text_splitter.split_text(full_text_content)
        except Exception as e_split:
            msg = f"Error during text splitting for document {doc_id}: {e_split}"
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0

        if not self.embedding_model:
            return False, "Embedding model not available. Cannot process document.", doc_id, 0
        if not self.collection:
            return False, "Vector database (ChromaDB collection) not available. Cannot process document.", doc_id, 0

        chunk_texts_to_embed = []
        chunk_metadatas = []
        chunk_ids = []

        for i, chunk_text_content in enumerate(raw_chunks):
            chunk_id_val = str(uuid.uuid4())
            chunk_texts_to_embed.append(chunk_text_content)
            page_num_est = (i * (self.chunk_size - self.chunk_overlap)) // self.chunk_size + 1
            chunk_metadatas.append({
                'doc_id': doc_id, # Original document ID (filename stem)
                'pdf_path': str(pdf_path), # Store full path to original PDF
                'page_number': page_num_est,
                'chunk_index': i,
                'text_length_chars': len(chunk_text_content),
                'timestamp': time.time()
            })
            chunk_ids.append(chunk_id_val) # Unique ID for each chunk

        if not chunk_texts_to_embed:
            msg = f"No text chunks produced for document {doc_id} after splitting."
            print(f"[PdfRetriever] {msg}")
            return True, msg, doc_id, 0

        try:
            print(f"[PdfRetriever] Embedding {len(chunk_texts_to_embed)} chunks for document {doc_id}...")
            embeddings = self.embedding_model.encode(chunk_texts_to_embed).tolist()

            print(f"[PdfRetriever] Adding {len(chunk_ids)} chunks to ChromaDB collection '{self.db_collection_name}' for document {doc_id}...")
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts_to_embed, # Store the actual text of the chunk
                metadatas=chunk_metadatas
            )
            num_processed_chunks = len(chunk_ids)
            msg = f"Successfully extracted, embedded, and stored {num_processed_chunks} chunks for document: {doc_id} in collection '{self.db_collection_name}'."
            print(f"[PdfRetriever] {msg}")
            return True, msg, doc_id, num_processed_chunks
        except Exception as e:
            msg = f"Error during embedding or storing chunks for document {doc_id}: {e}"
            print(f"[PdfRetriever] {msg}")
            # import traceback; traceback.print_exc() # For debugging
            return False, msg, doc_id, 0


    def retrieve_from_pdf(self, query_text: str, doc_ids: t.Optional[t.List[str]] = None, top_k: int = 3) -> t.List[t.Dict[str, str]]:
        print(f"\n[PdfRetriever] retrieve_from_pdf called. Query: '{query_text[:50]}...', Target Docs: {doc_ids or 'all'}, Top K: {top_k}")

        if not self.embedding_model:
            print("[PdfRetriever] Error: Embedding model not available for retrieval.")
            return []
        if not self.collection:
            print("[PdfRetriever] Error: Vector database (ChromaDB collection) not available for retrieval.")
            return []
        if not self.tokenizer:
            # This was set to BasicCharTokenizer in __init__ if tiktoken failed, so it should always exist.
            # However, if it somehow became None, or if BasicCharTokenizer is deemed insufficient:
            print("[PdfRetriever] Error: Tokenizer not available for budgeting. Cannot reliably size snippets.")
            return []

        try:
            print(f"[PdfRetriever] Embedding query for retrieval: '{query_text[:100]}...'")
            query_embedding = self.embedding_model.encode(query_text).tolist()
        except Exception as e:
            print(f"[PdfRetriever] Error embedding query '{query_text[:100]}...': {e}")
            return []

        chroma_filter: t.Optional[t.Dict[str, t.Any]] = None
        if doc_ids:
            if len(doc_ids) == 1:
                chroma_filter = {"doc_id": doc_ids[0]}
            else:
                # Ensure doc_ids are strings, as ChromaDB metadata values are typically str, int, float, bool
                chroma_filter = {"doc_id": {"$in": [str(did) for did in doc_ids]}}
            print(f"[PdfRetriever] Applying ChromaDB filter: {chroma_filter}")

        try:
            # Retrieve more candidates for potential re-ranking.
            # Use self.rerank_top_n_candidates if re-ranker is present, otherwise a smaller set.
            num_initial_candidates = self.rerank_top_n_candidates if self.cross_encoder else top_k * 3

            print(f"[PdfRetriever] Querying ChromaDB collection (requesting {num_initial_candidates} candidates).")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_initial_candidates,
                where=chroma_filter,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            print(f"[PdfRetriever] Error querying ChromaDB: {e}")
            # import traceback; traceback.print_exc()
            return []

        retrieved_snippets: t.List[t.Dict[str, str]] = []
        current_token_count = 0

        if not results or not results.get('ids') or not results['ids'][0]:
            print("[PdfRetriever] No initial results from ChromaDB query for the given criteria.")
            return []

        # Process results (ids, documents, metadatas, distances are all lists of lists, take the first list for our single query)
        result_ids = results['ids'][0]
        result_documents = results['documents'][0] if results['documents'] else [""] * len(result_ids)
        result_metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(result_ids)
        result_distances = results['distances'][0] if results['distances'] else [float('inf')] * len(result_ids)

        candidate_items = []
        for i in range(len(result_ids)):
            doc_text = result_documents[i] if result_documents[i] is not None else "Content not available"
            metadata = result_metadatas[i] if result_metadatas[i] is not None else {}
            distance = result_distances[i] if result_distances[i] is not None else float('inf')

            if doc_text.strip().lower() == query_text.strip().lower(): # Avoid exact match of query
                print(f"  Skipping retrieved chunk identical to query: '{doc_text[:50]}...'")
                continue

            candidate_items.append({
                "text": doc_text,
                "metadata": metadata,
                "distance": distance,
                "id": result_ids[i]
            })

        # Re-ranking step
        if self.cross_encoder and candidate_items:
            print(f"[PdfRetriever] Re-ranking {len(candidate_items)} candidates with CrossEncoder: {self.cross_encoder_model_name}")
            try:
                pairs = [(query_text, item['text']) for item in candidate_items]
                cross_scores = self.cross_encoder.predict(pairs)

                for i, item in enumerate(candidate_items):
                    item['cross_score'] = cross_scores[i]

                # Sort by cross_score in descending order (higher is better)
                candidate_items.sort(key=lambda x: x.get('cross_score', -float('inf')), reverse=True)
                print(f"[PdfRetriever] Re-ranking complete. Top score: {candidate_items[0].get('cross_score', 'N/A'):.4f} if results exist.")
            except Exception as e_rerank:
                print(f"[PdfRetriever] Error during CrossEncoder prediction/re-ranking: {e_rerank}. Proceeding with original ranking.")
        else:
            # If no cross-encoder, ChromaDB results are already sorted by distance (ascending, lower is better)
            # No explicit sort needed here as we iterate and take top_k.
            pass


        for res_item in candidate_items: # Iterate through potentially re-ranked items
            if len(retrieved_snippets) >= top_k: # Apply final top_k limit
                break

            doc_id_meta = res_item['metadata'].get('doc_id', 'UnknownDoc')
            page_num_meta = res_item['metadata'].get('page_number', 'N/A')
            cross_score_val = res_item.get('cross_score', 'N/A')
            if isinstance(cross_score_val, float):
                cross_score_str = f"{cross_score_val:.4f}"
            else:
                cross_score_str = "N/A"

            snippet_text = (f"Retrieved from '{doc_id_meta}' "
                            f"(Page {page_num_meta}, ~distance {res_item['distance']:.4f}, ~rerank_score {cross_score_str}): "
                            f"\"{res_item['text']}\"")

            try:
                # Ensure tokenizer is available and has an encode method
                if hasattr(self.tokenizer, 'encode'):
                    snippet_tokens = len(self.tokenizer.encode(snippet_text))
                elif hasattr(self.tokenizer, 'count_tokens'): # Fallback to count_tokens if encode not present
                    snippet_tokens = self.tokenizer.count_tokens(snippet_text)
                else: # Last resort, character count
                    print(f"[PdfRetriever] Warning: Tokenizer has no 'encode' or 'count_tokens' method. Using char length for budgeting.")
                    snippet_tokens = len(snippet_text)
            except Exception as e_tok:
                print(f"[PdfRetriever] Warning: Tokenizer error for snippet budgeting ('{str(e_tok)}'). Falling back to char length.")
                snippet_tokens = len(snippet_text)

            if current_token_count + snippet_tokens <= self.recall_budget_tokens:
                retrieved_snippets.append({'r': 'retrieved_pdf_chunk', 'c': snippet_text})
                current_token_count += snippet_tokens
                log_score = f"cross_score: {cross_score_str}" if 'cross_score' in res_item else f"dist: {res_item['distance']:.4f}"
                print(f"  Added snippet ({log_score}, tokens: {snippet_tokens}): '{snippet_text[:100]}...'")
            else:
                print(f"[PdfRetriever] Recall budget ({self.recall_budget_tokens} tokens) reached. Current: {current_token_count}, Snippet: {snippet_tokens}. Stopping.")
                break

        print(f"[PdfRetriever] Returning {len(retrieved_snippets)} PDF snippets. Total tokens (estimated): {current_token_count}/{self.recall_budget_tokens}")
        return retrieved_snippets


if __name__ == '__main__':
    print("--- Testing PdfRetriever with Embedding and ChromaDB (Placeholder Retrieval) ---")
    dummy_pdf_root = Path("data/dummy_pdfs_for_retriever_demo") # Changed demo path slightly
    dummy_pdf_root.mkdir(parents=True, exist_ok=True)

    # Define a more specific DB path for the demo to allow easy cleanup
    DEMO_DB_BASE_PATH = "data/vector_dbs_demo" # Base for all demo DBs
    # PdfRetriever will create a subdirectory under this named "pdf_rag_db" by default or per its DEFAULT_DB_SUBDIR if that exists
    # For this demo, we'll assume the default internal logic for PdfRetriever path construction.
    # The actual path will be DEMO_DB_BASE_PATH / "pdf_rag_db" (if PdfRetriever uses its own default subdir logic)
    # Or simply DEMO_DB_BASE_PATH if it doesn't add a subdir.
    # Let's use the retriever's constructed path for cleanup.

    # Check library availability for demo
    if not LANGCHAIN_TEXT_SPLITTERS_AVAILABLE:
        print("\n!!! Langchain text splitters not available. Demo will be limited. !!!")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n!!! SentenceTransformers not available. Demo will be limited (no embeddings). !!!")
    if not CHROMADB_AVAILABLE:
        print("\n!!! ChromaDB not available. Demo will be limited (no vector storage). !!!")


    # Initialize retriever
    pdf_retriever = PdfRetriever(
        vector_db_path=DEMO_DB_BASE_PATH, # Pass the base path
        chunk_size=150, # Smaller chunk size for more chunks in demo
        chunk_overlap=30,
        recall_budget_tokens=500
    )

    # Determine the actual DB path used by the retriever instance for cleanup
    actual_demo_db_path = pdf_retriever.vector_db_path
    if pdf_retriever.db_client: # If client initialized, it means path was resolved
        # ChromaDB client path might be different if it appends stuff, but PersistentClient takes the exact path.
        # The PdfRetriever itself forms the full path including its default subdir.
        # So, pdf_retriever.vector_db_path is the one to clean.
        # The init logic makes vector_db_path = Path(vector_db_path), so it's already a Path object.
        pass # actual_demo_db_path is already correct.

    print(f"Demo will use ChromaDB path: {actual_demo_db_path.resolve()}")
    # Cleanup before demo if it exists from a previous failed run
    if actual_demo_db_path.exists():
        print(f"Cleaning up existing demo DB at: {actual_demo_db_path}")
        shutil.rmtree(actual_demo_db_path)
    actual_demo_db_path.mkdir(parents=True, exist_ok=True)


    dummy_pdf_path1 = dummy_pdf_root / "climate_report_demo.pdf"
    dummy_pdf_path2 = dummy_pdf_root / "ai_ethics_demo.pdf"

    # Create dummy text files. PyMuPDF will fail to open these, which is fine for this demo's purpose
    # as we are testing the library switch and error handling, not successful PDF parsing in the demo.
    # A real test with actual PDFs would be separate.
    with open(dummy_pdf_path1, 'w') as f:
        f.write("This is not a real PDF, but a text file for demo purposes. PyMuPDF should gracefully fail to open it.")
    with open(dummy_pdf_path2, 'w') as f:
        f.write("Another text file disguised as a PDF for the PdfRetriever demo.")

    if pdf_retriever.embedding_model and pdf_retriever.collection and pdf_retriever.text_splitter:
        print("\n--- Upload Documents (will attempt embedding and ChromaDB storage) ---")
        # Wrap upload_document in try-except as PyMuPDF will likely fail on these text files
        try:
            print(f"Attempting to upload (expected to fail gracefully with PyMuPDF): {dummy_pdf_path1}")
            s1, m1, id1, nc1 = pdf_retriever.upload_document(str(dummy_pdf_path1))
            print(f"Upload 1 ('{id1}'): Success={s1}, Chunks={nc1}, Msg: {m1}")
        except Exception as e_upload1:
            print(f"Caught exception during upload of {dummy_pdf_path1}: {e_upload1}")

        try:
            print(f"Attempting to upload (expected to fail gracefully with PyMuPDF): {dummy_pdf_path2}")
            s2, m2, id2, nc2 = pdf_retriever.upload_document(str(dummy_pdf_path2))
            print(f"Upload 2 ('{id2}'): Success={s2}, Chunks={nc2}, Msg: {m2}")
        except Exception as e_upload2:
            print(f"Caught exception during upload of {dummy_pdf_path2}: {e_upload2}")

        # Since uploads are expected to fail with text files, collection count will be 0
        if pdf_retriever.collection:
            print(f"\nTotal items in ChromaDB collection '{pdf_retriever.db_collection_name}': {pdf_retriever.collection.count()}")

        # Retrieval attempts will also likely yield no results due to failed uploads
        print("\n--- Retrieve from all PDFs (query: 'climate change mitigation') ---")
        retrieved_all = pdf_retriever.retrieve_from_pdf("climate change mitigation", top_k=3)
        if not retrieved_all: print("  (Retrieval expected to yield no results due to upload failures or empty DB)")

    else:
        print("\n[Demo] PdfRetriever not fully initialized (missing SentenceTransformer, ChromaDB, or TextSplitter). Skipping upload/retrieve demo.")

    # Cleanup dummy files and DB directory (same as before)
    try:
        if dummy_pdf_path1.exists(): os.remove(dummy_pdf_path1)
        if dummy_pdf_path2.exists(): os.remove(dummy_pdf_path2)
        if not any(dummy_pdf_root.iterdir()):
            os.rmdir(dummy_pdf_root)

        if actual_demo_db_path.exists():
            print(f"\nCleaning up demo DB at: {actual_demo_db_path}")
            if hasattr(pdf_retriever, 'collection') and pdf_retriever.collection: del pdf_retriever.collection
            if hasattr(pdf_retriever, 'db_client') and pdf_retriever.db_client: del pdf_retriever.db_client
            time.sleep(0.1)
            shutil.rmtree(actual_demo_db_path)
            print(f"Successfully removed demo DB directory: {actual_demo_db_path}")
            if actual_demo_db_path.parent == Path(DEMO_DB_BASE_PATH).resolve() and not os.listdir(DEMO_DB_BASE_PATH):
                 shutil.rmtree(DEMO_DB_BASE_PATH)
                 print(f"Successfully removed demo base DB directory: {DEMO_DB_BASE_PATH}")
    except OSError as e:
        print(f"Error during demo cleanup: {e}")

    print("\nPdfRetriever demo complete.")
    # The following was a duplicate of the demo setup from an earlier version, removing it.
    # dummy_pdf_root.mkdir(parents=True, exist_ok=True)
    # ... (rest of duplicated demo code removed) ...
