# llm_context_os/retriever/pdf_retriever.py
import typing as t
from pathlib import Path
import uuid
import os
from PyPDF2 import PdfReader # Added for PDF text extraction
import shutil # For demo cleanup

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
                 recall_budget_tokens: int = 1024, # Retained for retrieve method, not used in upload
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 collection_name: t.Optional[str] = None):
        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        self.tokenizer = tokenizer
        self.recall_budget_tokens = recall_budget_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_collection_name = collection_name or self.DEFAULT_COLLECTION_NAME

        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"[PdfRetriever] Initialized SentenceTransformer model: {self.embedding_model_name}")
            except Exception as e:
                print(f"[PdfRetriever] Error initializing SentenceTransformer model '{self.embedding_model_name}': {e}")
        else:
            print("[PdfRetriever] SentenceTransformers library not available. Embedding model not loaded.")

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
            reader = PdfReader(pdf_path)
            extracted_text_parts = []
            # page_map = {} # To map char offset to page number for chunks - for future refinement
            # current_char_offset = 0
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text: # Ensure text was extracted
                        extracted_text_parts.append(page_text)
                        # page_map[current_char_offset] = page_num + 1
                        # current_char_offset += len(page_text)
                except Exception as e_page: # Handle errors on specific pages
                    print(f"[PdfRetriever] Warning: Could not extract text from page {page_num + 1} of {doc_id}: {e_page}")

            if not extracted_text_parts:
                msg = f"Error: No text could be extracted from PDF: {pdf_path}"
                print(f"[PdfRetriever] {msg}")
                return False, msg, doc_id, 0 # Return doc_id as it was identified

            full_text_content = "\n".join(extracted_text_parts) # Join pages with newlines
            print(f"[PdfRetriever] Extracted ~{len(full_text_content)} characters from {len(reader.pages)} pages in {doc_id}.")

        except Exception as e:
            msg = f"Error parsing PDF file {pdf_path}: {e}"
            print(f"[PdfRetriever] {msg}")
            return False, msg, None, 0 # No doc_id if parsing failed early

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
            # Retrieve more initially to allow for re-ranking or filtering if needed, then apply top_k.
            # Also gives more options if some results are too long for the token budget.
            num_candidates = top_k * 3
            print(f"[PdfRetriever] Querying ChromaDB collection (requesting {num_candidates} candidates).")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_candidates,
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
        result_documents = results['documents'][0] if results['documents'] else [""] * len(result_ids) # Handle empty documents list
        result_metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(result_ids)
        result_distances = results['distances'][0] if results['distances'] else [float('inf')] * len(result_ids)

        processed_results = []
        for i in range(len(result_ids)):
            doc_text = result_documents[i] if result_documents[i] is not None else "Content not available"
            metadata = result_metadatas[i] if result_metadatas[i] is not None else {}
            distance = result_distances[i] if result_distances[i] is not None else float('inf')

            if doc_text.strip().lower() == query_text.strip().lower(): # Avoid exact match of query
                print(f"  Skipping retrieved chunk identical to query: '{doc_text[:50]}...'")
                continue

            processed_results.append({
                "text": doc_text,
                "metadata": metadata,
                "distance": distance
            })

        # ChromaDB query results are typically sorted by distance already.

        for res_item in processed_results:
            if len(retrieved_snippets) >= top_k: # Apply final top_k limit
                break

            doc_id_meta = res_item['metadata'].get('doc_id', 'UnknownDoc')
            page_num_meta = res_item['metadata'].get('page_number', 'N/A')

            snippet_text = f"Retrieved from '{doc_id_meta}' (Page {page_num_meta}, ~distance {res_item['distance']:.4f}): \"{res_item['text']}\""

            try:
                snippet_tokens = len(self.tokenizer.encode(snippet_text))
            except Exception as e_tok:
                print(f"[PdfRetriever] Warning: Tokenizer error for snippet budgeting ('{str(e_tok)}'). Falling back to char length.")
                snippet_tokens = len(snippet_text)

            if current_token_count + snippet_tokens <= self.recall_budget_tokens:
                retrieved_snippets.append({'r': 'retrieved_pdf_chunk', 'c': snippet_text})
                current_token_count += snippet_tokens
                print(f"  Added snippet (dist: {res_item['distance']:.4f}, tokens: {snippet_tokens}): '{snippet_text[:100]}...'")
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

    # Create dummy text files (PyPDF2 will attempt to parse them)
    with open(dummy_pdf_path1, 'w') as f:
        f.write("Climate Change Overview Document.\nSection 1: Introduction to climate science. The earth's climate is warming significantly. "
                "Section 2: Impacts of global warming on ecosystems and human society. Rising sea levels are a major global concern. We must act now. "
                "Section 3: Mitigation strategies including renewable energy and carbon capture technologies. International cooperation is absolutely key for success.")
    with open(dummy_pdf_path2, 'w') as f:
        f.write("The Future of AI.\nChapter 1: Current state of artificial intelligence. LLMs show great promise for humanity. "
                "Chapter 2: AI Ethics and Governance. Ensuring fairness, accountability, and transparency is crucial. "
                "Chapter 3: Speculative future applications of AGI. Potential for solving complex global problems like disease and poverty.")

    if pdf_retriever.embedding_model and pdf_retriever.collection and pdf_retriever.text_splitter:
        print("\n--- Upload Documents (will attempt embedding and ChromaDB storage) ---")
        s1, m1, id1, nc1 = pdf_retriever.upload_document(str(dummy_pdf_path1))
        print(f"Upload 1 ('{id1}'): Success={s1}, Chunks={nc1}, Msg: {m1}")
        s2, m2, id2, nc2 = pdf_retriever.upload_document(str(dummy_pdf_path2))
        print(f"Upload 2 ('{id2}'): Success={s2}, Chunks={nc2}, Msg: {m2}")

        if id1 and pdf_retriever.collection:
            print(f"\nTotal items in ChromaDB collection '{pdf_retriever.db_collection_name}': {pdf_retriever.collection.count()}")

        print("\n--- Retrieve from all PDFs (query: 'climate change mitigation') ---")
        # This will currently print "not implemented" and return []
        retrieved_all = pdf_retriever.retrieve_from_pdf("climate change mitigation", top_k=3)
        if not retrieved_all: print("  (Retrieval not yet implemented or no results)")

        if id2:
            print(f"\n--- Retrieve from '{id2}' only (query: 'AI ethics LLMs') ---")
            retrieved_specific = pdf_retriever.retrieve_from_pdf("AI ethics LLMs", doc_ids=[id2], top_k=2)
            if not retrieved_specific: print("  (Retrieval not yet implemented or no results)")
    else:
        print("\n[Demo] PdfRetriever not fully initialized (missing SentenceTransformer, ChromaDB, or TextSplitter). Skipping upload/retrieve demo.")

    # Cleanup dummy files and DB directory
    try:
        if dummy_pdf_path1.exists(): os.remove(dummy_pdf_path1)
        if dummy_pdf_path2.exists(): os.remove(dummy_pdf_path2)
        if not any(dummy_pdf_root.iterdir()): # Check if directory is empty
            os.rmdir(dummy_pdf_root)

        # Cleanup the specific DB path used by this retriever instance
        if actual_demo_db_path.exists():
            print(f"\nCleaning up demo DB at: {actual_demo_db_path}")
            # Ensure client/collection are not locking files if they were initialized
            if hasattr(pdf_retriever, 'collection') and pdf_retriever.collection: del pdf_retriever.collection
            if hasattr(pdf_retriever, 'db_client') and pdf_retriever.db_client: del pdf_retriever.db_client
            time.sleep(0.1) # Brief pause for file handles
            shutil.rmtree(actual_demo_db_path)
            print(f"Successfully removed demo DB directory: {actual_demo_db_path}")
            # Clean up base path if it's now empty and was part of the demo structure
            if actual_demo_db_path.parent == Path(DEMO_DB_BASE_PATH).resolve() and not os.listdir(DEMO_DB_BASE_PATH):
                 shutil.rmtree(DEMO_DB_BASE_PATH)
                 print(f"Successfully removed demo base DB directory: {DEMO_DB_BASE_PATH}")

    except OSError as e:
        print(f"Error during demo cleanup: {e}")

    print("\nPdfRetriever demo complete.")
    dummy_pdf_root.mkdir(parents=True, exist_ok=True)

    dummy_pdf_path1 = dummy_pdf_root / "climate_change_overview.pdf"
    dummy_pdf_path2 = dummy_pdf_root / "future_of_ai_report.pdf"

    # Create dummy PDF files with some content for the demo
    with open(dummy_pdf_path1, 'w') as f:
        f.write("Climate Change Overview Document. Section 1: Introduction to climate science. The earth's climate is warming. "
                "Section 2: Impacts of global warming on ecosystems and human society. Rising sea levels are a major concern. "
                "Section 3: Mitigation strategies including renewable energy and carbon capture technologies. International cooperation is key.")
    with open(dummy_pdf_path2, 'w') as f:
        f.write("The Future of AI. Chapter 1: Current state of artificial intelligence. LLMs show great promise. "
                "Chapter 2: AI Ethics and Governance. Ensuring fairness and transparency. "
                "Chapter 3: Speculative future applications of AGI. Potential for solving complex global problems.")

    # Initialize retriever with a smaller chunk size for more demo chunks
    pdf_retriever = PdfRetriever(
        vector_db_path="data/pdf_retriever_test_db_main",
        chunk_size=100, # Small chunk size for demo
        chunk_overlap=20, # Small overlap for demo
        recall_budget_tokens=500 # Budget for total length of returned snippets
    )
    if LANGCHAIN_TEXT_SPLITTERS_AVAILABLE:
        print("RecursiveCharacterTextSplitter is available for this demo.")
    else:
        print("RecursiveCharacterTextSplitter is NOT available. Demo might not fully work as expected for splitting.")


    print("\n--- Upload Documents ---")
    s1, m1, id1, nc1 = pdf_retriever.upload_document(str(dummy_pdf_path1))
    print(f"Upload 1: {s1}, {m1[:50]}..., ID: {id1}, Chunks: {nc1}")
    s2, m2, id2, nc2 = pdf_retriever.upload_document(str(dummy_pdf_path2))
    print(f"Upload 2: {s2}, {m2[:50]}..., ID: {id2}, Chunks: {nc2}")

    print(f"\nTotal documents processed: {len(pdf_retriever.document_chunks)}")

    print("\n--- Retrieve from all PDFs (query: 'climate change mitigation') ---")
    retrieved_all = pdf_retriever.retrieve_from_pdf("climate change mitigation", top_k=3)
    for i, snippet in enumerate(retrieved_all):
        print(f"  Snippet {i+1}: {snippet['c'][:120]}... ({len(snippet['c'])} chars)")

    if id2:
        print(f"\n--- Retrieve from '{id2}' only (query: 'AI ethics LLMs') ---")
        retrieved_specific = pdf_retriever.retrieve_from_pdf("AI ethics LLMs", doc_ids=[id2], top_k=2)
        for i, snippet in enumerate(retrieved_specific):
            print(f"  Snippet {i+1}: {snippet['c'][:120]}... ({len(snippet['c'])} chars)")

    print("\n--- Retrieve with query that might not match well ---")
    retrieved_nomatch = pdf_retriever.retrieve_from_pdf("quantum physics explained", top_k=2)
    print(f"  Number of snippets for no match: {len(retrieved_nomatch)}")


    # Cleanup dummy files and directory
    # try:
    #     os.remove(dummy_pdf_path1)
    #     os.remove(dummy_pdf_path2)
    #     if not any(dummy_pdf_root.iterdir()): # Check if directory is empty
    #         os.rmdir(dummy_pdf_root)
    #     if pdf_retriever.vector_db_path.exists(): # Clean up test DB dir
    #         shutil.rmtree(pdf_retriever.vector_db_path)
    # except OSError as e:
    #     print(f"Error during cleanup: {e}")

    print("\nPdfRetriever demo complete. Manual cleanup of 'data/dummy_pdfs_for_retriever' and 'data/pdf_retriever_test_db_main' may be needed.")
