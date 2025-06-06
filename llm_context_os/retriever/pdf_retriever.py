# yawl/retriever/pdf_retriever.py
import typing as t
from pathlib import Path
import uuid
import os
import fitz  # PyMuPDF
import shutil # For demo cleanup
import time # For timestamp in metadata and demo cleanup delay
import nltk
import numpy as np

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

# Attempt to import BM25Okapi
BM25OKAPI_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    BM25OKAPI_AVAILABLE = True
    print("Successfully imported BM25Okapi from rank_bm25.")
except ImportError:
    print("Warning: rank_bm25 library not found. BM25 retrieval will not function.")
    BM25Okapi = None # Placeholder


class PdfRetriever:
    DEFAULT_COLLECTION_NAME = "pdf_document_chunks_v1"
    def __init__(self,
                 vector_db_path: str = "data/vector_dbs/pdf_rag_db",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 tokenizer = None,
                 recall_budget_tokens: int = 1024,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 collection_name: t.Optional[str] = None,
                 cross_encoder_model_name: t.Optional[str] = "ms-marco-MiniLM-L-6-v2",
                 rerank_top_n_candidates: int = 20,
                 enable_hybrid_search: bool = True, # New default
                 rrf_k_constant: int = 60, # New default
                 # Semantic Chunking specific parameters
                 chunking_strategy: str = "recursive",
                 semantic_chunker_embedding_model: t.Optional[str] = None,
                 semantic_chunker_breakpoint_threshold_type: str = "percentile",
                 semantic_chunker_breakpoint_threshold_amount: float = 5.0, # e.g. 5 for 5th percentile
                 semantic_chunker_min_chunk_sentences: int = 2,
                 enable_hyde: bool = False # HyDE specific
                 ):
        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        self.enable_hyde = enable_hyde # Store HyDE setting
        self.tokenizer = tokenizer
        self.recall_budget_tokens = recall_budget_tokens
        self.chunk_size = chunk_size # Used for 'recursive' strategy
        self.chunk_overlap = chunk_overlap # Used for 'recursive' strategy
        self.db_collection_name = collection_name or self.DEFAULT_COLLECTION_NAME

        # Semantic chunking settings
        self.chunking_strategy = chunking_strategy
        self.semantic_chunker_embedding_model_name = semantic_chunker_embedding_model
        self.semantic_chunker_breakpoint_threshold_type = semantic_chunker_breakpoint_threshold_type
        self.semantic_chunker_breakpoint_threshold_amount = semantic_chunker_breakpoint_threshold_amount
        self.semantic_chunker_min_chunk_sentences = semantic_chunker_min_chunk_sentences
        self.semantic_sentence_embedder = None

        self.cross_encoder_model_name = cross_encoder_model_name
        self.rerank_top_n_candidates = rerank_top_n_candidates
        self.cross_encoder = None

        self.enable_hybrid_search = enable_hybrid_search
        self.rrf_k_constant = rrf_k_constant

        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"[PdfRetriever] Initialized main SentenceTransformer (bi-encoder) model: {self.embedding_model_name}")
            except Exception as e:
                print(f"[PdfRetriever] Error initializing main SentenceTransformer model '{self.embedding_model_name}': {e}")

            if self.cross_encoder_model_name:
                try:
                    self.cross_encoder = SentenceTransformer(self.cross_encoder_model_name)
                    print(f"[PdfRetriever] Initialized CrossEncoder model: {self.cross_encoder_model_name}")
                except Exception as e:
                    print(f"[PdfRetriever] Error initializing CrossEncoder model '{self.cross_encoder_model_name}': {e}")
                    self.cross_encoder = None # Ensure it's None if init fails

            if self.chunking_strategy == "semantic":
                sem_embed_model_to_load = self.semantic_chunker_embedding_model_name or self.embedding_model_name
                try:
                    self.semantic_sentence_embedder = SentenceTransformer(sem_embed_model_to_load)
                    print(f"[PdfRetriever] Initialized SentenceTransformer for semantic chunking: {sem_embed_model_to_load}")
                except Exception as e:
                    print(f"[PdfRetriever] Error initializing semantic chunker SentenceTransformer model '{sem_embed_model_to_load}': {e}. Defaulting to 'recursive' chunking.")
                    self.chunking_strategy = "recursive" # Fallback strategy
                    self.semantic_sentence_embedder = None
        else:
            print("[PdfRetriever] SentenceTransformers library not available. Embedding, CrossEncoder, and Semantic Chunking models not loaded.")
            if self.chunking_strategy == "semantic":
                print("[PdfRetriever] Semantic chunking disabled due to missing SentenceTransformers. Defaulting to 'recursive'.")
                self.chunking_strategy = "recursive" # Fallback strategy

        # NLTK Punkt tokenizer download check (relevant for semantic chunking)
        if self.chunking_strategy == "semantic":
            try:
                nltk.data.find('tokenizers/punkt')
            except nltk.downloader.DownloadError:
                print("[PdfRetriever] NLTK 'punkt' tokenizer not found. Attempting to download...")
                try:
                    nltk.download('punkt', quiet=True)
                    print("[PdfRetriever] NLTK 'punkt' tokenizer downloaded successfully.")
                except Exception as e_nltk:
                    print(f"[PdfRetriever] Error downloading NLTK 'punkt': {e_nltk}. Semantic chunking may fail if 'punkt' is not available.")
            except Exception as e_nltk_find: # Catch other potential errors from find()
                 print(f"[PdfRetriever] Error finding NLTK 'punkt' tokenizer: {e_nltk_find}. Semantic chunking may fail.")


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

        # BM25 specific initializations
        self.bm25_corpus_texts: t.List[str] = []
        self.bm25_chunk_ids: t.List[str] = [] # To map BM25 results back to original chunk IDs
        self.bm25_index: t.Optional[BM25Okapi] = None
        self.bm25_corpus_tokenized: t.Optional[t.List[t.List[str]]] = None
        self.is_bm25_index_built: bool = False

        if not BM25OKAPI_AVAILABLE:
            print("[PdfRetriever] BM25Okapi not available from rank_bm25. BM25 features will be disabled.")


    def _tokenize_for_bm25(self, text: str) -> t.List[str]:
        # Basic tokenizer for BM25: lowercase and split by space.
        # Can be replaced with a more sophisticated tokenizer if needed.
        return text.lower().split()

    def _build_bm25_index_if_needed(self):
        if not BM25OKAPI_AVAILABLE: # Do nothing if library isn't there
            return

        if not self.is_bm25_index_built and self.bm25_corpus_texts:
            print(f"[PdfRetriever] Building BM25 index for {len(self.bm25_corpus_texts)} chunks...")
            try:
                self.bm25_corpus_tokenized = [self._tokenize_for_bm25(text) for text in self.bm25_corpus_texts]
                self.bm25_index = BM25Okapi(self.bm25_corpus_tokenized)
                self.is_bm25_index_built = True
                print(f"[PdfRetriever] BM25 index built successfully.")
            except Exception as e_bm25_build:
                print(f"[PdfRetriever] Error building BM25 index: {e_bm25_build}. BM25 will be unavailable.")
                self.bm25_index = None # Ensure it's None if build fails
                self.is_bm25_index_built = False # Mark as not built
        elif not self.bm25_corpus_texts:
            print("[PdfRetriever] No corpus texts for BM25 index. Skipping build.")
            self.bm25_index = None
            self.is_bm25_index_built = False


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

        # This check was originally here, but text_splitter is only used if strategy is 'recursive'
        # if not self.text_splitter:
        #     msg = "Error: Text splitter not available (e.g., langchain-text-splitters not installed). Cannot process document."
        #     print(f"[PdfRetriever] {msg}")
        #     return False, msg, doc_id, 0

        print(f"[PdfRetriever] Splitting text for document {doc_id} using '{self.chunking_strategy}' strategy...")
        try:
            if self.chunking_strategy == "semantic" and self.semantic_sentence_embedder:
                raw_chunks = self._chunk_semantically(full_text_content)
                print(f"[PdfRetriever] Semantic chunking produced {len(raw_chunks)} chunks for {doc_id}.")
            elif self.text_splitter: # Existing recursive chunking or fallback if semantic_sentence_embedder is None
                if self.chunking_strategy == "semantic" and not self.semantic_sentence_embedder:
                    print(f"[PdfRetriever] Warning: Semantic chunking strategy selected, but semantic embedder not available. Falling back to recursive text splitter for {doc_id}.")
                raw_chunks = self.text_splitter.split_text(full_text_content)
                print(f"[PdfRetriever] Recursive chunking produced {len(raw_chunks)} chunks for {doc_id}.")
            else: # Fallback if no splitter configured or semantic failed to init and text_splitter is also None
                raw_chunks = [full_text_content] # Treat full document as one chunk
                print(f"[PdfRetriever] Warning: No text splitter available (semantic or recursive). Using full document as one chunk for {doc_id}.")
        except Exception as e_split:
            msg = f"Error during text splitting for document {doc_id} with strategy '{self.chunking_strategy}': {e_split}"
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0

        if not self.embedding_model:
            msg = "Embedding model not available. Cannot process document."
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0
        if not self.collection:
            msg = "Vector database (ChromaDB collection) not available. Cannot process document."
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0

        current_doc_chunk_texts = []
        current_doc_chunk_ids = []

        chunk_embeddings_to_add = []
        chunk_docs_to_add = []
        chunk_metadatas_to_add = []
        chunk_unique_ids_to_add = []

        for i, chunk_text_content in enumerate(raw_chunks):
            chunk_id_val = str(uuid.uuid4()) # This is the ID for ChromaDB

            current_doc_chunk_texts.append(chunk_text_content)
            current_doc_chunk_ids.append(chunk_id_val) # Store mapping for BM25

            chunk_docs_to_add.append(chunk_text_content)
            # Estimate page number based on chunk index and typical chunk content. This is a rough estimate.
            # A more accurate method would involve linking chunks back to specific PDF page numbers during extraction.
            page_num_est = "N/A" # Default
            if self.chunking_strategy == "recursive" and self.chunk_size > 0:
                 page_num_est = (i * (self.chunk_size - self.chunk_overlap)) // self.chunk_size + 1 # Basic estimation for recursive
            # For semantic, page number estimation is harder without more info from chunking.

            chunk_metadatas_to_add.append({
                'doc_id': doc_id,
                'pdf_path': str(pdf_path),
                'page_number': page_num_est, # Store estimated page number
                'chunk_index': i,
                'text_length_chars': len(chunk_text_content),
                'timestamp': time.time()
            })
            chunk_unique_ids_to_add.append(chunk_id_val)

        if not chunk_docs_to_add:
            msg = f"No text chunks produced for document {doc_id} after splitting."
            print(f"[PdfRetriever] {msg}")
            return True, msg, doc_id, 0

        try:
            print(f"[PdfRetriever] Embedding {len(chunk_docs_to_add)} chunks for document {doc_id}...")
            chunk_embeddings_to_add = self.embedding_model.encode(chunk_docs_to_add).tolist()

            print(f"[PdfRetriever] Adding {len(chunk_unique_ids_to_add)} chunks to ChromaDB for document {doc_id}...")
            self.collection.add(
                ids=chunk_unique_ids_to_add,
                embeddings=chunk_embeddings_to_add,
                documents=chunk_docs_to_add,
                metadatas=chunk_metadatas_to_add
            )

            # Add to BM25 corpus only after successful ChromaDB add
            self.bm25_corpus_texts.extend(current_doc_chunk_texts)
            self.bm25_chunk_ids.extend(current_doc_chunk_ids)
            self.is_bm25_index_built = False # Mark BM25 index for rebuild

            num_processed_chunks = len(chunk_unique_ids_to_add)
            msg = f"Successfully extracted, embedded, and stored {num_processed_chunks} chunks for document: {doc_id}."
            print(f"[PdfRetriever] {msg}")
            return True, msg, doc_id, num_processed_chunks
        except Exception as e:
            msg = f"Error during embedding or storing chunks for document {doc_id}: {e}"
            print(f"[PdfRetriever] {msg}")
            return False, msg, doc_id, 0


    def _chunk_semantically(self, text: str) -> t.List[str]:
        if not self.semantic_sentence_embedder:
            print("[PdfRetriever] Semantic sentence embedder not available. Falling back to single chunk.")
            return [text]
        if not nltk: # Should not happen if init checks passed, but good for safety
            print("[PdfRetriever] NLTK not available for semantic chunking. Falling back.")
            return [text]

        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e_sent_tokenize:
            print(f"[PdfRetriever] Error tokenizing sentences with NLTK: {e_sent_tokenize}. Falling back to single chunk.")
            return [text]

        if len(sentences) < self.semantic_chunker_min_chunk_sentences:
            print(f"[PdfRetriever] Number of sentences ({len(sentences)}) is less than min_chunk_sentences ({self.semantic_chunker_min_chunk_sentences}). Returning as single chunk.")
            return [" ".join(sentences)] if sentences else [] # Join sentences if any, else empty list

        try:
            print(f"[PdfRetriever] Embedding {len(sentences)} sentences for semantic chunking...")
            sentence_embeddings = self.semantic_sentence_embedder.encode(sentences)
        except Exception as e_embed:
            print(f"[PdfRetriever] Error embedding sentences for semantic chunking: {e_embed}. Falling back.")
            return [text] # Fallback to full text as one chunk

        if len(sentence_embeddings) <= 1: # Need at least 2 embeddings for similarity
            return [" ".join(sentences)] if sentences else []

        # Calculate cosine similarities between adjacent sentence embeddings
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            emb1 = sentence_embeddings[i]
            emb2 = sentence_embeddings[i+1]
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0: # Handle zero vectors, though unlikely for ST
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            similarities.append(similarity)

        if not similarities: # Only one sentence after filtering, or some other edge case
             return [" ".join(sentences)] if sentences else []

        # Determine split points based on threshold type
        if self.semantic_chunker_breakpoint_threshold_type == "percentile":
            if not similarities: return [" ".join(sentences)] # Avoid error on empty similarities
            threshold_val = np.percentile(similarities, self.semantic_chunker_breakpoint_threshold_amount)
            print(f"[PdfRetriever] Semantic chunking: Percentile ({self.semantic_chunker_breakpoint_threshold_amount}th) threshold for similarity: {threshold_val:.4f}")
        # Add other threshold types (std_dev, distance) here if implemented later
        else: # Default or unknown, treat as a direct similarity threshold (though not fully specified)
            print(f"[PdfRetriever] Warning: Unsupported semantic_chunker_breakpoint_threshold_type: {self.semantic_chunker_breakpoint_threshold_type}. Using raw amount as similarity threshold.")
            threshold_val = self.semantic_chunker_breakpoint_threshold_amount # Assuming amount is a direct similarity value

        chunks = []
        current_chunk_sentences = []
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            # Check if this is a breakpoint (i.e., split *after* this sentence)
            # Similarities list is one shorter than sentences list. similarities[i] is between sentences[i] and sentences[i+1]
            if i < len(similarities):
                if similarities[i] < threshold_val:
                    if len(current_chunk_sentences) >= self.semantic_chunker_min_chunk_sentences:
                        chunks.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = []
                    # else: continue accumulating to meet min_chunk_sentences, even if it crosses a low similarity point.
                    # This logic might need refinement: do we prioritize min_length or similarity break?
                    # Current: prioritize min_length if current chunk is too short, otherwise split.
                    # If we split, the next chunk starts fresh.
            # For the last sentence, if there's an accumulated chunk, add it.
            elif i == len(sentences) - 1 and current_chunk_sentences:
                 # If the last chunk is too short after a split, merge it with the previous one.
                if chunks and len(current_chunk_sentences) < self.semantic_chunker_min_chunk_sentences:
                    chunks[-1] += " " + " ".join(current_chunk_sentences)
                else: # Add as a new chunk if it meets min length or is the only chunk
                    chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [] # Clear for safety, though loop ends

        # If loop finishes and current_chunk_sentences is not empty (e.g. no breakpoint hit for a while)
        if current_chunk_sentences:
            if chunks and len(current_chunk_sentences) < self.semantic_chunker_min_chunk_sentences : # If last chunk is too small and there's a previous one
                 chunks[-1] += " " + " ".join(current_chunk_sentences) # Append to previous
            elif not chunks and len(current_chunk_sentences) < self.semantic_chunker_min_chunk_sentences: # Only one chunk, but too small
                 chunks.append(" ".join(current_chunk_sentences)) # Still add it
            elif len(current_chunk_sentences) >= self.semantic_chunker_min_chunk_sentences:
                 chunks.append(" ".join(current_chunk_sentences))


        return chunks if chunks else [" ".join(sentences)] # Fallback to single chunk if logic produced nothing


    def retrieve_from_pdf(self,
                          query_text: str,
                          doc_ids: t.Optional[t.List[str]] = None,
                          top_k: int = 3,
                          query_embedding_override: t.Optional[t.List[float]] = None
                          ) -> t.List[t.Dict[str, str]]:
        print(f"\n[PdfRetriever] retrieve_from_pdf called. Query: '{query_text[:50]}...', Target Docs: {doc_ids or 'all'}, Top K: {top_k}, HyDE active: {query_embedding_override is not None}")

        self._build_bm25_index_if_needed() # Ensure BM25 index is ready

        if not self.embedding_model:
            print("[PdfRetriever] Error: Embedding model not available for retrieval.")
            return []
        if not self.collection:
            print("[PdfRetriever] Error: Vector database (ChromaDB collection) not available for retrieval.")
            return []
        if not self.tokenizer:
            print("[PdfRetriever] Error: Tokenizer not available for budgeting. Cannot reliably size snippets.")
            return []

        doc_details_cache = {} # To store full details for RRF reconstruction & final snippet generation
        candidate_items_for_cross_encoder = []
        dense_results_list = [] # To hold items from dense search if hybrid is off

        # --- Dense Retrieval (ChromaDB) ---
        try:
            if query_embedding_override is not None:
                query_embedding = query_embedding_override
                print(f"[PdfRetriever] Using HyDE provided query embedding for dense retrieval.")
            else:
                print(f"[PdfRetriever] Embedding original query for dense retrieval: '{query_text[:100]}...'")
                query_embedding = self.embedding_model.encode(query_text).tolist()

            chroma_filter: t.Optional[t.Dict[str, t.Any]] = None
            if doc_ids:
                chroma_filter = {"doc_id": {"$in": [str(did) for did in doc_ids]}} if len(doc_ids) > 1 else {"doc_id": str(doc_ids[0])}
                print(f"[PdfRetriever] Applying ChromaDB filter for dense search: {chroma_filter}")

            num_dense_to_fetch = self.rerank_top_n_candidates
            print(f"[PdfRetriever] Querying ChromaDB for dense results (requesting {num_dense_to_fetch} candidates).")
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_dense_to_fetch,
                where=chroma_filter,
                include=['documents', 'metadatas', 'distances'] # Ensure 'ids' is implicitly included or add it
            )

            if chroma_results and chroma_results.get('ids') and chroma_results['ids'][0]:
                for i in range(len(chroma_results['ids'][0])):
                    chunk_id = chroma_results['ids'][0][i]
                    doc_text = chroma_results['documents'][0][i] if chroma_results['documents'] and chroma_results['documents'][0][i] is not None else "Content not available"
                    metadata = chroma_results['metadatas'][0][i] if chroma_results['metadatas'] else {}
                    distance = chroma_results['distances'][0][i] if chroma_results['distances'] else float('inf')

                    if doc_text.strip().lower() == query_text.strip().lower(): continue

                    # Store with a score where higher is better (e.g., negative distance)
                    item_details = {"id": chunk_id, "text": doc_text, "metadata": metadata, "dense_score": -distance, "source": "dense"}
                    dense_results_list.append(item_details) # Add to list for non-hybrid path
                    doc_details_cache[chunk_id] = item_details # Add to cache for hybrid path

                dense_results_list.sort(key=lambda x: x['dense_score'], reverse=True) # Sort for non-hybrid path
                print(f"[PdfRetriever] Dense retrieval found {len(doc_details_cache)} candidates.")
        except Exception as e:
            print(f"[PdfRetriever] Error during dense retrieval: {e}")

        if self.enable_hybrid_search and self.bm25_index and self.bm25_chunk_ids:
            print(f"[PdfRetriever] Performing BM25 sparse retrieval for query: '{query_text[:100]}...'")
            try:
                tokenized_query = self._tokenize_for_bm25(query_text)
                bm25_scores = self.bm25_index.get_scores(tokenized_query)

                # Combine BM25 scores with existing doc_details or add new entries
                for i, chunk_id in enumerate(self.bm25_chunk_ids):
                    score = bm25_scores[i]
                    if score > 0: # Only consider documents with a positive BM25 score
                        if chunk_id in doc_details_cache:
                            doc_details_cache[chunk_id]['sparse_score'] = score
                            doc_details_cache[chunk_id]['source'] += ",sparse"
                        else: # Chunk found by BM25 but not by dense
                            # Need to fetch text and metadata for these.
                            # This requires bm25_corpus_texts and a way to map bm25_chunk_ids to metadata.
                            # For now, we'll assume if not in dense, we can't easily get metadata for sparse-only.
                            # This means sparse-only results might lack some details for cross-encoder or snippet.
                            # A get_by_ids from Chroma could fetch metadata for these if needed.
                            doc_details_cache[chunk_id] = {
                                "id": chunk_id,
                                "text": self.bm25_corpus_texts[i], # Text from BM25 corpus
                                "metadata": {}, # Placeholder metadata for sparse-only
                                "sparse_score": score,
                                "source": "sparse_only"
                            }
                print(f"[PdfRetriever] BM25 scores computed/merged for {len(self.bm25_chunk_ids)} items. Cache size: {len(doc_details_cache)}")
            except Exception as e_bm25:
                print(f"[PdfRetriever] Error during BM25 retrieval: {e_bm25}")

            # --- Reciprocal Rank Fusion (RRF) ---
            fused_scores: t.Dict[str, float] = {}

            # Dense results ranking (higher dense_score is better)
            # Sort by dense_score to get ranks for RRF
            sorted_dense_for_rrf = sorted([d for d in doc_details_cache.values() if 'dense_score' in d], key=lambda x: x['dense_score'], reverse=True)
            for rank, doc in enumerate(sorted_dense_for_rrf):
                fused_scores[doc["id"]] = fused_scores.get(doc["id"], 0.0) + (1.0 / (self.rrf_k_constant + rank))

            # Sparse results ranking (higher sparse_score is better)
            sorted_sparse_for_rrf = sorted([d for d in doc_details_cache.values() if 'sparse_score' in d], key=lambda x: x['sparse_score'], reverse=True)
            for rank, doc in enumerate(sorted_sparse_for_rrf):
                fused_scores[doc["id"]] = fused_scores.get(doc["id"], 0.0) + (1.0 / (self.rrf_k_constant + rank))

            if fused_scores:
                sorted_fused_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
                print(f"[PdfRetriever] RRF resulted in {len(sorted_fused_ids)} unique candidates. Top RRF score: {fused_scores[sorted_fused_ids[0]]:.4f} if any.")

                # Prepare candidates for cross-encoder using RRF sorted IDs
                for chunk_id in sorted_fused_ids[:self.rerank_top_n_candidates]: # Take top N from RRF
                    if chunk_id in doc_details_cache:
                        item = doc_details_cache[chunk_id]
                        item['rrf_score'] = fused_scores[chunk_id]
                        candidate_items_for_cross_encoder.append(item)
                    # Else: ID from fused_scores somehow not in cache, should not happen.
            else: # No scores to fuse, means neither dense nor sparse returned anything with score
                print("[PdfRetriever] No results from dense or sparse retrieval to fuse.")
                # candidate_items_for_cross_encoder will remain empty.

        else: # Hybrid search disabled, use only dense results
            print("[PdfRetriever] Hybrid search disabled. Using only dense retrieval results.")
            # Populate candidate_items_for_cross_encoder from dense_results_list directly
            # dense_results_list is already sorted by dense_score (descending, higher is better)
            candidate_items_for_cross_encoder = dense_results_list[:self.rerank_top_n_candidates]


        if not candidate_items_for_cross_encoder:
            print("[PdfRetriever] No candidates to process for cross-encoding or snippet generation.")
            return []

        # --- Cross-encoder Re-ranking ---
        # The list `candidate_items_for_cross_encoder` now holds the items to be re-ranked.
        # It's either top N from RRF (if hybrid) or top N from dense (if not hybrid).
        if self.cross_encoder and candidate_items_for_cross_encoder:
            print(f"[PdfRetriever] Cross-encoding {len(candidate_items_for_cross_encoder)} candidates with: {self.cross_encoder_model_name}")
            try:
                pairs = [(query_text, item['text']) for item in candidate_items_for_cross_encoder]
                cross_scores = self.cross_encoder.predict(pairs)

                for i, item in enumerate(candidate_items_for_cross_encoder): # Iterate over the correct list
                    item['cross_score'] = cross_scores[i]

                # Sort by cross_score in descending order (higher is better)
                candidate_items_for_cross_encoder.sort(key=lambda x: x.get('cross_score', -float('inf')), reverse=True)
                if candidate_items_for_cross_encoder:
                    print(f"[PdfRetriever] Cross-encoding complete. Top cross_score: {candidate_items_for_cross_encoder[0].get('cross_score', 'N/A'):.4f}")
            except Exception as e_rerank:
                print(f"[PdfRetriever] Error during CrossEncoder prediction/re-ranking: {e_rerank}. Proceeding with pre-cross-encoder sorted list.")

        # Snippet generation from final candidate_items (which are now sorted by cross-encoder if enabled, else by RRF/dense)
        final_candidates_for_snippets = candidate_items_for_cross_encoder

        retrieved_snippets: t.List[t.Dict[str, str]] = []
        current_token_count = 0
        for res_item in final_candidates_for_snippets:
            if len(retrieved_snippets) >= top_k: # Apply final top_k limit from original request
                break

            doc_id_meta = res_item['metadata'].get('doc_id', 'UnknownDoc')
            page_num_meta = res_item['metadata'].get('page_number', 'N/A')

            dense_score_val = res_item.get('dense_score') # This is negative distance
            dense_dist_str = f"{-dense_score_val:.4f}" if dense_score_val is not None else "N/A"
            sparse_score_str = f"{res_item['sparse_score']:.4f}" if 'sparse_score' in res_item else "N/A"
            rrf_score_str = f"{res_item['rrf_score']:.4f}" if 'rrf_score' in res_item else "N/A"
            cross_score_str = f"{res_item['cross_score']:.4f}" if 'cross_score' in res_item else "N/A"

            snippet_text = (f"Retrieved from '{doc_id_meta}' "
                            f"(Page {page_num_meta}, dense_dist: {dense_dist_str}, sparse: {sparse_score_str}, rrf: {rrf_score_str}, cross: {cross_score_str}): "
                            f"\"{res_item['text']}\"")
            try:
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
                # Corrected log_score to use res_item directly
                log_score_val = res_item.get('cross_score', res_item.get('rrf_score', res_item.get('dense_score', -float('inf'))))
                log_score_type = "cross" if 'cross_score' in res_item else ("rrf" if 'rrf_score' in res_item else ("dense" if 'dense_score' in res_item else "unknown"))
                log_score_str = f"{log_score_val:.4f}" if isinstance(log_score_val, float) else "N/A"
                print(f"  Added snippet ({log_score_type}: {log_score_str}, tokens: {snippet_tokens}): '{snippet_text[:100]}...'")
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

# End of llm_context_os/retriever/pdf_retriever.py
