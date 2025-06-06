# llm_context_os/retriever/chat_history.py
import uuid
import time
import typing as t
import os # For path operations
import shutil # For demo cleanup

# Attempt to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None # Placeholder
    print("Warning: sentence-transformers library not found. ChatHistoryRetriever may not function as expected.")

# Attempt to import ChromaDB
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None # Placeholder
    print("Warning: chromadb library not found. ChatHistoryRetriever may not function as expected.")

# Attempt to import tiktoken for default tokenizer
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None # Placeholder
    print("Warning: tiktoken library not found. Will use basic char count if no tokenizer provided.")

# Attempt to import BM25Okapi
BM25OKAPI_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    BM25OKAPI_AVAILABLE = True
    print("Successfully imported BM25Okapi from rank_bm25.")
except ImportError:
    print("Warning: rank_bm25 library not found. BM25 retrieval will not function for chat history.")
    BM25Okapi = None # Placeholder


class BasicCharTokenizer: # Fallback if tiktoken is not available and no tokenizer is passed
    def encode(self, text: str) -> t.List[int]: # Mock encode to return list of char codes
        return [ord(c) for c in text]
    def decode(self, tokens: t.List[int]) -> str: # Mock decode
        return "".join([chr(t) for t in tokens])
    # For budgeting, we need a count_tokens method equivalent.
    # tiktoken.Encoding has .encode() which returns list of ints. len(that_list) is token count.
    # So, this basic tokenizer's "token count" will be char count.
    def count_tokens(self, text:str) -> int:
        return len(self.encode(text))


class ChatHistoryRetriever:
    """
    Retrieves relevant snippets from chat history using vector embeddings.
    Uses SentenceTransformers for embeddings and ChromaDB for storage and querying.
    """
    DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    DEFAULT_DB_SUBDIR = "chat_history_db"
    DEFAULT_COLLECTION_NAME = "chat_history_v2"

    def __init__(self,
                 recall_budget_tokens: int = 1024, # Increased default budget
                 tokenizer: t.Any = None,
                 vector_db_path: str = 'data/vector_dbs',
                 embedding_model_name: t.Optional[str] = None,
                 collection_name: t.Optional[str] = None,
                 cross_encoder_model_name: t.Optional[str] = "ms-marco-MiniLM-L-6-v2",
                 rerank_top_n_candidates: int = 20,
                 enable_hybrid_search: bool = True, # New default
                 rrf_k_constant: int = 60): # New default
        """
        Initializes the ChatHistoryRetriever.
        """
        self.recall_budget_tokens = recall_budget_tokens
        self.embedding_model_name = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL
        self.cross_encoder_model_name = cross_encoder_model_name
        self.rerank_top_n_candidates = rerank_top_n_candidates
        self.enable_hybrid_search = enable_hybrid_search
        self.rrf_k_constant = rrf_k_constant

        self.db_collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.vector_db_full_path = os.path.join(vector_db_path, self.DEFAULT_DB_SUBDIR)

        self.embedding_model = None
        self.cross_encoder = None
        self.db_client = None
        self.collection = None

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"Error: SentenceTransformers library is required but not installed. Embedding and re-ranking will not function.")
            return
        if not CHROMADB_AVAILABLE:
            print(f"Error: ChromaDB library is required but not installed. Vector storage will not function.")
            return

        try:
            print(f"Initializing SentenceTransformer (bi-encoder) model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"Error initializing SentenceTransformer (bi-encoder) model '{self.embedding_model_name}': {e}")
            return # Critical failure if bi-encoder can't load

        if self.cross_encoder_model_name:
            try:
                print(f"Initializing CrossEncoder model: {self.cross_encoder_model_name}")
                self.cross_encoder = SentenceTransformer(self.cross_encoder_model_name)
            except Exception as e:
                print(f"Warning: Error initializing CrossEncoder model '{self.cross_encoder_model_name}': {e}. Re-ranking will be disabled.")
                self.cross_encoder = None # Ensure it's None if init fails

        try:
            print(f"Initializing ChromaDB client at path: {self.vector_db_full_path}")
            # Ensure the directory exists
            os.makedirs(self.vector_db_full_path, exist_ok=True)
            self.db_client = chromadb.PersistentClient(path=self.vector_db_full_path)

            print(f"Getting or creating ChromaDB collection: {self.db_collection_name}")
            # Note: ChromaDB's default embedding function can be resource-intensive if not managed.
            # We are providing our own embeddings, so embedding_function=None is not directly used here.
            # However, if one were to use collection.add(documents=[...]) without embeddings,
            # it would try to use its default. For clarity, we explicitly manage embeddings.
            self.collection = self.db_client.get_or_create_collection(name=self.db_collection_name)
            print(f"ChromaDB collection '{self.db_collection_name}' loaded/created. Count: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return

        if tokenizer:
            self.tokenizer = tokenizer
        elif TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                print("Using tiktoken cl100k_base for token budgeting.")
            except Exception as e:
                print(f"Error initializing tiktoken, falling back to BasicCharTokenizer: {e}")
                self.tokenizer = BasicCharTokenizer()
        else:
            print("tiktoken not available, falling back to BasicCharTokenizer for token budgeting.")
            self.tokenizer = BasicCharTokenizer()

        print(f"ChatHistoryRetriever initialized with {type(self.tokenizer).__name__}.")

        # BM25 specific initializations
        self.bm25_corpus_texts: t.List[str] = []
        self.bm25_message_ids: t.List[str] = [] # To map BM25 results back to original message IDs
        self.bm25_index: t.Optional[BM25Okapi] = None
        self.bm25_corpus_tokenized: t.Optional[t.List[t.List[str]]] = None
        self.is_bm25_index_built: bool = False

        if not BM25OKAPI_AVAILABLE:
            print("[ChatHistoryRetriever] BM25Okapi not available. BM25 features will be disabled.")

    def _tokenize_for_bm25(self, text: str) -> t.List[str]:
        return text.lower().split()

    def _build_bm25_index_if_needed(self):
        if not BM25OKAPI_AVAILABLE:
            return

        if not self.is_bm25_index_built and self.bm25_corpus_texts:
            print(f"[ChatHistoryRetriever] Building BM25 index for {len(self.bm25_corpus_texts)} messages...")
            try:
                self.bm25_corpus_tokenized = [self._tokenize_for_bm25(text) for text in self.bm25_corpus_texts]
                self.bm25_index = BM25Okapi(self.bm25_corpus_tokenized)
                self.is_bm25_index_built = True
                print(f"[ChatHistoryRetriever] BM25 index built successfully.")
            except Exception as e_bm25_build:
                print(f"[ChatHistoryRetriever] Error building BM25 index: {e_bm25_build}. BM25 will be unavailable.")
                self.bm25_index = None
                self.is_bm25_index_built = False
        elif not self.bm25_corpus_texts:
            print("[ChatHistoryRetriever] No corpus texts for BM25 index. Skipping build.")
            self.bm25_index = None
            self.is_bm25_index_built = False


    def add_message(self, message_text: str, role: str, message_id: t.Optional[str] = None) -> t.Optional[str]:
        """ Adds a message to the history, embeds it, and stores it in ChromaDB and BM25 corpus. """
        if not self.embedding_model or not self.collection:
            print("Error: Retriever not properly initialized (model or DB collection missing). Cannot add message.")
            return None

        msg_id = message_id or str(uuid.uuid4())

        try:
            if not isinstance(message_text, str):
                message_text = str(message_text)

            embedding = self.embedding_model.encode(message_text).tolist()

            # Prepare for BM25 before adding to ChromaDB (in case ChromaDB add fails)
            # No, add to BM25 corpus only after successful ChromaDB add.

            metadata = {
                'role': role,
                'timestamp': time.time(),
                'text_length_chars': len(message_text),
                'source': 'chat_history'
            }
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)

            self.collection.add(
                ids=[msg_id],
                embeddings=[embedding],
                documents=[message_text],
                metadatas=[metadata]
            )

            # Add to BM25 corpus and invalidate index
            self.bm25_corpus_texts.append(message_text)
            self.bm25_message_ids.append(msg_id)
            self.is_bm25_index_built = False

            print(f"Added message ID {msg_id} to ChromaDB & BM25 corpus: '{role}: {message_text[:100]}...'")
            return msg_id
        except Exception as e:
            print(f"Error adding message ID {msg_id} to ChromaDB/BM25 corpus: {e}")
            return None


    def retrieve(self, query_text: str, current_chat_history: t.Optional[t.List[t.Dict[str, str]]] = None, n_results: int = 5) -> t.List[t.Dict[str, str]]:
        self._build_bm25_index_if_needed() # Ensure BM25 index is ready

        if not self.embedding_model or not self.collection or not self.tokenizer:
            print("Error: Retriever not properly initialized. Cannot retrieve.")
            return []

        # Check if there's anything to search in the first place
        has_vector_content = self.collection.count() > 0
        has_bm25_content = self.bm25_index and self.bm25_corpus_texts
        if not has_vector_content and not (self.enable_hybrid_search and has_bm25_content):
            print("No messages in history (vector DB or BM25 corpus if hybrid) to retrieve from.")
            return []

        doc_details_cache = {} # To store full details for RRF reconstruction & final snippet generation
        candidate_items_for_cross_encoder = [] # This will hold items for cross-encoder input
        dense_results_list = [] # Specifically for dense results before fusion/selection

        # --- Dense Retrieval (ChromaDB) ---
        if has_vector_content:
            try:
                print(f"[ChatHistoryRetriever] Embedding query for dense retrieval: '{query_text[:100]}...'")
                query_embedding = self.embedding_model.encode(query_text).tolist()

                num_dense_to_fetch = self.rerank_top_n_candidates

                query_n_dense = min(num_dense_to_fetch, self.collection.count())
                # Ensure query_n_dense is at least 1 if collection is not empty, to avoid ChromaDB error with n_results=0
                if query_n_dense == 0 and self.collection.count() > 0: query_n_dense = 1

                if query_n_dense > 0: # Proceed only if there's something to query for
                    chroma_results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=query_n_dense,
                        include=['documents', 'metadatas', 'distances']
                    )
                    if chroma_results and chroma_results.get('ids') and chroma_results['ids'][0]:
                        for i in range(len(chroma_results['ids'][0])):
                            msg_id = chroma_results['ids'][0][i]
                            text = chroma_results['documents'][0][i]
                            meta = chroma_results['metadatas'][0][i]
                            dist = chroma_results['distances'][0][i]

                            if text == query_text: continue
                            if current_chat_history and any(h.get('c') == text for h in current_chat_history): continue

                            item_details = {"id": msg_id, "text": text, "metadata": meta, "dense_score": -dist}
                            dense_results_list.append(item_details) # Keep a separate list for dense ranked by score
                            doc_details_cache[msg_id] = item_details # Cache for RRF and final object construction

                        dense_results_list.sort(key=lambda x: x['dense_score'], reverse=True) # Higher is better
                        print(f"[ChatHistoryRetriever] Dense retrieval found {len(dense_results_list)} candidates.")
            except Exception as e:
                print(f"Error during dense retrieval in ChatHistoryRetriever: {e}")

        if self.enable_hybrid_search and self.bm25_index and self.bm25_message_ids:
            sparse_results_ranked_list = []
            print(f"[ChatHistoryRetriever] Performing BM25 sparse retrieval for query: '{query_text[:100]}...'")
            try:
                tokenized_query = self._tokenize_for_bm25(query_text)
                bm25_scores = self.bm25_index.get_scores(tokenized_query)
                scored_bm25_ids = sorted(
                    [(self.bm25_message_ids[i], bm25_scores[i]) for i in range(len(bm25_scores)) if bm25_scores[i] > 0],
                    key=lambda x: x[1],
                    reverse=True
                )
                for msg_id, score in scored_bm25_ids[:self.rerank_top_n_candidates]: # Consider top N for RRF
                    if msg_id not in doc_details_cache:
                        try:
                            original_idx = self.bm25_message_ids.index(msg_id)
                            text = self.bm25_corpus_texts[original_idx]
                            if text == query_text: continue
                            if current_chat_history and any(h.get('c') == text for h in current_chat_history): continue
                            doc_details_cache[msg_id] = {"id": msg_id, "text": text, "metadata": {}, "sparse_score": score, "source": "sparse_only"}
                        except ValueError: continue
                    else: # Already in cache from dense search
                        if doc_details_cache[msg_id]['text'] == query_text: continue
                        if current_chat_history and any(h.get('c') == doc_details_cache[msg_id]['text'] for h in current_chat_history): continue
                        doc_details_cache[msg_id]["sparse_score"] = score
                        if "source" in doc_details_cache[msg_id] and doc_details_cache[msg_id]["source"] == "dense":
                             doc_details_cache[msg_id]["source"] += ",sparse"
                        else: # Should not happen if dense ran first and populated
                             doc_details_cache[msg_id]["source"] = "sparse_only_but_somehow_cached"


                    sparse_results_ranked_list.append({"id": msg_id, "sparse_score": score})
                print(f"[ChatHistoryRetriever] BM25 retrieval found {len(sparse_results_ranked_list)} candidates for RRF.")
            except Exception as e_bm25:
                print(f"[ChatHistoryRetriever] Error during BM25 retrieval: {e_bm25}")

            fused_scores: t.Dict[str, float] = {}
            for rank, doc in enumerate(dense_results_list): # dense_results_list is sorted
                fused_scores[doc["id"]] = fused_scores.get(doc["id"], 0.0) + (1.0 / (self.rrf_k_constant + rank))
            for rank, item in enumerate(sparse_results_ranked_list): # sparse_results_ranked_list is sorted
                fused_scores[item["id"]] = fused_scores.get(item["id"], 0.0) + (1.0 / (self.rrf_k_constant + rank))

            if fused_scores:
                sorted_fused_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
                print(f"[ChatHistoryRetriever] RRF resulted in {len(sorted_fused_ids)} unique candidates.")
                for msg_id in sorted_fused_ids[:self.rerank_top_n_candidates]:
                    if msg_id in doc_details_cache:
                        item = doc_details_cache[msg_id]
                        item['rrf_score'] = fused_scores[msg_id]
                        candidate_items_for_cross_encoder.append(item)
            else:
                print("[ChatHistoryRetriever] No results from dense or sparse retrieval to fuse.")

        else:
            print("[ChatHistoryRetriever] Hybrid search disabled or BM25 not ready. Using only dense retrieval results.")
            candidate_items_for_cross_encoder = dense_results_list[:self.rerank_top_n_candidates]

        if not candidate_items_for_cross_encoder:
            print("[ChatHistoryRetriever] No candidates to process for cross-encoding or snippet generation.")
            return []

        if self.cross_encoder and candidate_items_for_cross_encoder:
            print(f"[ChatHistoryRetriever] Cross-encoding {len(candidate_items_for_cross_encoder)} candidates: {self.cross_encoder_model_name}")
            try:
                pairs_for_reranking = [(query_text, item['text']) for item in candidate_items_for_cross_encoder]
                cross_scores = self.cross_encoder.predict(pairs_for_reranking)

                for i, item in enumerate(candidate_items):
                    item['cross_score'] = cross_scores[i]

                candidate_items_for_cross_encoder.sort(key=lambda x: x.get('cross_score', -float('inf')), reverse=True)
                if candidate_items_for_cross_encoder:
                    print(f"[ChatHistoryRetriever] Cross-encoding complete. Top cross_score: {candidate_items_for_cross_encoder[0].get('cross_score', 'N/A'):.4f}")
            except Exception as e_rerank:
                print(f"[ChatHistoryRetriever] Error during CrossEncoder prediction/re-ranking: {e_rerank}. Proceeding with pre-cross-encoder sorted list.")

        final_candidates_for_snippets = candidate_items_for_cross_encoder

        # Snippet generation from final candidate_items
        retrieved_snippets: t.List[t.Dict[str, str]] = []
        current_token_count = 0
        for item in final_candidates_for_snippets:
            if len(retrieved_snippets) >= n_results: # Apply final n_results limit from original request
                break

            role = item['metadata'].get('role', 'context')
            timestamp_val = item['metadata'].get('timestamp', 0)
            timestamp_str = f" (at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp_val))})" if timestamp_val else ""

            dense_score_val = item.get('dense_score') # This is negative distance
            dense_dist_str = f"{-dense_score_val:.4f}" if dense_score_val is not None else "N/A"
            sparse_score_str = f"{item['sparse_score']:.4f}" if 'sparse_score' in item else "N/A"
            rrf_score_str = f"{item['rrf_score']:.4f}" if 'rrf_score' in item else "N/A"
            cross_score_str = f"{item['cross_score']:.4f}" if 'cross_score' in item else "N/A"

            snippet_text = (f"Previously, {role} said{timestamp_str} "
                            f"(dense_dist: {dense_dist_str}, sparse: {sparse_score_str}, rrf: {rrf_score_str}, cross: {cross_score_str}): "
                            f"\"{item['text']}\"")
            try:
                snippet_tokens = len(self.tokenizer.encode(snippet_text))
            except Exception as e:
                print(f"Warning: Error tokenizing snippet for budgeting ('{str(e)}'), falling back to char count.")
                snippet_tokens = len(snippet_text)

            if current_token_count + snippet_tokens <= self.recall_budget_tokens:
                retrieved_snippets.append({'r': 'retrieved_context', 'c': snippet_text})
                current_token_count += snippet_tokens
                # Log primarily the score that determined its final position (cross if available, else rrf)
                final_sort_score = cross_score_str if item.get('cross_score') is not None else rrf_score_str
                print(f"  Added snippet (ID: {item['id']}, final_sort_score: {final_sort_score}, tokens: {snippet_tokens}): '{snippet_text[:100]}...'")
            else:
                print(f"  Budget exceeded with snippet ID {item['id']} (tokens: {snippet_tokens}). Total: {current_token_count}. Stopping.")
                break

        return retrieved_snippets


if __name__ == '__main__':
    print("--- ChatHistoryRetriever Functional Demo ---")

    # Ensure a clean environment for the demo
    DEMO_DB_BASE_PATH = 'data/vector_dbs_demo'
    # Get the specific sub-path the retriever will use
    DEMO_DB_FULL_PATH = os.path.join(DEMO_DB_BASE_PATH, ChatHistoryRetriever.DEFAULT_DB_SUBDIR)

    if os.path.exists(DEMO_DB_FULL_PATH):
        print(f"Cleaning up existing demo DB at: {DEMO_DB_FULL_PATH}")
        shutil.rmtree(DEMO_DB_FULL_PATH)

    # Attempt to use HF Tokenizer if available (example, not strictly needed if tiktoken is primary)
    # For this demo, we'll rely on the internal tiktoken or BasicCharTokenizer logic.
    # from llm_context_os.context.token_estimator import HFTokenEstimator
    # test_tokenizer = None
    # try:
    #     test_tokenizer = HFTokenEstimator("gpt2").tokenizer # Get the actual tokenizer object
    # except Exception:
    #     print("HF Tokenizer for gpt2 not available for demo, will use tiktoken or char count.")
    #     pass

    # Initialize retriever
    # If SentenceTransformer or ChromaDB is not available, this will print errors and not fully init
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not CHROMADB_AVAILABLE:
        print("Cannot run demo: SentenceTransformers or ChromaDB is not installed.")
    else:
        retriever = ChatHistoryRetriever(
            recall_budget_tokens=200, # Budget for retrieved snippets
            vector_db_path=DEMO_DB_BASE_PATH, # Pass base path
            # tokenizer=test_tokenizer # Pass a tokenizer if you have one, else it uses tiktoken/basic
        )

        if retriever.collection: # Check if initialization was successful
            print("\n--- Adding messages to ChromaDB ---")
            retriever.add_message(message_text="The sky is blue today.", role="user", message_id="msg1")
            retriever.add_message(message_text="Indeed, it's a beautiful day for a walk in the park.", role="assistant", message_id="msg2")
            retriever.add_message(message_text="I'm thinking of going to a cafe later.", role="user", message_id="msg3")
            retriever.add_message(message_text="Oh, which cafe are you considering?", role="assistant", message_id="msg4")
            retriever.add_message(message_text="Maybe 'The Code Cup' or 'The Byte Bar'.", role="user", message_id="msg5")

            print(f"\nTotal messages in DB: {retriever.collection.count()}")

            print("\n--- Retrieving based on a query: 'cafe suggestions' ---")
            query1 = "Any good cafe suggestions?"
            retrieved_snippets1 = retriever.retrieve(query_text=query1, n_results=3)
            if retrieved_snippets1:
                for snippet in retrieved_snippets1:
                    print(f"  Snippet: {snippet['r']}: {snippet['c']}")
            else:
                print("  No snippets retrieved for query1.")

            print("\n--- Retrieving based on a query: 'weather or park' ---")
            query2 = "What was said about the weather or the park?"
            retrieved_snippets2 = retriever.retrieve(query_text=query2, n_results=3)
            if retrieved_snippets2:
                for snippet in retrieved_snippets2:
                    print(f"  Snippet: {snippet['r']}: {snippet['c']}")
            else:
                print("  No snippets retrieved for query2.")

            # Clean up demo DB after run
            if os.path.exists(DEMO_DB_FULL_PATH):
                print(f"\nCleaning up demo DB at: {DEMO_DB_FULL_PATH}")
                # Make sure client is not using the DB anymore if needed by chromadb
                del retriever.collection
                del retriever.db_client # Explicitly delete client to release file locks if any
                # Small delay might be needed on some OS for file locks to release
                time.sleep(0.1)
                try:
                    shutil.rmtree(DEMO_DB_FULL_PATH)
                    print(f"Successfully removed demo DB directory: {DEMO_DB_FULL_PATH}")
                    # Also remove parent if it's now empty and was created by us
                    if DEMO_DB_BASE_PATH != DEMO_DB_FULL_PATH and not os.listdir(DEMO_DB_BASE_PATH):
                        shutil.rmtree(DEMO_DB_BASE_PATH)
                        print(f"Successfully removed demo DB base directory: {DEMO_DB_BASE_PATH}")

                except Exception as e:
                    print(f"Error cleaning up demo DB: {e}")
        else:
            print("Retriever initialization failed. Skipping message adding and retrieval.")

    print("\nChatHistoryRetriever functional demo complete.")
