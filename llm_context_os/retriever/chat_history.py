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
                 vector_db_path: str = 'data/vector_dbs', # Base path for all vector DBs
                 embedding_model_name: t.Optional[str] = None,
                 collection_name: t.Optional[str] = None):
        """
        Initializes the ChatHistoryRetriever.
        """
        self.recall_budget_tokens = recall_budget_tokens
        self.embedding_model_name = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL

        self.db_collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.vector_db_full_path = os.path.join(vector_db_path, self.DEFAULT_DB_SUBDIR)

        self.embedding_model = None
        self.db_client = None
        self.collection = None

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"Error: SentenceTransformers library is required but not installed.")
            # Or raise an exception, depending on desired strictness
            return
        if not CHROMADB_AVAILABLE:
            print(f"Error: ChromaDB library is required but not installed.")
            return

        try:
            print(f"Initializing SentenceTransformer model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"Error initializing SentenceTransformer model '{self.embedding_model_name}': {e}")
            # Fallback or re-raise
            return

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


    def add_message(self, message_text: str, role: str, message_id: t.Optional[str] = None) -> t.Optional[str]:
        """ Adds a message to the history, embeds it, and stores it in ChromaDB. """
        if not self.embedding_model or not self.collection:
            print("Error: Retriever not properly initialized (model or DB collection missing). Cannot add message.")
            return None

        msg_id = message_id or str(uuid.uuid4())

        try:
            # Ensure message_text is a string, handle if it's not (e.g. None)
            if not isinstance(message_text, str):
                message_text = str(message_text) # Attempt to cast, or handle error

            embedding = self.embedding_model.encode(message_text).tolist()

            metadata = {
                'role': role,
                'timestamp': time.time(),
                'text_length_chars': len(message_text), # Store original char length
                'source': 'chat_history' # Generic source type
            }
            # Ensure all metadata values are valid ChromaDB types (str, int, float, bool)
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)


            self.collection.add(
                ids=[msg_id],
                embeddings=[embedding],
                documents=[message_text], # Store original text
                metadatas=[metadata]
            )
            print(f"Added message ID {msg_id} to ChromaDB: '{role}: {message_text[:100]}...'")
            return msg_id
        except Exception as e:
            print(f"Error adding message ID {msg_id} to ChromaDB: {e}")
            # import traceback; traceback.print_exc() # For detailed debugging
            return None


    def retrieve(self, query_text: str, current_chat_history: t.Optional[t.List[t.Dict[str, str]]] = None, n_results: int = 5) -> t.List[t.Dict[str, str]]:
        if not self.embedding_model or not self.collection or not self.tokenizer:
            print("Error: Retriever not properly initialized. Cannot retrieve.")
            return []

        if self.collection.count() == 0:
            print("No messages in history to retrieve from.")
            return []

        try:
            query_embedding = self.embedding_model.encode(query_text).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()), # Ensure n_results <= collection size
                include=['documents', 'metadatas', 'distances'] # Request these fields
            )
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

        retrieved_snippets: t.List[t.Dict[str, str]] = []
        current_token_count = 0

        # Results are lists of lists (one list per query embedding, we have one query)
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] # Lower distance is better

        print(f"\nChatHistoryRetriever: Querying for '{query_text[:100]}...', found {len(ids)} candidates.")

        for i in range(len(ids)):
            doc_id = ids[i]
            doc_text = documents[i]
            doc_meta = metadatas[i]
            doc_dist = distances[i]

            # Avoid retrieving messages that are identical to the query or very recent in `current_chat_history`
            # (Simple check, more sophisticated logic could be added)
            if doc_text == query_text: # Simple exact match skip
                print(f"  Skipping identical document ID {doc_id}.")
                continue
            if current_chat_history: # Check if snippet is already in recent context
                 if any(hist_msg.get('c') == doc_text for hist_msg in current_chat_history):
                    print(f"  Skipping document ID {doc_id} as it's in current_chat_history.")
                    continue


            role = doc_meta.get('role', 'context')
            timestamp_str = f" (at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(doc_meta.get('timestamp', 0)))})" \
                            if doc_meta.get('timestamp') else ""

            # Format snippet for context injection
            # Using a more structured way for retrieved context
            snippet_text = f"Previously, {role} said{timestamp_str}: \"{doc_text}\""

            try:
                # Use the .encode() method of the tokenizer for counting, as it's more consistent
                # with how tiktoken's Encoding object works (len(encoding.encode(text)))
                snippet_tokens = len(self.tokenizer.encode(snippet_text))
            except Exception as e:
                print(f"Warning: Error tokenizing snippet, falling back to char count for budgeting: {e}")
                snippet_tokens = len(snippet_text) # Fallback for budgeting if tokenizer fails

            if current_token_count + snippet_tokens <= self.recall_budget_tokens:
                retrieved_snippets.append({'r': 'retrieved_context', 'c': snippet_text})
                current_token_count += snippet_tokens
                print(f"  Added snippet (ID: {doc_id}, dist: {doc_dist:.4f}, tokens: {snippet_tokens}): '{snippet_text[:100]}...'")
            else:
                print(f"  Budget exceeded with snippet ID {doc_id} (tokens: {snippet_tokens}). Total: {current_token_count}. Stopping.")
                break

        # Optional: Re-sort snippets by original timestamp if desired, though ChromaDB results are by relevance.
        # For chat history, chronological order of retrieved snippets might be useful.
        # This would require parsing timestamp from snippet_text or having it in metadata consistently.
        # For now, they are relevance-sorted.

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
