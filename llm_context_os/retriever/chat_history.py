# llm_context_os/retriever/chat_history.py
import uuid
import typing as t

# Placeholder for a real tokenizer if needed for more accurate budgeting
class SimpleCharTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

class ChatHistoryRetriever:
    """
    Placeholder for retrieving relevant snippets from chat history (micro-RAG).
    In a real implementation, this would involve:
    - Embedding messages using a sentence transformer.
    - Storing embeddings and messages in a vector database (e.g., ChromaDB).
    - Querying the database with the current input to find relevant historical snippets.
    - Managing a token budget for retrieved snippets.
    """

    def __init__(self,
                 recall_budget_tokens: int = 512,
                 tokenizer: t.Any = None, # Should be a proper tokenizer (e.g., from HF)
                 vector_db_path: str = 'data/chroma_db',
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the ChatHistoryRetriever.

        Args:
            recall_budget_tokens (int): Max tokens for retrieved snippets.
            tokenizer (t.Any): Tokenizer instance for budgeting. Uses SimpleCharTokenizer if None.
            vector_db_path (str): Path to the vector database.
            embedding_model_name (str): Name of the sentence transformer model.
        """
        self.recall_budget_tokens = recall_budget_tokens
        self.tokenizer = tokenizer if tokenizer else SimpleCharTokenizer()
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model_name

        self.stored_messages: t.List[t.Dict[str, str]] = [] # Stores dicts like {'role': role, 'text': text, 'id': id}

        # Placeholder for actual initialization of SentenceTransformer and ChromaDB
        # from sentence_transformers import SentenceTransformer
        # import chromadb
        # self.embedding_model = SentenceTransformer(self.embedding_model_name)
        # self.db_client = chromadb.PersistentClient(path=self.vector_db_path)
        # self.collection = self.db_client.get_or_create_collection(name="chat_history")

        print(f"ChatHistoryRetriever placeholder initialized:")
        print(f"  Recall Budget: {self.recall_budget_tokens} tokens")
        print(f"  Tokenizer: {type(self.tokenizer).__name__}")
        print(f"  Vector DB Path: {self.vector_db_path}")
        print(f"  Embedding Model: {self.embedding_model_name}")
        print("  (Note: Actual SentenceTransformer and ChromaDB not initialized in this placeholder)")

    def add_message(self, message_text: str, role: str, message_id: t.Optional[str] = None) -> str:
        """
        Adds a message to the history for potential retrieval.
        In a real implementation, this would embed the message and store it.

        Args:
            message_text (str): The content of the message.
            role (str): The role of the speaker (e.g., 'user', 'assistant').
            message_id (t.Optional[str]): Optional unique ID for the message. Generates one if None.

        Returns:
            str: The ID of the added message.
        """
        msg_id = message_id or str(uuid.uuid4())
        print(f"ChatHistoryRetriever placeholder: Would embed and store message (ID: {msg_id}): '{role}: {message_text[:100]}...'")

        message_data = {'role': role, 'text': message_text, 'id': msg_id}
        self.stored_messages.append(message_data)

        # Real implementation:
        # embedding = self.embedding_model.encode(message_text)
        # self.collection.add(
        #     ids=[msg_id],
        #     embeddings=[embedding.tolist()], # Store embedding
        #     documents=[message_text],         # Store original text
        #     metadatas=[{'role': role, 'timestamp': time.time()}] # Store metadata
        # )
        return msg_id

    def retrieve(self, query_text: str, current_chat_history: t.Optional[t.List[t.Dict[str, str]]] = None) -> t.List[t.Dict[str, str]]:
        """
        Retrieves relevant snippets from stored history based on the query.

        Args:
            query_text (str): The current user query or context to find relevant history for.
            current_chat_history (t.Optional[t.List[t.Dict[str, str]]]):
                The immediate preceding messages in the current conversation turn.
                Could be used to avoid retrieving messages already in the immediate context. (Not used in this placeholder)

        Returns:
            t.List[t.Dict[str, str]]: A list of dictionaries, where each dictionary
                                      is a retrieved message/snippet formatted for context insertion
                                      (e.g., {'r': 'retrieved_context', 'c': 'snippet_text'}).
        """
        print(f"\nChatHistoryRetriever placeholder: Retrieving snippets for query: '{query_text[:100]}...'")
        # Real implementation:
        # 1. Embed query_text:
        #    query_embedding = self.embedding_model.encode(query_text)
        # 2. Query ChromaDB:
        #    results = self.collection.query(
        #        query_embeddings=[query_embedding.tolist()],
        #        n_results=5 # Get a few candidates
        #        # where_filter can be used to exclude recent messages if their IDs are known
        #    )
        # 3. Process results:
        #    - Iterate through `results['documents']` or `results['ids']` and their distances/scores.
        #    - Format them into snippets.
        #    - Ensure total token count of snippets respects `self.recall_budget_tokens` using `self.tokenizer`.
        #    - Potentially re-rank or filter based on diversity, relevance, etc.

        if not self.stored_messages:
            print("  No messages in history to retrieve from.")
            return []

        # Placeholder logic: return the first stored message as a snippet
        first_message = self.stored_messages[0]
        snippet_text = f"Retrieved (placeholder): First stored message was '{first_message['role']}: {first_message['text'][:50]}...'"

        # Simulate token budgeting (very crudely)
        snippet_tokens = self.tokenizer.count_tokens(snippet_text)
        if snippet_tokens > self.recall_budget_tokens:
            # Crude truncation for placeholder
            max_len = self.tokenizer.count_tokens(f"Retrieved (placeholder): First stored message was '{first_message['role']}: ...'")
            available_chars = self.recall_budget_tokens - max_len
            if available_chars > 0 :
                 snippet_text = f"Retrieved (placeholder): First stored message was '{first_message['role']}: {first_message['text'][:available_chars]}...' (truncated)"
            else: # Should not happen if budget is reasonable for the prefix
                 snippet_text = f"Retrieved (placeholder): ..."


        print(f"  Returning fixed snippet based on the first stored message.")
        return [{'r': 'retrieved_context', 'c': snippet_text}]

if __name__ == '__main__':
    print("--- ChatHistoryRetriever Demo ---")
    retriever = ChatHistoryRetriever(recall_budget_tokens=100) # Small budget for testing truncation

    print("\n--- Adding messages ---")
    msg_id1 = retriever.add_message(message_text="Hello, this is the first message from the user.", role="user")
    msg_id2 = retriever.add_message(message_text="Hi Bob, this is Alice responding to your first message with a lot of details about our project plan which is very extensive and detailed.", role="assistant")
    msg_id3 = retriever.add_message(message_text="Okay Alice, thanks for the update. I have a follow-up question about the timeline.", role="user")

    print(f"\nTotal messages stored: {len(retriever.stored_messages)}")

    print("\n--- Retrieving based on a query ---")
    query1 = "What was the project plan discussion about?"
    retrieved_snippets1 = retriever.retrieve(query_text=query1)
    for snippet in retrieved_snippets1:
        print(f"  Snippet for query1: {snippet['r']}: {snippet['c']}")

    print("\n--- Retrieving with empty history (after creating new retriever) ---")
    empty_retriever = ChatHistoryRetriever()
    retrieved_snippets_empty = empty_retriever.retrieve("Any query")
    print(f"  Snippets from empty history: {retrieved_snippets_empty}")

    print("\n--- Retrieving with different budget (example of truncation) ---")
    # First message: "Hello, this is the first message from the user."
    # Placeholder format: "Retrieved (placeholder): First stored message was 'user: Hello, this is the first message from the user....'"
    # len("Retrieved (placeholder): First stored message was 'user: ...'") approx 55 chars
    # len("Hello, this is the first message from the user.") = 48 chars
    # If budget is 60, text[:(60-55)] = text[:5] = "Hello"
    # Expected: "Retrieved (placeholder): First stored message was 'user: Hello...' (truncated)"

    low_budget_retriever = ChatHistoryRetriever(recall_budget_tokens=60)
    low_budget_retriever.add_message(message_text="Hello, this is the first message from the user.", role="user")
    retrieved_low_budget = low_budget_retriever.retrieve("test")
    for snippet in retrieved_low_budget:
        print(f"  Snippet (low budget): {snippet['r']}: {snippet['c']}")


    print("\nChatHistoryRetriever demo complete.")
