# Retrievers in LLM Context OS

The LLM Context OS employs retrievers to fetch relevant information for augmenting the context provided to Large Language Models (LLMs). This enhances the LLM's ability to answer questions and generate text based on specific documents or chat history.

Two primary retrievers are currently implemented: `PdfRetriever` and `ChatHistoryRetriever`. Both leverage vector embeddings for semantic search and now include a re-ranking step for improved relevance.

## General Retriever Configuration

### Embedding Models
Both retrievers use sentence-transformer models to generate embeddings for text chunks. These embeddings are stored in a vector database (ChromaDB) to enable similarity searches.

*   **Default Embedding Model:** The default model has been updated to `"BAAI/bge-base-en-v1.5"`, which offers strong performance. This can be configured in `config.yaml`.
*   **Changing Models:** If you change the `embedding_model_name` after documents or chat history have been processed and embedded, you will need to re-process (re-embed) that data to ensure consistency and optimal search results with the new model. The system does not automatically re-embed existing data upon model change.

### Re-ranking
To improve the quality of retrieved results, a re-ranking step has been introduced:
1.  An initial set of candidates is fetched from the vector database based on semantic similarity (vector distance).
2.  A more computationally intensive cross-encoder model then re-evaluates these initial candidates against the query.
3.  The candidates are re-sorted based on the cross-encoder's relevance scores, providing a more refined set of results to the LLM.

This two-stage process aims to balance speed (initial vector search) with accuracy (cross-encoder re-ranking).

## PdfRetriever

The `PdfRetriever` is responsible for extracting text from PDF documents, chunking it, embedding the chunks, and retrieving relevant chunks based on a query.

### Key Features:
*   **PDF Parsing:** Uses `PyMuPDF (fitz)` for robust and efficient text extraction from PDF files. This replaces the previous PyPDF2 implementation.
*   **Chunking:** Splits extracted text into manageable chunks suitable for embedding.
*   **Embedding & Storage:** Embeds text chunks and stores them in a ChromaDB vector database.
*   **Retrieval & Re-ranking:**
    *   Retrieves an initial set of candidate chunks using vector similarity.
    *   Re-ranks these candidates using a configurable cross-encoder model.
*   **Configuration (in `config.yaml` under `pdf_retriever`):**
    *   `embedding_model_name`: The sentence-transformer model for embedding PDF chunks (e.g., `"BAAI/bge-base-en-v1.5"`).
    *   `vector_db_path`: Filesystem path for the ChromaDB vector store for PDF chunks.
    *   `chunk_size`, `chunk_overlap`: Parameters for text splitting.
    *   `recall_budget_tokens`: Maximum total tokens for snippets to be returned.
    *   `cross_encoder_model_name`: The cross-encoder model for re-ranking (e.g., `"cross-encoder/ms-marco-MiniLM-L-6-v2"`).
    *   `rerank_top_n_candidates`: The number of initial candidates to fetch from the vector DB for the cross-encoder to re-rank.

## ChatHistoryRetriever

The `ChatHistoryRetriever` stores and retrieves messages from past conversations, enabling the LLM to maintain context over longer interactions.

### Key Features:
*   **Message Storage:** Stores individual chat messages (user and assistant) with their embeddings in a ChromaDB collection.
*   **Retrieval & Re-ranking:**
    *   Retrieves relevant past messages based on the current query or conversation context.
    *   Re-ranks these messages using a configurable cross-encoder model for improved contextual relevance.
*   **Configuration (in `config.yaml` under `chat_history_retriever`):**
    *   `embedding_model_name`: The sentence-transformer model for embedding chat messages (e.g., `"BAAI/bge-base-en-v1.5"`).
    *   `vector_db_path`: Filesystem path for the ChromaDB vector store for chat history.
    *   `recall_budget_tokens`: Maximum total tokens for snippets to be returned.
    *   `cross_encoder_model_name`: The cross-encoder model for re-ranking (e.g., `"cross-encoder/ms-marco-MiniLM-L-6-v2"`).
    *   `rerank_top_n_candidates`: The number of initial candidates to fetch from the vector DB for the cross-encoder to re-rank.

By using these retrievers, the LLM Context OS can provide more accurate, contextually relevant, and well-informed responses.
