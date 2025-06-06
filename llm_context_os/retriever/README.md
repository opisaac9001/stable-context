# Retrievers in LLM Context OS

The LLM Context OS employs retrievers to fetch relevant information for augmenting the context provided to Large Language Models (LLMs). This enhances the LLM's ability to answer questions and generate text based on specific documents or chat history.

Two primary retrievers are currently implemented: `PdfRetriever` and `ChatHistoryRetriever`. Both leverage vector embeddings for semantic search and can be enhanced with hybrid search and re-ranking stages for improved relevance.

## General Retriever Configuration

### Embedding Models
Both retrievers use sentence-transformer models to generate embeddings for text chunks. These embeddings are stored in a vector database (ChromaDB) to enable similarity searches (dense retrieval).

*   **Default Embedding Model:** The default model is `"BAAI/bge-base-en-v1.5"`, which offers strong performance. This can be configured in `config.yaml`.
*   **Changing Models:** If you change the `embedding_model_name` after documents or chat history have been processed and embedded, you will need to re-process (re-embed) that data to ensure consistency and optimal search results with the new model. The system does not automatically re-embed existing data upon model change.

### Hybrid Search (BM25 + Dense Fusion)
To combine the strengths of keyword-based search and semantic search, retrievers can optionally use hybrid search:

1.  **Dense Retrieval:** Fetches initial candidates based on semantic similarity using vector embeddings (as described above).
2.  **Sparse Retrieval (BM25):** Simultaneously, a BM25Okapi index (from the `rank_bm25` library, a new dependency) is used to fetch candidates based on keyword overlap with the query. The BM25 index is built in memory from the text corpus.
3.  **Reciprocal Rank Fusion (RRF):** The results from dense and sparse retrieval are then fused using RRF. This method combines the ranks of documents from different result lists to produce a single, more robust ranking.
    *   The `rrf_k_constant` in the configuration fine-tunes the RRF algorithm.

This hybrid approach helps retrieve documents that are both semantically similar and contain relevant keywords.

### Re-ranking (Cross-Encoder)
After the initial retrieval (either dense-only or hybrid-fused), a further re-ranking step can be applied:

1.  The top candidates from the previous stage (e.g., RRF output or top dense results) are taken.
2.  A more computationally intensive cross-encoder model then re-evaluates these candidates against the query.
3.  The candidates are re-sorted based on the cross-encoder's relevance scores, providing a highly refined set of results to be used for snippet generation.

This multi-stage process (Dense/Sparse -> RRF Fusion -> Cross-Encoder Re-ranking) aims to maximize relevance.

## PdfRetriever

The `PdfRetriever` is responsible for extracting text from PDF documents, chunking it, embedding the chunks, and retrieving relevant chunks based on a query.

### Key Features:
*   **PDF Parsing:** Uses `PyMuPDF (fitz)` for robust and efficient text extraction from PDF files.
*   **Chunking & Embedding:** Splits text and stores embeddings in ChromaDB.
*   **Retrieval Pipeline:**
    *   Optional Hybrid Search (BM25 + Dense with RRF fusion).
    *   Optional Cross-Encoder re-ranking on the results of the previous stage.
*   **Configuration (in `config.yaml` under `pdf_retriever`):**
    *   `embedding_model_name`: Model for embedding PDF chunks (e.g., `"BAAI/bge-base-en-v1.5"`).
    *   `vector_db_path`: Path for ChromaDB store.
    *   `chunk_size`, `chunk_overlap`: Text splitting parameters.
    *   `recall_budget_tokens`: Max tokens for returned snippets.
    *   `enable_hybrid_search`: `true` or `false` to enable/disable BM25 + RRF fusion.
    *   `rrf_k_constant`: Constant for the RRF algorithm (e.g., `60`).
    *   `cross_encoder_model_name`: Cross-encoder model for re-ranking (e.g., `"cross-encoder/ms-marco-MiniLM-L-6-v2"`).
    *   `rerank_top_n_candidates`: Number of candidates passed to the cross-encoder (also used for initial dense/sparse fetches if hybrid search is on).

## ChatHistoryRetriever

The `ChatHistoryRetriever` stores and retrieves messages from past conversations.

### Key Features:
*   **Message Storage:** Stores messages with embeddings in ChromaDB.
*   **Retrieval Pipeline:**
    *   Optional Hybrid Search (BM25 + Dense with RRF fusion).
    *   Optional Cross-Encoder re-ranking.
*   **Configuration (in `config.yaml` under `chat_history_retriever`):**
    *   `embedding_model_name`: Model for embedding messages (e.g., `"BAAI/bge-base-en-v1.5"`).
    *   `vector_db_path`: Path for ChromaDB store.
    *   `recall_budget_tokens`: Max tokens for returned snippets.
    *   `enable_hybrid_search`: `true` or `false`.
    *   `rrf_k_constant`: Constant for RRF (e.g., `60`).
    *   `cross_encoder_model_name`: Cross-encoder model for re-ranking.
    *   `rerank_top_n_candidates`: Number of candidates for cross-encoder.

By using these configurable retrieval stages, the LLM Context OS can provide highly relevant information to the LLM. Remember to install `rank_bm25` if using the hybrid search feature.
