# Retrievers in YAWL

YAWL (Yet Another Wrapper for Llama) employs retrievers to fetch relevant information for augmenting the context provided to Large Language Models (LLMs). This enhances the LLM's ability to answer questions and generate text based on specific documents or chat history.

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
    *   `recall_budget_tokens`: Max tokens for returned snippets.
    *   `chunking_strategy`: How text is split. Options: `"recursive"`, `"semantic"`. (See "Chunking Strategies" below).
    *   `chunk_size`, `chunk_overlap`: Parameters for the `"recursive"` chunking strategy.
    *   `semantic_chunker_embedding_model`, `semantic_chunker_breakpoint_threshold_type`, `semantic_chunker_breakpoint_threshold_amount`, `semantic_chunker_min_chunk_sentences`: Parameters for the `"semantic"` chunking strategy.
    *   `enable_hybrid_search`: `true` or `false` to enable/disable BM25 + RRF fusion.
    *   `rrf_k_constant`: Constant for the RRF algorithm (e.g., `60`).
    *   `cross_encoder_model_name`: Cross-encoder model for re-ranking (e.g., `"cross-encoder/ms-marco-MiniLM-L-6-v2"`).
    *   `rerank_top_n_candidates`: Number of candidates passed to the cross-encoder (also used for initial dense/sparse fetches if hybrid search is on).
    *   `enable_hyde`: `true` or `false` to enable Hypothetical Document Embeddings (HyDE) for this retriever.

### Chunking Strategies for PdfRetriever

The `PdfRetriever` supports different strategies for chunking PDF text content before embedding:

1.  **`recursive` (Default):**
    *   Uses `langchain_text_splitters.RecursiveCharacterTextSplitter`.
    *   This method splits text based on a list of separators (e.g., newlines, spaces) and aims to create chunks of a fixed size.
    *   Configuration options:
        *   `chunk_size`: Approximate size of each chunk in characters.
        *   `chunk_overlap`: Number of characters to overlap between adjacent chunks to maintain context.

2.  **`semantic` (Experimental):**
    *   Aims to create more contextually coherent chunks by splitting text based on semantic similarity between sentences.
    *   **How it works:**
        1.  The document text is first split into individual sentences using `nltk.sent_tokenize`.
        2.  Each sentence is then embedded using a SentenceTransformer model.
        3.  Cosine similarities between adjacent sentence embeddings are calculated.
        4.  Breakpoints (splits) are identified where the similarity between adjacent sentences drops below a certain threshold. This indicates a potential shift in topic.
        5.  Sentences are grouped into chunks based on these breakpoints.
    *   **Goal:** To produce chunks that are more semantically self-contained, potentially improving retrieval quality by better aligning chunk content with query intent.
    *   **Configuration options (in `config.yaml` under `pdf_retriever`):**
        *   `chunking_strategy: "semantic"` (to enable this strategy).
        *   `semantic_chunker_embedding_model`: The SentenceTransformer model used to embed sentences for similarity calculation. If `null`, the main `embedding_model_name` of the `PdfRetriever` is used. This can be set to a different, possibly lighter-weight, model optimized for sentence similarity tasks.
        *   `semantic_chunker_breakpoint_threshold_type`: Method to determine the similarity threshold for splitting. Currently, `"percentile"` is the primary supported method.
            *   `"percentile"`: Splits if the similarity is below the Nth percentile of all calculated adjacent sentence similarities.
        *   `semantic_chunker_breakpoint_threshold_amount`: The value for the chosen threshold type. For `"percentile"`, this is N (e.g., `5` for the 5th percentile). A lower percentile value (e.g., 5) means splits occur at points of very low similarity, leading to larger, more distinct chunks. A higher value (e.g., 25) will create more, smaller chunks.
        *   `semantic_chunker_min_chunk_sentences`: The minimum number of sentences required to form a valid chunk. This prevents overly short or fragmented chunks.
    *   **Dependencies:** This strategy requires the `nltk` library and its `punkt` sentence tokenizer data. The system will attempt to download `punkt` automatically if it's not found, but an internet connection would be needed for this first-time download.

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
    *   `enable_hyde`: `true` or `false` to enable Hypothetical Document Embeddings (HyDE) for this retriever.

By using these configurable retrieval stages, YAWL can provide highly relevant information to the LLM. Remember to install `rank_bm25` if using the hybrid search feature.

## Advanced Retrieval Techniques

### Hypothetical Document Embeddings (HyDE)

**Concept:**
HyDE is a technique that aims to improve the relevance of dense retrieval, especially for queries that might be vague, use different terminology than the source documents, or require inferential reasoning. Instead of directly embedding the user's raw query, HyDE first uses a Large Language Model (LLM) to generate a "hypothetical" document that *could* answer the query. This generated document is then embedded, and its embedding is used for the dense vector search against the actual document chunks.

**Benefits:**
*   **Improved Semantic Matching:** The hypothetical document often captures the underlying intent and expected information structure better than the raw query, leading to embeddings that are closer to relevant document chunks in the vector space.
*   **Addresses Keyword Mismatch:** Helpful when the query uses synonyms or related concepts not explicitly present in the target documents.

**Configuration:**
HyDE can be enabled independently for `PdfRetriever` and `ChatHistoryRetriever` via the `enable_hyde: true` flag in their respective sections in `config.yaml`.

Global HyDE settings are configured under a top-level `hyde:` section in `config.yaml`:
*   `llm_identifier`: Specifies the LLM to use for generating the hypothetical documents. `"default"` uses the currently loaded main model in the `ModelManager`. (Future versions might allow specifying a dedicated, possibly smaller/faster model for HyDE).
*   `prompt_template`: The template used to instruct the LLM. It must include a `{query}` placeholder, which will be replaced by the user's actual query. Example: `"Generate a concise, relevant document that could answer the following question...: {query}"`
*   `max_tokens_hyde_doc`: The maximum number of tokens for the generated hypothetical document. Keeping this relatively low (e.g., 64-256) is recommended for conciseness and speed.

**Interaction with Other Retrieval Stages:**
*   **Dense Search:** The embedding of the *hypothetical document* is used for the dense vector search (e.g., ChromaDB query).
*   **BM25 Sparse Search:** If hybrid search is enabled, BM25 always uses the *original user query* for keyword matching.
*   **Cross-Encoder Re-ranking:** The cross-encoder, if enabled, always re-ranks candidates against the *original user query* to ensure final relevance to what the user actually asked.

This ensures that while HyDE assists in finding semantically relevant candidates, the final filtering and ranking stages are still grounded in the original query.
