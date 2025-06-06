# YAWL (Yet Another Wrapper for Llama)

YAWL (Yet Another Wrapper for Llama) is a comprehensive system designed to streamline and enhance interactions with Large Language Models (LLMs). It provides a structured environment for managing LLM context, integrating diverse data sources via advanced RAG capabilities, leveraging various LLM runners, and utilizing OpenAI-compatible tool calls.

## Key Features

*   **Advanced Context Management:** Sophisticated control over conversational history and LLM context windows.
*   **Versatile Model Runner Support:** Compatibility with multiple LLM backends and formats (e.g., GGUF, AWQ, EXL2, API-based models like OpenAI, vLLM).
*   **Retrieval Augmented Generation (RAG):**
    *   Retrieval from PDF documents.
    *   Retrieval from chat history.
    *   Hybrid search (BM25 + dense vector search with RRF fusion).
    *   Cross-encoder re-ranking for enhanced relevance.
    *   Semantic chunking for PDF documents.
    *   Hypothetical Document Embeddings (HyDE) for query transformation.
*   **Tool Calling:** OpenAI-compatible tool/function calling with local and potentially remote (MCP) tool dispatch.
*   **Model Management:**
    *   API endpoints for loading models and managing their lifecycle (e.g., idle auto-unload).
    *   Scanning configured directories to discover available local models.
    *   Downloading models directly from Hugging Face Hub.
*   **Flexible Interaction:**
    *   FastAPI backend providing a comprehensive API.
    *   Tauri-based GUI frontend for desktop interaction.
*   **Performance & Efficiency:**
    *   KV cache preloading and management.
    *   (Planned) Speculative decoding.

## Getting Started

*(Detailed setup, installation, and basic usage instructions will be added here.)*

For now, ensure you have Python 3.10+ and consider setting up a virtual environment. Install dependencies using:
```bash
pip install -r requirements.txt
```
To run the backend API server:
```bash
cd yawl/api
python main.py
```
(Or use `uvicorn yawl.api.main:app --reload` from the project root if `yawl` is in your PYTHONPATH).

The GUI can be started by navigating to `yawl/gui/` and following the instructions in its README.md.

## Downloading Models from Hugging Face

YAWL provides an API endpoint to download models and specific files directly from Hugging Face Hub repositories. This feature simplifies acquiring models for local use.

**API Endpoint:** `/models/download` (POST)

**Functionality:**
This endpoint allows you to specify a Hugging Face `repo_id` and optionally a `filename` and `revision` (branch, tag, or commit hash) to download.
- If only `repo_id` is provided, the entire repository snapshot is downloaded.
- If `filename` is also provided, only that specific file is downloaded (common for GGUF).

Downloaded content is saved into a subdirectory (named after the repository) within the directory specified by `model_download_dir` in the `config.yaml` file.

**Configuration:**
Several settings in `config.yaml` (located in `yawl/config/config.yaml`) control the download behavior:
- `model_download_dir`: Base directory for all downloads (Default: `models/downloaded/`).
- `hf_token`: Your Hugging Face API token. Essential for private or gated models. Can also be set via the `HUGGING_FACE_HUB_TOKEN` environment variable.
- `hf_snapshot_ignore_patterns`: Glob patterns for files to ignore when downloading a full repository snapshot (e.g., `["*.md", "*.txt"]`).
- `hf_snapshot_allow_patterns`: Glob patterns for files to exclusively include in a snapshot download (e.g., `["*.gguf", "*.json"]`). Takes precedence over ignore patterns.
- `hf_repo_type`: Typically auto-detected (e.g., "model", "dataset"); usually can be left as `null`.

Refer to the comments in `config.yaml` for more details on these settings.

**Dependency:**
This feature requires the `huggingface_hub` Python library. Ensure it is included in your `requirements.txt` and installed in your environment:
```bash
pip install huggingface_hub
```
(Note: `nltk` is also a new dependency for the semantic chunking feature in PDFRetriever, ensure it's in `requirements.txt`).

**Making Downloaded Models Discoverable:**
For downloaded models to be listed by the `/models/available` endpoint and usable by the system, the directory specified in `model_download_dir` (e.g., `yawl/models/downloaded/`) or the specific path where a model was downloaded (e.g., `yawl/models/downloaded/TheBloke_Mistral-7B-Instruct-v0.1-GGUF/`) must be included in the `model_scan_directories` list in your `config.yaml`.

*Example `config.yaml` snippet:*
```yaml
model_scan_directories:
  - "models/" # This will scan subdirectories, including 'models/downloaded/' by default

model_download_dir: "models/downloaded/" # Path relative to project root (yawl/)

# If you set model_download_dir to an absolute path like "/data/my_llms/",
# you would also need to add "/data/my_llms/" (or a parent) to model_scan_directories.
```

**Usage Examples (`curl`):**

*   **Download a single GGUF file:**
    ```bash
    curl -X POST http://localhost:8000/models/download -H "Content-Type: application/json" -d \
    '{
      "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
      "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
      "revision": "main"
    }'
    ```

*   **Download an entire model repository snapshot (e.g., for safetensors models):**
    ```bash
    curl -X POST http://localhost:8000/models/download -H "Content-Type: application/json" -d \
    '{
      "repo_id": "meta-llama/Llama-2-7b-chat-hf",
      "revision": "main"
    }'
    ```
    (Note: For large snapshots, configure `hf_snapshot_allow_patterns` in `config.yaml` to target specific file types like `["*.safetensors", "*.json", "tokenizer.model"]` to avoid downloading unnecessary files.)

## Project Structure Overview (Conceptual - after rename)

- `yawl/api/`: FastAPI backend application.
- `yawl/config/`: Default configurations.
- `yawl/context/`: Context management and token estimation.
- `yawl/gui/`: Tauri-based React frontend.
- `yawl/retriever/`: RAG components (PDF, chat history).
- `yawl/runners/`: LLM model runners.
- `yawl/tools/`: Tool dispatching and built-in tools.
- `yawl/utils/`: Utility functions like model downloading and scanning.
- `yawl/tests/`: Unit and integration tests.

## License

*(License information to be added here. Assume MIT or Apache 2.0 unless specified otherwise.)*
