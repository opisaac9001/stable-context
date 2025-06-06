# yawl/api/main.py
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import time
import json
import asyncio # For potential sleep in streaming, and async generator
from typing import List, Optional, AsyncGenerator, Dict, Any
from pathlib import Path
import tempfile
import shutil

# App-specific imports
import yaml # For config loading
from yawl.api.schemas import (
    LoadModelRequest,
    ChatRequest,
    StatusResponse,
    ChatRequest,
    StatusResponse,
    ChatResponse,
    UploadPdfResponse,
    GenerationParams,
    GlobalSettings, # For GET /settings
    UpdateSettingsRequest, # For PUT /settings
    # Import individual settings models for type hints during propagation
    ContextManagerSettings,
    ChatHistoryRetrieverSettings,
    PdfRetrieverSettings,
    ModelManagerSettings,
    TokenEstimatorConfigSettings,
    ModelListResponse, # For /models/available
    ToolInfo, ToolListResponse, ToggleToolRequest, # Added for tool management
    DownloadModelRequest # Added for model download
)
from yawl.utils.hf_downloader import download_model_from_hf # Added for model download
from yawl.context.context_manager import ContextManager, MockTokenizer
from yawl.context.token_estimator import TikTokenEstimator, HFTokenEstimator # For RAG tokenizer
from yawl.runners.manager import ModelManager
from yawl.runners.base import BaseRunner
from yawl.retriever.chat_history import ChatHistoryRetriever
from yawl.retriever.pdf_retriever import PdfRetriever
from yawl.tools.tool_dispatcher import ToolDispatcher

# --- Application Setup ---
app = FastAPI(
    title="YAWL API",
    description="YAWL (Yet Another Wrapper for Llama): API for managing local LLM models, context, RAG, and tool-enhanced generation. "
                "Supports OpenAI-compatible tool calling and schema retrieval.",
    version="0.1.3" # Incremented version for tool call and RAG enhancements
)

# --- Configuration Loading ---
DEFAULT_CONFIG = {
    "context_manager": {"system_prompt": "You are a helpful AI assistant.", "max_tokens": 4096},
    "chat_history_retriever": {
        "recall_budget_tokens": 512,
        "embedding_model_name": "BAAI/bge-base-en-v1.5",
        "vector_db_path": "data/vector_dbs/api_default_chat_history",
        "cross_encoder_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", # Default from class
        "rerank_top_n_candidates": 20, # Default from class
        "enable_hybrid_search": True,
        "rrf_k_constant": 60,
        "enable_hyde": False # Default for HyDE
    },
    "pdf_retriever": {
        "vector_db_path": "data/vector_dbs/api_default_pdf_rag",
        "embedding_model_name": "BAAI/bge-base-en-v1.5",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "cross_encoder_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", # Default from class
        "rerank_top_n_candidates": 20, # Default from class
        "enable_hybrid_search": True,
        "rrf_k_constant": 60,
        # New semantic chunking defaults
        "chunking_strategy": "recursive",
        "semantic_chunker_embedding_model": None,
        "semantic_chunker_breakpoint_threshold_type": "percentile",
        "semantic_chunker_breakpoint_threshold_amount": 5, # e.g., 5th percentile
        "semantic_chunker_min_chunk_sentences": 2,
        "enable_hyde": False # Default for HyDE
    },
    "model_manager": {"default_idle_unload_sec": 900},
    "token_estimator_for_rag_budgeting": {"type": "tiktoken", "model_name": "cl100k_base"},
    "model_discovery": { # Added model_discovery default
        "scan_directories": [] # Default to no scan directories
    },
    "hyde": { # New HyDE defaults
        "llm_identifier": "default",
        "prompt_template": "Generate a concise, relevant document that could answer the following question. Focus on providing factual-sounding information directly related to the query's core subject: {query}",
        "max_tokens_hyde_doc": 128
    }
}
CONFIG = DEFAULT_CONFIG.copy() # Start with defaults
try:
    # Assuming config.yaml is in yawl/config/config.yaml relative to project root
    # For robustness, resolve path from this file's location.
    # __file__ is yawl/api/main.py
    # So, parent is api/, parent.parent is yawl/
    config_file_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_file_path.exists():
        print(f"Loading configuration from: {config_file_path}")
        with open(config_file_path, 'r') as f:
            loaded_config_yaml = yaml.safe_load(f)
        if loaded_config_yaml: # Merge loaded config into defaults
            for key, value in loaded_config_yaml.items():
                if key in CONFIG and isinstance(CONFIG[key], dict) and isinstance(value, dict):
                    # Deep merge for one level of nesting
                    CONFIG[key].update(value)
                else:
                    CONFIG[key] = value
            print("Configuration loaded and merged successfully.")
    else:
        print(f"Warning: config.yaml not found at {config_file_path}. Using default API configurations.")
except Exception as e:
    print(f"Error loading or parsing config.yaml: {e}. Using default API configurations.")

# --- Global Instances Initialized from CONFIG ---

# Tokenizer for RAG and ContextManager budgeting
rag_tokenizer_config = CONFIG.get('token_estimator_for_rag_budgeting', DEFAULT_CONFIG['token_estimator_for_rag_budgeting'])
rag_tokenizer = None
print(f"Attempting to load RAG tokenizer based on config: {rag_tokenizer_config}")
if rag_tokenizer_config['type'] == 'tiktoken':
    try:
        rag_tokenizer = TikTokenEstimator(model_name=rag_tokenizer_config.get('model_name', 'cl100k_base'))
        print(f"Using TikTokenEstimator ('{rag_tokenizer_config.get('model_name', 'cl100k_base')}') for RAG budgeting.")
    except Exception as e:
        print(f"Warning: Could not load TikTokenEstimator for RAG: {e}")
elif rag_tokenizer_config['type'] == 'hf':
    try:
        rag_tokenizer = HFTokenEstimator(model_name=rag_tokenizer_config.get('model_name', 'gpt2'))
        print(f"Using HFTokenEstimator ('{rag_tokenizer_config.get('model_name', 'gpt2')}') for RAG budgeting.")
    except Exception as e:
        print(f"Warning: Could not load HFTokenEstimator for RAG: {e}")

if not rag_tokenizer:
    print("Warning: Using fallback MockTokenizer for RAG/Context budgeting due to previous errors.")
    rag_tokenizer = MockTokenizer()

# Context Manager
ctx_mgr_config = CONFIG.get('context_manager', DEFAULT_CONFIG['context_manager'])
ctx_mgr = ContextManager(
    system_prompt=ctx_mgr_config['system_prompt'],
    tokenizer=rag_tokenizer, # Use the RAG tokenizer for context manager too
    max_tokens=ctx_mgr_config['max_tokens']
)
print(f"ContextManager initialized with: system_prompt='{ctx_mgr_config['system_prompt'][:50]}...', max_tokens={ctx_mgr_config['max_tokens']}, tokenizer={type(rag_tokenizer).__name__}")


# Chat History Retriever
chr_config = CONFIG.get('chat_history_retriever', DEFAULT_CONFIG['chat_history_retriever'])
chat_history_retriever = ChatHistoryRetriever(
    recall_budget_tokens=chr_config.get('recall_budget_tokens'), # Use .get for robustness
    tokenizer=rag_tokenizer,
    vector_db_path=chr_config.get('vector_db_path'),
    embedding_model_name=chr_config.get('embedding_model_name'),
    cross_encoder_model_name=chr_config.get('cross_encoder_model_name'),
    rerank_top_n_candidates=chr_config.get('rerank_top_n_candidates'),
    enable_hybrid_search=chr_config.get('enable_hybrid_search', True),
    rrf_k_constant=chr_config.get('rrf_k_constant', 60),
    enable_hyde=chr_config.get('enable_hyde', False) # Pass HyDE setting
)
print(f"ChatHistoryRetriever initialized with: budget={chat_history_retriever.recall_budget_tokens}, "
      f"db_path='{chat_history_retriever.vector_db_full_path}', model='{chat_history_retriever.embedding_model_name}', "
      f"enable_hyde='{chat_history_retriever.enable_hyde}', "
      f"cross_encoder='{chat_history_retriever.cross_encoder_model_name}', rerank_top_n={chat_history_retriever.rerank_top_n_candidates}, "
      f"hybrid_search={chat_history_retriever.enable_hybrid_search}, rrf_k={chat_history_retriever.rrf_k_constant}")

# PDF Retriever
pdf_retriever_config = CONFIG.get('pdf_retriever', DEFAULT_CONFIG['pdf_retriever'])
pdf_retriever = PdfRetriever(
    vector_db_path=pdf_retriever_config.get('vector_db_path'),
    embedding_model_name=pdf_retriever_config.get('embedding_model_name'),
    tokenizer=rag_tokenizer,
    chunk_size=pdf_retriever_config.get('chunk_size'),
    chunk_overlap=pdf_retriever_config.get('chunk_overlap'),
    cross_encoder_model_name=pdf_retriever_config.get('cross_encoder_model_name'),
    rerank_top_n_candidates=pdf_retriever_config.get('rerank_top_n_candidates'),
    enable_hybrid_search=pdf_retriever_config.get('enable_hybrid_search', True),
    rrf_k_constant=pdf_retriever_config.get('rrf_k_constant', 60),
    # Semantic chunking parameters
    chunking_strategy=pdf_retriever_config.get('chunking_strategy', 'recursive'),
    semantic_chunker_embedding_model=pdf_retriever_config.get('semantic_chunker_embedding_model'),
    semantic_chunker_breakpoint_threshold_type=pdf_retriever_config.get('semantic_chunker_breakpoint_threshold_type', 'percentile'),
    semantic_chunker_breakpoint_threshold_amount=pdf_retriever_config.get('semantic_chunker_breakpoint_threshold_amount', 5),
    semantic_chunker_min_chunk_sentences=pdf_retriever_config.get('semantic_chunker_min_chunk_sentences', 2),
    enable_hyde=pdf_retriever_config.get('enable_hyde', False) # Pass HyDE setting
)
print(f"PdfRetriever initialized with: db_path='{pdf_retriever.vector_db_path}', model='{pdf_retriever.embedding_model_name}', "
      f"chunk_strategy='{pdf_retriever.chunking_strategy}', enable_hyde='{pdf_retriever.enable_hyde}', "
      f"cross_encoder='{pdf_retriever.cross_encoder_model_name}', rerank_top_n={pdf_retriever.rerank_top_n_candidates}, "
      f"hybrid_search={pdf_retriever.enable_hybrid_search}, rrf_k={pdf_retriever.rrf_k_constant}")


# Model Manager
model_mgr_config = CONFIG.get('model_manager', DEFAULT_CONFIG['model_manager'])
model_mgr = ModelManager(default_idle_unload_sec=model_mgr_config['default_idle_unload_sec'])
print(f"ModelManager initialized with default_idle_unload_sec={model_mgr_config['default_idle_unload_sec']}")

# Tool Dispatcher (currently no config needed from file for its __init__)
tool_dispatcher = ToolDispatcher()
print("ToolDispatcher initialized.")

# --- Project Root for resolving relative paths in config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # yawl directory
print(f"Project root determined for resolving relative paths: {PROJECT_ROOT}")

# --- Helper for deep merging dictionaries for settings updates ---
def deep_merge_dicts(source: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_merge_dicts(source[key], value)
        else:
            source[key] = value
    return source

# --- API Endpoints ---

@app.get("/settings", response_model=GlobalSettings)
async def get_settings():
    """
    Retrieves the current application settings as loaded from config.yaml and defaults.
    Note: This endpoint returns the configuration values as understood by the application.
    It does not include sensitive information like API keys if they were handled solely by runners.
    """
    # Construct GlobalSettings Pydantic model from the CONFIG dictionary
    # This ensures that only fields defined in GlobalSettings schema are returned
    # and that they are validated.

    # Default GenerationParams needs to be constructed if present in CONFIG
    gen_defaults_dict = CONFIG.get('generation_defaults')
    gen_defaults_model = GenerationParams(**gen_defaults_dict) if gen_defaults_dict else None

    settings_to_return = GlobalSettings(
        context_manager=CONFIG.get('context_manager'),
        chat_history_retriever=CONFIG.get('chat_history_retriever'),
        pdf_retriever=CONFIG.get('pdf_retriever'),
        model_manager=CONFIG.get('model_manager'),
        token_estimator_for_rag_budgeting=CONFIG.get('token_estimator_for_rag_budgeting'),
        generation_defaults=gen_defaults_model # Use the Pydantic model instance
    )
    return settings_to_return

@app.put("/settings", response_model=StatusResponse)
async def update_settings_endpoint(updated_values: UpdateSettingsRequest):
    """
    Updates global application settings.
    Changes are applied to the in-memory CONFIG and propagated to live service instances where possible.
    Some changes (e.g., vector_db_path, embedding_model_name for retrievers, RAG tokenizer type)
    may require an application restart to take full effect on existing components.
    """
    global CONFIG # Ensure we are modifying the global CONFIG

    update_data = updated_values.model_dump(exclude_unset=True)
    if not update_data:
        return StatusResponse(status="ok", message="No settings provided to update.")

    print(f"Received /settings PUT request with data: {update_data}")

    # Perform a deep merge of the new settings into the existing CONFIG
    CONFIG = deep_merge_dicts(CONFIG, update_data)
    print("Global CONFIG updated in memory.")

    applied_notes = []
    restart_notes = []

    # Propagate changes to live instances
    if "context_manager" in update_data:
        cm_updates = CONFIG.get('context_manager', {})
        if "max_tokens" in cm_updates and ctx_mgr.max_tokens != cm_updates["max_tokens"]:
            ctx_mgr.max_tokens = cm_updates["max_tokens"]
            applied_notes.append(f"ContextManager max_tokens updated to {ctx_mgr.max_tokens}.")
        if "system_prompt" in cm_updates and ctx_mgr.system_prompt != cm_updates["system_prompt"]:
            ctx_mgr.system_prompt = cm_updates["system_prompt"]
            applied_notes.append("ContextManager system_prompt updated.")

    if "model_manager" in update_data:
        mm_updates = CONFIG.get('model_manager', {})
        if "default_idle_unload_sec" in mm_updates and model_mgr.default_idle_unload_sec != mm_updates["default_idle_unload_sec"]:
            model_mgr.default_idle_unload_sec = mm_updates["default_idle_unload_sec"]
            if model_mgr.current_runner and model_mgr.current_runner_config:
                 pass
            applied_notes.append(f"ModelManager default_idle_unload_sec updated to {model_mgr.default_idle_unload_sec}.")

    if "chat_history_retriever" in update_data:
        chr_updates = CONFIG.get('chat_history_retriever', {})
        if "recall_budget_tokens" in chr_updates and chat_history_retriever.recall_budget_tokens != chr_updates["recall_budget_tokens"]:
            chat_history_retriever.recall_budget_tokens = chr_updates["recall_budget_tokens"]
            applied_notes.append(f"ChatHistoryRetriever recall_budget_tokens updated to {chat_history_retriever.recall_budget_tokens}.")
        if "embedding_model_name" in chr_updates or "vector_db_path" in chr_updates:
            restart_notes.append("ChatHistoryRetriever 'embedding_model_name' or 'vector_db_path' changes require a restart or re-initialization to take full effect on the running instance.")

    if "pdf_retriever" in update_data:
        pr_updates = CONFIG.get('pdf_retriever', {})
        if "chunk_size" in pr_updates and pdf_retriever.chunk_size != pr_updates["chunk_size"]:
            pdf_retriever.chunk_size = pr_updates["chunk_size"]
            applied_notes.append(f"PdfRetriever chunk_size updated to {pdf_retriever.chunk_size}.")
        if "chunk_overlap" in pr_updates and pdf_retriever.chunk_overlap != pr_updates["chunk_overlap"]:
            pdf_retriever.chunk_overlap = pr_updates["chunk_overlap"]
            applied_notes.append(f"PdfRetriever chunk_overlap updated to {pdf_retriever.chunk_overlap}.")
        if "embedding_model_name" in pr_updates or "vector_db_path" in pr_updates:
            restart_notes.append("PdfRetriever 'embedding_model_name' or 'vector_db_path' changes require a restart or re-initialization to take full effect on the running instance.")

    if "token_estimator_for_rag_budgeting" in update_data:
        restart_notes.append("Changes to 'token_estimator_for_rag_budgeting' require an application restart to affect ContextManager, ChatHistoryRetriever, and PdfRetriever.")

    if "generation_defaults" in update_data:
        applied_notes.append("Default generation parameters updated. New chat sessions will use these defaults if not overridden in the request.")

    message = "Settings updated."
    if applied_notes:
        message += " Applied live: " + "; ".join(applied_notes)
    if restart_notes:
        message += " Require restart/re-init for full effect: " + "; ".join(list(set(restart_notes)))

    return StatusResponse(status="ok", message=message)

@app.post("/models/download", response_model=StatusResponse)
async def download_model_endpoint(req: DownloadModelRequest):
    """
    Initiates a model download from a Hugging Face repository.

    The endpoint uses the `hf_downloader` utility to fetch models or specific files.
    Models are saved into a subdirectory (named after the repository) within the
    globally configured `model_download_dir` (see `config.yaml`).

    The download process is synchronous in the current implementation.

    Request Body (`DownloadModelRequest`):
    - **repo_id** (str, required): The Hugging Face repository ID.
      Example: `"TheBloke/Mistral-7B-Instruct-v0.1-GGUF"`.
    - **filename** (str, optional): The specific file to download from the repository.
      If provided, only this file will be downloaded. This is typically used for
      GGUF models or specific configuration files. If `None`, the entire repository
      (a "snapshot") will be downloaded, respecting ignore/allow patterns from `config.yaml`.
      Example: `"mistral-7b-instruct-v0.1.Q4_K_M.gguf"`.
    - **model_type** (str, optional): Expected model type (e.g., "gguf", "safetensors", "awq").
      This field is primarily for organizational purposes or future validation. It can
      also inform the `repo_type` parameter for `huggingface_hub` functions if needed,
      though `hf_repo_type` in `config.yaml` or auto-detection often handles this.
    - **revision** (str, optional): The specific model revision to download.
      This can be a branch name (e.g., `"main"`), a tag (e.g., `"v1.0.0"`),
      or a commit hash. If `None`, the default revision (usually "main") is used.
      Example: `"gguf-Q4_K_M"`.
    - **target_path** (str, optional): Not currently used by this endpoint as downloads
      are always placed within the configured `model_download_dir`. Reserved for future use.


    Successful Response (HTTP 200):
    - `StatusResponse` with `status: "ok"` and a message indicating successful
      download, including the path to the downloaded model file or repository snapshot.
      Example:
      ```json
      {
        "status": "ok",
        "message": "Download for 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF' (File: mistral-7b-instruct-v0.1.Q4_K_M.gguf, Revision: main) completed. Saved to: /path/to/models/downloaded/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
      }
      ```

    Error Responses (HTTP 200 with `status: "error"` in body, or HTTP 500 for server errors):
    - If the download fails due to issues like file not found, repository not found,
      network errors, or permissions, a `StatusResponse` with `status: "error"`
      and a descriptive message from the downloader is returned.
      Example:
      ```json
      {
        "status": "error",
        "message": "File 'nonexistent.gguf' not found in repo 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF' (revision: main)."
      }
      ```
    - If there's an internal server error during the process (e.g., cannot create
      base download directory), an HTTP 500 error might be raised directly by FastAPI.
    """
    print(f"Received download request: repo_id='{req.repo_id}', filename='{req.filename}', revision='{req.revision}', model_type='{req.model_type}'")

    # Retrieve configurations
    configured_download_dir_str = CONFIG.get('model_download_dir', 'models/downloaded/') # Relative to project root or absolute
    hf_auth_token = CONFIG.get('hf_token') # Optional, for private repos
    ignore_patterns = CONFIG.get('hf_snapshot_ignore_patterns') # For snapshot downloads
    allow_patterns = CONFIG.get('hf_snapshot_allow_patterns') # For snapshot downloads
    default_hf_repo_type = CONFIG.get('hf_repo_type') # General repo type from config (e.g., "model")

    # Determine target directory for the downloader
    # The hf_downloader will create a model-specific subdirectory inside this.
    base_download_dir = Path(configured_download_dir_str)
    if not base_download_dir.is_absolute():
        base_download_dir = PROJECT_ROOT / base_download_dir

    # Ensure the base directory exists; hf_downloader handles its own subdirectories.
    try:
        base_download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Base download directory ensured at: {base_download_dir.resolve()}")
    except Exception as e:
        print(f"Error creating base download directory {base_download_dir}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error creating base download directory: {str(e)}")

    # Determine repo_type for hf_hub_download.
    # If req.model_type is provided and is a valid HF repo_type (e.g., "model", "dataset", "space"), it could be used.
    # For now, we'll prefer a specific hf_repo_type from config if available, else None (downloader default).
    # A more sophisticated mapping from req.model_type to hf repo_type might be needed if they differ.
    hf_repo_type_to_use = default_hf_repo_type # Could also be informed by req.model_type if desired

    print(f"Calling download_model_from_hf with target_dir: {base_download_dir}")
    success, result_path_or_msg = download_model_from_hf(
        repo_id=req.repo_id,
        target_dir=base_download_dir, # hf_downloader creates repo_id subdir here
        filename=req.filename,
        hf_token=hf_auth_token,
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        repo_type=hf_repo_type_to_use, # Pass the determined repo_type
        revision=req.revision # Pass revision from request
    )

    if success:
        return StatusResponse(
            status="ok",
            message=f"Download for '{req.repo_id}' (File: {req.filename or 'snapshot'}, Revision: {req.revision or 'main'}) completed. Saved to: {result_path_or_msg}"
        )
    else:
        # Log the error server-side as hf_downloader already prints.
        # Determine if it's a client error (4xx) or server error (5xx) based on message
        error_msg_str = str(result_path_or_msg)
        status_code = 400 # Default to client error
        if "not found" in error_msg_str.lower() or "invalid" in error_msg_str.lower():
            status_code = 404 # Or 400 for bad request
        elif "http error" in error_msg_str.lower() or "network" in error_msg_str.lower():
            status_code = 502 # Bad Gateway / upstream error
        elif "unexpected error" in error_msg_str.lower() or "permission" in error_msg_str.lower():
            status_code = 500 # Internal server error

        # For this implementation, we'll return a StatusResponse with error message.
        # Raising HTTPException is also a good option for more RESTful error handling.
        # Example: raise HTTPException(status_code=status_code, detail=error_msg_str)
        return StatusResponse(status="error", message=error_msg_str)

@app.get("/models/available", response_model=ModelListResponse)
async def get_available_models_endpoint():
    """
    Scans configured local directories for models and returns a list of available models.
    """
    model_discovery_config = CONFIG.get('model_discovery', {}) # Ensure this key exists in CONFIG/DEFAULT_CONFIG
    scan_paths_from_config = model_discovery_config.get('scan_directories', []) # Use the correct key based on config

    absolute_scan_paths = []
    if scan_paths_from_config:
        for p_str in scan_paths_from_config:
            path_obj = Path(p_str)
            if not path_obj.is_absolute():
                path_obj = PROJECT_ROOT / p_str

            if path_obj.exists() and path_obj.is_dir():
                absolute_scan_paths.append(str(path_obj.resolve()))
            else:
                print(f"Warning: Configured scan directory does not exist or is not a directory: {path_obj}")

    if not absolute_scan_paths:
        print("No valid scan directories configured or found.")
        return ModelListResponse(models=[])

    try:
        # Assuming model_scanner is imported correctly
        from yawl.utils import model_scanner # Ensure this import is at the top
        found_models = model_scanner.scan_model_directories(absolute_scan_paths)
        return ModelListResponse(models=found_models)
    except Exception as e:
        print(f"Error during model scanning: {e}")
        # import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to scan model directories: {str(e)}")


@app.get("/tools", response_model=ToolListResponse)
async def list_tools_endpoint():
    """
    Lists all available tools (local and potentially MCP) and their current states.
    """
    tools = tool_dispatcher.list_tools()
    return ToolListResponse(tools=tools)

@app.get("/tools/openai_schemas", response_model=t.List[t.Dict[str, t.Any]])
async def get_openai_tool_schemas_endpoint():
    """
    Returns a list of OpenAI-compatible tool schemas for currently enabled local tools.

    This endpoint allows applications to discover the available tools and their
    expected parameters in a format compatible with OpenAI's function calling feature.
    The schemas can be used to inform an LLM about available functions.
    """
    try:
        schemas = tool_dispatcher.get_openai_tool_schemas()
        return schemas
    except Exception as e:
        print(f"Error generating OpenAI tool schemas: {e}")
        # import traceback; traceback.print_exc() # For debugging
        raise HTTPException(status_code=500, detail=f"Failed to generate OpenAI tool schemas: {str(e)}")

@app.post("/tools/toggle", response_model=StatusResponse)
async def toggle_tool_endpoint(req: ToggleToolRequest):
    """
    Enables or disables a specified tool.
    """
    tool_name = req.tool_name
    action_taken_msg = ""
    success = False

    if req.enable:
        success = tool_dispatcher.enable_tool(tool_name)
        action_taken_msg = f"Tool '{tool_name}' enabled."
    else:
        success = tool_dispatcher.disable_tool(tool_name)
        action_taken_msg = f"Tool '{tool_name}' disabled."

    # The current enable/disable_tool always returns True as it just updates state.
    # A more robust check might involve seeing if tool_name is in list_tools().
    # For now, we assume success in setting the state.
    if success: # This will always be true with current implementation
        # Check if the tool is known (i.e., was listed) to provide more context.
        # This requires calling list_tools() again, or modifying enable/disable to return more info.
        # For simplicity, we'll keep the message generic.
        # If the tool was not previously known (e.g. a new MCP tool name being toggled),
        # its state is now recorded.
        return StatusResponse(status="ok", message=action_taken_msg)
    else:
        # This path is not reachable with current enable/disable_tool logic.
        return StatusResponse(status="error", message=f"Failed to toggle tool '{tool_name}'. It might not be a known tool type that can be toggled.")


@app.post("/load_model", response_model=StatusResponse)
async def load_model_endpoint(req: LoadModelRequest):
    # ... (load_model_endpoint remains the same as previous version)
    print(f"Received /load_model request: {req.model_dump()}")
    try:
        runner_params = req.runner_params if req.runner_params is not None else {}
        idle_unload_sec = runner_params.pop('idle_unload_sec', None)
        default_prefix_text = runner_params.pop('default_prefix_text', None) # If passed here

        model_mgr.load(
            model_type=req.model_type,
            model_path_or_name=req.model_path_or_name,
            idle_unload_sec=idle_unload_sec,
            default_prefix_text=default_prefix_text, # Pass it on
            **runner_params
        )
        if model_mgr.current_runner:
            return StatusResponse(status="ok", message=f"Model '{model_mgr.current_model_identifier}' loaded successfully.")
        else:
            return StatusResponse(status="error", message=f"Failed to load model '{req.model_path_or_name}'. Runner not available.")
    except Exception as e:
        print(f"Error in /load_model endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during model load: {str(e)}")

@app.post("/upload_document", response_model=UploadPdfResponse)
async def upload_document_api(file: UploadFile = File(...)):
    """
    Uploads a PDF document for RAG processing.
    The document is temporarily saved, processed by PdfRetriever, then deleted.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = Path(temp_dir) / file.filename

        print(f"[API /upload_document] Saving uploaded file to temporary path: {temp_pdf_path}")
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[API /upload_document] Processing document with PdfRetriever: {temp_pdf_path}")
        success, msg, doc_id, num_chunks = pdf_retriever.upload_document(str(temp_pdf_path))

        if success:
            return UploadPdfResponse(
                doc_id=doc_id,
                filename=file.filename,
                message=msg,
                num_chunks_processed=num_chunks,
                status="success"
            )
        else:
            # PdfRetriever.upload_document already prints error, so just return it
            raise HTTPException(status_code=400, detail=msg) # Or 500 if server-side processing issue

    except Exception as e:
        print(f"Error in /upload_document endpoint: {e}")
        # import traceback; traceback.print_exc(); # For detailed debugging
        raise HTTPException(status_code=500, detail=f"Internal server error during file upload: {str(e)}")
    finally:
        if file: # Ensure file object is closed
            await file.close()
        if temp_dir and Path(temp_dir).exists():
            print(f"[API /upload_document] Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Handles chat requests, supporting both synchronous and streaming responses,
    RAG capabilities, and OpenAI-style tool calls.

    If the loaded LLM supports tool calling and identifies a need to use a tool,
    this endpoint will:
    1. Receive the LLM's intent to call tools (as `tool_calls` in the conceptual response).
    2. Execute the requested tools using the `ToolDispatcher`.
    3. Send the tool results back to the LLM.
    4. The LLM then generates a final textual response based on the tool outputs.

    **Tool Calling Notes:**
    - The system primarily uses an OpenAI-compatible `tool_calls` mechanism.
    - The older `[FUNCALL]` string-based method is being phased out but might still exist in some parts of the ToolDispatcher for backward compatibility.
      For new integrations, relying on OpenAI-style tool calls is recommended.

    **Streaming SSE Events for Tool Calls:**
    If `stream=True` and tool calls occur, the following Server-Sent Events (SSE) may be emitted in sequence:
    - `event: prompt_info` (for the initial user prompt)
    - `event: tool_calls_processing` (contains the `tool_calls` objects requested by the LLM)
    - `event: tool_result` (one for each tool call, contains `tool_call_id`, `name`, and `result` or `error`)
    - `event: prompt_info` (for the prompt sent to LLM after tool results are incorporated)
    - `data: {...}` (standard text chunks for the LLM's final response)
    - `event: stream_end` (final summary)
    """
    print(f"Received /chat request: {req.model_dump(exclude={'generation_params', 'stream'})}, Stream: {req.stream}") # Log stream state

    runner = model_mgr.get()
    if not runner:
        raise HTTPException(status_code=400, detail="No model is currently loaded.")

    # --- Shared logic for context preparation and initial prompt building ---
    def _prepare_context_and_initial_prompt(
        current_req: ChatRequest,
        current_ctx_mgr: ContextManager,
        current_chat_history_retriever: ChatHistoryRetriever,
        current_pdf_retriever: PdfRetriever,
        runner_for_counting: Optional[BaseRunner] # Pass runner for token counting
    ) -> tuple[str, int]: # Return prompt_text and its token count
        first_image_path: Optional[str] = None
        if current_req.image_paths and len(current_req.image_paths) > 0:
            first_image_path = current_req.image_paths[0]

        # Add user message to context manager first
        current_ctx_mgr.add(role="user", content=current_req.message, image_path=first_image_path)
        # Then add to chat history retriever
        # (Let CHR generate its own ID, or pass one if available/needed)
        try:
            chat_history_retriever.add_message(message_text=current_req.message, role="user")
        except Exception as e_chr_add:
            print(f"Warning: Failed to add user message to ChatHistoryRetriever: {e_chr_add}")

        retrieved_snippets = []
        if current_req.use_rag and not current_req.pdf_doc_ids_for_rag:
            print('[API /chat] Chat History RAG enabled. Retrieving snippets...')
            history_for_rag = current_ctx_mgr._messages[:-1] if len(current_ctx_mgr._messages) > 1 else []

            # HyDE for Chat History
            hyde_query_embedding_chat = None
            # Check instance attribute on the CHR, which was set from its specific config section
            if chat_history_retriever.enable_hyde:
                hyde_config = CONFIG.get('hyde', {}) # Get global HyDE settings
                # hyde_llm_id = hyde_config.get('llm_identifier', 'default') # Currently only default runner is used
                hyde_prompt_template = hyde_config.get('prompt_template')
                hyde_max_tokens = hyde_config.get('max_tokens_hyde_doc', 128)

                if hyde_prompt_template:
                    hyde_prompt = hyde_prompt_template.format(query=current_req.message)
                    # Use the main loaded runner for HyDE generation.
                    # A dedicated, possibly smaller/faster model for HyDE is a future enhancement.
                    hyde_runner = runner # Use the already fetched main runner for the chat endpoint
                    if hyde_runner:
                        try:
                            print(f"[API HyDE Chat] Generating hypothetical doc for query: '{current_req.message[:50]}...'")
                            # Assuming runner.generate returns a dict with 'text' or a tuple (text, usage_dict)
                            hypothetical_doc_response_obj = hyde_runner.generate(prompt=hyde_prompt, max_new_tokens=hyde_max_tokens)

                            hypothetical_doc_text = None
                            if isinstance(hypothetical_doc_response_obj, dict) and "text" in hypothetical_doc_response_obj:
                                hypothetical_doc_text = hypothetical_doc_response_obj["text"]
                            elif isinstance(hypothetical_doc_response_obj, tuple) and len(hypothetical_doc_response_obj) > 0 and isinstance(hypothetical_doc_response_obj[0], str):
                                hypothetical_doc_text = hypothetical_doc_response_obj[0]
                            elif isinstance(hypothetical_doc_response_obj, str): # Direct string response
                                 hypothetical_doc_text = hypothetical_doc_response_obj

                            if hypothetical_doc_text and chat_history_retriever.embedding_model:
                                hyde_query_embedding_chat = chat_history_retriever.embedding_model.encode(hypothetical_doc_text).tolist()
                                print(f"[API HyDE Chat] Generated & embedded hypothetical doc (len {len(hypothetical_doc_text)}): '{hypothetical_doc_text[:100]}...'")
                            elif not hypothetical_doc_text:
                                print("[API HyDE Chat] Warning: HyDE LLM generated empty or invalid document.")
                        except Exception as e_hyde:
                            print(f"[API HyDE Chat] Error during HyDE document generation: {e_hyde}")

            chat_snippets = current_chat_history_retriever.retrieve(
                query_text=current_req.message,
                query_embedding_override=hyde_query_embedding_chat, # Pass potential HyDE embedding
                current_chat_history=history_for_rag
            )
            retrieved_snippets.extend(chat_snippets)
            print(f'[API /chat] Retrieved {len(chat_snippets)} CHAT snippets.')

        if current_req.use_rag and current_req.pdf_doc_ids_for_rag:
            print(f'[API /chat] PDF RAG enabled for doc IDs: {current_req.pdf_doc_ids_for_rag}. Retrieving snippets...')

            # HyDE for PDF
            hyde_query_embedding_pdf = None
            if pdf_retriever.enable_hyde: # Check instance attribute
                hyde_config = CONFIG.get('hyde', {}) # Global HyDE settings
                # hyde_llm_id = hyde_config.get('llm_identifier', 'default')
                hyde_prompt_template = hyde_config.get('prompt_template')
                hyde_max_tokens = hyde_config.get('max_tokens_hyde_doc', 128)

                if hyde_prompt_template:
                    hyde_prompt = hyde_prompt_template.format(query=current_req.message)
                    hyde_runner = runner # Use the main chat runner
                    if hyde_runner:
                        try:
                            print(f"[API HyDE PDF] Generating hypothetical doc for query: '{current_req.message[:50]}...'")
                            hypothetical_doc_response_obj = hyde_runner.generate(prompt=hyde_prompt, max_new_tokens=hyde_max_tokens)

                            hypothetical_doc_text = None
                            if isinstance(hypothetical_doc_response_obj, dict) and "text" in hypothetical_doc_response_obj:
                                hypothetical_doc_text = hypothetical_doc_response_obj["text"]
                            elif isinstance(hypothetical_doc_response_obj, tuple) and len(hypothetical_doc_response_obj) > 0 and isinstance(hypothetical_doc_response_obj[0], str):
                                hypothetical_doc_text = hypothetical_doc_response_obj[0]
                            elif isinstance(hypothetical_doc_response_obj, str):
                                 hypothetical_doc_text = hypothetical_doc_response_obj

                            if hypothetical_doc_text and pdf_retriever.embedding_model:
                                hyde_query_embedding_pdf = pdf_retriever.embedding_model.encode(hypothetical_doc_text).tolist()
                                print(f"[API HyDE PDF] Generated & embedded hypothetical doc (len {len(hypothetical_doc_text)}): '{hypothetical_doc_text[:100]}...'")
                            elif not hypothetical_doc_text:
                                print("[API HyDE PDF] Warning: HyDE LLM generated empty or invalid document.")
                        except Exception as e_hyde_pdf:
                             print(f"[API HyDE PDF] Error during HyDE document generation: {e_hyde_pdf}")

            pdf_snippets = current_pdf_retriever.retrieve_from_pdf(
                query_text=current_req.message,
                query_embedding_override=hyde_query_embedding_pdf, # Pass potential HyDE embedding
                doc_ids=current_req.pdf_doc_ids_for_rag
            )
            retrieved_snippets.extend(pdf_snippets)
            print(f'[API /chat] Retrieved {len(pdf_snippets)} PDF snippets.')

        try:
            prompt_text = current_ctx_mgr.build_prompt(extra_messages=retrieved_snippets)
            print(f"Built prompt for model (len {len(prompt_text)} chars):\n{prompt_text[:500]}...")

            prompt_token_count = 0
            if runner_for_counting and hasattr(runner_for_counting, 'count_tokens'): # Check if method exists
                try:
                    count = runner_for_counting.count_tokens(prompt_text)
                    prompt_token_count = count if count is not None else 0
                except Exception as e_count:
                    print(f"Warning: Could not count tokens for prompt: {e_count}")
            return prompt_text, prompt_token_count
        except Exception as e:
            print(f"Error building prompt: {e}")
            raise HTTPException(status_code=500, detail="Error building prompt.")

    # --- Shared logic for handling tool calls ---
    def _handle_tool_call(
        tool_call_str: str,
        current_ctx_mgr: ContextManager,
        current_tool_dispatcher: ToolDispatcher,
        runner_for_counting: Optional[BaseRunner] # Pass runner for token counting
    ) -> tuple[str, str, str, int]: # Returns (second_prompt_text, tool_name, tool_params_json_str, second_prompt_token_count)
        print(f'[API /chat] Detected function call: {tool_call_str}')
        function_call_json_str = tool_call_str.replace('[FUNCALL]', '').strip()

        tool_name_for_event = "unknown_tool"
        try:
            tool_params = json.loads(function_call_json_str)
            tool_name_for_event = tool_params.get("tool_name", "unknown_tool")
        except json.JSONDecodeError:
            tool_name_for_event = "unknown_tool_json_decode_error"

        tool_result_dict = current_tool_dispatcher.dispatch(function_call_json_str)
        print(f'[API /chat] Tool dispatch result: {tool_result_dict}')
        tool_message_content = json.dumps(tool_result_dict)
        current_ctx_mgr.add(role='tool_result', content=tool_message_content)

        print('[API /chat] Added tool result to context. Building second prompt...')
        second_prompt_text = current_ctx_mgr.build_prompt()
        print(f"Built second prompt for model (len {len(second_prompt_text)} chars):\n{second_prompt_text[:500]}...")

        second_prompt_token_count = 0
        if runner_for_counting and hasattr(runner_for_counting, 'count_tokens'):
            try:
                count = runner_for_counting.count_tokens(second_prompt_text)
                second_prompt_token_count = count if count is not None else 0
            except Exception as e_count:
                print(f"Warning: Could not count tokens for second prompt: {e_count}")
        return second_prompt_text, tool_name_for_event, function_call_json_str, second_prompt_token_count


    # --- Streaming Response Logic ---
    if req.stream:
        async def sse_generator() -> AsyncGenerator[str, None]:
            final_prompt_tokens_for_response = 0
            accumulated_completion_tokens = 0
            full_assistant_reply_for_history: List[str] = [] # Stores text parts of assistant's reply
            active_tool_calls_info: List[Dict[str, Any]] = [] # Stores OpenAI tool_call objects

            initial_prompt_data_for_runner: Any # Could be str or List[Dict] depending on runner
            initial_prompt_token_count_calc = 0

            try:
                # _prepare_context_and_initial_prompt now returns a list of messages if ctx_mgr.build_prompt does.
                # For now, we assume runner.stream expects a string for the first call,
                # so _prepare_context_and_initial_prompt needs to return that or be adapted.
                # Let's assume _prepare_context_and_initial_prompt still returns (str, int) for now.
                # This is a temporary inconsistency that runner adaptation will resolve.
                # For this subtask, we focus on sse_generator logic given this assumption.

                # TODO: Resolve if initial_prompt_text should be str or List[Dict] based on runner capabilities
                # For now, assuming _prepare_context_and_initial_prompt provides a string for the first call.
                # If it provides List[Dict], runner.stream() needs to handle `messages=` for first call too.
                initial_prompt_text_str, initial_prompt_token_count_calc = _prepare_context_and_initial_prompt(
                    req, ctx_mgr, chat_history_retriever, pdf_retriever, runner
                )
                initial_prompt_data_for_runner = initial_prompt_text_str # Assuming string for now

                final_prompt_tokens_for_response = initial_prompt_token_count_calc
                yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': initial_prompt_token_count_calc, 'context': 'initial_prompt'})}\n\n"

            except HTTPException as e:
                error_content = json.dumps({"error": e.detail, "status_code": e.status_code})
                yield f"event: error\ndata: {error_content}\n\n"
                return
            except Exception as e_prep: # Catch other prep errors
                print(f"Error in _prepare_context_and_initial_prompt or initial yield: {e_prep}")
                error_content = json.dumps({"error": str(e_prep), "status_code": 500})
                yield f"event: error\ndata: {error_content}\n\n"
                return

            # Main execution block for streaming and tool calls
            try:
                # First LLM stream
                # Assuming runner.stream can take `prompt` (string) or `messages` (list)
                # Based on current _prepare_context_and_initial_prompt, it's a string.
                stream_iterator_initial = runner.stream(
                    prompt=initial_prompt_data_for_runner, # type: ignore
                    image_paths=req.image_paths,
                    **req.generation_params.model_dump()
                )

                for item in stream_iterator_initial:
                    if isinstance(item, dict):
                        if "prompt_tokens" in item:
                            actual_prompt_tokens_from_runner = item["prompt_tokens"]
                            if final_prompt_tokens_for_response != actual_prompt_tokens_from_runner:
                                print(f"Info: Runner's initial stream prompt token count ({actual_prompt_tokens_from_runner}) differs from pre-calculated ({final_prompt_tokens_for_response}). Using runner's value.")
                                final_prompt_tokens_for_response = actual_prompt_tokens_from_runner
                                yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': final_prompt_tokens_for_response, 'context': 'runner_provided_initial'})}\n\n"

                        elif "tool_calls" in item: # Runner yields a complete tool_calls list
                            openai_tool_calls = item["tool_calls"]
                            if openai_tool_calls: # Ensure not empty
                                print(f"[API sse_generator] Received tool_calls: {openai_tool_calls}")
                                # Add assistant's turn with tool_calls to context
                                ctx_mgr.add(role="assistant", content=None, tool_calls=openai_tool_calls)
                                # Store for history and processing
                                # The assistant's message in history will be one with tool_calls, not text.
                                full_assistant_reply_for_history.append(json.dumps({"tool_calls": openai_tool_calls})) # Store as JSON string for history
                                active_tool_calls_info = openai_tool_calls
                                break # Exit first stream loop to process tool calls
                        # Potentially handle other dict types like usage stats if yielded separately

                    elif isinstance(item, tuple) and len(item) == 2: # (text_chunk, tokens_in_chunk)
                        text_chunk, tokens_in_chunk = item
                        if text_chunk: # Ensure text_chunk is not empty
                            full_assistant_reply_for_history.append(text_chunk)
                            accumulated_completion_tokens += tokens_in_chunk if tokens_in_chunk is not None else 0
                            yield f"data: {json.dumps({'text': text_chunk, 'tokens_in_chunk': tokens_in_chunk})}\n\n"
                            await asyncio.sleep(0.01) # Small sleep for stream flushing
                    else:
                        print(f"Warning: Unknown item type from initial stream: {item}")

                # Tool Execution Phase (if active_tool_calls_info is populated)
                if active_tool_calls_info:
                    yield f"event: tool_calls_processing\ndata: {json.dumps(active_tool_calls_info)}\n\n"

                    # Clear previous assistant reply (which was the tool_calls request)
                    # The next LLM call will generate the actual textual response.
                    full_assistant_reply_for_history.clear()
                    accumulated_completion_tokens = 0 # Reset for the generation after tool calls

                    for tool_call in active_tool_calls_info:
                        tool_call_id = tool_call.get("id")
                        tool_function = tool_call.get("function", {})
                        tool_name = tool_function.get("name")
                        arguments_json_str = tool_function.get("arguments")

                        if not tool_call_id or not tool_name or arguments_json_str is None:
                            print(f"Warning: Malformed tool_call object in sse_generator: {tool_call}")
                            tool_content_str = "Error: Malformed tool_call object from LLM."
                            # Add to context even if malformed, so LLM knows an error occurred for this ID
                            ctx_mgr.add(role="tool", content=tool_content_str, name=tool_name or "error_tool", tool_call_id=tool_call_id or "unknown_malformed_id")
                            yield f"event: tool_result\ndata: {json.dumps({'tool_call_id': tool_call_id, 'name': tool_name, 'error': tool_content_str})}\n\n"
                            continue

                        try:
                            print(f"[API sse_generator] Dispatching OpenAI tool: ID='{tool_call_id}', Name='{tool_name}', Args='{arguments_json_str}'")
                            dispatch_result = tool_dispatcher.dispatch_openai_tool_call(tool_name, arguments_json_str)

                            tool_content_str = ""
                            if dispatch_result.get("status") == "success":
                                tool_content_str = str(dispatch_result.get("result", ""))
                            else:
                                tool_content_str = str(dispatch_result.get("error", "Tool execution failed."))

                            ctx_mgr.add(role="tool", content=tool_content_str, name=tool_name, tool_call_id=tool_call_id)
                            yield f"event: tool_result\ndata: {json.dumps({'tool_call_id': tool_call_id, 'name': tool_name, 'result': tool_content_str})}\n\n"
                        except Exception as e_dispatch:
                            print(f"Error dispatching tool '{tool_name}': {e_dispatch}")
                            error_content_str = f"Error executing tool {tool_name}: {str(e_dispatch)}"
                            ctx_mgr.add(role="tool", content=error_content_str, name=tool_name, tool_call_id=tool_call_id)
                            yield f"event: tool_result\ndata: {json.dumps({'tool_call_id': tool_call_id, 'name': tool_name, 'error': error_content_str})}\n\n"

                    # Second LLM Stream (after tool calls)
                    # ctx_mgr.build_prompt() now returns List[Dict[str, Any]]
                    prompt_messages_after_tools: List[Dict[str, Any]] = ctx_mgr.build_prompt()

                    # Assuming runner.stream can now take `messages` argument
                    stream_iterator_second = runner.stream(
                        messages=prompt_messages_after_tools, # type: ignore
                        image_paths=req.image_paths, # Pass images again if needed
                        **req.generation_params.model_dump()
                    )

                    for item in stream_iterator_second:
                        if isinstance(item, dict):
                            if "prompt_tokens" in item: # Runner reports prompt tokens for the second call
                                second_call_prompt_tokens = item["prompt_tokens"]
                                if final_prompt_tokens_for_response != second_call_prompt_tokens:
                                    # This usually means the runner has a different way of counting the full conversation
                                    print(f"Info: Runner's second stream prompt token count ({second_call_prompt_tokens}) used. Initial was ({final_prompt_tokens_for_response}).")
                                    final_prompt_tokens_for_response = second_call_prompt_tokens
                                yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': final_prompt_tokens_for_response, 'context': 'runner_provided_after_tools'})}\n\n"
                            # Recursive tool calls not handled in this iteration.

                        elif isinstance(item, tuple) and len(item) == 2: # (text_chunk, tokens_in_chunk)
                            text_chunk, tokens_in_chunk = item
                            if text_chunk:
                                full_assistant_reply_for_history.append(text_chunk)
                                accumulated_completion_tokens += tokens_in_chunk if tokens_in_chunk is not None else 0
                                yield f"data: {json.dumps({'text': text_chunk, 'tokens_in_chunk': tokens_in_chunk})}\n\n"
                                await asyncio.sleep(0.01)
                        else:
                             print(f"Warning: Unknown item type from second stream: {item}")

                # If no tool calls, the first stream completed, and full_assistant_reply_for_history contains its output.

            except Exception as e_stream_main:
                print(f"Error during main streaming logic: {e_stream_main}")
                # import traceback; traceback.print_exc() # For debugging
                error_content = json.dumps({"error": str(e_stream_main), "status_code": 500})
                yield f"event: error\ndata: {error_content}\n\n"

            finally: # This block executes whether there was an exception or not
                if full_assistant_reply_for_history:
                    # Determine if the reply was a tool request or text
                    # If active_tool_calls_info is populated, the last "message" from assistant was tool_calls.
                    # If not, it was text.
                    final_message_content_for_history = "".join(full_assistant_reply_for_history)
                    if active_tool_calls_info and not any(isinstance(chunk, str) and chunk.strip() for chunk in full_assistant_reply_for_history):
                        # If tool calls happened AND the final text content is empty/whitespace,
                        # it implies the "reply" was the tool calls themselves.
                        # The full_assistant_reply_for_history might contain the JSON representation of tool_calls if we stored it earlier.
                        # The ContextManager already added the assistant's tool_calls object.
                        # ChatHistoryRetriever needs the textual summary or representation.
                        # For now, if tools were called, the last textual output from the LLM is what we save.
                        # If that's empty, we might save a placeholder or the stringified tool_calls.
                        pass # The content is already in full_assistant_reply_for_history.

                    try:
                        # If the last thing the assistant did was call tools, full_assistant_reply_for_history
                        # would be the *textual response after tools*. If it only called tools and then said nothing,
                        # it would be empty here.
                        # If it only outputted text (no tools), that text is here.
                        if final_message_content_for_history: # Only add if there's actual text content
                             chat_history_retriever.add_message(message_text=final_message_content_for_history, role="assistant")
                        elif active_tool_calls_info: # Assistant made tool calls but produced no further text
                             # Add a representation of the tool call action to history if desired
                             # For now, we assume ctx_mgr.add(role="assistant", content=None, tool_calls=...) is sufficient
                             # and CHR doesn't need a separate entry for the tool calling itself beyond the subsequent tool results.
                             # Or, we could add the JSON string of tool_calls that was in full_assistant_reply_for_history before clearing.
                             pass
                    except Exception as e_chr_add_assist:
                        print(f"Warning: Failed to add assistant's streamed reply to ChatHistoryRetriever: {e_chr_add_assist}")

                final_token_summary = {
                    "message": "Stream ended.",
                    "final_prompt_tokens": final_prompt_tokens_for_response,
                    "total_generated_tokens": accumulated_completion_tokens
                }
                yield f"event: stream_end\ndata: {json.dumps(final_token_summary)}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # --- Synchronous (Non-Streaming) Response Logic ---
    else:
        start_time = time.time()
        final_reply_text: Optional[str] = None # Initialize to None or empty string
        prompt_tokens_count = 0
        generated_tokens_count = 0

        try:
            prompt_for_model, initial_prompt_tokens_estimate = _prepare_context_and_initial_prompt(
                req, ctx_mgr, chat_history_retriever, pdf_retriever, runner
            )
            # This initial_prompt_tokens_estimate is from our calculation.
            # We will use the actuals from the runner's response.

            # First LLM call
            llm_response_initial: Dict[str, Any] = runner.generate(
                prompt=prompt_for_model,
                image_paths=req.image_paths,
                **req.generation_params.model_dump()
            )

            actual_prompt_tokens = llm_response_initial.get("usage", {}).get("prompt_tokens", 0)
            initial_completion_tokens = llm_response_initial.get("usage", {}).get("completion_tokens", 0)

            prompt_tokens_count = actual_prompt_tokens
            generated_tokens_count = initial_completion_tokens
            final_reply_text = llm_response_initial.get("text") # This might be None if tool_calls are present

            openai_tool_calls = llm_response_initial.get("tool_calls")

            if openai_tool_calls:
                print(f"[API /chat] Detected OpenAI tool calls: {openai_tool_calls}")
                # Add the LLM's message that included tool_calls to context
                # This assumes the LLM's message (even if it's just tool_calls) should be part of history.
                # The exact content for role='assistant' when tool_calls are present might need refinement.
                # For now, we add a placeholder or skip if final_reply_text was None.
                # If final_reply_text was not None (e.g. text + tool_call), it's already set.
                # If only tool_calls, the assistant "message" is the call itself.
                # This part of context manager interaction might need adjustment based on how
                # OpenAI compatible models structure this. For now, we assume ctx_mgr can handle
                # assistant messages that are primarily tool_calls.
                # Let's assume for now, the assistant's "turn" that requests tool calls is implicitly added
                # to context by the act of then adding the "tool" role messages.
                # Or, if the model also returns text along with tool_calls, that text is the assistant's message.
                # For now, we don't add a specific assistant message here if final_reply_text is None.
                # The `ctx_mgr.add(role="assistant", tool_calls=openai_tool_calls)` would be ideal if supported.

                for tool_call in openai_tool_calls:
                    tool_call_id = tool_call.get("id")
                    tool_function = tool_call.get("function", {})
                    tool_name = tool_function.get("name")
                    arguments_json_str = tool_function.get("arguments")

                    if not tool_call_id or not tool_name or arguments_json_str is None:
                        print(f"Warning: Malformed tool_call object: {tool_call}")
                        # Potentially add an error message to context for this tool_call_id
                        ctx_mgr.add(
                            role="tool",
                            content="Error: Malformed tool_call object from LLM.",
                            name="error_tool", # Placeholder name
                            tool_call_id=tool_call_id or "unknown_id"
                        )
                        continue

                    print(f"[API /chat] Dispatching OpenAI tool: ID='{tool_call_id}', Name='{tool_name}', Args='{arguments_json_str}'")
                    dispatch_result = tool_dispatcher.dispatch_openai_tool_call(tool_name, arguments_json_str)

                    tool_content_str = ""
                    if dispatch_result.get("status") == "success":
                        tool_content_str = str(dispatch_result.get("result", ""))
                    else:
                        tool_content_str = str(dispatch_result.get("error", "Tool execution failed."))

                    print(f"[API /chat] Tool '{tool_name}' result (ID: {tool_call_id}): {tool_content_str[:200]}")
                    ctx_mgr.add(role="tool", content=tool_content_str, name=tool_name, tool_call_id=tool_call_id)

                # All tool calls dispatched and results added to context. Now make the second LLM call.
                prompt_after_tools = ctx_mgr.build_prompt()
                print(f"[API /chat] Built second prompt after tool calls (len {len(prompt_after_tools)} chars):\n{prompt_after_tools[:500]}...")

                llm_response_second: Dict[str, Any] = runner.generate(
                    prompt=prompt_after_tools,
                    image_paths=req.image_paths, # Pass images again if needed, though less common for tool-result processing
                    **req.generation_params.model_dump()
                )

                final_reply_text = llm_response_second.get("text", "") # Get the final text response

                # Token accounting for the second call
                second_prompt_tokens = llm_response_second.get("usage", {}).get("prompt_tokens", 0)
                second_completion_tokens = llm_response_second.get("usage", {}).get("completion_tokens", 0)

                # How to sum up tokens?
                # Option 1: OpenAI style - prompt_tokens for second call includes everything.
                # So, total_prompt_tokens = second_prompt_tokens. Total_completion_tokens = initial_completion_tokens + second_completion_tokens.
                prompt_tokens_count = second_prompt_tokens # This should reflect the full conversation up to that point.
                generated_tokens_count += second_completion_tokens # Add to initial completion tokens.

                print(f"[API /chat] Final reply after OpenAI tool calls: {final_reply_text[:100]}...")

            # Ensure final_reply_text is not None if it was never set (e.g. first call only had tool_calls and second call somehow returned no text)
            if final_reply_text is None:
                final_reply_text = "" # Default to empty string if no text was generated.

            # Add final assistant reply to chat history retriever
            if final_reply_text:
                try:
                    chat_history_retriever.add_message(message_text=final_reply_text, role="assistant")
                except Exception as e_chr_add_assist_sync:
                    print(f"Warning: Failed to add assistant's sync reply to ChatHistoryRetriever: {e_chr_add_assist_sync}")

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error during model generation or tool call: {e}")
            # import traceback; traceback.print_exc(); # For debugging
            raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return ChatResponse(
            reply=final_reply_text, # Ensure final_reply_text is a string
            request_details=req,
            generated_tokens=generated_tokens_count,
            prompt_tokens=prompt_tokens_count,
            latency_ms=round(latency_ms, 2)
        )


if __name__ == "__main__":
    print("Starting Uvicorn server for YAWL API...")
    # ... (Uvicorn run command) ...

    # Updated curl examples:
    # 1. Upload a PDF document:
    # curl -X POST http://localhost:8000/upload_document -F "file=@/path/to/your/document.pdf"
    #
    # (Assuming a model like LLaVA or a GGUF model is loaded via /load_model)
    #
    # 2. Chat with PDF RAG:
    # curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "Summarize the key findings from document 'your_document_stem'",
    #   "use_rag": true,
    #   "pdf_doc_ids_for_rag": ["your_document_stem"]
    # }'
    #
    # 3. Chat with PDF RAG and an image (if using a multimodal model like LLaVA):
    # curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "Based on this image and the document 'reportX', what is the main conclusion?",
    #   "image_paths": ["/path/on/server/to/image.jpg"],
    #   "use_rag": true,
    #   "pdf_doc_ids_for_rag": ["reportX"]
    # }'
    #
    # 4. Chat with streaming response:
    # curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "Tell me a very long story about a brave knight who fights a dragon.",
    #   "stream": true,
    #   "generation_params": {"max_new_tokens": 300}
    # }'
    #
    # 5. Chat with streaming, RAG, and a tool call (e.g., if a 'get_weather' tool is available):
    # curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "What is the weather in London and then tell me a short poem about it?",
    #   "stream": true,
    #   "use_rag": false
    # }'
    # (This assumes the LLM is prompted or fine-tuned to use a get_weather tool when appropriate)
    #
    # 6. Get current settings:
    # curl -X GET http://localhost:8000/settings
    #
    # 7. Update some settings:
    # curl -X PUT http://localhost:8000/settings -H "Content-Type: application/json" -d \
    # '{
    #   "context_manager": {"max_tokens": 3000, "system_prompt": "You are a concise assistant."},
    #   "model_manager": {"default_idle_unload_sec": 600},
    #   "chat_history_retriever": {"recall_budget_tokens": 256}
    # }'
    #
    # Check settings again after update:
    # curl -X GET http://localhost:8000/settings
    #
    # 8. List available models (after configuring scan_directories and placing models)
    # curl -X GET http://localhost:8000/models/available
    # Remember to create dummy files like ./models/gguf_files/dummy.gguf
    # or dirs like ./models/awq_model_dirs/my_awq_model_dir for the scan to find anything based on default config.
    #
    # 9. List available tools:
    # curl -X GET http://localhost:8000/tools
    #
    # 10. Disable a tool (e.g., get_weather):
    # curl -X POST http://localhost:8000/tools/toggle -H "Content-Type: application/json" -d '{"tool_name": "get_weather", "enable": false}'
    #
    # 11. Attempt to use the disabled tool via /chat (should result in an error or different behavior):
    # curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "What is the weather in London? [FUNCALL] {\"name\": \"get_weather\", \"arguments\": {\"location\": \"London\"}}", "stream": false}'
    # (Note: The above curl for chat with FUNCALL is conceptual; normally the LLM generates the FUNCALL string.)
    #
    # 12. Re-enable the tool:
    # curl -X POST http://localhost:8000/tools/toggle -H "Content-Type: application/json" -d '{"tool_name": "get_weather", "enable": true}'
    #
    # 13. Request a model download (placeholder):
    # curl -X POST http://localhost:8000/models/download -H "Content-Type: application/json" -d \
    # '{
    #   "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    #   "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #   "model_type": "gguf"
    # }'


    uvicorn.run(app, host="0.0.0.0", port=8000)
