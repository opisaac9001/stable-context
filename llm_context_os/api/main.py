# llm_context_os/api/main.py
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
from llm_context_os.api.schemas import (
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
from llm_context_os.context.context_manager import ContextManager, MockTokenizer
from llm_context_os.context.token_estimator import TikTokenEstimator, HFTokenEstimator # For RAG tokenizer
from llm_context_os.runners.manager import ModelManager
from llm_context_os.runners.base import BaseRunner
from llm_context_os.retriever.chat_history import ChatHistoryRetriever
from llm_context_os.retriever.pdf_retriever import PdfRetriever
from llm_context_os.tools.tool_dispatcher import ToolDispatcher

# --- Application Setup ---
app = FastAPI(
    title="LLM Context OS API",
    description="API for managing local LLM models, context, RAG, and generation.",
    version="0.1.2" # Incremented version for config changes
)

# --- Configuration Loading ---
DEFAULT_CONFIG = {
    "context_manager": {"system_prompt": "You are a helpful AI assistant.", "max_tokens": 4096},
    "chat_history_retriever": {
        "recall_budget_tokens": 512,
        "embedding_model_name": "all-MiniLM-L6-v2",
        "vector_db_path": "data/vector_dbs/api_default_chat_history"
    },
    "pdf_retriever": {
        "vector_db_path": "data/vector_dbs/api_default_pdf_rag",
        "embedding_model_name": "all-MiniLM-L6-v2",
        "chunk_size": 500,
        "chunk_overlap": 50
    },
    "model_manager": {"default_idle_unload_sec": 900},
    "token_estimator_for_rag_budgeting": {"type": "tiktoken", "model_name": "cl100k_base"},
    "model_discovery": { # Added model_discovery default
        "scan_directories": [] # Default to no scan directories
    }
}
CONFIG = DEFAULT_CONFIG.copy() # Start with defaults
try:
    # Assuming config.yaml is in llm_context_os/config/config.yaml relative to project root
    # For robustness, resolve path from this file's location.
    # __file__ is llm_context_os/api/main.py
    # So, parent is api/, parent.parent is llm_context_os/
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
    recall_budget_tokens=chr_config['recall_budget_tokens'],
    tokenizer=rag_tokenizer,
    vector_db_path=chr_config['vector_db_path'],
    embedding_model_name=chr_config['embedding_model_name']
)
print(f"ChatHistoryRetriever initialized with: budget={chr_config['recall_budget_tokens']}, db_path='{chr_config['vector_db_path']}', model='{chr_config['embedding_model_name']}'")

# PDF Retriever
pdf_retriever_config = CONFIG.get('pdf_retriever', DEFAULT_CONFIG['pdf_retriever'])
pdf_retriever = PdfRetriever(
    vector_db_path=pdf_retriever_config['vector_db_path'],
    embedding_model_name=pdf_retriever_config['embedding_model_name'],
    tokenizer=rag_tokenizer, # Pass tokenizer for potential future use (e.g. chunk size by tokens)
    chunk_size=pdf_retriever_config['chunk_size'],
    chunk_overlap=pdf_retriever_config['chunk_overlap']
)
print(f"PdfRetriever initialized with: db_path='{pdf_retriever_config['vector_db_path']}', model='{pdf_retriever_config['embedding_model_name']}'")


# Model Manager
model_mgr_config = CONFIG.get('model_manager', DEFAULT_CONFIG['model_manager'])
model_mgr = ModelManager(default_idle_unload_sec=model_mgr_config['default_idle_unload_sec'])
print(f"ModelManager initialized with default_idle_unload_sec={model_mgr_config['default_idle_unload_sec']}")

# Tool Dispatcher (currently no config needed from file for its __init__)
tool_dispatcher = ToolDispatcher()
print("ToolDispatcher initialized.")

# --- Project Root for resolving relative paths in config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # llm_context_os directory
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
    (Placeholder) Initiates a model download from a Hugging Face repository.
    Actual download logic to be implemented later.
    """
    print(f"Received download request for model repo: {req.repo_id}, filename: {req.filename or 'all files (repo)'}")
    print(f"  Requested model type: {req.model_type or 'any'}")
    print(f"  Requested target path: {req.target_path or 'default location'}")

    # Placeholder: In a real implementation, this would trigger an async download task
    # from huggingface_hub import hf_hub_download
    # For now, just acknowledge the request.

    # Determine a default target path if not provided, e.g., based on model_type and repo_id
    # For example: models/<model_type>/<repo_id_user>/<repo_id_name>
    # Ensure this path is within allowed configurable base model directories.

    return StatusResponse(
        status="ok",
        message=f"Download request for '{req.repo_id}' (file: {req.filename or 'all files'}) received. "
                f"Simulated download initiated. Model would appear at a predefined location based on target_path or defaults."
    )

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
        from llm_context_os.utils import model_scanner # Ensure this import is at the top
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
            chat_snippets = current_chat_history_retriever.retrieve(query_text=current_req.message, current_chat_history=history_for_rag)
            retrieved_snippets.extend(chat_snippets)
            print(f'[API /chat] Retrieved {len(chat_snippets)} CHAT snippets.')

        if current_req.use_rag and current_req.pdf_doc_ids_for_rag:
            print(f'[API /chat] PDF RAG enabled for doc IDs: {current_req.pdf_doc_ids_for_rag}. Retrieving snippets...')
            pdf_snippets = current_pdf_retriever.retrieve_from_pdf(
                query_text=current_req.message,
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
            full_assistant_reply_for_history = []
            accumulated_completion_tokens = 0
            final_prompt_tokens_for_response = 0

            initial_prompt_text = ""
            try:
                initial_prompt_text, initial_prompt_tokens_calc = _prepare_context_and_initial_prompt(
                    req, ctx_mgr, chat_history_retriever, pdf_retriever, runner
                )
                final_prompt_tokens_for_response = initial_prompt_tokens_calc
                yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': initial_prompt_tokens_calc, 'context': 'initial_prompt'})}\n\n"
            except HTTPException as e:
                error_content = json.dumps({"error": e.detail, "status_code": e.status_code})
                yield f"event: error\ndata: {error_content}\n\n"
                return

            initial_response_buffer = []
            func_call_str_detected = None

            try:
                # First LLM call (streaming)
                stream_iterator_initial = runner.stream(prompt=initial_prompt_text, image_paths=req.image_paths, **req.generation_params.model_dump())

                # Process prompt_tokens info from the runner's stream
                first_item_initial_stream = next(stream_iterator_initial, None)
                if isinstance(first_item_initial_stream, dict) and "prompt_tokens" in first_item_initial_stream:
                    if final_prompt_tokens_for_response != first_item_initial_stream["prompt_tokens"]:
                        print(f"Info: Runner's initial stream prompt token count ({first_item_initial_stream['prompt_tokens']}) differs from pre-calculated ({final_prompt_tokens_for_response}). Using runner's value.")
                        final_prompt_tokens_for_response = first_item_initial_stream["prompt_tokens"]
                        # Optionally re-yield if it changed significantly
                        yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': final_prompt_tokens_for_response, 'context': 'runner_provided_initial'})}\n\n"
                elif first_item_initial_stream: # First item was not dict, assume it's a (chunk, count) tuple
                    text_chunk, tokens_in_chunk = first_item_initial_stream
                    initial_response_buffer.append(text_chunk) # Buffer for FUNCALL detection
                    # Check for FUNCALL immediately, even in the first chunk
                    current_buffered_text_check = "".join(initial_response_buffer)
                    if "[FUNCALL]" in current_buffered_text_check and (current_buffered_text_check.endswith("}") or len(current_buffered_text_check) > 2048): # Heuristic for completion
                        func_call_str_detected = current_buffered_text_check
                    else: # Not a FUNCALL or potentially incomplete
                        full_assistant_reply_for_history.append(text_chunk)
                        accumulated_completion_tokens += tokens_in_chunk if tokens_in_chunk is not None else 0
                        yield f"data: {json.dumps({'text': text_chunk, 'tokens_in_chunk': tokens_in_chunk})}\n\n"
                        initial_response_buffer.clear() # Clear since this part is yielded
                        await asyncio.sleep(0.01)

                if not func_call_str_detected: # Continue with the rest of the stream if no FUNCALL yet
                    for item in stream_iterator_initial:
                        text_chunk, tokens_in_chunk = item
                        initial_response_buffer.append(text_chunk)
                        current_buffered_text_check = "".join(initial_response_buffer)
                        if "[FUNCALL]" in current_buffered_text_check:
                            if not current_buffered_text_check.endswith("}") and len(current_buffered_text_check) <= 2048 : continue
                            func_call_str_detected = current_buffered_text_check; break

                        full_assistant_reply_for_history.append(text_chunk)
                        accumulated_completion_tokens += tokens_in_chunk if tokens_in_chunk is not None else 0
                        yield f"data: {json.dumps({'text': text_chunk, 'tokens_in_chunk': tokens_in_chunk})}\n\n"
                        initial_response_buffer.clear()
                        await asyncio.sleep(0.01)

                if func_call_str_detected:
                    # Extract FUNCALL and any preceding text from the buffer
                    # The buffer might contain parts already yielded if FUNCALL was detected late.
                    # This part needs careful management to avoid duplicate text yields.
                    # For simplicity, assume func_call_str_detected contains the full relevant string.
                    preceding_text_in_buffer, _, actual_func_call_payload = func_call_str_detected.partition("[FUNCALL]")
                    if preceding_text_in_buffer and "".join(full_assistant_reply_for_history) != preceding_text_in_buffer:
                        # This implies some part of preceding_text_in_buffer was not yielded.
                        # This logic is complex; the sse_generator in previous steps was simpler.
                        # For now, we assume if preceding_text_in_buffer exists, it's new.
                        # This is a known simplification area.
                        # TODO: Refine handling of text preceding FUNCALL when it spans multiple chunks.
                        # For now, we focus on the tool call itself.
                         pass # Avoid yielding potentially duplicated preceding_text from buffer.

                    full_func_call_command = f"[FUNCALL]{actual_func_call_payload}"
                    second_prompt, tool_name, tool_params_json_str, second_prompt_tokens_calc = _handle_tool_call(
                        full_func_call_command, ctx_mgr, tool_dispatcher, runner
                    )
                    final_prompt_tokens_for_response = second_prompt_tokens_calc

                    parsed_tool_params = {};
                    try: parsed_tool_params = json.loads(tool_params_json_str)
                    except: pass
                    yield f"event: tool_call\ndata: {json.dumps({'tool_name': tool_name, 'tool_params': parsed_tool_params})}\n\n"
                    yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': second_prompt_tokens_calc, 'context': 'after_tool_call'})}\n\n"

                    full_assistant_reply_for_history.clear()
                    accumulated_completion_tokens = 0

                    stream_iterator_after_tool = runner.stream(prompt=second_prompt, **req.generation_params.model_dump())
                    second_stream_prompt_info = next(stream_iterator_after_tool, None)

                    if isinstance(second_stream_prompt_info, dict) and "prompt_tokens" in second_stream_prompt_info:
                        if final_prompt_tokens_for_response != second_stream_prompt_info["prompt_tokens"]:
                             final_prompt_tokens_for_response = second_stream_prompt_info["prompt_tokens"]
                             yield f"event: prompt_info\ndata: {json.dumps({'prompt_tokens': final_prompt_tokens_for_response, 'context': 'after_tool_call_runner_provided'})}\n\n"
                    elif second_stream_prompt_info:
                        text_chunk, tokens_in_chunk = second_stream_prompt_info
                        full_assistant_reply_for_history.append(text_chunk)
                        accumulated_completion_tokens += tokens_in_chunk if tokens_in_chunk is not None else 0
                        yield f"data: {json.dumps({'text': text_chunk, 'tokens_in_chunk': tokens_in_chunk})}\n\n"

                    for final_chunk, tokens_in_chunk_final in stream_iterator_after_tool:
                        full_assistant_reply_for_history.append(final_chunk)
                        accumulated_completion_tokens += tokens_in_chunk_final if tokens_in_chunk_final is not None else 0
                        yield f"data: {json.dumps({'text': final_chunk, 'tokens_in_chunk': tokens_in_chunk_final})}\n\n"
                        await asyncio.sleep(0.01)
                # else: No tool call, all chunks (if any after first) were yielded.

            except Exception as e_stream:
                print(f"Error during streaming: {e_stream}")
                error_content = json.dumps({"error": str(e_stream)})
                yield f"event: error\ndata: {error_content}\n\n"
            finally:
                if full_assistant_reply_for_history:
                    try:
                        chat_history_retriever.add_message(message_text="".join(full_assistant_reply_for_history), role="assistant")
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
        final_reply_text = ""
        prompt_tokens_count = 0
        generated_tokens_count = 0

        try:
            prompt_for_model, initial_prompt_tokens = _prepare_context_and_initial_prompt(
                req, ctx_mgr, chat_history_retriever, pdf_retriever, runner
            )
            prompt_tokens_count = initial_prompt_tokens # Initial prompt tokens

            reply_text_tuple = runner.generate( # runner.generate now returns a tuple
                prompt=prompt_for_model,
                image_paths=req.image_paths,
                **req.generation_params.model_dump()
            )
            final_reply_text = reply_text
            prompt_tokens_count = token_counts1.get("prompt_tokens", 0)
            generated_tokens_count = token_counts1.get("completion_tokens", 0)

            if reply_text.startswith('[FUNCALL]'):
                second_prompt, _, _, _ = _handle_tool_call(reply_text, ctx_mgr, tool_dispatcher)
                # For sync, the second call's tokens overwrite the first for simplicity of reporting one set.
                # Or, one could sum them if that's more meaningful. Here, we report tokens for the final text-generating call.
                final_reply_text, token_counts2 = runner.generate(second_prompt, **req.generation_params.model_dump())
                prompt_tokens_count = token_counts2.get("prompt_tokens", 0) # Update with second prompt's tokens
                generated_tokens_count = token_counts2.get("completion_tokens", 0) # Update with second call's completion
                print(f"[API /chat] Final reply after tool call: {final_reply_text[:100]}...")

            if final_reply_text:
                try:
                    chat_history_retriever.add_message(message_text=final_reply_text, role="assistant")
                except Exception as e_chr_add_assist_sync:
                    print(f"Warning: Failed to add assistant's sync reply to ChatHistoryRetriever: {e_chr_add_assist_sync}")

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error during model generation or tool call: {e}")
            raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return ChatResponse(
            reply=final_reply_text,
            request_details=req,
            generated_tokens=generated_tokens_count,
            prompt_tokens=prompt_tokens_count,
            latency_ms=round(latency_ms, 2)
        )


if __name__ == "__main__":
    print("Starting Uvicorn server for LLM Context OS API...")
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
