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
    ChatResponse,
    UploadPdfResponse,
    GenerationParams
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
    "token_estimator_for_rag_budgeting": {"type": "tiktoken", "model_name": "cl100k_base"}
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

# --- API Endpoints ---

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
        current_chat_history_retriever: ChatHistoryRetriever, # Now global, but passed for explicitness
        current_pdf_retriever: PdfRetriever         # Now global, but passed for explicitness
    ) -> str:
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
            return prompt_text
        except Exception as e:
            print(f"Error building prompt: {e}")
            raise HTTPException(status_code=500, detail="Error building prompt.")

    # --- Shared logic for handling tool calls ---
    def _handle_tool_call(
        tool_call_str: str, # The raw "[FUNCALL]..." string
        current_ctx_mgr: ContextManager,         # Now global, but passed for explicitness
        current_tool_dispatcher: ToolDispatcher  # Now global, but passed for explicitness
    ) -> tuple[str, str, str]: # Returns (second_prompt_text, tool_name, tool_params_json_str)
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
        # Also add tool result to chat history retriever? This could be noisy.
        # For now, only user/assistant turns are added to CHR. This can be reviewed.
        # e.g., if tool_result_dict.get("status") == "success":
        #    chat_history_retriever.add_message(message_text=f"Tool {tool_name_for_event} result: {tool_result_dict.get('result')}", role="tool_result_summary")

        print('[API /chat] Added tool result to context. Building second prompt...')
        second_prompt_text = current_ctx_mgr.build_prompt()
        print(f"Built second prompt for model (len {len(second_prompt_text)} chars):\n{second_prompt_text[:500]}...")
        return second_prompt_text, tool_name_for_event, function_call_json_str


    # --- Streaming Response Logic ---
    if req.stream:
        async def sse_generator() -> AsyncGenerator[str, None]:
            full_assistant_reply_for_history = [] # Accumulate chunks for CHR
            prompt_for_model = ""
            try:
                # Use global instances directly in the generator context
                prompt_for_model = _prepare_context_and_initial_prompt(req, ctx_mgr, chat_history_retriever, pdf_retriever)
            except HTTPException as e:
                error_content = json.dumps({"error": e.detail, "status_code": e.status_code})
                yield f"event: error\ndata: {error_content}\n\n"
                return

            initial_response_buffer = []
            func_call_str_detected = None

            try:
                # Iterate over the stream from the runner
                # Note: runner.stream is a synchronous generator.
                # FastAPI can handle yielding from sync generators in async routes,
                # but for long-running generations, this will block the event loop thread.
                # For truly non-blocking behavior, runner.stream would need to be async
                # or run in a thread pool (e.g., using asyncio.to_thread).
                # For this implementation, we'll assume simple iteration is acceptable for now.
                for chunk in runner.stream(prompt=prompt_for_model, image_paths=req.image_paths, **req.generation_params.model_dump()):
                    if func_call_str_detected: # If we broke from loop due to funcall, but more chunks came
                        print(f"Warning: Received chunk '{chunk}' after FUNCALL was detected and buffer processed. Ignoring.")
                        continue # Or handle as part of the funcall string if it can be multi-chunk (complex)

                    initial_response_buffer.append(chunk)
                    current_buffered_text = "".join(initial_response_buffer)

                    # Simple check for FUNCALL. More robust parsing might be needed if it can span chunks.
                    if "[FUNCALL]" in current_buffered_text:
                        # Try to find the full FUNCALL block. This assumes it's not too large.
                        # A more robust solution might involve a timeout or max buffer size for FUNCALL detection.
                        if not current_buffered_text.endswith("}"): # Simple check if JSON might be incomplete
                             # Wait for more chunks if it looks like an incomplete JSON
                             if len(current_buffered_text) > 2048: # Safety break for too long buffer
                                 print("Warning: FUNCALL detected but buffer is very long and might be incomplete. Processing as is.")
                                 func_call_str_detected = current_buffered_text # Process potentially truncated
                                 break
                             else:
                                 continue # Accumulate more chunks for potentially complete JSON

                        func_call_str_detected = current_buffered_text
                        break # Stop accumulating initial response, proceed to tool call

                    # If no FUNCALL detected yet, yield the chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.01) # Small sleep to allow other tasks, if any

                if func_call_str_detected:
                    # We have a function call
                    # The part of the FUNCALL string before "[FUNCALL]" is considered preceding text.
                    preceding_text, _, actual_func_call_payload = func_call_str_detected.partition("[FUNCALL]")
                    if preceding_text: # Yield any text that came before the [FUNCALL] marker
                        yield f"data: {json.dumps({'text': preceding_text})}\n\n"

                    # Reconstruct the "[FUNCALL]..." string for the handler
                    full_func_call_command = f"[FUNCALL]{actual_func_call_payload}"

                    # Use global instances here too
                    second_prompt, tool_name, tool_params_json_str = _handle_tool_call(full_func_call_command, ctx_mgr, tool_dispatcher)
                    parsed_tool_params = {}
                    try: parsed_tool_params = json.loads(tool_params_json_str)
                    except: pass # Keep it empty if not valid JSON for some reason

                    yield f"event: tool_call\ndata: {json.dumps({'tool_name': tool_name, 'tool_params': parsed_tool_params})}\n\n"

                    # Stream the final response after tool call
                    for final_chunk in runner.stream(prompt=second_prompt, **req.generation_params.model_dump()):
                        full_assistant_reply_for_history.append(final_chunk)
                        yield f"data: {json.dumps({'text': final_chunk})}\n\n"
                        await asyncio.sleep(0.01)
                else: # No tool call, initial stream is the final response
                    full_assistant_reply_for_history = initial_response_buffer # Already contains all chunks

            except Exception as e_stream:
                print(f"Error during streaming: {e_stream}")
                error_content = json.dumps({"error": str(e_stream)})
                yield f"event: error\ndata: {error_content}\n\n"
            finally:
                # Add accumulated assistant reply to history AFTER stream is fully processed
                if full_assistant_reply_for_history:
                    try:
                        chat_history_retriever.add_message(message_text="".join(full_assistant_reply_for_history), role="assistant")
                    except Exception as e_chr_add_assist:
                        print(f"Warning: Failed to add assistant's streamed reply to ChatHistoryRetriever: {e_chr_add_assist}")

                yield f"event: stream_end\ndata: {json.dumps({'message': 'Stream ended.'})}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # --- Synchronous (Non-Streaming) Response Logic ---
    else:
        start_time = time.time()
        final_reply_text = "" # Ensure it's always defined

        try:
            # Use global instances
            prompt_for_model = _prepare_context_and_initial_prompt(req, ctx_mgr, chat_history_retriever, pdf_retriever)

            reply_text = runner.generate(
                prompt=prompt_for_model,
                image_paths=req.image_paths,
                **req.generation_params.model_dump()
            )
            final_reply_text = reply_text

            if reply_text.startswith('[FUNCALL]'):
                second_prompt, _, _ = _handle_tool_call(reply_text, ctx_mgr, tool_dispatcher)
                final_reply_text = runner.generate(second_prompt, **req.generation_params.model_dump())
                print(f"[API /chat] Final reply after tool call: {final_reply_text[:100]}...")

            # Add final assistant reply to chat history retriever
            if final_reply_text: # Ensure there is a reply to add
                try:
                    chat_history_retriever.add_message(message_text=final_reply_text, role="assistant")
                except Exception as e_chr_add_assist_sync:
                    print(f"Warning: Failed to add assistant's sync reply to ChatHistoryRetriever: {e_chr_add_assist_sync}")

        except HTTPException: # Re-raise HTTPExceptions from helpers
            raise
        except Exception as e:
            print(f"Error during model generation or tool call: {e}")
            # import traceback; traceback.print_exc() # For debugging
            raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return ChatResponse(
            reply=final_reply_text,
            request_details=req, # req now includes 'stream' field, which is fine
            tokens_generated=None, # Placeholder, could be filled if runner returns count
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


    uvicorn.run(app, host="0.0.0.0", port=8000)
