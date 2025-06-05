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
from llm_context_os.api.schemas import (
    LoadModelRequest,
    ChatRequest,
    StatusResponse,
    ChatResponse,
    UploadPdfResponse,
    GenerationParams
)
from llm_context_os.context.context_manager import ContextManager, MockTokenizer # Ensure correct import path
from llm_context_os.runners.manager import ModelManager
from llm_context_os.runners.base import BaseRunner
from llm_context_os.retriever.chat_history import ChatHistoryRetriever
from llm_context_os.retriever.pdf_retriever import PdfRetriever # Added
from llm_context_os.tools.tool_dispatcher import ToolDispatcher

# --- Application Setup ---
app = FastAPI(
    title="LLM Context OS API",
    description="API for managing local LLM models, context, RAG, and generation.",
    version="0.1.1" # Incremented version
)

# --- Global Instances ---
model_mgr = ModelManager()
mock_tokenizer_for_ctx = MockTokenizer()
ctx_mgr = ContextManager(
    system_prompt="You are a helpful AI assistant.",
    tokenizer=mock_tokenizer_for_ctx,
    max_tokens=1024
)
chat_history_retriever = ChatHistoryRetriever() # Renamed for clarity
pdf_retriever = PdfRetriever() # Added
tool_dispatcher = ToolDispatcher()

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
        current_chat_history_retriever: ChatHistoryRetriever,
        current_pdf_retriever: PdfRetriever
    ) -> str:
        first_image_path: Optional[str] = None
        if current_req.image_paths and len(current_req.image_paths) > 0:
            first_image_path = current_req.image_paths[0]
            # Log image path if needed

        current_ctx_mgr.add(role="user", content=current_req.message, image_path=first_image_path)

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
        current_ctx_mgr: ContextManager,
        current_tool_dispatcher: ToolDispatcher
    ) -> str: # Returns the second prompt
        print(f'[API /chat] Detected function call: {tool_call_str}')
        function_call_json_str = tool_call_str.replace('[FUNCALL]', '').strip()

        # Attempt to parse the JSON string early to catch errors
        try:
            tool_params = json.loads(function_call_json_str) # To extract tool name for event
            tool_name_for_event = tool_params.get("tool_name", "unknown_tool")
        except json.JSONDecodeError:
            tool_name_for_event = "unknown_tool_json_decode_error"
            # Let tool_dispatcher handle the potentially malformed string if it can/wants to

        # Yield tool call event (for streaming) - this helper is sync, so can't yield directly.
        # The streaming generator will yield this.

        tool_result_dict = current_tool_dispatcher.dispatch(function_call_json_str)
        print(f'[API /chat] Tool dispatch result: {tool_result_dict}')
        tool_message_content = json.dumps(tool_result_dict) # Convert result back to JSON string for context
        current_ctx_mgr.add(role='tool_result', content=tool_message_content)
        print('[API /chat] Added tool result to context. Building second prompt...')

        second_prompt_text = current_ctx_mgr.build_prompt()
        print(f"Built second prompt for model (len {len(second_prompt_text)} chars):\n{second_prompt_text[:500]}...")
        return second_prompt_text, tool_name_for_event, function_call_json_str


    # --- Streaming Response Logic ---
    if req.stream:
        async def sse_generator() -> AsyncGenerator[str, None]:
            prompt_for_model = ""
            try:
                prompt_for_model = _prepare_context_and_initial_prompt(req, ctx_mgr, chat_history_retriever, pdf_retriever)
            except HTTPException as e: # Catch HTTPException from prompt building
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

                    second_prompt, tool_name, tool_params_str = _handle_tool_call(full_func_call_command, ctx_mgr, tool_dispatcher)
                    yield f"event: tool_call\ndata: {json.dumps({'tool_name': tool_name, 'tool_params': json.loads(tool_params_str) if tool_params_str else None})}\n\n"

                    # Stream the final response after tool call
                    for final_chunk in runner.stream(prompt=second_prompt, **req.generation_params.model_dump()):
                        yield f"data: {json.dumps({'text': final_chunk})}\n\n"
                        await asyncio.sleep(0.01)

                # If loop finished and no func_call_str_detected, it means all initial chunks were yielded.
                # (Handled by the initial loop's yield)

            except Exception as e_stream:
                print(f"Error during streaming: {e_stream}")
                # import traceback; traceback.print_exc() # For debugging
                error_content = json.dumps({"error": str(e_stream)})
                yield f"event: error\ndata: {error_content}\n\n"
            finally:
                yield f"event: stream_end\ndata: {json.dumps({'message': 'Stream ended.'})}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # --- Synchronous (Non-Streaming) Response Logic ---
    else:
        start_time = time.time()
        final_reply_text = ""

        try:
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
