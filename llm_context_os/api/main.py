# llm_context_os/api/main.py
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile # Added File, UploadFile
import time
import json
from typing import List, Optional
from pathlib import Path # Added
import tempfile # Added
import shutil # Added

# App-specific imports
from llm_context_os.api.schemas import (
    LoadModelRequest,
    ChatRequest,
    StatusResponse,
    ChatResponse,
    UploadPdfResponse, # Added
    GenerationParams
)
from llm_context_os.context.context_manager import ContextManager, MockTokenizer
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
    print(f"Received /chat request: {req.model_dump(exclude={'generation_params'})}")
    # ... (logging as before) ...

    start_time = time.time()
    final_reply_text = ""

    first_image_path: Optional[str] = None
    if req.image_paths and len(req.image_paths) > 0:
        first_image_path = req.image_paths[0]
        # ... (logging as before) ...

    ctx_mgr.add(role="user", content=req.message, image_path=first_image_path)

    retrieved_snippets = []
    # RAG from Chat History (conceptual, assuming chat_history_retriever is used if use_rag is general)
    if req.use_rag and not req.pdf_doc_ids_for_rag: # Example: only use chat history RAG if no PDF RAG
        print('[API /chat] Chat History RAG enabled. Retrieving snippets...')
        history_for_rag = ctx_mgr._messages[:-1] if len(ctx_mgr._messages) > 1 else []
        # Assuming chat_history_retriever exists and has a similar interface
        chat_snippets = chat_history_retriever.retrieve(query_text=req.message, current_chat_history=history_for_rag)
        retrieved_snippets.extend(chat_snippets)
        print(f'[API /chat] Retrieved {len(chat_snippets)} CHAT snippets.')

    # RAG from PDF documents
    if req.use_rag and req.pdf_doc_ids_for_rag:
        print(f'[API /chat] PDF RAG enabled for doc IDs: {req.pdf_doc_ids_for_rag}. Retrieving snippets...')
        pdf_snippets = pdf_retriever.retrieve_from_pdf(
            query_text=req.message,
            doc_ids=req.pdf_doc_ids_for_rag
            # top_k can be passed from req or use pdf_retriever's default
        )
        retrieved_snippets.extend(pdf_snippets) # Append PDF snippets
        print(f'[API /chat] Retrieved {len(pdf_snippets)} PDF snippets.')
        # Optional: Add logic here to de-duplicate or prioritize if both chat and PDF snippets exist

    try:
        prompt_for_model = ctx_mgr.build_prompt(extra_messages=retrieved_snippets)
    except Exception as e:
        print(f"Error building prompt: {e}")
        raise HTTPException(status_code=500, detail="Error building prompt.")

    print(f"Built prompt for model (len {len(prompt_for_model)} chars):\n{prompt_for_model[:500]}...")

    runner = model_mgr.get()
    if not runner:
        raise HTTPException(status_code=400, detail="No model is currently loaded.")

    try:
        reply_text = runner.generate(
            prompt=prompt_for_model,
            image_paths=req.image_paths,
            **req.generation_params.model_dump()
        )
        final_reply_text = reply_text

        if reply_text.startswith('[FUNCALL]'):
            # ... (tool calling logic remains the same as previous version) ...
            print(f'[API /chat] Detected function call: {reply_text}')
            function_call_json = reply_text.replace('[FUNCALL]', '').strip()
            tool_result_dict = tool_dispatcher.dispatch(function_call_json)
            print(f'[API /chat] Tool dispatch result: {tool_result_dict}')
            tool_message_content = json.dumps(tool_result_dict)
            ctx_mgr.add(role='tool_result', content=tool_message_content)
            print('[API /chat] Added tool result to context. Re-generating...')
            second_prompt = ctx_mgr.build_prompt()
            print(f"Built second prompt for model (len {len(second_prompt)} chars):\n{second_prompt[:500]}...")
            final_reply_text = runner.generate(second_prompt, **req.generation_params.model_dump())
            print(f"[API /chat] Final reply after tool call: {final_reply_text[:100]}...")

    except Exception as e:
        print(f"Error during model generation or tool call: {e}")
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    return ChatResponse(
        reply=final_reply_text,
        request_details=req,
        tokens_generated=None,
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
