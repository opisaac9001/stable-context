# llm_context_os/api/main.py
import uvicorn
from fastapi import FastAPI, HTTPException
import time
import json
from typing import List, Optional # Added for type hints

from llm_context_os.api.schemas import (
    LoadModelRequest,
    ChatRequest,
    StatusResponse,
    ChatResponse,
    GenerationParams
)
from llm_context_os.context.context_manager import ContextManager, MockTokenizer
from llm_context_os.runners.manager import ModelManager
from llm_context_os.runners.base import BaseRunner
from llm_context_os.retriever.chat_history import ChatHistoryRetriever
from llm_context_os.tools.tool_dispatcher import ToolDispatcher

# --- Application Setup ---
app = FastAPI(
    title="LLM Context OS API",
    description="API for managing local LLM models, context, and generation.",
    version="0.1.0"
)

# --- Global Instances ---
model_mgr = ModelManager()
mock_tokenizer_for_ctx = MockTokenizer()
ctx_mgr = ContextManager(
    system_prompt="You are a helpful AI assistant.",
    tokenizer=mock_tokenizer_for_ctx,
    max_tokens=1024
)
retriever = ChatHistoryRetriever()
tool_dispatcher = ToolDispatcher()

# --- API Endpoints ---

@app.post("/load_model", response_model=StatusResponse)
async def load_model_endpoint(req: LoadModelRequest):
    print(f"Received /load_model request: {req.model_dump()}")
    try:
        runner_params = req.runner_params if req.runner_params is not None else {}
        # Extract idle_unload_sec from runner_params if present, as load() expects it as a direct arg
        idle_unload_sec = runner_params.pop('idle_unload_sec', None)

        model_mgr.load(
            model_type=req.model_type,
            model_path_or_name=req.model_path_or_name,
            idle_unload_sec=idle_unload_sec,
            **runner_params
        )
        if model_mgr.current_runner:
            return StatusResponse(status="ok", message=f"Model '{model_mgr.current_model_identifier}' loaded successfully.")
        else:
            # Attempt to get a more specific error from logs if model_mgr.load failed silently to set runner
            # For now, a generic message based on current_runner being None
            return StatusResponse(status="error", message=f"Failed to load model '{req.model_path_or_name}'. Runner not available.")
    except Exception as e:
        print(f"Error in /load_model endpoint: {e}")
        # Consider logging the traceback for detailed debugging
        # import traceback; traceback.print_exc();
        raise HTTPException(status_code=500, detail=f"Internal server error during model load: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"Received /chat request: {req.model_dump(exclude={'generation_params'})}")
    if req.image_paths:
        print(f"  Image paths received: {req.image_paths}")
    print(f"Generation params for chat: {req.generation_params.model_dump()}")

    start_time = time.time()
    final_reply_text = ""

    # 1. Add user message (and potentially first image) to context
    # For simplicity, ContextManager.add takes only one image_path.
    # A more advanced ContextManager could handle a list or structured image content.
    first_image_path: Optional[str] = None
    if req.image_paths and len(req.image_paths) > 0:
        first_image_path = req.image_paths[0]
        if len(req.image_paths) > 1:
            print(f"[API /chat] Multiple images received, but ContextManager.add currently handles only the first: {first_image_path}")
            # The remaining images are still passed to the runner.

    ctx_mgr.add(role="user", content=req.message, image_path=first_image_path)

    # 2. RAG: Retrieve snippets if requested
    retrieved_snippets = []
    if req.use_rag:
        print('[API /chat] RAG enabled. Retrieving snippets...')
        history_for_rag = ctx_mgr._messages[:-1] if len(ctx_mgr._messages) > 1 else []
        retrieved_snippets = retriever.retrieve(query_text=req.message, current_chat_history=history_for_rag)
        print(f'[API /chat] Retrieved {len(retrieved_snippets)} snippets.')

    # 3. Build the prompt (ContextManager._fmt will include the first_image_path placeholder if added)
    try:
        prompt_for_model = ctx_mgr.build_prompt(extra_messages=retrieved_snippets)
    except Exception as e:
        print(f"Error building prompt: {e}")
        raise HTTPException(status_code=500, detail="Error building prompt.")

    print(f"Built prompt for model (len {len(prompt_for_model)} chars):\n{prompt_for_model[:500]}...")

    # 4. Get the model runner
    runner = model_mgr.get()
    if not runner:
        raise HTTPException(status_code=400, detail="No model is currently loaded. Please load a model first via /load_model.")

    # 5. Generate response (first pass)
    try:
        # Pass all image_paths to the runner, as runner might handle multiple images.
        reply_text = runner.generate(
            prompt=prompt_for_model,
            image_paths=req.image_paths, # Pass full list here
            **req.generation_params.model_dump()
        )
        final_reply_text = reply_text

        # 6. Tool Calling Placeholder Logic
        if reply_text.startswith('[FUNCALL]'):
            print(f'[API /chat] Detected function call: {reply_text}')
            function_call_json = reply_text.replace('[FUNCALL]', '').strip()
            tool_result_dict = tool_dispatcher.dispatch(function_call_json)
            print(f'[API /chat] Tool dispatch result: {tool_result_dict}')
            tool_message_content = json.dumps(tool_result_dict)

            # Add tool result to context. ContextManager.add doesn't take image for tool_result.
            ctx_mgr.add(role='tool_result', content=tool_message_content)
            print('[API /chat] Added tool result to context. Re-generating...')

            second_prompt = ctx_mgr.build_prompt()
            print(f"Built second prompt for model (len {len(second_prompt)} chars):\n{second_prompt[:500]}...")

            # For re-generation after tool call, typically images are not passed again
            # unless the tool interaction specifically requires re-evaluating them.
            # For now, not passing image_paths on the second call.
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

# --- Main block for Uvicorn ---
if __name__ == "__main__":
    print("Starting Uvicorn server for LLM Context OS API...")
    # Example to run from project root (llm_context_os/):
    # PYTHONPATH=$(pwd) python llm_context_os/api/main.py
    # Or: uvicorn llm_context_os.api.main:app --reload --port 8000

    # Example curl commands:
    # 1. Load a model (e.g., LLaVA Cpp placeholder)
    # curl -X POST http://localhost:8000/load_model -H "Content-Type: application/json" -d \
    # '{
    #   "model_type": "llava_cpp",
    #   "model_path_or_name": "path/to/llava-model.gguf",
    #   "runner_params": {
    #     "mmproj_path": "path/to/llava-mmproj.gguf",
    #     "n_gpu_layers": 30
    #   }
    # }'
    #
    # 2. Send a chat message with an image
    # curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "What color is the object in this image?",
    #   "image_paths": ["/path/on/server/to/your_image.jpg"],
    #   "use_rag": false,
    #   "generation_params": {
    #     "temperature": 0.5,
    #     "max_new_tokens": 75
    #   }
    # }'
    #
    # 3. Send a chat message with multiple images
    # curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "Compare these two images.",
    #   "image_paths": ["/server/img1.png", "/server/img2.jpeg"],
    #   "use_rag": false
    # }'

    uvicorn.run(app, host="0.0.0.0", port=8000)
