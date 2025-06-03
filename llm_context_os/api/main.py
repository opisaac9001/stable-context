# llm_context_os/api/main.py
import uvicorn
from fastapi import FastAPI, HTTPException
import time # For latency calculation
import json # For tool_dispatcher integration

# App-specific imports
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
from llm_context_os.retriever.chat_history import ChatHistoryRetriever # Added
from llm_context_os.tools.tool_dispatcher import ToolDispatcher # Added

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
retriever = ChatHistoryRetriever() # Added
tool_dispatcher = ToolDispatcher() # Added

# --- API Endpoints ---

@app.post("/load_model", response_model=StatusResponse)
async def load_model_endpoint(req: LoadModelRequest):
    print(f"Received /load_model request: {req.model_dump()}")
    try:
        runner_params = req.runner_params if req.runner_params is not None else {}
        model_mgr.load(
            model_type=req.model_type,
            model_path_or_name=req.model_path_or_name,
            # Pass idle_unload_sec from runner_params if provided, or let manager use its default
            idle_unload_sec=runner_params.pop('idle_unload_sec', None),
            **runner_params
        )
        if model_mgr.current_runner:
            return StatusResponse(status="ok", message=f"Model '{req.model_path_or_name}' of type '{req.model_type}' loaded successfully.")
        else:
            return StatusResponse(status="error", message=f"Failed to load model '{req.model_path_or_name}'. Runner not available after load attempt.")
    except Exception as e:
        print(f"Error in /load_model endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"Received /chat request: {req.model_dump(exclude={'generation_params'})}")
    print(f"Generation params for chat: {req.generation_params.model_dump()}")

    start_time = time.time()
    final_reply_text = "" # Initialize

    # 1. Add user message to context
    ctx_mgr.add(role="user", content=req.message)

    # 2. RAG: Retrieve snippets if requested
    retrieved_snippets = []
    if req.use_rag:
        print('[API /chat] RAG enabled. Retrieving snippets...')
        # Pass current messages (excluding the latest user query which is part of ctx_mgr)
        # This is a conceptual detail; placeholder retriever doesn't use it yet.
        history_for_rag = ctx_mgr._messages[:-1] if len(ctx_mgr._messages) > 1 else []
        retrieved_snippets = retriever.retrieve(query_text=req.message, current_chat_history=history_for_rag)
        print(f'[API /chat] Retrieved {len(retrieved_snippets)} snippets.')
        if retrieved_snippets:
            # Add snippets to context before building the main prompt for the LLM
            # The ContextManager's build_prompt will format these.
            # For now, we just pass them as 'extra_messages'.
            pass # extra_messages will be passed to build_prompt later

    # 3. Build the prompt
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
        reply_text = runner.generate(prompt=prompt_for_model, **req.generation_params.model_dump())
        final_reply_text = reply_text # Assume this is the final reply unless a tool call happens

        # 6. Tool Calling Placeholder Logic
        if reply_text.startswith('[FUNCALL]'): # Placeholder for detecting function call
            print(f'[API /chat] Detected function call: {reply_text}')
            function_call_json = reply_text.replace('[FUNCALL]', '').strip()

            tool_result_dict = tool_dispatcher.dispatch(function_call_json)
            print(f'[API /chat] Tool dispatch result: {tool_result_dict}')

            # Format tool result for context (can be simple JSON string or more structured)
            tool_message_content = json.dumps(tool_result_dict)

            ctx_mgr.add(role='tool_result', content=tool_message_content) # Add tool result to context
            print('[API /chat] Added tool result to context. Re-generating...')

            # Re-build prompt with tool result included
            second_prompt = ctx_mgr.build_prompt() # extra_messages not typically used for re-gen
            print(f"Built second prompt for model (len {len(second_prompt)} chars):\n{second_prompt[:500]}...")

            # Generate final response from the LLM after tool execution
            final_reply_text = runner.generate(second_prompt, **req.generation_params.model_dump())
            print(f"[API /chat] Final reply after tool call: {final_reply_text[:100]}...")

        # Optional: Add AI's final reply to context manager for multi-turn conversation history
        # ctx_mgr.add(role="assistant", content=final_reply_text)

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
