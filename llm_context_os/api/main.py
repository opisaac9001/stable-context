# llm_context_os/api/main.py
import uvicorn
from fastapi import FastAPI, HTTPException
import time # For latency calculation

# App-specific imports
# Assuming the script is run from a context where llm_context_os is a package
# or PYTHONPATH is set appropriately.
from llm_context_os.api.schemas import (
    LoadModelRequest,
    ChatRequest,
    StatusResponse,
    ChatResponse,
    GenerationParams # Imported for type hinting if needed, though ChatRequest embeds it
)
from llm_context_os.context.context_manager import ContextManager, MockTokenizer
from llm_context_os.runners.manager import ModelManager
from llm_context_os.runners.base import BaseRunner # For type hinting

# --- Application Setup ---
app = FastAPI(
    title="LLM Context OS API",
    description="API for managing local LLM models, context, and generation.",
    version="0.1.0"
)

# --- Global Instances ---
# These would ideally be managed with more sophisticated dependency injection
# or application lifecycle events for production, but are fine for now.
model_mgr = ModelManager()

# Initialize a default context manager.
# This could be made configurable per session/user in a more complex app.
mock_tokenizer_for_ctx = MockTokenizer() # Using a simple char-based tokenizer for context
# System prompt and max_tokens could come from a config file.
# For now, the ContextManager uses character length for token counting with MockTokenizer.
# If a real tokenizer (e.g., from Hugging Face) were used, this would be more accurate.
# Max tokens for ContextManager should ideally align with model's actual context window,
# minus space for generated output if the model's limit is strict on input+output.
# Let's use a smaller max_tokens for context to test truncation easily.
ctx_mgr = ContextManager(
    system_prompt="You are a helpful AI assistant.",
    tokenizer=mock_tokenizer_for_ctx,
    max_tokens=1024 # Example: context window size for the prompt itself
)

# --- API Endpoints ---

@app.post("/load_model", response_model=StatusResponse)
async def load_model_endpoint(req: LoadModelRequest):
    """
    Loads a model into the ModelManager.
    The `runner_params` in the request will be passed as keyword arguments
    to the specific runner's constructor.

    Example `runner_params` for 'gguf': `{"n_gpu_layers": 20, "n_ctx": 4096}`
    Example `runner_params` for 'api': `{"api_url": "...", "api_key": "..."}`
    """
    print(f"Received /load_model request: {req.model_dump()}")
    try:
        runner_params = req.runner_params if req.runner_params is not None else {}
        # The model_path_or_name from LoadModelRequest is the main identifier
        model_mgr.load(
            model_type=req.model_type,
            model_path_or_name=req.model_path_or_name,
            **runner_params
        )
        # Check if loading was successful by trying to get the runner (optional check)
        if model_mgr.get() is not None: # get() prints success/failure already
             # model_mgr.load also prints success/failure.
             # We trust model_mgr.load to have set current_runner to None on failure.
            if model_mgr.current_runner: # Check if a runner was actually set
                return StatusResponse(status="ok", message=f"Model '{req.model_path_or_name}' of type '{req.model_type}' loaded successfully.")
            else: # Should not happen if load() clears current_runner on any failure path
                return StatusResponse(status="error", message=f"Failed to load model '{req.model_path_or_name}'. Runner not available after load attempt.")
        else: # get() returned None
             return StatusResponse(status="error", message=f"Failed to load model '{req.model_path_or_name}'. See server logs for details.")

    except Exception as e:
        # This top-level exception catch is a fallback.
        # ModelManager.load() should ideally handle its own errors more gracefully.
        print(f"Error in /load_model endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Handles a chat request, generates a response using the currently loaded model.
    """
    print(f"Received /chat request: {req.model_dump(exclude={'generation_params'})}")
    print(f"Generation params for chat: {req.generation_params.model_dump()}")

    start_time = time.time()

    # 1. Add user message to context
    ctx_mgr.add(role="user", content=req.message) # Assuming "user" role

    # 2. (Placeholder) Retrieve RAG snippets if requested
    snippets = []
    if req.use_rag:
        print("RAG requested, but not yet implemented. Proceeding without RAG.")
        # In future:
        # snippets = rag_retriever.retrieve(req.message, budget=config.rag_budget_tokens)
        # This would likely involve another component, e.g. a RetrieverManager.

    # 3. Build the prompt
    try:
        # Note: build_prompt expects `extra` not `extra_messages` based on ContextManager
        prompt_for_model = ctx_mgr.build_prompt(extra_messages=snippets)
    except Exception as e:
        print(f"Error building prompt: {e}")
        raise HTTPException(status_code=500, detail="Error building prompt.")

    print(f"Built prompt for model (len {len(prompt_for_model)} chars):\n{prompt_for_model[:500]}...") # Log first 500 chars

    # 4. Get the model runner
    runner = model_mgr.get()
    if not runner:
        # ctx_mgr.remove_last() # Optional: roll back context if model not available?
        raise HTTPException(status_code=400, detail="No model is currently loaded. Please load a model first via /load_model.")

    # 5. Generate response
    try:
        # Pass generation_params as kwargs by unpacking the Pydantic model
        # Placeholder runners will return fixed strings for now.
        reply_text = runner.generate(prompt=prompt_for_model, **req.generation_params.model_dump())

        # (Optional) Add AI's reply to context manager for multi-turn conversation
        # ctx_mgr.add(role="assistant", content=reply_text)

    except Exception as e:
        print(f"Error during model generation: {e}")
        # ctx_mgr.remove_last() # Optional: roll back context if generation failed
        raise HTTPException(status_code=500, detail=f"Error during model generation: {str(e)}")

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # For simplicity, tokens_generated is not calculated here with placeholder runners.
    # A real implementation would get this from the runner's response or tokenizer.

    # 6. Return response
    return ChatResponse(
        reply=reply_text,
        request_details=req, # Echo back the request for context
        tokens_generated=None, # Placeholder
        latency_ms=round(latency_ms, 2)
    )

# --- Main block for Uvicorn ---
if __name__ == "__main__":
    print("Starting Uvicorn server for LLM Context OS API...")
    # Note: For development, consider using `reload=True`.
    # The host "0.0.0.0" makes it accessible on the network.
    # For production, a more robust ASGI server like Gunicorn with Uvicorn workers is common.
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Example to run: python -m llm_context_os.api.main
    # (Ensure PYTHONPATH includes the parent directory of llm_context_os)
    # Or, if llm_context_os is installed or in current dir: python llm_context_os/api/main.py
    # Or, from parent dir of llm_context_os: uvicorn llm_context_os.api.main:app --reload --port 8000

    # To test endpoints (e.g., with curl or HTTPie):
    # Load a model (using a dummy API runner example):
    # curl -X POST http://localhost:8000/load_model -H "Content-Type: application/json" -d \
    # '{
    #   "model_type": "api",
    #   "model_path_or_name": "dummy-gpt",
    #   "runner_params": {
    #     "api_url": "http://localhost:1111/v1",
    #     "api_key": "sk-dummy"
    #   }
    # }'
    #
    # Send a chat message:
    # curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \
    # '{
    #   "message": "Hello, world!",
    #   "use_rag": false,
    #   "generation_params": {
    #     "temperature": 0.5,
    #     "max_new_tokens": 50
    #   }
    # }'
