# llm_context_os/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class GenerationParams(BaseModel):
    """
    Common parameters for controlling text generation.
    """
    temperature: float = Field(0.7, description="Sampling temperature.", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling parameter.", ge=0.0, le=1.0)
    # max_tokens in blueprint, but runners use max_new_tokens. Let's use max_new_tokens for clarity.
    max_new_tokens: int = Field(256, description="Maximum number of new tokens to generate.", gt=0)
    use_kv_cache: bool = Field(True, description="Whether to use KV caching for generation.")
    # Add other common params like top_k, repeat_penalty if needed later
    # top_k: Optional[int] = Field(None, description="Top-k sampling parameter.", gt=0)
    # repeat_penalty: Optional[float] = Field(None, description="Repetition penalty.", gt=0.0)

class LoadModelRequest(BaseModel):
    """
    Request to load a model into the ModelManager.
    """
    model_type: str = Field(..., description="Type of the model to load (e.g., 'gguf', 'awq', 'api', 'exl2').")
    model_path_or_name: str = Field(..., description="Path to the model file/directory or unique model name/identifier.")

    # context_params from blueprint seems generic, could be runner-specific init args.
    # Using a more generic 'runner_params' to pass to runner constructors.
    runner_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the runner constructor (e.g., n_gpu_layers, api_url, api_key, device, n_ctx).")

    # gpu_layers was specified in blueprint's FastAPI example for /load_model.
    # It's specific to some runners (like LlamaCpp).
    # Better to include it in runner_params for generality, e.g., runner_params={"n_gpu_layers": 20}
    # However, if it's a very common top-level param for many local runners, it could stay.
    # For now, let's assume it's passed via runner_params.
    # If context_params was meant for ContextManager, that's a different concern.
    # Assuming 'context_params' from blueprint's /load_model req.ctx -> runner_params for now.

class ChatRequest(BaseModel):
    """
    Request for a chat interaction.
    """
    message: str = Field(..., description="User's message.")
    use_rag: bool = Field(False, description="Whether to use RAG (Retrieval Augmented Generation).")
    # generation_params from blueprint was GenerationParams = GenerationParams()
    # This means if not provided, it uses defaults from GenerationParams.
    generation_params: GenerationParams = Field(default_factory=GenerationParams, description="Parameters for text generation.")
    # stream: bool = Field(False, description="Whether to stream the response.") # Add if separate endpoint for stream vs. non-stream

class StatusResponse(BaseModel):
    """
    Generic status response.
    """
    status: str = Field(..., description="Status of the operation (e.g., 'ok', 'error').")
    message: Optional[str] = Field(None, description="Optional message providing more details.")

class ChatResponse(BaseModel):
    """
    Response from a chat interaction.
    """
    reply: str = Field(..., description="The LLM's reply.")
    request_details: Optional[ChatRequest] = Field(None, description="Original request details (for context or logging).")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated in the reply.", gt=-1) # Can be 0 for empty or error
    latency_ms: Optional[float] = Field(None, description="Time taken to generate the response in milliseconds.", ge=0.0)
    # Could add other details like RAG sources used, etc.

if __name__ == '__main__':
    # Example Usage and Validation
    print("--- GenerationParams ---")
    params_default = GenerationParams()
    print(f"Default: {params_default.model_dump_json(indent=2)}")
    params_custom = GenerationParams(temperature=0.9, max_new_tokens=50)
    print(f"Custom: {params_custom.model_dump_json(indent=2)}")
    try:
        GenerationParams(temperature=-0.5)
    except ValueError as e:
        print(f"Validation error (temp): {e}")
    try:
        GenerationParams(max_new_tokens=0)
    except ValueError as e:
        print(f"Validation error (max_new_tokens): {e}")

    print("\n--- LoadModelRequest ---")
    load_req_gguf = LoadModelRequest(
        model_type="gguf",
        model_path_or_name="models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        runner_params={"n_gpu_layers": 20, "n_ctx": 4096}
    )
    print(f"GGUF Load: {load_req_gguf.model_dump_json(indent=2)}")
    load_req_api = LoadModelRequest(
        model_type="api",
        model_path_or_name="gpt-4o",
        runner_params={"api_url": "https://api.openai.com/v1", "api_key": "sk-..."}
    )
    print(f"API Load: {load_req_api.model_dump_json(indent=2)}")

    print("\n--- ChatRequest ---")
    chat_req_simple = ChatRequest(message="Hello there!")
    print(f"Simple Chat: {chat_req_simple.model_dump_json(indent=2)}")
    chat_req_custom_gen = ChatRequest(
        message="Tell me a story.",
        use_rag=True,
        generation_params=GenerationParams(temperature=0.5, max_new_tokens=512)
    )
    print(f"Custom Gen Chat: {chat_req_custom_gen.model_dump_json(indent=2)}")

    print("\n--- StatusResponse ---")
    status_ok = StatusResponse(status="ok", message="Model loaded successfully.")
    print(f"OK Status: {status_ok.model_dump_json(indent=2)}")
    status_err = StatusResponse(status="error", message="Failed to load model.")
    print(f"Error Status: {status_err.model_dump_json(indent=2)}")

    print("\n--- ChatResponse ---")
    chat_resp = ChatResponse(
        reply="The weather is sunny.",
        request_details=chat_req_simple,
        tokens_generated=5,
        latency_ms=120.5
    )
    print(f"Chat Resp: {chat_resp.model_dump_json(indent=2)}")

    print("\nSchema definitions and examples complete.")
