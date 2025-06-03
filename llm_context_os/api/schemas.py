# llm_context_os/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List # Added List

class GenerationParams(BaseModel):
    """
    Common parameters for controlling text generation.
    """
    temperature: float = Field(0.7, description="Sampling temperature.", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling parameter.", ge=0.0, le=1.0)
    max_new_tokens: int = Field(256, description="Maximum number of new tokens to generate.", gt=0)
    use_kv_cache: bool = Field(True, description="Whether to use KV caching for generation.")

class LoadModelRequest(BaseModel):
    """
    Request to load a model into the ModelManager.
    """
    model_type: str = Field(..., description="Type of the model to load (e.g., 'gguf', 'awq', 'api', 'exl2', 'llava_cpp').")
    model_path_or_name: str = Field(..., description="Path to the model file/directory or unique model name/identifier.")
    runner_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the runner constructor.")

class ChatRequest(BaseModel):
    """
    Request for a chat interaction.
    """
    message: str = Field(..., description="User's message.")
    image_paths: Optional[List[str]] = Field(None, description="Optional list of image paths or URLs relevant to the message.") # Added
    use_rag: bool = Field(False, description="Whether to use RAG (Retrieval Augmented Generation).")
    generation_params: GenerationParams = Field(default_factory=GenerationParams, description="Parameters for text generation.")

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
    # request_details will now implicitly include image_paths if ChatRequest has it
    request_details: Optional[ChatRequest] = Field(None, description="Original request details (for context or logging).")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated in the reply.", gt=-1)
    latency_ms: Optional[float] = Field(None, description="Time taken to generate the response in milliseconds.", ge=0.0)

if __name__ == '__main__':
    print("--- GenerationParams ---")
    params_default = GenerationParams()
    print(f"Default: {params_default.model_dump_json(indent=2)}")

    print("\n--- LoadModelRequest ---")
    load_req_gguf = LoadModelRequest(
        model_type="llava_cpp",
        model_path_or_name="models/llava-v1.5-7b.gguf",
        runner_params={"mmproj_path": "models/llava-v1.5-7b-mmproj-f16.gguf", "n_gpu_layers": 35}
    )
    print(f"LLaVA Load: {load_req_gguf.model_dump_json(indent=2)}")

    print("\n--- ChatRequest ---")
    chat_req_simple = ChatRequest(message="Hello there!")
    print(f"Simple Chat: {chat_req_simple.model_dump_json(indent=2)}")

    chat_req_multimodal = ChatRequest(
        message="What is in this image?",
        image_paths=["/path/to/image1.jpg", "https://example.com/image2.png"],
        generation_params=GenerationParams(max_new_tokens=100)
    )
    print(f"Multimodal Chat: {chat_req_multimodal.model_dump_json(indent=2)}")
    assert chat_req_multimodal.image_paths is not None
    assert len(chat_req_multimodal.image_paths) == 2

    print("\n--- StatusResponse ---")
    status_ok = StatusResponse(status="ok", message="Model loaded successfully.")
    print(f"OK Status: {status_ok.model_dump_json(indent=2)}")

    print("\n--- ChatResponse ---")
    chat_resp = ChatResponse(
        reply="The image shows a cat.",
        request_details=chat_req_multimodal, # Now includes image_paths
        tokens_generated=5,
        latency_ms=120.5
    )
    print(f"Chat Resp: {chat_resp.model_dump_json(indent=2)}")
    assert chat_resp.request_details.image_paths == ["/path/to/image1.jpg", "https://example.com/image2.png"]

    print("\nSchema definitions and examples complete.")
