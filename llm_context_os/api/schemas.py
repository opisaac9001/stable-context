# llm_context_os/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class GenerationParams(BaseModel):
    temperature: float = Field(0.7, description="Sampling temperature.", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling parameter.", ge=0.0, le=1.0)
    max_new_tokens: int = Field(256, description="Maximum number of new tokens to generate.", gt=0)
    use_kv_cache: bool = Field(True, description="Whether to use KV caching for generation.")

class LoadModelRequest(BaseModel):
    model_type: str = Field(..., description="Type of the model to load.")
    model_path_or_name: str = Field(..., description="Path to the model file/directory or unique model name/identifier.")
    runner_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the runner constructor.")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message.")
    image_paths: Optional[List[str]] = Field(None, description="Optional list of image paths or URLs relevant to the message.")
    use_rag: bool = Field(False, description="Whether to use RAG from chat history or PDFs.")
    pdf_doc_ids_for_rag: Optional[List[str]] = Field(None, description="Optional list of document IDs (e.g., PDF filenames without extension) to use for PDF RAG.") # Added
    stream: bool = Field(False, description="Enable Server-Sent Events streaming for the response.")
    generation_params: GenerationParams = Field(default_factory=GenerationParams, description="Parameters for text generation.")

class StatusResponse(BaseModel):
    status: str = Field(..., description="Status of the operation (e.g., 'ok', 'error').")
    message: Optional[str] = Field(None, description="Optional message providing more details.")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="The LLM's reply.")
    request_details: Optional[ChatRequest] = Field(None, description="Original request details.")
    generated_tokens: Optional[int] = Field(None, description="Number of tokens generated in the reply.", gt=-1) # Renamed
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the input prompt processed by the model.", gt=-1) # Added
    latency_ms: Optional[float] = Field(None, description="Time taken to generate the response in milliseconds.", ge=0.0)

# New Schema for PDF Upload Response
class UploadPdfResponse(BaseModel):
    doc_id: Optional[str] = Field(None, description="Document ID assigned after processing (e.g., filename stem).")
    filename: str = Field(..., description="Original filename of the uploaded PDF.")
    message: str = Field(..., description="Status message from the processing.")
    num_chunks_processed: Optional[int] = Field(None, description="Number of text chunks processed from the PDF.")
    status: str = Field("success", description="Overall status ('success' or 'error').")

# --- Global Application Settings Schemas ---

class ContextManagerSettings(BaseModel):
    system_prompt: Optional[str] = Field(None, description="Default system prompt for the context manager.")
    max_tokens: Optional[int] = Field(None, description="Default maximum context tokens for the context manager.")

class ChatHistoryRetrieverSettings(BaseModel):
    recall_budget_tokens: Optional[int] = Field(None, description="Token budget for retrieved chat history snippets.")
    embedding_model_name: Optional[str] = Field(None, description="SentenceTransformer model name for chat history embeddings.")
    vector_db_path: Optional[str] = Field(None, description="Path to the vector database for chat history.")

class PdfRetrieverSettings(BaseModel):
    vector_db_path: Optional[str] = Field(None, description="Path to the vector database for PDF RAG.")
    embedding_model_name: Optional[str] = Field(None, description="SentenceTransformer model name for PDF embeddings.")
    chunk_size: Optional[int] = Field(None, description="Chunk size for PDF text processing.")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap for PDF text processing.")

class ModelManagerSettings(BaseModel):
    default_idle_unload_sec: Optional[int] = Field(None, description="Default idle time in seconds before unloading a model.")

class TokenEstimatorConfigSettings(BaseModel):
    type: Optional[str] = Field(None, description="Type of token estimator ('tiktoken' or 'hf').")
    model_name: Optional[str] = Field(None, description="Model name for the token estimator (e.g., 'cl100k_base' for tiktoken, 'gpt2' for hf).")

class GlobalSettings(BaseModel):
    context_manager: Optional[ContextManagerSettings] = None
    chat_history_retriever: Optional[ChatHistoryRetrieverSettings] = None
    pdf_retriever: Optional[PdfRetrieverSettings] = None
    model_manager: Optional[ModelManagerSettings] = None
    token_estimator_for_rag_budgeting: Optional[TokenEstimatorConfigSettings] = None
    generation_defaults: Optional[GenerationParams] = Field(None, description="Default generation parameters for chat operations.")
    # Add other top-level config sections here if they exist in config.yaml

# For API endpoint to update settings
UpdateSettingsRequest = GlobalSettings


# --- Model Listing Schemas ---
class AvailableModel(BaseModel):
    model_id: str = Field(..., description="A unique identifier for the model, could be filename or a derived ID.")
    model_type: str = Field(..., description="Type of the model (e.g., 'gguf', 'awq', 'exl2', 'api_provided').")
    path_or_identifier: str = Field(..., description="Filesystem path to the model or unique name for API models.")
    name: Optional[str] = Field(None, description="A user-friendly name, could be derived from path.")
    description: Optional[str] = Field(None, description="Brief description of the model.")
    details: Optional[Dict[str, Any]] = Field(None, description="Extra information like file size, quantization, family, etc.")

class ModelListResponse(BaseModel):
    models: List[AvailableModel] = Field(..., description="List of available models.")

# --- Tool Listing & Management Schemas ---
class ToolInfo(BaseModel):
    name: str = Field(..., description="Unique name of the tool.")
    type: str = Field(..., description="Type of the tool (e.g., 'local', 'mcp').")
    description: Optional[str] = Field(None, description="Description of what the tool does.")
    is_enabled: bool = Field(True, description="Whether the tool is currently enabled for dispatch.")
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON schema of parameters the tool accepts.")

class ToolListResponse(BaseModel):
    tools: List[ToolInfo] = Field(..., description="List of available tools and their status.")

class ToggleToolRequest(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to enable or disable.")
    enable: bool = Field(..., description="Set to true to enable the tool, false to disable.")

# --- Model Download Schemas ---
class DownloadModelRequest(BaseModel):
    repo_id: str = Field(..., description="The Hugging Face repository ID (e.g., 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF').")
    model_type: Optional[str] = Field(None, description="Expected model type (e.g., 'gguf', 'awq', 'exl2'). Helps in organizing downloaded files or validation.")
    filename: Optional[str] = Field(None, description="Specific filename to download from the repo (especially for GGUF). If None, attempts to download suitable files or the whole repo based on type.")
    target_path: Optional[str] = Field(None, description="Optional local relative path within a configured models directory to save the model. If None, a default path will be constructed.")


if __name__ == '__main__':
    print("\n--- ChatRequest with PDF RAG and Stream---")
    chat_req_pdf_rag = ChatRequest(
        message="What did the climate report say about mitigation?",
        use_rag=True,
        pdf_doc_ids_for_rag=["climate_change_overview", "another_report"],
        stream=True
    )
    print(f"Chat with PDF RAG and Stream: {chat_req_pdf_rag.model_dump_json(indent=2)}")
    assert chat_req_pdf_rag.pdf_doc_ids_for_rag == ["climate_change_overview", "another_report"]
    assert chat_req_pdf_rag.stream is True

    # Test default for stream
    chat_req_no_stream = ChatRequest(message="Hello")
    assert chat_req_no_stream.stream is False


    print("\n--- UploadPdfResponse ---")
    upload_resp_success = UploadPdfResponse(
        doc_id="climate_change_overview",
        filename="climate_change_overview.pdf",
        message="Successfully processed and stored 3 chunks for document: climate_change_overview",
        num_chunks_processed=3,
        status="success"
    )
    print(f"Upload Success Resp: {upload_resp_success.model_dump_json(indent=2)}")

    upload_resp_error = UploadPdfResponse(
        filename="missing_document.pdf",
        message="Error: PDF file not found or is not a file: missing_document.pdf",
        status="error"
    )
    print(f"Upload Error Resp: {upload_resp_error.model_dump_json(indent=2)}")
    assert upload_resp_error.doc_id is None
    assert upload_resp_error.num_chunks_processed is None

    # Test existing schemas remain valid
    print("\n--- GenerationParams (existing) ---")
    params_default = GenerationParams()
    print(f"Default GenParams: {params_default.model_dump_json(indent=2)}")

    print("\n--- ChatResponse (Updated for token counts) ---")
    chat_resp_updated = ChatResponse(
        reply="The PDF discusses several mitigation strategies.",
        request_details=chat_req_pdf_rag, # Using existing complex request
        generated_tokens=20,
        prompt_tokens=150, # Example prompt token count
        latency_ms=150.0
    )
    print(f"Updated ChatResponse: {chat_resp_updated.model_dump_json(indent=2)}")
    assert chat_resp_updated.generated_tokens == 20
    assert chat_resp_updated.prompt_tokens == 150


    print("\nSchema definitions and examples updated (GlobalSettings, etc.).")

    print("\n--- GlobalSettings Example (Partial) ---")
    partial_settings = GlobalSettings(
        context_manager={"max_tokens": 2048}, # Pass as dict
        model_manager=ModelManagerSettings(default_idle_unload_sec=600) # Pass as model instance
    )
    print(partial_settings.model_dump_json(indent=2, exclude_unset=True))
    # Example of how to access nested values:
    if partial_settings.context_manager:
        assert partial_settings.context_manager.max_tokens == 2048
    if partial_settings.model_manager:
        assert partial_settings.model_manager.default_idle_unload_sec == 600


    print("\n--- UpdateSettingsRequest Example (showing partial update structure) ---")
    update_req = UpdateSettingsRequest(
        chat_history_retriever=ChatHistoryRetrieverSettings(recall_budget_tokens=256, embedding_model_name="new_embed_model")
    )
    print(update_req.model_dump_json(indent=2, exclude_unset=True))
    if update_req.chat_history_retriever:
        assert update_req.chat_history_retriever.recall_budget_tokens == 256
        assert update_req.chat_history_retriever.embedding_model_name == "new_embed_model"

    print("\n--- Empty GlobalSettings (all fields None) ---")
    empty_settings = GlobalSettings()
    print(empty_settings.model_dump_json(indent=2, exclude_unset=True)) # Should be {}
    assert empty_settings.model_dump(exclude_unset=True) == {}

    print("\n--- Model Listing Schemas Examples ---")
    available_model_ex = AvailableModel(
        model_id="model_abc_123",
        model_type="gguf",
        path_or_identifier="/path/to/models/model_abc.gguf",
        name="My Awesome GGUF Model",
        description="A GGUF version of a popular LLM.",
        details={"quantization": "Q5_K_M", "size_gb": 7.5, "family": "LlamaFam"}
    )
    print(f"AvailableModel: {available_model_ex.model_dump_json(indent=2)}")

    model_list_resp_ex = ModelListResponse(models=[available_model_ex])
    print(f"ModelListResponse: {model_list_resp_ex.model_dump_json(indent=2)}")
    assert len(model_list_resp_ex.models) == 1

    print("\n--- Tool Listing & Management Schemas Examples ---")
    tool_info_ex = ToolInfo(
        name="get_weather",
        type="local",
        description="Gets the current weather for a specified location.",
        is_enabled=True,
        parameters={"type": "object", "properties": {"location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}
    )
    print(f"ToolInfo: {tool_info_ex.model_dump_json(indent=2)}")

    tool_list_resp_ex = ToolListResponse(tools=[tool_info_ex])
    print(f"ToolListResponse: {tool_list_resp_ex.model_dump_json(indent=2)}")

    toggle_tool_req_ex = ToggleToolRequest(tool_name="get_weather", enable=False)
    print(f"ToggleToolRequest: {toggle_tool_req_ex.model_dump_json(indent=2)}")

    print("\n--- DownloadModelRequest Example ---")
    download_req_ex = DownloadModelRequest(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type="gguf",
        filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        target_path="downloaded_ggufs/"
    )
    print(f"DownloadModelRequest: {download_req_ex.model_dump_json(indent=2)}")
    assert download_req_ex.filename == "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
