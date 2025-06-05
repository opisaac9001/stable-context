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
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated in the reply.", gt=-1)
    latency_ms: Optional[float] = Field(None, description="Time taken to generate the response in milliseconds.", ge=0.0)

# New Schema for PDF Upload Response
class UploadPdfResponse(BaseModel):
    doc_id: Optional[str] = Field(None, description="Document ID assigned after processing (e.g., filename stem).")
    filename: str = Field(..., description="Original filename of the uploaded PDF.")
    message: str = Field(..., description="Status message from the processing.")
    num_chunks_processed: Optional[int] = Field(None, description="Number of text chunks processed from the PDF.")
    status: str = Field("success", description="Overall status ('success' or 'error').")


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

    print("\n--- ChatResponse with updated ChatRequest (existing) ---")
    chat_resp = ChatResponse(
        reply="The PDF discusses several mitigation strategies.",
        request_details=chat_req_pdf_rag,
        tokens_generated=20,
        latency_ms=150.0
    )
    print(f"Chat Resp with PDF RAG in details: {chat_resp.model_dump_json(indent=2)}")
    assert chat_resp.request_details.pdf_doc_ids_for_rag == ["climate_change_overview", "another_report"]

    print("\nSchema definitions and examples updated.")
