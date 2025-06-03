# llm_context_os/runners/api_runner.py
import typing as t
from .base import BaseRunner

class APIRunner(BaseRunner):
    def __init__(self, model_name: str, api_url: str, api_key: t.Optional[str] = None, **kwargs: t.Any):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"APIRunner initialized for model '{self.model_name}' at URL: {self.api_url}")
        if self.api_key:
            print(f"API Key: {'*' * (len(self.api_key) - 4) + self.api_key[-4:] if len(self.api_key) > 4 else '****'}")

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> str:
        print(f"\n--- APIRunner ({self.model_name}) Generating ---")
        print(f"Target URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: APIRunner placeholder doesn't process images with prompt text currently)")
        response = f"[API Response from {self.model_name} to: {prompt[:49]}...]"
        if image_paths:
            response += f" (images: {', '.join(image_paths)})"
        print(f"Full Response: {response}")
        return response

    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[str, None, None]:
        print(f"\n--- APIRunner ({self.model_name}) Streaming ---")
        print(f"Target URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: APIRunner placeholder doesn't process images with stream text currently)")
        print(f"Streaming Config: {kwargs}")
        yield f"[Chunk 1 from {self.model_name} for '{prompt[:25]}...'] "
        if image_paths:
            yield f"[Images received: {len(image_paths)}] "
        yield f"[Chunk 2 from {self.model_name}, params: {kwargs.get('temperature', 'default_temp')}] "
        yield f"[End of stream from {self.model_name}]"
        print("Streaming complete.")

    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        print(f"\n--- APIRunner ({self.model_name}) Preloading KV Cache ---")
        print(f"Prompt for KV: {prompt}")
        print("[APIRunner] preload_kv called. Typically, KV caching is managed by the remote API endpoint or not applicable for stateless API calls.")
        super().preload_kv(prompt, **kwargs)

    def export_kv_cache(self) -> t.Any:
        print(f"\n--- APIRunner ({self.model_name}) Exporting KV Cache ---")
        print("API runners typically do not manage or expose exportable KV cache directly from the client side.")
        return None

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print(f"\n--- APIRunner ({self.model_name}) Importing KV Cache ---")
        print("Importing KV cache is generally not applicable to stateless API runners from the client side.")
        if cache_data:
            print(f"  (Received cache_data of type: {type(cache_data)}, but it will not be used.)")

    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Loading LoRA Adapter ---")
        print("  LoRA management is typically server-side for API-based models. This call is a no-op for APIRunner.")
        return False

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Unloading LoRA Adapter ---")
        print("  LoRA management is typically server-side for API-based models. This call is a no-op for APIRunner.")
        return False

    def get_active_lora_adapters(self) -> t.List[str]:
        print(f"\n--- APIRunner ({self.model_name}) Getting Active LoRA Adapters ---")
        print("  APIRunner does not manage LoRA adapters on the client-side.")
        return []

    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Merging LoRA Adapters ---")
        print("  Merging LoRAs is a server-side operation for API-based models. This call is a no-op for APIRunner.")
        return False

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Unmerging LoRA Adapters ---")
        print("  Unmerging LoRAs is a server-side operation for API-based models. This call is a no-op for APIRunner.")
        return False


if __name__ == '__main__':
    dummy_api_url = "https://api.example.com/v1/chat/completions"
    dummy_api_key = "sk-dummy_key_for_testing_1234"
    api_model = APIRunner(model_name="gpt-dummy-3.5", api_url=dummy_api_url, api_key=dummy_api_key)

    generation_params = {"temperature": 0.8, "max_tokens": 100}
    example_image_paths = ["/path/to/image1.jpg", "/path/to/image2.png"]

    response_text = api_model.generate("Describe the weather and these images.", image_paths=example_image_paths, **generation_params)
    print(f"Generate call returned: '{response_text}'")

    print("\nCollecting stream from API Runner:")
    full_api_streamed_response = []
    for chunk in api_model.stream("Tell me a short story about these pictures.", image_paths=example_image_paths, **generation_params):
        print(f"Received API chunk: '{chunk}'")
        full_api_streamed_response.append(chunk)
    print(f"Full API streamed response: {''.join(full_api_streamed_response)}")

    print("\nAPIRunner placeholder demonstration complete.")
