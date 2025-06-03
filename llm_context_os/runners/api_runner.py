# llm_context_os/runners/api_runner.py
import typing as t
from .base import BaseRunner

class APIRunner(BaseRunner):
    """
    A placeholder runner for models accessed via an API (OpenAI-style).
    """
    def __init__(self, model_name: str, api_url: str, api_key: t.Optional[str] = None, **kwargs: t.Any):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"APIRunner initialized for model '{self.model_name}' at URL: {self.api_url}")
        if self.api_key:
            print(f"API Key: {'*' * (len(self.api_key) - 4) + self.api_key[-4:] if len(self.api_key) > 4 else '****'}")

    def generate(self, prompt: str, **kwargs: t.Any) -> str:
        print(f"\n--- APIRunner ({self.model_name}) Generating ---")
        print(f"Target URL: {self.api_url}")
        # print(f"Prompt (full for debug): '{prompt}' (len {len(prompt)})")
        # print(f"Prompt slice for response: '{prompt[:49]}'")
        response = f"[API Response from {self.model_name} to: {prompt[:49]}...]"
        print(f"Full Response: {response}")
        return response

    def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
        print(f"\n--- APIRunner ({self.model_name}) Streaming ---")
        print(f"Target URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        print(f"Streaming Config: {kwargs}")
        yield f"[Chunk 1 from {self.model_name} for '{prompt[:25]}...'] "
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


if __name__ == '__main__':
    dummy_api_url = "https://api.example.com/v1/chat/completions"
    dummy_api_key = "sk-dummy_key_for_testing_1234"
    api_model = APIRunner(model_name="gpt-dummy-3.5", api_url=dummy_api_url, api_key=dummy_api_key)

    generation_params = {"temperature": 0.8, "max_tokens": 100}
    response_text = api_model.generate("What is the weather like in London today?", **generation_params)
    print(f"Generate call returned: '{response_text}'")

    print("\nCollecting stream from API Runner:")
    full_api_streamed_response = []
    for chunk in api_model.stream("Tell me a short story about a robot explorer.", **generation_params):
        print(f"Received API chunk: '{chunk}'")
        full_api_streamed_response.append(chunk)
    print(f"Full API streamed response: {''.join(full_api_streamed_response)}")

    api_model.preload_kv("Common context for API calls.")

    exported_cache = api_model.export_kv_cache()
    print(f"Exported cache from API runner: {exported_cache}")

    api_model.import_kv_cache({"some_data": "data_val"})
    print("Import called on API runner.")

    print("\nAPIRunner placeholder demonstration complete.")
