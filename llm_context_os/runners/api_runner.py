# llm_context_os/runners/api_runner.py
import typing as t
from .base import BaseRunner # Assuming base.py is in the same directory

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
        # print(f"Prompt (full for debug): '{prompt}' (len {len(prompt)})") # Debug line
        # print(f"Prompt slice for response: '{prompt[:49]}'") # Debug line
        response = f"[API Response from {self.model_name} to: {prompt[:49]}...]" # Using :49
        print(f"Full Response: {response}")
        return response

    def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
        print(f"\n--- APIRunner ({self.model_name}) Streaming ---")
        print(f"Target URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        print(f"Streaming Config: {kwargs}")
        yield f"[Chunk 1 from {self.model_name} for '{prompt[:25]}...'] " # Shorter slice for stream
        yield f"[Chunk 2 from {self.model_name}, params: {kwargs.get('temperature', 'default_temp')}] "
        yield f"[End of stream from {self.model_name}]"
        print("Streaming complete.")

    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        print(f"\n--- APIRunner ({self.model_name}) Preloading KV Cache ---")
        print("Note: KV cache preloading is typically not applicable for standard external APIs.")
        super().preload_kv(prompt, **kwargs)

if __name__ == '__main__':
    dummy_api_url = "https://api.example.com/v1/chat/completions"
    dummy_api_key = "sk-dummy_key_for_testing_1234"
    api_model = APIRunner(model_name="gpt-dummy-3.5", api_url=dummy_api_url, api_key=dummy_api_key)
    generation_params = {"temperature": 0.8, "max_tokens": 100}
    # Test prompt: "You are a helpful AI assistant.\nuser: Hello, model!\n" (len 55)
    # prompt[:49] = "You are a helpful AI assistant.\nuser: Hello, mod"
    response_text = api_model.generate("You are a helpful AI assistant.\nuser: Hello, model!\n", **generation_params)
    # Expected: "[API Response from gpt-dummy-3.5 to: You are a helpful AI assistant.\nuser: Hello, mod...]"
    print(f"Generate call returned: '{response_text}'")
