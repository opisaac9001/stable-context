# llm_context_os/runners/base.py
from abc import ABC, abstractmethod
import typing as t

class BaseRunner(ABC):
    """
    Abstract base class for all model runners.
    Defines the interface for generating text and streaming responses.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: t.Any) -> str:
        """
        Generates a single text response from the given prompt.
        kwargs can include generation parameters like temperature, max_tokens, etc.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
        """
        Streams text responses from the given prompt.
        Yields chunks of text as they are generated.
        kwargs can include generation parameters.
        """
        pass

    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        """
        Optional method to preload the Key-Value cache for the model
        with a given prompt. This can speed up generation for subsequent calls
        that share a common prefix.
        Not all runners may support or require this.
        """
        pass # Default implementation does nothing

if __name__ == '__main__':
    class DummyRunner(BaseRunner):
        def __init__(self, model_name: str):
            self.model_name = model_name
            print(f"DummyRunner initialized with model: {self.model_name}")

        def generate(self, prompt: str, **kwargs: t.Any) -> str:
            print(f"\n--- {self.model_name} Generating ---")
            print(f"Prompt: {prompt}")
            print(f"Config: {kwargs}")
            response = f"Response from {self.model_name} to: '{prompt[:20]}...'"
            print(f"Full Response: {response}")
            return response

        def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
            print(f"\n--- {self.model_name} Streaming ---")
            print(f"Prompt: {prompt}")
            print(f"Config: {kwargs}")
            yield f"Stream chunk 1 from {self.model_name} for '{prompt[:10]}...' "
            yield f"Stream chunk 2 from {self.model_name} for '{prompt[:10]}...' "
            yield "Done."
            print("Streaming complete.")

        def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
            print(f"\n--- {self.model_name} Preloading KV Cache ---")
            print(f"Prompt for KV: {prompt}")
            print(f"Config: {kwargs}")
            print("KV cache preloaded (simulated).")

    # Example Usage
    dummy_model = DummyRunner(model_name="TestModel-001")

    # Test generate
    generation_config = {"temperature": 0.7, "max_new_tokens": 50}
    dummy_model.generate("This is a test prompt for generation.", **generation_config)

    # Test stream
    streaming_config = {"temperature": 0.5, "top_p": 0.9}
    print("\nCollecting stream:")
    full_streamed_response = []
    for chunk in dummy_model.stream("This is a test prompt for streaming.", **streaming_config):
        print(f"Received chunk: '{chunk}'")
        full_streamed_response.append(chunk)
    print(f"Full streamed response: {''.join(full_streamed_response)}")

    # Test preload_kv
    kv_config = {"user_id": "test_user"}
    dummy_model.preload_kv("This is a common prefix to preload.", **kv_config)
    # Subsequent calls might reuse this (in a real scenario)
    dummy_model.generate("This is a test prompt for generation after KV preload.", **generation_config)

    print("\nBaseRunner and DummyRunner demonstration complete.")
