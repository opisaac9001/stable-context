# llm_context_os/runners/base.py
from abc import ABC, abstractmethod
import typing as t

class BaseRunner(ABC):
    """
    Abstract base class for all model runners.
    Defines the interface for generating text, streaming responses,
    and managing KV cache.
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

    @abstractmethod
    def export_kv_cache(self) -> t.Any:
        """
        Exports the current Key-Value cache state from the model.
        The format of cache_data is runner-specific.
        Returns:
            t.Any: The KV cache data. Can be None if not supported or cache is empty.
        """
        pass

    @abstractmethod
    def import_kv_cache(self, cache_data: t.Any) -> None:
        """
        Imports a previously exported Key-Value cache state into the model.
        The format of cache_data is runner-specific.
        Args:
            cache_data (t.Any): The KV cache data to import.
        """
        pass


if __name__ == '__main__':
    class DummyRunner(BaseRunner):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.kv_cache_data: t.Optional[t.Any] = None # Simulate a KV cache
            print(f"DummyRunner initialized with model: {self.model_name}")

        def generate(self, prompt: str, **kwargs: t.Any) -> str:
            print(f"\n--- {self.model_name} Generating ---")
            print(f"Prompt: {prompt}")
            print(f"Config: {kwargs}")
            if self.kv_cache_data:
                print(f"  (Simulating using imported KV cache: {self.kv_cache_data})")
            response = f"Response from {self.model_name} to: '{prompt[:20]}...'"
            # Simulate updating KV cache after generation
            self.kv_cache_data = {"prompt_prefix": prompt[:10], "generated_tokens": 5, "internal_state": "simulated_state"}
            print(f"Full Response: {response}")
            return response

        def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
            print(f"\n--- {self.model_name} Streaming ---")
            print(f"Prompt: {prompt}")
            print(f"Config: {kwargs}")
            if self.kv_cache_data:
                print(f"  (Simulating using imported KV cache for streaming: {self.kv_cache_data})")
            yield f"Stream chunk 1 from {self.model_name} for '{prompt[:10]}...' "
            yield f"Stream chunk 2 from {self.model_name} for '{prompt[:10]}...' "
            # Simulate updating KV cache during streaming
            self.kv_cache_data = {"prompt_prefix": prompt[:15], "generated_tokens": 10, "internal_state": "simulated_streaming_state"}
            yield "Done."
            print("Streaming complete.")

        def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
            print(f"\n--- {self.model_name} Preloading KV Cache ---")
            print(f"Prompt for KV: {prompt}")
            print(f"Config: {kwargs}")
            self.kv_cache_data = {"preloaded_prompt": prompt, "internal_state": "preloaded_simulated_state"}
            print(f"KV cache preloaded (simulated). Current cache: {self.kv_cache_data}")

        def export_kv_cache(self) -> t.Any:
            print(f"\n--- {self.model_name} Exporting KV Cache ---")
            if self.kv_cache_data:
                print(f"Exporting cache data: {self.kv_cache_data}")
                return self.kv_cache_data
            else:
                print("KV cache is empty or not supported for export.")
                return None

        def import_kv_cache(self, cache_data: t.Any) -> None:
            print(f"\n--- {self.model_name} Importing KV Cache ---")
            if cache_data:
                self.kv_cache_data = cache_data
                print(f"Imported cache data: {self.kv_cache_data}")
            else:
                print("No cache data provided to import, or import failed.")


    # Example Usage
    dummy_model = DummyRunner(model_name="TestModel-KV")

    # Test generate (populates KV cache)
    generation_config = {"temperature": 0.7, "max_new_tokens": 50}
    dummy_model.generate("This is a test prompt for generation.", **generation_config)

    # Test export_kv_cache
    exported_cache = dummy_model.export_kv_cache()
    assert exported_cache is not None
    assert exported_cache["generated_tokens"] == 5

    # Simulate clearing or changing the cache for the next step
    dummy_model.kv_cache_data = None
    print("\nSimulated clearing KV cache in model.")

    # Test import_kv_cache
    imported_data = {"prompt_prefix": "imported", "generated_tokens": 100, "internal_state": "imported_state"}
    dummy_model.import_kv_cache(imported_data)
    assert dummy_model.kv_cache_data["generated_tokens"] == 100

    # Test generate again to see if it uses the imported cache
    dummy_model.generate("Another prompt after importing cache.", **generation_config)

    # Test stream (also populates KV cache)
    streaming_config = {"temperature": 0.5, "top_p": 0.9}
    print("\nCollecting stream:")
    full_streamed_response = []
    for chunk in dummy_model.stream("This is a test prompt for streaming.", **streaming_config):
        print(f"Received chunk: '{chunk}'")
        full_streamed_response.append(chunk)
    print(f"Full streamed response: {''.join(full_streamed_response)}")

    exported_cache_after_stream = dummy_model.export_kv_cache()
    assert exported_cache_after_stream["generated_tokens"] == 10 # From dummy stream logic

    # Test preload_kv
    kv_config = {"user_id": "test_user"}
    dummy_model.preload_kv("This is a common prefix to preload.", **kv_config)
    assert dummy_model.kv_cache_data["preloaded_prompt"] == "This is a common prefix to preload."
    dummy_model.generate("Generate after preload.", **generation_config)


    print("\nBaseRunner and DummyRunner with KV cache methods demonstration complete.")
