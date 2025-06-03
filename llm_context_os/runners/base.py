# llm_context_os/runners/base.py
from abc import ABC, abstractmethod
import typing as t

class BaseRunner(ABC):
    """
    Abstract base class for all model runners.
    Defines the interface for generating text, streaming responses,
    managing KV cache, and LoRA adapter management.
    Now includes basic multimodal input capability via image_paths.
    """

    @abstractmethod
    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> str:
        """
        Generates a single text response from the given prompt and optional images.
        Args:
            prompt (str): The text prompt.
            image_paths (t.Optional[t.List[str]]): A list of paths to images relevant to the prompt.
            **kwargs: Additional generation parameters.
        Returns:
            str: The generated text response.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[str, None, None]:
        """
        Streams text responses from the given prompt and optional images.
        Args:
            prompt (str): The text prompt.
            image_paths (t.Optional[t.List[str]]): A list of paths to images relevant to the prompt.
            **kwargs: Additional generation parameters.
        Yields:
            str: Chunks of text as they are generated.
        """
        pass

    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None: # image_paths not typically part of KV preloading for text-based prefixes
        pass

    @abstractmethod
    def export_kv_cache(self) -> t.Any:
        pass

    @abstractmethod
    def import_kv_cache(self, cache_data: t.Any) -> None:
        pass

    @abstractmethod
    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        pass

    @abstractmethod
    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        pass

    @abstractmethod
    def get_active_lora_adapters(self) -> t.List[str]:
        pass

    @abstractmethod
    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        pass

    @abstractmethod
    def unmerge_lora_adapters(self, **kwargs) -> bool:
        pass


if __name__ == '__main__':
    class DummyRunner(BaseRunner):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.kv_cache_data: t.Optional[t.Any] = None
            self.active_loras: t.Dict[str, t.Dict[str, t.Any]] = {}
            self.is_merged: bool = False
            print(f"DummyRunner initialized with model: {self.model_name}")

        def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> str:
            print(f"\n--- {self.model_name} Generating ---")
            print(f"Prompt: {prompt}")
            if image_paths:
                print(f"  Image Paths: {image_paths}")
            print(f"Config: {kwargs}")
            if self.active_loras: print(f"  Active LoRAs: {list(self.active_loras.keys())}")
            if self.is_merged: print("  (Model is LoRA-merged)")
            if self.kv_cache_data: print(f"  (Simulating using imported KV cache: {self.kv_cache_data})")

            response_content = f"Response from {self.model_name} to: '{prompt[:20]}...'"
            if image_paths:
                response_content += f" (processed {len(image_paths)} image(s): {', '.join(image_paths)})"
            self.kv_cache_data = {"prompt_prefix": prompt[:10], "generated_tokens": 5}
            return response_content

        def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[str, None, None]:
            print(f"\n--- {self.model_name} Streaming ---")
            print(f"Prompt: {prompt}")
            if image_paths:
                print(f"  Image Paths: {image_paths}")
            if self.active_loras: print(f"  Active LoRAs: {list(self.active_loras.keys())}")
            if self.is_merged: print("  (Model is LoRA-merged)")

            yield f"Stream chunk 1 from {self.model_name} for '{prompt[:10]}...' "
            if image_paths:
                yield f"(saw images: {', '.join(image_paths)}) "
            self.kv_cache_data = {"prompt_prefix": prompt[:15], "generated_tokens": 10}
            yield "Done."

        def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
            print(f"\n--- {self.model_name} Preloading KV Cache (text prompt only) ---")
            self.kv_cache_data = {"preloaded_prompt": prompt}
            print(f"KV cache preloaded (simulated). Current cache: {self.kv_cache_data}")

        def export_kv_cache(self) -> t.Any:
            print(f"\n--- {self.model_name} Exporting KV Cache ---")
            return self.kv_cache_data

        def import_kv_cache(self, cache_data: t.Any) -> None:
            print(f"\n--- {self.model_name} Importing KV Cache ---")
            self.kv_cache_data = cache_data
            print(f"Imported cache data: {self.kv_cache_data}")

        def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
            print(f"\n--- {self.model_name} Loading LoRA: {adapter_id} ---")
            self.active_loras[adapter_id] = {"path": adapter_path, **kwargs}
            return True
        def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
            print(f"\n--- {self.model_name} Unloading LoRA: {adapter_id} ---")
            return self.active_loras.pop(adapter_id, None) is not None
        def get_active_lora_adapters(self) -> t.List[str]: return list(self.active_loras.keys())
        def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
            print(f"\n--- {self.model_name} Merging LoRAs: {adapter_ids} ---"); self.is_merged = True
            for aid in adapter_ids: self.active_loras.pop(aid, None)
            return True
        def unmerge_lora_adapters(self, **kwargs) -> bool:
            print(f"\n--- {self.model_name} Unmerging LoRAs ---"); self.is_merged = False
            return True

    # Example Usage
    dummy_multimodal_model = DummyRunner(model_name="TestModel-Multimodal")

    print("\n--- Multimodal Generate/Stream Demo ---")
    img_paths1 = ["/path/to/image_a.jpg", "/path/to/image_b.png"]
    dummy_multimodal_model.generate("Describe these images.", image_paths=img_paths1)

    img_paths2 = ["/path/to/image_c.jpeg"]
    stream_output = list(dummy_multimodal_model.stream("What about this one?", image_paths=img_paths2))
    print(f"Streamed output for multimodal: {stream_output}")

    # ... (rest of LoRA and KV cache demos can remain or be adapted)
    print("\nBaseRunner with multimodal and LoRA methods demonstration complete.")
