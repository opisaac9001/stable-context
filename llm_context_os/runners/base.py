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
    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
        """
        Generates a single text response from the given prompt and optional images.
        Args:
            prompt (str): The text prompt.
            image_paths (t.Optional[t.List[str]]): A list of paths to images relevant to the prompt.
            **kwargs: Additional generation parameters.
        Returns:
            t.Tuple[str, t.Dict[str, int]]: A tuple containing:
                - The generated text response (str).
                - A dictionary with token counts, e.g., {"prompt_tokens": X, "completion_tokens": Y}.
                  Returns {"prompt_tokens": 0, "completion_tokens": 0} if counting is not supported.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
        """
        Streams text responses from the given prompt and optional images.
        The first item yielded should be a dictionary with prompt token information.
        Subsequent items should be tuples of (text_chunk, tokens_in_chunk).

        Args:
            prompt (str): The text prompt.
            image_paths (t.Optional[t.List[str]]): A list of paths to images relevant to the prompt.
            **kwargs: Additional generation parameters.
        Yields:
            t.Union[t.Dict[str, int], t.Tuple[str, int]]:
                - First yield: `{"prompt_tokens": X}` (int is token count for the prompt).
                - Subsequent yields: `(text_chunk: str, tokens_in_chunk: int)`.
                  `tokens_in_chunk` is 0 if not supported for the chunk.
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

    @abstractmethod
    def count_tokens(self, text: str) -> Optional[int]:
        '''Counts the number of tokens in the given text using the runner's specific tokenizer.'''
        pass


if __name__ == '__main__':
    class DummyRunner(BaseRunner):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.kv_cache_data: t.Optional[t.Any] = None
            self.active_loras: t.Dict[str, t.Dict[str, t.Any]] = {}
            self.is_merged: bool = False
            print(f"DummyRunner initialized with model: {self.model_name}")

        def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
            print(f"\n--- {self.model_name} Generating ---")
            print(f"Prompt: {prompt}")
            if image_paths: print(f"  Image Paths: {image_paths}")
            print(f"Config: {kwargs}")
            if self.active_loras: print(f"  Active LoRAs: {list(self.active_loras.keys())}")
            if self.is_merged: print("  (Model is LoRA-merged)")
            if self.kv_cache_data: print(f"  (Simulating using imported KV cache: {self.kv_cache_data})")

            response_content = f"Response from {self.model_name} to: '{prompt[:20]}...'"
            if image_paths: response_content += f" (processed {len(image_paths)} image(s))"

            prompt_tokens_dummy = len(prompt.split()) # Dummy prompt token count
            completion_tokens_dummy = len(response_content.split()) # Dummy completion token count
            token_counts = {"prompt_tokens": prompt_tokens_dummy, "completion_tokens": completion_tokens_dummy}

            self.kv_cache_data = {"prompt_prefix": prompt[:10], "generated_tokens": completion_tokens_dummy}
            print(f"  Returning token_counts: {token_counts}")
            return response_content, token_counts

        def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
            print(f"\n--- {self.model_name} Streaming ---")
            print(f"Prompt: {prompt}")
            if image_paths: print(f"  Image Paths: {image_paths}")
            if self.active_loras: print(f"  Active LoRAs: {list(self.active_loras.keys())}")
            if self.is_merged: print("  (Model is LoRA-merged)")

            prompt_tokens_dummy = len(prompt.split())
            yield {"prompt_tokens": prompt_tokens_dummy}
            print(f"  Yielded prompt_tokens: {prompt_tokens_dummy}")

            chunk1_text = f"Stream chunk 1 from {self.model_name} for '{prompt[:10]}...' "
            chunk1_tokens = len(chunk1_text.split())
            yield (chunk1_text, chunk1_tokens)
            print(f"  Yielded chunk1: '{chunk1_text}' with {chunk1_tokens} tokens")

            if image_paths:
                chunk_img_text = f"(saw images: {', '.join(image_paths)}) "
                chunk_img_tokens = len(chunk_img_text.split())
                yield (chunk_img_text, chunk_img_tokens)
                print(f"  Yielded image chunk: '{chunk_img_text}' with {chunk_img_tokens} tokens")

            self.kv_cache_data = {"prompt_prefix": prompt[:15], "generated_tokens": 10} # Dummy total for stream

            done_text = "Done."
            done_tokens = len(done_text.split())
            yield (done_text, done_tokens)
            print(f"  Yielded final chunk: '{done_text}' with {done_tokens} tokens")


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

        def count_tokens(self, text: str) -> Optional[int]:
            # Simple word count for the dummy runner
            count = len(text.split())
            print(f"\n--- {self.model_name} Counting Tokens ---")
            print(f"Text: '{text[:30]}...' | Word Count: {count}")
            return count

    # Example Usage
    dummy_multimodal_model = DummyRunner(model_name="TestModel-Multimodal")

    print("\n--- Multimodal Generate/Stream Demo ---")
    img_paths1 = ["/path/to/image_a.jpg", "/path/to/image_b.png"]
    dummy_multimodal_model.generate("Describe these images.", image_paths=img_paths1)

    img_paths2 = ["/path/to/image_c.jpeg"]
    stream_output = list(dummy_multimodal_model.stream("What about this one?", image_paths=img_paths2))
    print(f"Streamed output for multimodal: {stream_output}")

    print("\n--- Token Counting Demo ---")
    text_to_count = "This is a sample sentence for token counting."
    token_count = dummy_multimodal_model.count_tokens(text_to_count)
    if token_count is not None:
        print(f"'{text_to_count}' has {token_count} tokens (dummy count).")
    else:
        print(f"Token counting not available for '{text_to_count}'.")


    # ... (rest of LoRA and KV cache demos can remain or be adapted)
    print("\nBaseRunner with multimodal and LoRA methods demonstration complete.")
