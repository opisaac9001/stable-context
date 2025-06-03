# llm_context_os/runners/base.py
from abc import ABC, abstractmethod
import typing as t

class BaseRunner(ABC):
    """
    Abstract base class for all model runners.
    Defines the interface for generating text, streaming responses,
    managing KV cache, and LoRA adapter management.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: t.Any) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
        pass

    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        pass

    @abstractmethod
    def export_kv_cache(self) -> t.Any:
        pass

    @abstractmethod
    def import_kv_cache(self, cache_data: t.Any) -> None:
        pass

    # --- LoRA Adapter Methods ---
    @abstractmethod
    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        """
        Loads a LoRA adapter into the model.
        Args:
            adapter_id (str): A unique identifier for the LoRA adapter.
            adapter_path (str): Filesystem path or HF Hub identifier for the adapter.
            **kwargs: Additional runner-specific arguments (e.g., scaling factor).
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        pass

    @abstractmethod
    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        """
        Unloads a LoRA adapter from the model.
        Args:
            adapter_id (str): The unique identifier of the LoRA adapter to unload.
            **kwargs: Additional runner-specific arguments.
        Returns:
            bool: True if unloading was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_active_lora_adapters(self) -> t.List[str]:
        """
        Returns a list of unique identifiers for currently active LoRA adapters.
        Returns:
            t.List[str]: A list of active LoRA adapter IDs.
        """
        pass

    @abstractmethod
    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        """
        Merges one or more LoRA adapters into the base model weights.
        This is typically a permanent change for the current instance.
        Args:
            adapter_ids (t.List[str]): List of adapter IDs to merge.
            **kwargs: Runner-specific merge parameters (e.g., scaling, new save path).
        Returns:
            bool: True if merging was successful, False otherwise.
        """
        pass

    @abstractmethod
    def unmerge_lora_adapters(self, **kwargs) -> bool:
        """
        Unmerges LoRA adapters from the base model, if supported by the runner
        (e.g., by reloading original weights or specific unmerge functions).
        Args:
            **kwargs: Runner-specific unmerge parameters.
        Returns:
            bool: True if unmerging was successful, False otherwise.
        """
        pass


if __name__ == '__main__':
    class DummyRunner(BaseRunner):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.kv_cache_data: t.Optional[t.Any] = None
            self.active_loras: t.Dict[str, t.Dict[str, t.Any]] = {} # Store LoRAs by ID
            self.is_merged: bool = False
            print(f"DummyRunner initialized with model: {self.model_name}")

        def generate(self, prompt: str, **kwargs: t.Any) -> str:
            print(f"\n--- {self.model_name} Generating ---")
            print(f"Prompt: {prompt}")
            print(f"Config: {kwargs}")
            if self.active_loras:
                print(f"  Active LoRAs: {list(self.active_loras.keys())}")
            if self.is_merged:
                print("  (Model is LoRA-merged)")
            if self.kv_cache_data:
                print(f"  (Simulating using imported KV cache: {self.kv_cache_data})")
            response = f"Response from {self.model_name} to: '{prompt[:20]}...'"
            self.kv_cache_data = {"prompt_prefix": prompt[:10], "generated_tokens": 5}
            return response

        def stream(self, prompt: str, **kwargs: t.Any) -> t.Generator[str, None, None]:
            print(f"\n--- {self.model_name} Streaming ---")
            # ... (rest of stream implementation as before) ...
            if self.active_loras: print(f"  Active LoRAs: {list(self.active_loras.keys())}")
            if self.is_merged: print("  (Model is LoRA-merged)")
            yield f"Stream chunk 1 from {self.model_name} for '{prompt[:10]}...' "
            self.kv_cache_data = {"prompt_prefix": prompt[:15], "generated_tokens": 10}
            yield "Done."

        def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
            print(f"\n--- {self.model_name} Preloading KV Cache ---")
            self.kv_cache_data = {"preloaded_prompt": prompt}
            print(f"KV cache preloaded (simulated). Current cache: {self.kv_cache_data}")

        def export_kv_cache(self) -> t.Any:
            print(f"\n--- {self.model_name} Exporting KV Cache ---")
            return self.kv_cache_data

        def import_kv_cache(self, cache_data: t.Any) -> None:
            print(f"\n--- {self.model_name} Importing KV Cache ---")
            self.kv_cache_data = cache_data
            print(f"Imported cache data: {self.kv_cache_data}")

        # --- LoRA Dummy Implementations ---
        def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
            print(f"\n--- {self.model_name} Loading LoRA Adapter ---")
            print(f"  ID: {adapter_id}, Path: {adapter_path}, Params: {kwargs}")
            if adapter_id in self.active_loras:
                print(f"  Warning: LoRA adapter '{adapter_id}' already loaded.")
                return False # Or True if reloading is fine
            self.active_loras[adapter_id] = {"path": adapter_path, **kwargs, "status": "loaded"}
            print(f"  LoRA adapter '{adapter_id}' loaded successfully.")
            return True

        def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
            print(f"\n--- {self.model_name} Unloading LoRA Adapter ---")
            print(f"  ID: {adapter_id}, Params: {kwargs}")
            if adapter_id in self.active_loras:
                del self.active_loras[adapter_id]
                print(f"  LoRA adapter '{adapter_id}' unloaded successfully.")
                return True
            else:
                print(f"  Warning: LoRA adapter '{adapter_id}' not found.")
                return False

        def get_active_lora_adapters(self) -> t.List[str]:
            print(f"\n--- {self.model_name} Getting Active LoRA Adapters ---")
            adapter_ids = list(self.active_loras.keys())
            print(f"  Active adapters: {adapter_ids}")
            return adapter_ids

        def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
            print(f"\n--- {self.model_name} Merging LoRA Adapters ---")
            print(f"  IDs to merge: {adapter_ids}, Params: {kwargs}")
            # Simulate merging - in reality, this would modify base model weights
            # For dummy, just set a flag and clear active_loras if they are "consumed" by merge.
            valid_adapters = [aid for aid in adapter_ids if aid in self.active_loras]
            if not valid_adapters:
                print("  Error: No valid (loaded) adapters specified for merge.")
                return False

            print(f"  Merging adapters: {valid_adapters} into base model.")
            self.is_merged = True
            # Decide if merged LoRAs are still "active" or if they are consumed.
            # For simplicity, let's assume they are consumed and no longer individually active.
            for aid in valid_adapters:
                if aid in self.active_loras: del self.active_loras[aid]
            print("  Merge successful (simulated). Model is now LoRA-merged.")
            return True

        def unmerge_lora_adapters(self, **kwargs) -> bool:
            print(f"\n--- {self.model_name} Unmerging LoRA Adapters ---")
            print(f"  Params: {kwargs}")
            if not self.is_merged:
                print("  Model is not LoRA-merged. Nothing to unmerge.")
                return False
            self.is_merged = False
            # Unmerging might require reloading original weights. Here, just reset flag.
            print("  Unmerge successful (simulated). Model is no longer LoRA-merged.")
            # Active LoRAs are not restored by this dummy op, would need to be reloaded.
            return True

    # Example Usage
    dummy_lora_model = DummyRunner(model_name="TestModel-LoRA")
    dummy_lora_model.generate("Initial prompt before any LoRA.")

    print("\n--- LoRA Operations Demo ---")
    lora1_id = "lora_adapter_style_A"
    lora1_path = "/path/to/style_A_adapter"
    dummy_lora_model.load_lora_adapter(lora1_id, lora1_path, scaling=0.7)

    lora2_id = "lora_adapter_knowledge_B"
    lora2_id_path = "/path/to/knowledge_B_adapter"
    dummy_lora_model.load_lora_adapter(lora2_id, lora2_id_path, scaling=0.5)

    active_loras = dummy_lora_model.get_active_lora_adapters()
    assert lora1_id in active_loras and lora2_id in active_loras

    dummy_lora_model.generate("Prompt with LoRAs A and B active.")

    dummy_lora_model.unload_lora_adapter(lora1_id)
    active_loras_after_unload = dummy_lora_model.get_active_lora_adapters()
    assert lora1_id not in active_loras_after_unload and lora2_id in active_loras_after_unload

    dummy_lora_model.generate("Prompt with only LoRA B active.")

    # Merge remaining LoRA (LoRA B)
    dummy_lora_model.merge_lora_adapters([lora2_id])
    assert dummy_lora_model.is_merged
    assert not dummy_lora_model.get_active_lora_adapters() # LoRA B consumed by merge

    dummy_lora_model.generate("Prompt after LoRA B was merged.")

    # Try to load a new LoRA after merge - depends on runner impl if allowed
    lora3_id = "lora_adapter_C_after_merge"
    dummy_lora_model.load_lora_adapter(lora3_id, "/path/to/C", scaling=1.0)
    active_loras_after_merge_load = dummy_lora_model.get_active_lora_adapters()
    assert lora3_id in active_loras_after_merge_load

    dummy_lora_model.generate("Prompt with merged LoRA B and active LoRA C.")

    dummy_lora_model.unmerge_lora_adapters()
    assert not dummy_lora_model.is_merged
    # LoRA C might still be active or might need reloading depending on unmerge strategy
    print(f"Active LoRAs after unmerge: {dummy_lora_model.get_active_lora_adapters()}")


    print("\nBaseRunner and DummyRunner with LoRA methods demonstration complete.")
