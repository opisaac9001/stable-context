# llm_context_os/runners/manager.py
import typing as t
import time
from pathlib import Path
import shutil

from .base import BaseRunner
from .api_runner import APIRunner
from .llama_cpp_runner import LlamaCppRunner
from .awq_runner import AWQRunner
from .exl2_runner import EXL2Runner
from .speculative_runner import SpeculativeRunner
from .vllm_runner import VLLMRunner
from llm_context_os.caching.kv_cache_manager import KVCacheManager

class ModelManager:
    """
    Manages the loading and retrieval of different model runners.
    Includes basic idle auto-unload, KV cache, and LoRA management functionality.
    """
    def __init__(self):
        self.current_runner: t.Optional[BaseRunner] = None
        self.current_model_type: t.Optional[str] = None
        self.current_model_identifier: t.Optional[str] = None

        self.idle_unload_sec: t.Optional[int] = None
        self.last_accessed_time: t.Optional[float] = None

        self.kv_cache_mgr = KVCacheManager()

        print("ModelManager initialized.")
        print(f"  KV Cache Manager using directory: {self.kv_cache_mgr.cache_dir.resolve()}")


    def load(self,
             model_type: str,
             model_path_or_name: str,
             idle_unload_sec: t.Optional[int] = None,
             default_prefix_text: t.Optional[str] = None,
             **kwargs: t.Any) -> None:
        # ... (load method remains the same as the last version) ...
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name/ID: {model_path_or_name}")
        if idle_unload_sec is not None:
            print(f"  Idle Unload Sec: {idle_unload_sec}")
        if default_prefix_text:
            print(f"  Default Prefix Text: '{default_prefix_text[:50]}...'")

        processed_kwargs_for_print = {}
        for k, v in kwargs.items():
            if isinstance(v, BaseRunner):
                processed_kwargs_for_print[k] = type(v).__name__
            else:
                processed_kwargs_for_print[k] = v
        print(f"  Runner Constructor Params (kwargs): {processed_kwargs_for_print}")

        if self.current_runner:
            self.unload()

        try:
            runner_to_load: t.Optional[BaseRunner] = None
            mt_lower = model_type.lower()

            if mt_lower == 'api':
                api_url = kwargs.pop('api_url', None)
                api_key = kwargs.pop('api_key', None)
                if not api_url:
                    print("Error: 'api_url' is required for API runner.")
                    return
                runner_to_load = APIRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **kwargs)
            elif mt_lower == 'gguf':
                runner_to_load = LlamaCppRunner(model_path=model_path_or_name, **kwargs)
            elif mt_lower == 'awq':
                runner_to_load = AWQRunner(model_path_or_repo_id=model_path_or_name, **kwargs)
            elif mt_lower == 'exl2':
                runner_to_load = EXL2Runner(model_path=model_path_or_name, **kwargs)
            elif mt_lower == 'vllm':
                api_url = kwargs.pop('api_url', "http://localhost:8000")
                api_key = kwargs.pop('api_key', None)
                runner_to_load = VLLMRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **kwargs)
            elif mt_lower == 'speculative':
                draft_runner = kwargs.get('draft_runner')
                target_runner = kwargs.get('target_runner')
                speculative_k = kwargs.get('speculative_k', 5)

                if isinstance(draft_runner, BaseRunner) and isinstance(target_runner, BaseRunner):
                    runner_to_load = SpeculativeRunner(
                        draft_runner=draft_runner,
                        target_runner=target_runner,
                        speculative_k=speculative_k
                    )
                else:
                    print("Error: SpeculativeRunner requires pre-instantiated 'draft_runner' and 'target_runner' (BaseRunner instances) in parameters.")
                    return
            else:
                print(f"Error: Unknown model type '{model_type}'. No model loaded.")
                return

            self.current_runner = runner_to_load
            self.current_model_type = mt_lower

            if mt_lower == 'speculative' and isinstance(self.current_runner, SpeculativeRunner):
                 self.current_model_identifier = (
                    f"speculative(draft={type(self.current_runner.draft_runner).__name__},"
                    f"target={type(self.current_runner.target_runner).__name__})"
                    f"@{model_path_or_name}"
                )
            elif mt_lower == 'vllm' and hasattr(self.current_runner, 'api_url'):
                self.current_model_identifier = f'vllm_model({model_path_or_name} @ {self.current_runner.api_url})'
            else:
                self.current_model_identifier = model_path_or_name

            self.idle_unload_sec = idle_unload_sec
            self.last_accessed_time = time.time()
            print(f"Successfully instantiated {self.current_model_type} runner for: {self.current_model_identifier}")

            if default_prefix_text and self.current_runner:
                print(f'[ModelManager] Processing KV cache for prefix: "{default_prefix_text[:50]}..." for model {self.current_model_identifier}')
                cache_id = self.current_model_identifier
                if mt_lower == 'speculative' and isinstance(self.current_runner, SpeculativeRunner):
                    target = self.current_runner.target_runner
                    if hasattr(target, 'model_path_or_repo_id'): cache_id = target.model_path_or_repo_id
                    elif hasattr(target, 'model_path'): cache_id = target.model_path
                    elif hasattr(target, 'model_name'): cache_id = target.model_name
                    else: cache_id = f"target_of_{self.current_model_identifier}"
                    print(f"[ModelManager] Speculative mode: using target runner ID for cache: {cache_id}")

                loaded_cache = self.kv_cache_mgr.load_kv_cache(cache_id, default_prefix_text)
                if loaded_cache is not None:
                    print(f'[ModelManager] Found existing KV cache. Importing into runner.')
                    self.current_runner.import_kv_cache(loaded_cache)
                else:
                    print(f'[ModelManager] No existing KV cache. Preloading/generating in runner.')
                    self.current_runner.preload_kv(default_prefix_text)
                    exported_cache = self.current_runner.export_kv_cache()
                    if exported_cache is not None:
                        print(f'[ModelManager] Exported KV cache from runner. Saving for {cache_id}.')
                        self.kv_cache_mgr.save_kv_cache(exported_cache, cache_id, default_prefix_text)
                    else:
                        print(f'[ModelManager] Runner did not provide exportable KV cache after preload.')
            print(f"Model '{self.current_model_identifier}' fully ready.")
        except Exception as e:
            print(f"Error during model loading or runner initialization: {e}")
            self.current_runner = None
            self.current_model_type = None
            self.current_model_identifier = None
            self.idle_unload_sec = None
            self.last_accessed_time = None

    def get(self) -> t.Optional[BaseRunner]:
        if self.current_runner:
            self.last_accessed_time = time.time()
            return self.current_runner
        return None

    def unload(self) -> None:
        if self.current_runner:
            print(f"\nUnloading model: {self.current_model_identifier} ({self.current_model_type})")
            if hasattr(self.current_runner, '__del__'):
                try: self.current_runner.__del__()
                except Exception as e: print(f"Error during runner __del__: {e}")
        self.current_runner = None
        self.current_model_type = None
        self.current_model_identifier = None
        self.idle_unload_sec = None
        self.last_accessed_time = None

    def check_idle(self) -> bool:
        if self.current_runner and \
           self.idle_unload_sec is not None and self.idle_unload_sec > 0 and \
           self.last_accessed_time is not None:
            idle_time = time.time() - self.last_accessed_time
            if idle_time > self.idle_unload_sec:
                print(f"\nModel '{self.current_model_identifier}' idle for {idle_time:.2f}s (limit: {self.idle_unload_sec}s), unloading.")
                self.unload()
                return True
        return False

    # --- LoRA Management Methods ---
    def load_lora_on_current_runner(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"[ModelManager] Attempting to load LoRA '{adapter_id}' from '{adapter_path}' onto current runner.")
        if not self.current_runner:
            print("[ModelManager] Error: No model currently loaded. Cannot load LoRA.")
            return False
        self.last_accessed_time = time.time() # Consider LoRA load an access
        return self.current_runner.load_lora_adapter(adapter_id, adapter_path, **kwargs)

    def unload_lora_on_current_runner(self, adapter_id: str, **kwargs) -> bool:
        print(f"[ModelManager] Attempting to unload LoRA '{adapter_id}' from current runner.")
        if not self.current_runner:
            print("[ModelManager] Error: No model currently loaded. Cannot unload LoRA.")
            return False
        self.last_accessed_time = time.time()
        return self.current_runner.unload_lora_adapter(adapter_id, **kwargs)

    def get_active_loras_on_current_runner(self) -> t.List[str]:
        print(f"[ModelManager] Getting active LoRAs from current runner.")
        if not self.current_runner:
            print("[ModelManager] Error: No model currently loaded.")
            return []
        self.last_accessed_time = time.time()
        return self.current_runner.get_active_lora_adapters()

    def merge_loras_on_current_runner(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"[ModelManager] Attempting to merge LoRAs {adapter_ids} on current runner.")
        if not self.current_runner:
            print("[ModelManager] Error: No model currently loaded. Cannot merge LoRAs.")
            return False
        self.last_accessed_time = time.time()
        return self.current_runner.merge_lora_adapters(adapter_ids, **kwargs)

    def unmerge_loras_on_current_runner(self, **kwargs) -> bool:
        print(f"[ModelManager] Attempting to unmerge LoRAs on current runner.")
        if not self.current_runner:
            print("[ModelManager] Error: No model currently loaded. Cannot unmerge LoRAs.")
            return False
        self.last_accessed_time = time.time()
        return self.current_runner.unmerge_lora_adapters(**kwargs)


if __name__ == '__main__':
    kv_cache_dir_for_demo = Path("data/kv_cache_manager_demo_main")
    if kv_cache_dir_for_demo.exists():
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up old demo cache dir: {kv_cache_dir_for_demo}")

    manager = ModelManager()
    manager.kv_cache_mgr = KVCacheManager(cache_dir=str(kv_cache_dir_for_demo))

    print(f"--- Initial state: Runner loaded? {manager.get() is not None} ---")

    # --- Demonstrate LoRA Management with a compatible runner (e.g., LlamaCppRunner placeholder) ---
    print("\n--- Loading LlamaCppRunner for LoRA Demo ---")
    gguf_model_path = "dummy-lora-test-model.gguf"
    manager.load(
        model_type='gguf',
        model_path_or_name=gguf_model_path,
        n_gpu_layers=0 # Example kwarg for LlamaCppRunner
    )

    current_runner = manager.get()
    if current_runner:
        print(f"Runner for LoRA demo: {type(current_runner).__name__}")

        lora_id1 = "style_adapter_1"
        lora_path1 = "/path/to/style_adapter_1"
        print(f"\nLoading LoRA: {lora_id1}")
        load_ok = manager.load_lora_on_current_runner(lora_id1, lora_path1, alpha=0.7)
        print(f"LoRA load status: {load_ok}")

        active_loras = manager.get_active_loras_on_current_runner()
        print(f"Active LoRAs: {active_loras}")
        assert lora_id1 in active_loras if load_ok else lora_id1 not in active_loras

        current_runner.generate("Test prompt with LoRA loaded.", max_tokens=5)

        print(f"\nMerging LoRAs: {[lora_id1]}")
        merge_ok = manager.merge_loras_on_current_runner([lora_id1])
        print(f"LoRA merge status: {merge_ok}")
        if merge_ok:
             active_loras_after_merge = manager.get_active_loras_on_current_runner()
             print(f"Active LoRAs after merge: {active_loras_after_merge}")
             assert lora_id1 not in active_loras_after_merge # Assuming merge consumes it

        current_runner.generate("Test prompt after LoRA merge.", max_tokens=5)

        print(f"\nUnmerging LoRAs")
        unmerge_ok = manager.unmerge_loras_on_current_runner()
        print(f"LoRA unmerge status: {unmerge_ok}")

        current_runner.generate("Test prompt after LoRA unmerge.", max_tokens=5)

        print(f"\nUnloading LoRA: {lora_id1}") # Might fail if merge consumed it and unmerge doesn't restore
        unload_ok = manager.unload_lora_on_current_runner(lora_id1)
        print(f"LoRA unload status: {unload_ok}")
        active_loras_final = manager.get_active_loras_on_current_runner()
        print(f"Final active LoRAs: {active_loras_final}")
        assert lora_id1 not in active_loras_final

    else:
        print("Failed to load GGUF runner for LoRA demo.")

    print("\nModelManager LoRA demo complete.")

    if kv_cache_dir_for_demo.exists(): # Cleanup
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up demo cache dir: {kv_cache_dir_for_demo}")
