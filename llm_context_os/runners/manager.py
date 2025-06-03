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
from .vllm_runner import VLLMRunner # Added
from llm_context_os.caching.kv_cache_manager import KVCacheManager

class ModelManager:
    """
    Manages the loading and retrieval of different model runners.
    Includes basic idle auto-unload and KV cache loading/saving functionality.
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
        """
        Loads a model runner, and optionally loads/generates/saves its KV cache for a default prefix.
        For 'speculative' type, expects 'draft_runner' and 'target_runner' in kwargs.
        For 'vllm' and 'api', 'api_url' can be in kwargs.
        """
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name/ID: {model_path_or_name}")
        if idle_unload_sec is not None:
            print(f"  Idle Unload Sec: {idle_unload_sec}")
        if default_prefix_text:
            print(f"  Default Prefix Text: '{default_prefix_text[:50]}...'")

        # Print kwargs, showing BaseRunner instances by type name for readability
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
                api_key = kwargs.pop('api_key', None) # Optional
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
            elif mt_lower == 'vllm': # Added vLLM
                api_url = kwargs.pop('api_url', "http://localhost:8000") # Default in VLLMRunner itself
                api_key = kwargs.pop('api_key', None) # Optional
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
                    f"@{model_path_or_name}" # Use the provided ID for the spec config
                )
            elif mt_lower == 'vllm':
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
                    # For speculative, cache operations target the 'target_runner'
                    # Construct a unique ID for the target_runner for caching purposes
                    target = self.current_runner.target_runner
                    if hasattr(target, 'model_path_or_repo_id'): cache_id = target.model_path_or_repo_id
                    elif hasattr(target, 'model_path'): cache_id = target.model_path
                    elif hasattr(target, 'model_name'): cache_id = target.model_name
                    else: cache_id = f"target_of_{self.current_model_identifier}" # Fallback cache ID
                    print(f"[ModelManager] Speculative mode: using target runner ID for cache: {cache_id}")


                loaded_cache = self.kv_cache_mgr.load_kv_cache(cache_id, default_prefix_text)

                if loaded_cache is not None:
                    print(f'[ModelManager] Found existing KV cache for prefix. Importing into runner.')
                    self.current_runner.import_kv_cache(loaded_cache)
                else:
                    print(f'[ModelManager] No existing KV cache. Preloading/generating in runner.')
                    self.current_runner.preload_kv(default_prefix_text)

                    exported_cache = self.current_runner.export_kv_cache()
                    if exported_cache is not None:
                        print(f'[ModelManager] Exported new KV cache from runner. Saving to disk for {cache_id}.')
                        self.kv_cache_mgr.save_kv_cache(exported_cache, cache_id, default_prefix_text)
                    else:
                        print(f'[ModelManager] Runner did not provide an exportable KV cache after preload (e.g., API/vLLM runners).')

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
        else:
            return None

    def unload(self) -> None:
        if self.current_runner:
            print(f"\nUnloading model: {self.current_model_identifier} ({self.current_model_type})")
            if hasattr(self.current_runner, '__del__'):
                try:
                    self.current_runner.__del__()
                except Exception as e:
                    print(f"Error during explicit runner __del__: {e}")
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

if __name__ == '__main__':
    kv_cache_dir_for_demo = Path("data/kv_cache_manager_demo_main") # Ensure Path is available
    if kv_cache_dir_for_demo.exists():
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up old demo cache dir: {kv_cache_dir_for_demo}")

    manager = ModelManager()
    manager.kv_cache_mgr = KVCacheManager(cache_dir=str(kv_cache_dir_for_demo))

    print(f"--- Initial state: Runner loaded? {manager.get() is not None} ---")

    # --- Demonstrate vLLM Runner Loading ---
    print("\n--- Loading vLLM Runner ---")
    vllm_model_id = "Mistral-7B-Instruct-v0.1-vLLM" # This is the model_name for VLLMRunner
    vllm_api_url = "http://localhost:2345/v1" # Example, if vLLM server is on non-default port or path

    manager.load(
        model_type='vllm',
        model_path_or_name=vllm_model_id,
        api_url=vllm_api_url, # Passed as kwarg, VLLMRunner's init will pick it up
        # No default_prefix_text for vLLM as KV cache is not client managed in this way
    )
    vllm_runner = manager.get()
    if vllm_runner:
        print(f"VLLM Runner loaded: {isinstance(vllm_runner, VLLMRunner)}")
        print(f"  VLLM Runner ID: {manager.current_model_identifier}")
        vllm_runner.generate("Test prompt for vLLM runner.", max_new_tokens=5)
    else:
        print(f"Failed to load VLLMRunner with ID '{vllm_model_id}'.")

    manager.unload() # Unload vLLM runner

    # --- Demonstrate Speculative Runner Loading (as before) ---
    print("\n--- Loading Speculative Runner ---")
    draft_runner_instance = APIRunner(model_name="dummy-draft-model", api_url="http://dummy/draft")
    target_runner_instance = LlamaCppRunner(model_path="dummy-target-model.gguf")
    speculative_id = "my_speculative_config"
    speculative_prefix = "This is the system prompt for the speculative model."

    manager.load(
        model_type='speculative',
        model_path_or_name=speculative_id,
        default_prefix_text=speculative_prefix,
        draft_runner=draft_runner_instance,
        target_runner=target_runner_instance,
        speculative_k=3
    )
    spec_runner = manager.get()
    if spec_runner:
        print(f"SpeculativeRunner loaded: {isinstance(spec_runner, SpeculativeRunner)}")
        spec_runner.generate("User asks a question to the speculative model.", max_new_tokens=10)
    else:
        print(f"Failed to load SpeculativeRunner with ID '{speculative_id}'.")

    print("\nModelManager with vLLM and SpeculativeRunner demo complete.")

    if kv_cache_dir_for_demo.exists(): # Cleanup
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up demo cache dir: {kv_cache_dir_for_demo}")
