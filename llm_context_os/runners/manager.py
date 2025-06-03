# llm_context_os/runners/manager.py
import typing as t
import time
from pathlib import Path # Ensure Path is imported if used in __main__ for cleanup
import shutil # Ensure shutil is imported if used in __main__ for cleanup

from .base import BaseRunner
from .api_runner import APIRunner
from .llama_cpp_runner import LlamaCppRunner
from .awq_runner import AWQRunner
from .exl2_runner import EXL2Runner
from .speculative_runner import SpeculativeRunner # Added
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
             model_path_or_name: str, # For 'speculative', this could be a config name or descriptive ID
             idle_unload_sec: t.Optional[int] = None,
             default_prefix_text: t.Optional[str] = None,
             **kwargs: t.Any) -> None: # runner_constructor_params are in kwargs
        """
        Loads a model runner, and optionally loads/generates/saves its KV cache for a default prefix.
        For 'speculative' type, expects 'draft_runner' and 'target_runner' in kwargs.
        """
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name/ID: {model_path_or_name}") # Changed label for clarity
        if idle_unload_sec is not None:
            print(f"  Idle Unload Sec: {idle_unload_sec}")
        if default_prefix_text:
            print(f"  Default Prefix Text: '{default_prefix_text[:50]}...'")
        print(f"  Runner Constructor Params (kwargs): { {k:type(v).__name__ if isinstance(v, BaseRunner) else v for k,v in kwargs.items()} }") # Print types for runners

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
            elif mt_lower == 'speculative':
                draft_runner = kwargs.get('draft_runner')
                target_runner = kwargs.get('target_runner')
                speculative_k = kwargs.get('speculative_k', 5) # Default k if not provided

                if isinstance(draft_runner, BaseRunner) and isinstance(target_runner, BaseRunner):
                    runner_to_load = SpeculativeRunner(
                        draft_runner=draft_runner,
                        target_runner=target_runner,
                        speculative_k=speculative_k
                    )
                    # model_path_or_name for speculative could be a descriptive ID like "spec(draft_model_id,target_model_id)"
                    # For now, we'll use the one passed.
                else:
                    print("Error: SpeculativeRunner requires pre-instantiated 'draft_runner' and 'target_runner' (BaseRunner instances) in parameters.")
                    return
            else:
                print(f"Error: Unknown model type '{model_type}'. No model loaded.")
                return

            self.current_runner = runner_to_load
            self.current_model_type = mt_lower
            # For speculative, model_path_or_name might be a config name.
            # A more detailed identifier could be constructed if needed.
            if mt_lower == 'speculative' and isinstance(self.current_runner, SpeculativeRunner):
                 self.current_model_identifier = (
                    f"speculative(draft={type(self.current_runner.draft_runner).__name__}@{model_path_or_name},"
                    f"target={type(self.current_runner.target_runner).__name__})"
                )
            else:
                self.current_model_identifier = model_path_or_name

            self.idle_unload_sec = idle_unload_sec
            self.last_accessed_time = time.time()
            print(f"Successfully instantiated {self.current_model_type} runner for: {self.current_model_identifier}")

            # KV Cache handling (not typically used for the SpeculativeRunner itself, but for its components)
            # If default_prefix_text is provided for a SpeculativeRunner, it will try to load/save
            # cache for the *target_runner* as per SpeculativeRunner's export/import_kv_cache logic.
            if default_prefix_text and self.current_runner:
                print(f'[ModelManager] Processing KV cache for prefix: "{default_prefix_text[:50]}..." for model {self.current_model_identifier}')
                # For speculative, the model_identifier used for caching should ideally be unique to the target model,
                # or the cache key should include target model info.
                # KVCacheManager uses the model_identifier passed to it.
                # If SpeculativeRunner's identifier is used, cache might clash if different target models are used
                # with same speculative config name.
                # For now, using self.current_model_identifier which is now more descriptive for speculative.
                # Or, better: use target_runner's identifier if available (needs it to have one)
                cache_id_for_speculative = self.current_model_identifier
                if mt_lower == 'speculative' and hasattr(self.current_runner.target_runner, 'model_path_or_repo_id'):
                    cache_id_for_speculative = self.current_runner.target_runner.model_path_or_repo_id
                elif mt_lower == 'speculative' and hasattr(self.current_runner.target_runner, 'model_path'):
                    cache_id_for_speculative = self.current_runner.target_runner.model_path
                elif mt_lower == 'speculative' and hasattr(self.current_runner.target_runner, 'model_name'):
                     cache_id_for_speculative = self.current_runner.target_runner.model_name


                loaded_cache = self.kv_cache_mgr.load_kv_cache(cache_id_for_speculative, default_prefix_text)

                if loaded_cache is not None:
                    print(f'[ModelManager] Found existing KV cache for prefix. Importing into runner (delegated by SpeculativeRunner).')
                    self.current_runner.import_kv_cache(loaded_cache) # SpeculativeRunner handles distributing this
                else:
                    print(f'[ModelManager] No existing KV cache. Preloading/generating in runner (delegated by SpeculativeRunner).')
                    self.current_runner.preload_kv(default_prefix_text)

                    exported_cache = self.current_runner.export_kv_cache() # SpeculativeRunner exports target's cache
                    if exported_cache is not None:
                        print(f'[ModelManager] Exported new KV cache from runner. Saving to disk for {cache_id_for_speculative}.')
                        self.kv_cache_mgr.save_kv_cache(exported_cache, cache_id_for_speculative, default_prefix_text)
                    else:
                        print(f'[ModelManager] Runner did not provide an exportable KV cache after preload.')

            print(f"Model '{self.current_model_identifier}' fully ready.")

        except Exception as e:
            print(f"Error during model loading or runner initialization: {e}")
            self.current_runner = None
            self.current_model_type = None
            self.current_model_identifier = None
            self.idle_unload_sec = None
            self.last_accessed_time = None

    def get(self) -> t.Optional[BaseRunner]:
        # (Previous get, unload, check_idle methods remain unchanged)
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
    kv_cache_dir_for_demo = "data/kv_cache_manager_demo_main"
    if Path(kv_cache_dir_for_demo).exists():
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up old demo cache dir: {kv_cache_dir_for_demo}")

    manager = ModelManager()
    # Override KVCacheManager to use a specific test directory for this demo
    manager.kv_cache_mgr = KVCacheManager(cache_dir=kv_cache_dir_for_demo)

    print(f"--- Initial state: Runner loaded? {manager.get() is not None} ---")

    # --- Demonstrate Speculative Runner Loading ---
    print("\n--- Loading Speculative Runner ---")
    # 1. Instantiate draft and target runners (using placeholders for this demo)
    # For real use, these would be actual LlamaCppRunner, AWQRunner, etc.
    # Using APIRunner as a lightweight placeholder for draft/target for this demo.
    draft_runner_instance = APIRunner(model_name="dummy-draft-model", api_url="http://dummy/draft")
    target_runner_instance = LlamaCppRunner(model_path="dummy-target-model.gguf") # Target often a GGUF/EXL2

    speculative_id = "my_speculative_config"
    speculative_prefix = "This is the system prompt for the speculative model."

    manager.load(
        model_type='speculative',
        model_path_or_name=speculative_id, # Name for this speculative configuration
        default_prefix_text=speculative_prefix, # Prefix for KV caching (applies to target runner)
        # Runner constructor params for SpeculativeRunner:
        draft_runner=draft_runner_instance,
        target_runner=target_runner_instance,
        speculative_k=3
    )

    spec_runner = manager.get()
    if spec_runner:
        print(f"SpeculativeRunner loaded: {isinstance(spec_runner, SpeculativeRunner)}")
        print(f"  SpeculativeRunner ID: {manager.current_model_identifier}")
        # Test generate with speculative runner
        spec_runner.generate("User asks a question to the speculative model.", max_new_tokens=10)

        # Test KV cache export (delegates to target runner)
        kv_exported = spec_runner.export_kv_cache()
        print(f"KV cache exported from spec_runner (target's cache): {str(kv_exported)[:100]}...")

    else:
        print(f"Failed to load SpeculativeRunner with ID '{speculative_id}'.")

    # --- Demonstrate KV Caching with Speculative Runner (interaction with target runner's cache) ---
    print("\n--- Unloading and Reloading Speculative Runner to test KV cache for target ---")
    manager.unload()

    # Re-instantiate draft/target for a clean load test if their state was modified by first load
    # (Our placeholder runners' states are modified, so re-instantiate)
    draft_runner_instance_2 = APIRunner(model_name="dummy-draft-model", api_url="http://dummy/draft")
    target_runner_instance_2 = LlamaCppRunner(model_path="dummy-target-model.gguf")


    manager.load(
        model_type='speculative',
        model_path_or_name=speculative_id,
        default_prefix_text=speculative_prefix, # Same prefix, should load target's cache
        draft_runner=draft_runner_instance_2,
        target_runner=target_runner_instance_2,
        speculative_k=3
    )
    spec_runner_reloaded = manager.get()
    if spec_runner_reloaded:
        print(f"SpeculativeRunner reloaded: {isinstance(spec_runner_reloaded, SpeculativeRunner)}")
        # Generate again; the LlamaCppRunner (target) should indicate if it used a preloaded/imported cache
        spec_runner_reloaded.generate("Another user question after reloading.", max_new_tokens=10)
    else:
        print(f"Failed to reload SpeculativeRunner with ID '{speculative_id}'.")

    print("\nModelManager with SpeculativeRunner demo complete.")

    if Path(kv_cache_dir_for_demo).exists():
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up demo cache dir: {kv_cache_dir_for_demo}")
