# llm_context_os/runners/manager.py
import typing as t
import time

from .base import BaseRunner
from .api_runner import APIRunner
from .llama_cpp_runner import LlamaCppRunner
from .awq_runner import AWQRunner
from .exl2_runner import EXL2Runner
from llm_context_os.caching.kv_cache_manager import KVCacheManager # Added

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

        self.kv_cache_mgr = KVCacheManager() # Instantiate KVCacheManager

        print("ModelManager initialized.")
        print(f"  KV Cache Manager using directory: {self.kv_cache_mgr.cache_dir.resolve()}")


    def load(self,
             model_type: str,
             model_path_or_name: str,
             idle_unload_sec: t.Optional[int] = None,
             default_prefix_text: t.Optional[str] = None, # New parameter
             **kwargs: t.Any) -> None:
        """
        Loads a model runner, and optionally loads/generates/saves its KV cache for a default prefix.
        """
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name: {model_path_or_name}")
        if idle_unload_sec is not None:
            print(f"  Idle Unload Sec: {idle_unload_sec}")
        if default_prefix_text:
            print(f"  Default Prefix Text: '{default_prefix_text[:50]}...'")
        print(f"  Additional args: {kwargs}")

        if self.current_runner:
            self.unload()

        try:
            # Instantiate the runner
            if model_type.lower() == 'api':
                api_url = kwargs.pop('api_url', None)
                api_key = kwargs.pop('api_key', None)
                if not api_url:
                    print("Error: 'api_url' is required for API runner.")
                    return
                self.current_runner = APIRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **kwargs)
            elif model_type.lower() == 'gguf':
                self.current_runner = LlamaCppRunner(model_path=model_path_or_name, **kwargs)
            elif model_type.lower() == 'awq':
                self.current_runner = AWQRunner(model_path_or_repo_id=model_path_or_name, **kwargs)
            elif model_type.lower() == 'exl2':
                self.current_runner = EXL2Runner(model_path=model_path_or_name, **kwargs)
            else:
                print(f"Error: Unknown model type '{model_type}'. No model loaded.")
                self.current_runner = None
                return

            # Store model info
            self.current_model_type = model_type.lower()
            self.current_model_identifier = model_path_or_name
            self.idle_unload_sec = idle_unload_sec
            self.last_accessed_time = time.time()
            print(f"Successfully instantiated {self.current_model_type} runner for: {self.current_model_identifier}")

            # KV Cache handling for the default_prefix_text
            if default_prefix_text and self.current_runner:
                print(f'[ModelManager] Processing KV cache for prefix: "{default_prefix_text[:50]}..."')
                loaded_cache = self.kv_cache_mgr.load_kv_cache(self.current_model_identifier, default_prefix_text)

                if loaded_cache is not None:
                    print(f'[ModelManager] Found existing KV cache for prefix. Importing into runner.')
                    self.current_runner.import_kv_cache(loaded_cache)
                else:
                    print(f'[ModelManager] No existing KV cache for prefix. Preloading/generating in runner.')
                    self.current_runner.preload_kv(default_prefix_text) # Runner populates its internal KV cache

                    exported_cache = self.current_runner.export_kv_cache()
                    if exported_cache is not None:
                        print(f'[ModelManager] Exported new KV cache from runner. Saving to disk.')
                        self.kv_cache_mgr.save_kv_cache(exported_cache, self.current_model_identifier, default_prefix_text)
                    else:
                        print(f'[ModelManager] Runner did not provide an exportable KV cache after preload (common for API runners).')

            print(f"Model '{self.current_model_identifier}' fully ready.")


        except Exception as e:
            print(f"Error during model loading or runner initialization: {e}")
            # Ensure partial state is cleared
            self.current_runner = None
            self.current_model_type = None
            self.current_model_identifier = None
            self.idle_unload_sec = None
            self.last_accessed_time = None

    def get(self) -> t.Optional[BaseRunner]:
        if self.current_runner:
            # print(f"\nRetrieved current runner: {self.current_model_identifier} ({self.current_model_type})") # Less verbose for get
            self.last_accessed_time = time.time()
            return self.current_runner
        else:
            # print("Error: No model is currently loaded. Call load() first.") # Less verbose
            return None

    def unload(self) -> None:
        if self.current_runner:
            print(f"\nUnloading model: {self.current_model_identifier} ({self.current_model_type})")
            if hasattr(self.current_runner, '__del__'):
                try:
                    self.current_runner.__del__()
                except Exception as e:
                    print(f"Error during explicit runner __del__: {e}")
        else:
            # print("No model to unload.") # Less verbose if called when already None
            pass

        self.current_runner = None
        self.current_model_type = None
        self.current_model_identifier = None
        self.idle_unload_sec = None
        self.last_accessed_time = None
        # print("Model unloaded and manager state reset.") # Less verbose

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
    # Use a test-specific cache directory for the demo
    # Note: KVCacheManager constructor will create it.
    # For repeated runs, you might want to clean this dir before/after.
    kv_cache_dir_for_demo = "data/kv_cache_manager_demo"

    # Clean up previous demo cache if it exists, for a clean run
    import shutil
    if Path(kv_cache_dir_for_demo).exists():
        shutil.rmtree(kv_cache_dir_for_demo)
        print(f"Cleaned up old demo cache dir: {kv_cache_dir_for_demo}")

    manager = ModelManager()
    manager.kv_cache_mgr = KVCacheManager(cache_dir=kv_cache_dir_for_demo) # Override for demo

    print(f"--- Initial state: Runner loaded? {manager.get() is not None} ---")

    model_id_for_kv_test = "gguf_model_for_kv_test.gguf"
    common_prefix = "The story of the three little pigs is a classic tale."

    # --- First Load: Cache should be generated and saved ---
    print("\n--- First Load: Generating and Saving KV Cache ---")
    manager.load(
        model_type='gguf',
        model_path_or_name=model_id_for_kv_test,
        default_prefix_text=common_prefix,
        n_gpu_layers=0 # Placeholder GGUF param
    )
    runner1 = manager.get()
    if runner1:
        print(f"Runner1 loaded: {type(runner1)}")
        # Simulate a generation that might use/confirm the preloaded cache
        runner1.generate(common_prefix + " The first pig built his house of straw.", max_tokens=5)
    else:
        print("Failed to load runner1.")

    # --- Unload the model ---
    print("\n--- Unloading the model ---")
    manager.unload()
    print(f"Runner after unload: {manager.get() is None}")

    # --- Second Load: Cache should be loaded from disk ---
    print("\n--- Second Load: Loading KV Cache from Disk ---")
    manager.load(
        model_type='gguf',
        model_path_or_name=model_id_for_kv_test,
        default_prefix_text=common_prefix,
        n_gpu_layers=0
    )
    runner2 = manager.get()
    if runner2:
        print(f"Runner2 loaded: {type(runner2)}")
        # Simulate a generation. If KV cache was loaded, runner2 might log it (placeholder runners do).
        runner2.generate(common_prefix + " The second pig built his house of sticks.", max_tokens=5)
    else:
        print("Failed to load runner2.")

    # --- Test Idle Unload with KV cache ---
    print("\n--- Testing Idle Unload with KV Cache Model ---")
    idle_test_model_id = "api_model_for_idle_kv.gguf" # Use API as it doesn't export cache
    idle_prefix = "This is a system prompt for an idle test."
    idle_wait_seconds = 1

    manager.load(
        model_type='api', # API runner does not export cache, so preload won't save.
        model_path_or_name=idle_test_model_id,
        api_url="http://dummy.api.kv/v1", # Required for APIRunner
        default_prefix_text=idle_prefix,
        idle_unload_sec=idle_wait_seconds
    )
    idle_runner = manager.get()
    if idle_runner:
        print(f"Idle Runner '{idle_runner.model_name if hasattr(idle_runner, 'model_name') else 'N/A'}' loaded. Waiting for {idle_wait_seconds + 1}s...")
        time.sleep(idle_wait_seconds + 1)
        unloaded = manager.check_idle()
        print(f"Model unloaded due to idle: {unloaded}")
        print(f"Runner after idle: {manager.get() is None}")
    else:
        print("Failed to load idle_runner.")

    print("\nModelManager KV Cache and Idle demo complete.")

    # Optional: Clean up the test cache directory after demo
    # if Path(kv_cache_dir_for_demo).exists():
    #     shutil.rmtree(kv_cache_dir_for_demo)
    #     print(f"Cleaned up demo cache dir: {kv_cache_dir_for_demo}")
