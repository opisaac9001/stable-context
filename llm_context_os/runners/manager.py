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
from .llava_cpp_runner import LlavaCppRunner
from llm_context_os.caching.kv_cache_manager import KVCacheManager
from llm_context_os.tuning.auto_tuner import AutoTuner # Added

class ModelManager:
    """
    Manages the loading and retrieval of different model runners.
    Includes basic idle auto-unload, KV cache, LoRA management, and auto-tuning functionality.
    """
    def __init__(self):
        self.current_runner: t.Optional[BaseRunner] = None
        self.current_model_type: t.Optional[str] = None
        self.current_model_identifier: t.Optional[str] = None

        self.idle_unload_sec: t.Optional[int] = None
        self.last_accessed_time: t.Optional[float] = None

        self.kv_cache_mgr = KVCacheManager()
        self.auto_tuner = AutoTuner() # Added AutoTuner instance

        print("ModelManager initialized.")
        print(f"  KV Cache Manager using directory: {self.kv_cache_mgr.cache_dir.resolve()}")
        print(f"  AutoTuner using results path: {self.auto_tuner.results_path.resolve()}")


    def load(self,
             model_type: str,
             model_path_or_name: str,
             idle_unload_sec: t.Optional[int] = None,
             default_prefix_text: t.Optional[str] = None,
             auto_tune: bool = False, # New parameter for auto-tuning
             **kwargs: t.Any) -> None:
        """
        Loads a model runner.
        Optionally uses AutoTuner to find optimal parameters for certain model types.
        **kwargs are passed as runner_constructor_params.
        """
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name/ID: {model_path_or_name}")
        if idle_unload_sec is not None: print(f"  Idle Unload Sec: {idle_unload_sec}")
        if default_prefix_text: print(f"  Default Prefix Text: '{default_prefix_text[:50]}...'")
        if auto_tune: print(f"  Auto-Tune Enabled: {auto_tune}")

        # Separate runner_params from kwargs for clarity before auto-tuning potentially modifies them
        runner_params = kwargs.copy()

        processed_kwargs_for_print = {}
        for k, v in runner_params.items(): # Use runner_params for printing
            if isinstance(v, BaseRunner):
                processed_kwargs_for_print[k] = type(v).__name__
            else:
                processed_kwargs_for_print[k] = v
        print(f"  Initial Runner Constructor Params: {processed_kwargs_for_print}")

        if self.current_runner:
            self.unload()

        try:
            runner_to_load: t.Optional[BaseRunner] = None
            mt_lower = model_type.lower()

            # Auto-tuning step (before runner instantiation for relevant types)
            if auto_tune and self.auto_tuner and mt_lower in ['gguf', 'llava_cpp', 'awq', 'exl2', 'llama_cpp']:
                print(f'[ModelManager] Auto-tuning enabled for {model_path_or_name} ({mt_lower})...')
                optimal_params = self.auto_tuner.get_optimal_settings(model_path_or_name, mt_lower)
                if optimal_params:
                    print(f'[ModelManager] Applying auto-tuned optimal_params: {optimal_params}')
                    # Update runner_params, optimal_params take precedence over initially passed kwargs
                    runner_params.update(optimal_params)
                    print(f"[ModelManager] Runner params after auto-tuning: {runner_params}")
                else:
                    print('[ModelManager] AutoTuner did not return optimal parameters. Using provided or default params.')

            # Instantiate the runner using potentially updated runner_params
            if mt_lower == 'api':
                api_url = runner_params.pop('api_url', None)
                api_key = runner_params.pop('api_key', None)
                if not api_url:
                    print("Error: 'api_url' is required for API runner.")
                    return
                runner_to_load = APIRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **runner_params)
            elif mt_lower == 'gguf' or mt_lower == 'llama_cpp': # llama_cpp as alias for gguf type
                runner_to_load = LlamaCppRunner(model_path=model_path_or_name, **runner_params)
            elif mt_lower == 'awq':
                runner_to_load = AWQRunner(model_path_or_repo_id=model_path_or_name, **runner_params)
            elif mt_lower == 'exl2':
                runner_to_load = EXL2Runner(model_path=model_path_or_name, **runner_params)
            elif mt_lower == 'vllm':
                api_url = runner_params.pop('api_url', "http://localhost:8000")
                api_key = runner_params.pop('api_key', None)
                runner_to_load = VLLMRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **runner_params)
            elif mt_lower == 'llava_cpp':
                mmproj_path = runner_params.pop('mmproj_path', None)
                if not mmproj_path:
                    print("Error: 'mmproj_path' is required for LlavaCppRunner.")
                    return
                runner_to_load = LlavaCppRunner(model_path=model_path_or_name, mmproj_path=mmproj_path, **runner_params)
            elif mt_lower == 'speculative':
                draft_runner = runner_params.pop('draft_runner', None) # Pop from runner_params
                target_runner = runner_params.pop('target_runner', None)
                speculative_k = runner_params.pop('speculative_k', 5)
                if isinstance(draft_runner, BaseRunner) and isinstance(target_runner, BaseRunner):
                    runner_to_load = SpeculativeRunner(
                        draft_runner=draft_runner, target_runner=target_runner,
                        speculative_k=speculative_k, **runner_params
                    )
                else:
                    print("Error: SpeculativeRunner requires 'draft_runner' and 'target_runner' (BaseRunner instances).")
                    return
            else:
                print(f"Error: Unknown model type '{model_type}'. No model loaded.")
                return

            self.current_runner = runner_to_load
            self.current_model_type = mt_lower
            # ... (rest of identifier setting and KV cache logic as before) ...
            if mt_lower == 'speculative' and isinstance(self.current_runner, SpeculativeRunner):
                 self.current_model_identifier = (
                    f"speculative(draft={type(self.current_runner.draft_runner).__name__},"
                    f"target={type(self.current_runner.target_runner).__name__})"
                    f"@{model_path_or_name}"
                )
            elif (mt_lower == 'vllm' or mt_lower == 'api') and hasattr(self.current_runner, 'api_url'):
                self.current_model_identifier = f'{mt_lower}({model_path_or_name} @ {self.current_runner.api_url})'
            elif mt_lower == 'llava_cpp' and hasattr(self.current_runner, 'mmproj_path'):
                 self.current_model_identifier = f'llava_cpp({model_path_or_name} + {self.current_runner.mmproj_path})'
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
            # import traceback; traceback.print_exc() # For debug
            self.current_runner = None
            self.current_model_type = None
            self.current_model_identifier = None
            self.idle_unload_sec = None
            self.last_accessed_time = None

    # --- get, unload, check_idle, LoRA methods remain the same ---
    def get(self) -> t.Optional[BaseRunner]:
        if self.current_runner: self.last_accessed_time = time.time(); return self.current_runner
        return None
    def unload(self) -> None:
        if self.current_runner: print(f"\nUnloading model: {self.current_model_identifier} ({self.current_model_type})")
        if hasattr(self.current_runner, '__del__'):
            try: self.current_runner.__del__()
            except Exception as e: print(f"Error during runner __del__: {e}")
        self.current_runner = None; self.current_model_type = None; self.current_model_identifier = None
        self.idle_unload_sec = None; self.last_accessed_time = None
    def check_idle(self) -> bool:
        if self.current_runner and self.idle_unload_sec is not None and self.idle_unload_sec > 0 and self.last_accessed_time is not None:
            if (time.time() - self.last_accessed_time) > self.idle_unload_sec:
                print(f"\nModel '{self.current_model_identifier}' idle, unloading."); self.unload(); return True
        return False
    def load_lora_on_current_runner(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        if not self.current_runner: print("[ModelManager] Error: No model loaded."); return False
        self.last_accessed_time = time.time(); return self.current_runner.load_lora_adapter(adapter_id, adapter_path, **kwargs)
    def unload_lora_on_current_runner(self, adapter_id: str, **kwargs) -> bool:
        if not self.current_runner: print("[ModelManager] Error: No model loaded."); return False
        self.last_accessed_time = time.time(); return self.current_runner.unload_lora_adapter(adapter_id, **kwargs)
    def get_active_loras_on_current_runner(self) -> t.List[str]:
        if not self.current_runner: print("[ModelManager] Error: No model loaded."); return []
        self.last_accessed_time = time.time(); return self.current_runner.get_active_lora_adapters()
    def merge_loras_on_current_runner(self, adapter_ids: t.List[str], **kwargs) -> bool:
        if not self.current_runner: print("[ModelManager] Error: No model loaded."); return False
        self.last_accessed_time = time.time(); return self.current_runner.merge_lora_adapters(adapter_ids, **kwargs)
    def unmerge_loras_on_current_runner(self, **kwargs) -> bool:
        if not self.current_runner: print("[ModelManager] Error: No model loaded."); return False
        self.last_accessed_time = time.time(); return self.current_runner.unmerge_lora_adapters(**kwargs)

if __name__ == '__main__':
    kv_cache_dir_for_demo = Path("data/kv_cache_manager_demo_main")
    autotuner_results_path = Path("data/tuning_results_manager_demo.json")

    if kv_cache_dir_for_demo.exists(): shutil.rmtree(kv_cache_dir_for_demo)
    if autotuner_results_path.exists(): autotuner_results_path.unlink()
    print(f"Cleaned up old demo cache/tuning files.")

    manager = ModelManager()
    manager.kv_cache_mgr = KVCacheManager(cache_dir=str(kv_cache_dir_for_demo))
    manager.auto_tuner = AutoTuner(results_path=str(autotuner_results_path)) # Ensure manager uses test tuner path

    print(f"--- Initial state: Runner loaded? {manager.get() is not None} ---")

    # --- Demonstrate Auto-Tuning with a GGUF model ---
    print("\n--- Loading GGUF Runner with Auto-Tune ---")
    gguf_model_id_for_tuning = "test-model-for-autotune.gguf"
    manager.load(
        model_type='gguf',
        model_path_or_name=gguf_model_id_for_tuning,
        auto_tune=True, # Enable auto-tuning
        # Provide some initial params that auto-tuner might override or complement
        n_ctx=2048
    )
    tuned_gguf_runner = manager.get()
    if tuned_gguf_runner:
        print(f"Tuned GGUF Runner loaded: {isinstance(tuned_gguf_runner, LlamaCppRunner)}")
        print(f"  Runner ID: {manager.current_model_identifier}")
        # Check if runner has attributes that might have been set by auto-tuner (e.g., n_gpu_layers)
        # This depends on what AutoTuner's placeholder returns and if LlamaCppRunner stores them.
        if hasattr(tuned_gguf_runner, 'n_gpu_layers'):
             print(f"  Runner n_gpu_layers (potentially auto-tuned): {tuned_gguf_runner.n_gpu_layers}")
        tuned_gguf_runner.generate("Test prompt for auto-tuned GGUF runner.", max_new_tokens=5)
    else:
        print(f"Failed to load GGUF runner with auto-tuning.")

    manager.unload()

    # --- Demonstrate loading another model type (e.g., API, which won't auto-tune) ---
    print("\n--- Loading API Runner (auto-tune should be skipped) ---")
    manager.load(
        model_type='api',
        model_path_or_name='demo-api-model',
        api_url='http://dummy.api.example.com/v1',
        auto_tune=True # Attempt auto-tune
    )
    api_runner_demo = manager.get()
    if api_runner_demo:
        print(f"API Runner loaded: {isinstance(api_runner_demo, APIRunner)}")
        api_runner_demo.generate("Test API runner after auto-tune attempt.")
    manager.unload()

    print("\nModelManager auto-tuning demonstration complete.")

    if kv_cache_dir_for_demo.exists(): shutil.rmtree(kv_cache_dir_for_demo)
    if autotuner_results_path.exists(): autotuner_results_path.unlink()
    print(f"Cleaned up demo cache/tuning files.")
