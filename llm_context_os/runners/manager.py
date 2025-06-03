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
from llm_context_os.tuning.auto_tuner import AutoTuner

class ModelManager:
    def __init__(self):
        self.current_runner: t.Optional[BaseRunner] = None
        self.current_model_type: t.Optional[str] = None
        self.current_model_identifier: t.Optional[str] = None
        self.idle_unload_sec: t.Optional[int] = None
        self.last_accessed_time: t.Optional[float] = None
        self.kv_cache_mgr = KVCacheManager()
        self.auto_tuner = AutoTuner()
        print("ModelManager initialized.")
        print(f"  KV Cache Manager using directory: {self.kv_cache_mgr.cache_dir.resolve()}")
        print(f"  AutoTuner using results path: {self.auto_tuner.results_path.resolve()}")

    def load(self,
             model_type: str,
             model_path_or_name: str,
             idle_unload_sec: t.Optional[int] = None,
             default_prefix_text: t.Optional[str] = None,
             auto_tune: bool = False,
             **kwargs: t.Any) -> None:
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name/ID: {model_path_or_name}")
        if idle_unload_sec is not None: print(f"  Idle Unload Sec: {idle_unload_sec}")
        if default_prefix_text: print(f"  Default Prefix Text: '{default_prefix_text[:50]}...'")
        if auto_tune: print(f"  Auto-Tune Enabled: {auto_tune}")

        runner_params = kwargs.copy()
        processed_kwargs_for_print = {k: type(v).__name__ if isinstance(v, BaseRunner) else v for k,v in runner_params.items()}
        print(f"  Initial Runner Constructor Params: {processed_kwargs_for_print}")

        if self.current_runner:
            self.unload()

        try:
            runner_to_load: t.Optional[BaseRunner] = None
            mt_lower = model_type.lower()

            if auto_tune and self.auto_tuner and mt_lower in ['gguf', 'llava_cpp', 'awq', 'exl2', 'llama_cpp']:
                print(f'[ModelManager] Auto-tuning enabled for {model_path_or_name} ({mt_lower})...')
                optimal_params = self.auto_tuner.get_optimal_settings(model_path_or_name, mt_lower)
                if optimal_params:
                    print(f'[ModelManager] Applying auto-tuned optimal_params: {optimal_params}')
                    runner_params.update(optimal_params)
                    print(f"[ModelManager] Runner params after auto-tuning: {runner_params}")
                else:
                    print('[ModelManager] AutoTuner did not return optimal parameters.')

            if mt_lower == 'api':
                api_url = runner_params.pop('api_url', None); api_key = runner_params.pop('api_key', None)
                if not api_url: print("Error: 'api_url' is required for API runner."); return
                runner_to_load = APIRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **runner_params)
            elif mt_lower == 'gguf' or mt_lower == 'llama_cpp':
                runner_to_load = LlamaCppRunner(model_path=model_path_or_name, **runner_params)
            elif mt_lower == 'awq':
                runner_to_load = AWQRunner(model_path_or_repo_id=model_path_or_name, **runner_params)
            elif mt_lower == 'exl2':
                runner_to_load = EXL2Runner(model_path=model_path_or_name, **runner_params)
            elif mt_lower == 'vllm':
                api_url = runner_params.pop('api_url', "http://localhost:8000"); api_key = runner_params.pop('api_key', None)
                runner_to_load = VLLMRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **runner_params)
            elif mt_lower == 'llava_cpp':
                mmproj_path = runner_params.pop('mmproj_path', None)
                if not mmproj_path: print("Error: 'mmproj_path' is required for LlavaCppRunner."); return
                runner_to_load = LlavaCppRunner(model_path=model_path_or_name, mmproj_path=mmproj_path, **runner_params)
            elif mt_lower == 'speculative':
                draft_runner = runner_params.pop('draft_runner', None)
                target_runner = runner_params.pop('target_runner', None)
                speculative_k = runner_params.pop('speculative_k', 5)
                if isinstance(draft_runner, BaseRunner) and isinstance(target_runner, BaseRunner):
                    runner_to_load = SpeculativeRunner(draft_runner=draft_runner, target_runner=target_runner, speculative_k=speculative_k, **runner_params)
                else: print("Error: SpeculativeRunner requires 'draft_runner' and 'target_runner'."); return
            else: print(f"Error: Unknown model type '{model_type}'."); return

            self.current_runner = runner_to_load
            self.current_model_type = mt_lower

            # Construct identifier
            if mt_lower == 'speculative' and isinstance(r := self.current_runner, SpeculativeRunner):
                 self.current_model_identifier = f"speculative(draft={type(r.draft_runner).__name__},target={type(r.target_runner).__name__})@{model_path_or_name}"
            elif hasattr(self.current_runner, 'api_url') and (mt_lower == 'vllm' or mt_lower == 'api'):
                self.current_model_identifier = f'{mt_lower}({model_path_or_name} @ {self.current_runner.api_url})'
            elif hasattr(self.current_runner, 'mmproj_path') and mt_lower == 'llava_cpp':
                 self.current_model_identifier = f'llava_cpp({model_path_or_name}+{self.current_runner.mmproj_path})'
            else:
                self.current_model_identifier = model_path_or_name

            self.idle_unload_sec = idle_unload_sec
            self.last_accessed_time = time.time()
            print(f"Successfully instantiated {self.current_model_type} runner for: {self.current_model_identifier}")

            # KV Cache handling
            if default_prefix_text and self.current_runner:
                print(f'[ModelManager] Processing KV cache for prefix: "{default_prefix_text[:50]}..." for model {self.current_model_identifier}')
                cache_id = self.current_model_identifier
                expected_suffix = ".kvcache.pkl" # Default for most runners (pickled data)

                if mt_lower == 'speculative' and isinstance(self.current_runner, SpeculativeRunner):
                    target = self.current_runner.target_runner
                    if hasattr(target, 'model_path_or_repo_id'): cache_id = target.model_path_or_repo_id
                    elif hasattr(target, 'model_path'): cache_id = target.model_path
                    elif hasattr(target, 'model_name'): cache_id = target.model_name
                    else: cache_id = f"target_of_{self.current_model_identifier}"
                    print(f"[ModelManager] Speculative mode: using target runner ID for cache: {cache_id}")
                    # Determine expected suffix based on target runner type for speculative
                    if isinstance(target, (LlamaCppRunner, LlavaCppRunner)): expected_suffix = ".kst"
                    # Add other types if they also save direct files with specific suffixes

                elif mt_lower in ['gguf', 'llama_cpp', 'llava_cpp']:
                    expected_suffix = ".kst" # LlamaCpp-based runners export .kst session files

                print(f"[ModelManager] Attempting to load KV cache with expected suffix: {expected_suffix}")
                loaded_cache = self.kv_cache_mgr.load_kv_cache(cache_id, default_prefix_text, expected_file_suffix=expected_suffix)

                if loaded_cache is not None:
                    print(f'[ModelManager] Found existing KV cache (type: {type(loaded_cache)}). Importing into runner.')
                    self.current_runner.import_kv_cache(loaded_cache)
                    # If loaded_cache is a path (e.g. for LlamaCppRunner), LlamaCppRunner.import_kv_cache handles it.
                    # If it's an object (e.g. for hypothetical other runners), they handle it.
                else:
                    print(f'[ModelManager] No existing KV cache for prefix. Preloading/generating in runner.')
                    self.current_runner.preload_kv(default_prefix_text)
                    exported_cache = self.current_runner.export_kv_cache()
                    if exported_cache is not None:
                        print(f'[ModelManager] Exported new KV cache from runner (type: {type(exported_cache)}). Saving for {cache_id}.')
                        self.kv_cache_mgr.save_kv_cache(exported_cache, cache_id, default_prefix_text)
                        # If exported_cache is a temp filepath (from LlamaCppRunner), KVCacheManager copies it.
                        # The temp file itself should be cleaned up by the runner if it created one.
                        if isinstance(exported_cache, str) and "temp" in exported_cache.lower() and Path(exported_cache).exists():
                            try: Path(exported_cache).unlink(); print(f"[ModelManager] Cleaned up temp exported KV cache file: {exported_cache}")
                            except Exception as e_clean: print(f"[ModelManager] Error cleaning temp KV cache file {exported_cache}: {e_clean}")
                    else:
                        print(f'[ModelManager] Runner did not provide an exportable KV cache after preload.')
            print(f"Model '{self.current_model_identifier}' fully ready.")
        except Exception as e:
            print(f"Error during model loading or runner initialization: {e}")
            self.current_runner = None; self.current_model_type = None; self.current_model_identifier = None
            self.idle_unload_sec = None; self.last_accessed_time = None

    def get(self) -> t.Optional[BaseRunner]: # ... (no change)
        if self.current_runner: self.last_accessed_time = time.time(); return self.current_runner
        return None
    def unload(self) -> None: # ... (no change)
        if self.current_runner: print(f"\nUnloading model: {self.current_model_identifier} ({self.current_model_type})")
        if hasattr(self.current_runner, '__del__'):
            try: self.current_runner.__del__()
            except Exception as e: print(f"Error during runner __del__: {e}")
        self.current_runner = None; self.current_model_type = None; self.current_model_identifier = None
        self.idle_unload_sec = None; self.last_accessed_time = None
    def check_idle(self) -> bool: # ... (no change)
        if self.current_runner and self.idle_unload_sec is not None and self.idle_unload_sec > 0 and \
           self.last_accessed_time is not None:
            if (time.time() - self.last_accessed_time) > self.idle_unload_sec:
                print(f"\nModel '{self.current_model_identifier}' idle, unloading."); self.unload(); return True
        return False
    # --- LoRA methods remain the same ---
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
    manager.auto_tuner = AutoTuner(results_path=str(autotuner_results_path))

    print(f"--- Initial state: Runner loaded? {manager.get() is not None} ---")

    # --- Demonstrate KV Caching with LlamaCppRunner (expects .kst file handling) ---
    model_id_for_kv_test = "dummy_gguf_for_kv_caching.gguf"
    # Create a dummy file for LlamaCppRunner to "load" to avoid init error for demo
    # Real LlamaCppRunner would error if this isn't a valid GGUF.
    with open(model_id_for_kv_test, 'wb') as f: f.write(b"GGUF")

    common_prefix = "The story of ancient Rome begins with Romulus and Remus."

    print("\n--- First Load (LlamaCpp): Generating and Saving KV Cache (as .kst file via KVCacheManager) ---")
    manager.load(
        model_type='gguf',
        model_path_or_name=model_id_for_kv_test,
        default_prefix_text=common_prefix,
        n_gpu_layers=0
    )
    runner1 = manager.get()
    if runner1 and runner1.model: # Check if model actually loaded (it won't fully with dummy GGUF)
        print(f"Runner1 loaded: {type(runner1).__name__}")
        runner1.generate(common_prefix + " And they were raised by a she-wolf.", max_new_tokens=5)
    elif runner1:
        print(f"Runner1 ({type(runner1).__name__}) instantiated but model load failed. KV Cache save from preload will be based on mock data if any.")
        # Manually simulate that preload created some mock data if model is None for demo of save
        if hasattr(runner1, 'mock_kv_cache_data') and runner1.mock_kv_cache_data:
             print(f"  Runner has mock_kv_cache_data: {runner1.mock_kv_cache_data}")
             # To test KVCacheManager's file copy, we need export_kv_cache to return a real temp file path
             # This part of demo is tricky if LlamaCppRunner.model is None.
             # For now, KVCacheManager.save will try to pickle the mock_kv_cache_data if export returns it.
    else:
        print("Failed to load runner1.")

    print("\n--- Unloading the model ---")
    manager.unload()
    print(f"Runner after unload: {manager.get() is not None}")

    print("\n--- Second Load (LlamaCpp): Loading KV Cache from Disk (e.g., .kst file via KVCacheManager) ---")
    manager.load(
        model_type='gguf',
        model_path_or_name=model_id_for_kv_test,
        default_prefix_text=common_prefix, # Same prefix
        n_gpu_layers=0
    )
    runner2 = manager.get()
    if runner2 and runner2.model:
        print(f"Runner2 loaded: {type(runner2).__name__}")
        # LlamaCppRunner.import_kv_cache would have been called if cache file was found by KVCacheManager.
        # The runner's internal print statements should indicate this.
        runner2.generate(common_prefix + " This time, the story continues differently.", max_new_tokens=5)
    elif runner2:
        print(f"Runner2 ({type(runner2).__name__}) instantiated but model load failed. KV Cache import would have attempted if file exists.")
    else:
        print("Failed to load runner2.")

    if Path(model_id_for_kv_test).exists(): Path(model_id_for_kv_test).unlink() # Clean dummy GGUF

    print("\nModelManager KV cache demonstration complete.")

    if kv_cache_dir_for_demo.exists(): shutil.rmtree(kv_cache_dir_for_demo)
    if autotuner_results_path.exists(): autotuner_results_path.unlink()
    print(f"Cleaned up demo cache/tuning files.")
