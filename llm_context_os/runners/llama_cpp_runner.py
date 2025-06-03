# llm_context_os/runners/llama_cpp_runner.py
import typing as t
from .base import BaseRunner
from llama_cpp import Llama, LlamaGrammar # LlamaGrammar might be useful later
import tempfile
import os
from pathlib import Path

class LlamaCppRunner(BaseRunner):
    """
    Runner for GGUF models using the llama-cpp-python library.
    Implements real KV cache handling and basic dynamic LoRA application at generation time.
    """
    def __init__(self,
                 model_path: str,
                 n_gpu_layers: int = 0,
                 n_ctx: int = 2048,
                 seed: int = -1,
                 n_threads: t.Optional[int] = None,
                 verbose: bool = False,
                 # LoRA base model path (if LoRAs are not trained on the main model_path)
                 # lora_base: t.Optional[str] = None, # For more advanced LoRA application at init
                 **kwargs: t.Any):

        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.seed = seed
        self.n_threads = n_threads
        self.verbose = verbose
        # self.lora_base_model_path = lora_base # Store if using LoRAs that need a different base

        self.llama_constructor_kwargs = {
            'n_batch': kwargs.pop('n_batch', 512),
            'logits_all': kwargs.pop('logits_all', False),
            'embedding': kwargs.pop('embedding', False),
            **kwargs
        }

        self.model: t.Optional[Llama] = None
        # Store active LoRAs: adapter_id -> (adapter_path, adapter_scale)
        self.active_loras: t.Dict[str, t.Tuple[str, float]] = {}
        # self.is_merged is removed as merging is typically offline for GGUF.

        print(f"[LlamaCppRunner] Initializing for model path: {self.model_path}")
        print(f"  Parameters: n_gpu_layers={self.n_gpu_layers}, n_ctx={self.n_ctx}, seed={self.seed}, verbose={self.verbose}"
              f"{f', n_threads={self.n_threads}' if self.n_threads is not None else ''}")
        if self.llama_constructor_kwargs:
            print(f"  Additional Llama params: {self.llama_constructor_kwargs}")

        try:
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                seed=self.seed,
                n_threads=self.n_threads,
                verbose=self.verbose,
                # lora_base=self.lora_base_model_path, # If loading a base LoRA at init
                **self.llama_constructor_kwargs
            )
            print(f"[LlamaCppRunner] Successfully loaded model: {self.model_path}")
            print(f"[LlamaCppRunner] Model context size (n_ctx_train): {self.model.n_ctx_train()}")
        except Exception as e:
            print(f"[LlamaCppRunner] Error loading model {self.model_path}: {e}")
            self.model = None

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> str:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Generating ---")
        if not self.model:
            print("[LlamaCppRunner] Error: Model not loaded. Cannot generate.")
            return "[LlamaCppRunner] Error: Model not loaded."

        if image_paths:
            print(f"[LlamaCppRunner] Warning: image_paths provided but LlamaCppRunner does not process them. Use LlavaCppRunner.")

        llm_params = {
            "prompt": prompt,
            "max_tokens": kwargs.get('max_new_tokens', kwargs.get('max_tokens', 128)),
            "temperature": kwargs.get('temperature', 0.8),
            "top_p": kwargs.get('top_p', 0.95),
            "top_k": kwargs.get('top_k', 40),
            "stop": kwargs.get('stop', None),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
            "presence_penalty": kwargs.get('presence_penalty', 0.0),
            "repeat_penalty": kwargs.get('repeat_penalty', 1.1),
        }
        if llm_params["stop"] is None: del llm_params["stop"]

        # LoRA application at generation time
        if self.active_loras:
            if len(self.active_loras) > 1:
                print(f"[LlamaCppRunner] Warning: Multiple LoRAs active ({list(self.active_loras.keys())}). Using the first one for this generation call.")
            # Get the first active LoRA (path, scale)
            lora_id, (lora_path, lora_scale) = list(self.active_loras.items())[0]
            llm_params['lora_path'] = lora_path
            llm_params['lora_scale'] = lora_scale
            print(f"  Applying LoRA '{lora_id}' (Path: {lora_path}, Scale: {lora_scale}) for this generation.")

        print(f"[LlamaCppRunner] Generating with params: { {k:v for k,v in llm_params.items() if k != 'prompt'} }")
        try:
            completion = self.model.create_completion(**llm_params)
            return completion['choices'][0]['text']
        except Exception as e:
            print(f"[LlamaCppRunner] Error during model generation: {e}")
            return f"[LlamaCppRunner] Error generating response: {e}"

    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[str, None, None]:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Streaming ---")
        if not self.model:
            print("[LlamaCppRunner] Error: Model not loaded. Cannot stream.")
            yield "[LlamaCppRunner] Error: Model not loaded."; return
        if image_paths:
            print(f"[LlamaCppRunner] Warning: image_paths provided but LlamaCppRunner does not process them for streaming.")

        llm_params = {
            "prompt": prompt,
            "max_tokens": kwargs.get('max_new_tokens', kwargs.get('max_tokens', 256)),
            "temperature": kwargs.get('temperature', 0.8),
            "top_p": kwargs.get('top_p', 0.95),
            "top_k": kwargs.get('top_k', 40),
            "stop": kwargs.get('stop', None),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
            "presence_penalty": kwargs.get('presence_penalty', 0.0),
            "repeat_penalty": kwargs.get('repeat_penalty', 1.1),
            "stream": True
        }
        if llm_params["stop"] is None: del llm_params["stop"]

        if self.active_loras:
            if len(self.active_loras) > 1:
                print(f"[LlamaCppRunner] Warning: Multiple LoRAs active ({list(self.active_loras.keys())}). Using the first one for this stream call.")
            lora_id, (lora_path, lora_scale) = list(self.active_loras.items())[0]
            llm_params['lora_path'] = lora_path
            llm_params['lora_scale'] = lora_scale
            print(f"  Applying LoRA '{lora_id}' (Path: {lora_path}, Scale: {lora_scale}) for this stream.")

        print(f"[LlamaCppRunner] Streaming with params: { {k:v for k,v in llm_params.items() if k not in ['prompt', 'stream']} }")
        try:
            completion_stream = self.model.create_completion(**llm_params)
            for chunk in completion_stream:
                yield chunk['choices'][0]['text']
        except Exception as e:
            print(f"[LlamaCppRunner] Error during model streaming: {e}")
            yield f"[LlamaCppRunner] Error streaming response: {e}"
        finally:
            print("[LlamaCppRunner] Streaming finished.")

    # --- KV Cache Methods (Real Implementation) ---
    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Preloading KV Cache ---")
        if not self.model: print("[LlamaCppRunner] Model not loaded, cannot preload KV."); return
        print(f"Prompt for KV preloading: '{prompt[:100]}...'")
        try:
            tokens = self.model.tokenize(prompt.encode('utf-8', errors='ignore'))
            print(f"[LlamaCppRunner] Tokenized preload prompt into {len(tokens)} tokens.")
            self.model.eval(tokens)
            print(f"[LlamaCppRunner] Successfully preloaded KV cache with {len(tokens)} tokens from prompt.")
        except Exception as e: print(f"[LlamaCppRunner] Error during preload_kv: {e}")

    def export_kv_cache(self) -> t.Optional[str]:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Exporting KV Cache ---")
        if not self.model: print("[LlamaCppRunner] Model not loaded, no KV cache to export."); return None
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.kst', delete=False) as tmp_f: temp_file_path = tmp_f.name
            self.model.save_session_file(temp_file_path)
            print(f"[LlamaCppRunner] Successfully saved KV cache session to temporary file: {temp_file_path}")
            return temp_file_path
        except Exception as e:
            print(f"[LlamaCppRunner] Error saving KV cache session: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as oe: print(f"[LlamaCppRunner] Error deleting temp cache file '{temp_file_path}' on error: {oe}")
            return None

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Importing KV Cache ---")
        if not self.model: print("[LlamaCppRunner] Model not loaded, cannot import KV cache."); return
        if not isinstance(cache_data, str) or not os.path.exists(cache_data):
            print(f"[LlamaCppRunner] Error: Invalid cache_data. Expected a valid file path, got: {cache_data}"); return
        try:
            self.model.load_session_file(str(cache_data))
            print(f"[LlamaCppRunner] Successfully loaded KV cache session from file: {cache_data}")
        except Exception as e: print(f"[LlamaCppRunner] Error loading KV cache session from '{cache_data}': {e}")

    # --- LoRA Adapter Methods (Initial Pass - dynamic application at generation) ---
    def load_lora_adapter(self, adapter_id: str, adapter_path: str, adapter_scale: float = 1.0, **kwargs) -> bool:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Loading LoRA Adapter ---")
        print(f"  ID: {adapter_id}, Path: {adapter_path}, Scale: {adapter_scale}, Other Params: {kwargs}")
        if not self.model: print("[LlamaCppRunner] Model not loaded, cannot 'load' LoRA (it's applied at generation)."); return False # Or True if just tracking

        # For llama-cpp-python, LoRAs are typically applied at generation time via create_completion params
        # or by loading a LlamaLora object, or by model.apply_lora() for persistent state change.
        # This method will just store it for use in generate/stream.
        if adapter_id in self.active_loras:
            print(f"  Warning: LoRA adapter '{adapter_id}' already tracked. Overwriting path and scale.")
        self.active_loras[adapter_id] = (adapter_path, adapter_scale)
        print(f"  LoRA adapter '{adapter_id}' registered for use at generation time.")
        return True

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Unloading LoRA Adapter ---")
        print(f"  ID: {adapter_id}")
        if not self.model: print("[LlamaCppRunner] Model not loaded."); return False
        if adapter_id in self.active_loras:
            del self.active_loras[adapter_id]
            print(f"  LoRA adapter '{adapter_id}' unregistered from active use.")
            # If model.disable_lora() or model.apply_lora(None) was used, call it here.
            return True
        else:
            print(f"  Warning: LoRA adapter '{adapter_id}' not found in active LoRAs.")
            return False

    def get_active_lora_adapters(self) -> t.List[str]:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Getting Active LoRA Adapters ---")
        if not self.model: print("[LlamaCppRunner] Model not loaded."); return []
        adapter_ids = list(self.active_loras.keys())
        print(f"  Tracked active adapters for generation: {adapter_ids}")
        return adapter_ids

    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Merging LoRA Adapters ---")
        print(f"  IDs to merge: {adapter_ids}")
        if not self.model: print("[LlamaCppRunner] Model not loaded, cannot merge."); return False
        print("[LlamaCppRunner] LoRA merging is typically an offline process for GGUF files (e.g., using 'export-lora' in main llama.cpp). Dynamic merging post-init for GGUFs is not directly supported by this runner's current llama-cpp-python interface for saving.")
        return False # Placeholder: Not supported for GGUF runtime merging in this way

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print(f"\n--- LlamaCppRunner ({self.model_path.split('/')[-1]}) Unmerging LoRA Adapters ---")
        if not self.model: print("[LlamaCppRunner] Model not loaded, cannot unmerge."); return False
        print("[LlamaCppRunner] LoRA unmerging for GGUFs usually means reloading the base model without LoRA. This is not dynamically supported here.")
        return False

if __name__ == '__main__':
    dummy_model_path = "dummy_model_for_lora_test.gguf"
    is_real_model_available = False
    if Path(dummy_model_path).exists():
        is_real_model_available = True
        print(f"--- Using existing model for demo: {dummy_model_path} ---")
    else:
        try:
            with open(dummy_model_path, 'wb') as f: f.write(b"GGUF_dummy")
            print(f"--- Created dummy file: {dummy_model_path} (Llama load will fail, for LoRA method call tests) ---")
        except Exception as e: print(f"Could not create dummy file: {e}")

    print(f"--- Attempting to initialize LlamaCppRunner with model: {dummy_model_path} ---")

    try:
        llama_model = LlamaCppRunner(model_path=dummy_model_path, n_gpu_layers=0, verbose=False, n_ctx=512)

        if llama_model.model:
            print("\n--- LlamaCppRunner Initialized (Model Found) ---")
            # Generate/Stream calls would work here if model is valid
        else:
            print("\n--- LlamaCppRunner Initialized (Model NOT Loaded or Invalid) ---")
            print("--- KV/LoRA methods will show 'Model not loaded' if they depend on self.model ---")

        print("\n--- LoRA Methods Demo on LlamaCppRunner ---")
        lora1_id = "style_adapter"
        lora1_path = "/path/to/style_adapter.bin" # Placeholder path
        lora1_scale = 0.8

        print(f"\nLoading LoRA: {lora1_id}")
        load_status = llama_model.load_lora_adapter(lora1_id, lora1_path, adapter_scale=lora1_scale) # Pass scale via kwargs
        print(f"Load LoRA status: {load_status}")

        active_loras = llama_model.get_active_lora_adapters()
        print(f"Active LoRAs: {active_loras}")
        if load_status and llama_model.model : assert lora1_id in active_loras # Only if model loaded and LoRA load successful

        print("\nGenerating with (simulated) LoRA applied:")
        # generate will pick up the first LoRA from self.active_loras if model is loaded
        response_with_lora = llama_model.generate("Prompt with LoRA.", max_new_tokens=10)
        print(f"Response with LoRA: {response_with_lora}")

        print(f"\nUnloading LoRA: {lora1_id}")
        unload_status = llama_model.unload_lora_adapter(lora1_id)
        print(f"Unload LoRA status: {unload_status}")
        active_loras_after_unload = llama_model.get_active_lora_adapters()
        print(f"Active LoRAs after unload: {active_loras_after_unload}")
        if load_status and llama_model.model: assert lora1_id not in active_loras_after_unload

        print("\nAttempting merge/unmerge (expected to indicate not supported):")
        merge_status = llama_model.merge_lora_adapters([lora1_id])
        print(f"Merge status: {merge_status}")
        unmerge_status = llama_model.unmerge_lora_adapters()
        print(f"Unmerge status: {unmerge_status}")

    except Exception as e:
        print(f"An unexpected error occurred during LlamaCppRunner demo: {e}")
    finally:
        if not is_real_model_available and Path(dummy_model_path).exists():
            Path(dummy_model_path).unlink()
            print(f"\nCleaned up dummy model file: {dummy_model_path}")

    print("\nLlamaCppRunner LoRA methods demonstration complete.")
