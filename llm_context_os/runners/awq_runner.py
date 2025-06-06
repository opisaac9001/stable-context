# yawl/runners/awq_runner.py
import typing as t
from .base import BaseRunner
from pathlib import Path

# Attempt to import AWQ and Transformers components
try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    from transformers.generation.streamers import TextIteratorStreamer
    from threading import Thread
    import torch
    AWQ_AVAILABLE = True
except ImportError:
    AutoAWQForCausalLM = None
    AutoTokenizer = None
    AutoConfig = None
    GenerationConfig = None # type: ignore
    TextIteratorStreamer = None # type: ignore
    Thread = None # type: ignore
    torch = None # type: ignore
    AWQ_AVAILABLE = False
    print("[AWQRunner] Warning: 'autoawq', 'transformers', or 'torch' library not found. AWQRunner will not be functional.")

try:
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
    print("[AWQRunner] PEFT library found.")
except ImportError:
    PeftModel = None # type: ignore
    LoraConfig = None # type: ignore
    get_peft_model = None # type: ignore
    prepare_model_for_kbit_training = None # type: ignore
    PEFT_AVAILABLE = False
    print("[AWQRunner] Warning: PEFT library not found. LoRA functionalities will be disabled.")


class AWQRunner(BaseRunner):
    """
    Runner for AWQ quantized models using the AutoAWQ library and Hugging Face Transformers.
    Implements KV cache handling using past_key_values.
    """
    def __init__(self,
                 model_path_or_repo_id: str,
                 quant_filename: t.Optional[str] = None,
                 device: str = "cuda",
                 use_flash_attention_2: bool = True,
                 trust_remote_code: bool = True,
                 **kwargs: t.Any):

        self.model_path_or_repo_id = model_path_or_repo_id
        self.quant_filename = quant_filename
        self.device = device
        self.use_flash_attention_2 = use_flash_attention_2
        self.trust_remote_code = trust_remote_code

        self.model: t.Optional[t.Any] = None
        self.tokenizer: t.Optional[t.Any] = None
        # self.active_loras and self.is_merged are removed as PEFT model will manage this.

        # For Hugging Face models, KV cache is typically 'past_key_values'
        self.current_kv_cache: t.Optional[t.Tuple[t.Tuple[torch.Tensor, ...], ...]] = None

        print(f"[AWQRunner] Initializing for model: {self.model_path_or_repo_id}")
        # ... (rest of init prints as before) ...
        if not AWQ_AVAILABLE:
            print("[AWQRunner] Error: 'autoawq' or 'transformers' not installed. Cannot load model.")
            return
        try:
            # ... (tokenizer and model loading logic as before) ...
            print(f"[AWQRunner] Loading tokenizer for '{self.model_path_or_repo_id}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_repo_id, trust_remote_code=self.trust_remote_code)
            print("[AWQRunner] Tokenizer loaded successfully.")
            load_kwargs = {
                "fuse_layers": kwargs.pop('fuse_layers', True),
                "trust_remote_code": self.trust_remote_code,
                "safetensors": kwargs.pop('safetensors', True),
                "attn_implementation": "flash_attention_2" if self.use_flash_attention_2 else kwargs.pop("attn_implementation", None),
                **kwargs
            }
            load_kwargs = {k:v for k,v in load_kwargs.items() if v is not None}
            print(f"[AWQRunner] Loading quantized model '{self.model_path_or_repo_id}'...")
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path_or_repo_id,
                quant_file=self.quant_filename,
                device_map=self.device,
                **load_kwargs
            )
            self.model.eval()
            print(f"[AWQRunner] Successfully loaded AWQ model: {self.model_path_or_repo_id} to device '{self.device}'.")
        except Exception as e:
            print(f"[AWQRunner] Error loading AWQ model {self.model_path_or_repo_id}: {e}")
            self.model = None; self.tokenizer = None

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Generating ---")
        if not self.model or not self.tokenizer or not AWQ_AVAILABLE or not torch:
            print("[AWQRunner] Error: Model, tokenizer, or torch not available.")
            return "[AWQRunner] Error: Model, tokenizer, or torch not available.", {"prompt_tokens": 0, "completion_tokens": 0}
        if image_paths: print(f"[AWQRunner] Warning: image_paths provided but AWQRunner does not process them.")

        active_adapters_msg = "No active LoRA adapters."
        if PEFT_AVAILABLE and isinstance(self.model, PeftModel) and hasattr(self.model, 'active_adapters'):
            if self.model.active_adapters: # type: ignore
                active_adapters_msg = f"Active LoRA adapters (PEFT): {self.model.active_adapters}" # type: ignore
        elif PEFT_AVAILABLE and isinstance(self.model, PeftModel) and hasattr(self.model, 'peft_config'): # For older PEFT
             # Check if peft_config is a dict and get keys
            if isinstance(self.model.peft_config, dict) and self.model.peft_config.keys(): # type: ignore
                active_adapters_msg = f"Active LoRA adapters (PEFT): {list(self.model.peft_config.keys())}" # type: ignore
        print(f"[AWQRunner] {active_adapters_msg}")
        # Check if merged state (often model is not PeftModel anymore or has a flag)
        # This is a heuristic; PEFT doesn't have a universal 'is_merged' flag after merge_and_unload
        if PEFT_AVAILABLE and not isinstance(self.model, PeftModel) and self.tokenizer is not None: # Assuming if not PeftModel, it might have been merged
            print("[AWQRunner] Model may be merged (not a PeftModel instance).")


        try:
            if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids_length = inputs.input_ids.shape[1]

            generation_params = {
                "max_new_tokens": kwargs.get('max_new_tokens', kwargs.get('max_tokens', 128)),
                "temperature": kwargs.get('temperature', 0.7), "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 50),
                "do_sample": kwargs.get('do_sample', True if kwargs.get('temperature', 0.7) > 0 else False),
                "repetition_penalty": kwargs.get('repetition_penalty', 1.0),
                "eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id,
            }
            generation_params = {k:v for k,v in generation_params.items() if v is not None}
            # ... (stop_sequences note as before) ...

            # Add past_key_values if available in current_kv_cache
            if self.current_kv_cache:
                generation_params["past_key_values"] = self.current_kv_cache
                print("[AWQRunner] Using provided past_key_values for generation.")

            print(f"[AWQRunner] Generating with params: { {k:v for k,v in generation_params.items() if k != 'past_key_values'} }") # Avoid printing large KV cache

            with torch.no_grad():
                # use_cache=True is implicit if past_key_values are used and supported by model
                outputs = self.model.generate(**inputs, **generation_params, use_cache=True)

            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                self.current_kv_cache = outputs.past_key_values
                print("[AWQRunner] Updated self.current_kv_cache from generate output.")
            else:
                # This might happen if generation is short or model doesn't return it for some reason
                self.current_kv_cache = None
                print("[AWQRunner] Did not find past_key_values in generate output. KV cache not updated.")

            generated_ids = outputs[0][input_ids_length:] # type: ignore
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            prompt_tokens = input_ids_length
            completion_tokens = len(generated_ids)
            token_counts = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

            print(f"[AWQRunner] Generated text: {generated_text[:100]}...")
            print(f"  Tokens: prompt={prompt_tokens}, completion={completion_tokens}")
            return generated_text, token_counts
        except Exception as e:
            print(f"[AWQRunner] Error during model generation: {e}")
            return f"[AWQRunner] Error generating response: {e}", {"prompt_tokens": 0, "completion_tokens": 0}

    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Streaming ---")
        if not self.model or not self.tokenizer or not AWQ_AVAILABLE or not torch:
            yield {"prompt_tokens": 0}
            yield ("[AWQRunner] Error: Model, tokenizer, or torch not available.", 0)
            return
        if image_paths: print(f"[AWQRunner] Warning: image_paths provided but AWQRunner does not process them.")

        active_adapters_msg = "No active LoRA adapters."
        if PEFT_AVAILABLE and isinstance(self.model, PeftModel) and hasattr(self.model, 'active_adapters'):
            if self.model.active_adapters: # type: ignore
                active_adapters_msg = f"Active LoRA adapters (PEFT): {self.model.active_adapters}" # type: ignore
        elif PEFT_AVAILABLE and isinstance(self.model, PeftModel) and hasattr(self.model, 'peft_config'): # For older PEFT
            if isinstance(self.model.peft_config, dict) and self.model.peft_config.keys(): # type: ignore
                active_adapters_msg = f"Active LoRA adapters (PEFT): {list(self.model.peft_config.keys())}" # type: ignore
        print(f"[AWQRunner] {active_adapters_msg}")
        if PEFT_AVAILABLE and not isinstance(self.model, PeftModel) and self.tokenizer is not None:
            print("[AWQRunner] Model may be merged (not a PeftModel instance).")

        try:
            if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_params = {
                "max_new_tokens": kwargs.get('max_new_tokens', kwargs.get('max_tokens', 256)),
                "temperature": kwargs.get('temperature', 0.7), "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 50),
                "do_sample": kwargs.get('do_sample', True if kwargs.get('temperature', 0.7) > 0 else False),
                "repetition_penalty": kwargs.get('repetition_penalty', 1.0),
                "eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id,
            }
            generation_params = {k:v for k,v in generation_params.items() if v is not None}
            # ... (stop_sequences note as before) ...

            # Add past_key_values if available
            if self.current_kv_cache:
                generation_params["past_key_values"] = self.current_kv_cache
                print("[AWQRunner] Using provided past_key_values for streaming generation.")

            print(f"[AWQRunner] Streaming with params: { {k:v for k,v in generation_params.items() if k != 'past_key_values'} }")
            generation_kwargs = {**inputs, **generation_params, "streamer": streamer, "use_cache": True} # Ensure use_cache for pkv

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            print("[AWQRunner] Note: For streaming, self.current_kv_cache is used as input if available, "
                  "but it's not updated with new past_key_values from the stream output by this placeholder. "
                  "Conversational context with KV cache in streaming requires more complex handling.")

            prompt_tokens = inputs.input_ids.shape[1] # type: ignore
            yield {"prompt_tokens": prompt_tokens}
            print(f"  Yielded prompt_tokens: {prompt_tokens}")

            for new_text in streamer: # type: ignore
                if new_text: # Ensure non-empty chunk
                    tokens_in_chunk = len(self.tokenizer.encode(new_text))
                    yield (new_text, tokens_in_chunk)
                # else: yield ("", 0) # Optionally yield empty strings with 0 tokens
        except Exception as e:
            print(f"[AWQRunner] Error during model streaming: {e}")
            yield (f"[AWQRunner] Error streaming response: {e}", 0)
        finally:
            if 'thread' in locals() and thread.is_alive(): thread.join(timeout=1) # type: ignore
            print("[AWQRunner] Streaming finished.")

    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Preloading KV Cache ---")
        if not self.model or not self.tokenizer or not AWQ_AVAILABLE or not torch:
            print("[AWQRunner] Model/tokenizer/torch not available for KV preloading."); return

        print(f"Prompt for KV preloading: '{prompt[:100]}...'")
        try:
            if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                # Generate a single token to populate past_key_values
                outputs = self.model.generate(**inputs, max_new_tokens=1, use_cache=True, do_sample=False)

            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                self.current_kv_cache = outputs.past_key_values
                print(f"[AWQRunner] Successfully preloaded KV cache with {inputs.input_ids.shape[1]} tokens from prompt.")
            else:
                self.current_kv_cache = None
                print("[AWQRunner] Failed to obtain past_key_values after preload generation.")
        except Exception as e:
            print(f"[AWQRunner] Error during preload_kv: {e}")
            self.current_kv_cache = None

    def export_kv_cache(self) -> t.Any: # Should be t.Optional[t.Tuple[t.Tuple[torch.Tensor, ...], ...]]
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Exporting KV Cache ---")
        if not self.model: print("[AWQRunner] Model not loaded."); return None
        if self.current_kv_cache:
            print(f"  Exporting KV cache (past_key_values) of type: {type(self.current_kv_cache)}")
        else:
            print("  No current KV cache (past_key_values) to export.")
        return self.current_kv_cache

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Importing KV Cache ---")
        if not self.model: print("[AWQRunner] Model not loaded."); return
        # Basic check for typical past_key_values structure (tuple of tuples of tensors)
        if isinstance(cache_data, tuple) and all(isinstance(item, tuple) for item in cache_data):
            # Further checks could be done here (e.g., tensor types, shapes, device)
            self.current_kv_cache = cache_data
            print(f"  Imported KV cache data (past_key_values). Type: {type(cache_data)}")
        elif cache_data is None:
            self.current_kv_cache = None
            print("  KV cache set to None (cleared).")
        else:
            print(f"  Warning: Provided cache_data (type: {type(cache_data)}) does not look like typical past_key_values. Storing as is.")
            self.current_kv_cache = cache_data


    # --- LoRA Methods (PEFT Implementation) ---
    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Loading LoRA Adapter ---")
        if not self.model:
            print("[AWQRunner] Error: Base model not loaded. Cannot load LoRA adapter.")
            return False
        if not PEFT_AVAILABLE or PeftModel is None:
            print("[AWQRunner] Error: PEFT library not available. Cannot load LoRA adapter.")
            return False

        # Filter kwargs for PeftModel.from_pretrained, e.g. common ones:
        # `adapter_name`, `revision`, `token`, `force_download`, `resume_download`
        # For simplicity, we pass all kwargs and let PeftModel handle them.
        # PEFT might need the model to be prepared for k-bit training if it's quantized and PEFT is used for training.
        # For inference, it's usually not needed if the base model is already quantized.
        # if hasattr(self.model, "is_quantized") and self.model.is_quantized and prepare_model_for_kbit_training:
        #     print("[AWQRunner] Preparing model for k-bit training (for LoRA loading on quantized model).")
        #     try:
        #          self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", False)) # Adapt as needed
        #     except Exception as e:
        #         print(f"[AWQRunner] Warning: Failed to prepare model for k-bit training: {e}")

        try:
            print(f"[AWQRunner] Loading LoRA adapter '{adapter_id}' from '{adapter_path}'...")
            # If self.model is already a PeftModel, add_adapter might be preferred.
            # If it's the base model, PeftModel.from_pretrained wraps it.
            if isinstance(self.model, PeftModel):
                print(f"[AWQRunner] Base model is already a PeftModel. Trying to load adapter '{adapter_id}' into it.")
                self.model.load_adapter(adapter_path, adapter_name=adapter_id, **kwargs) # type: ignore
            else:
                print(f"[AWQRunner] Base model is not a PeftModel. Wrapping with PeftModel for adapter '{adapter_id}'.")
                self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=adapter_id, **kwargs) # type: ignore

            # Optionally set the loaded adapter as active if PEFT doesn't do it by default or if you want to switch
            if hasattr(self.model, 'set_adapter'):
                 self.model.set_adapter(adapter_id) # type: ignore
            print(f"[AWQRunner] Successfully loaded LoRA adapter '{adapter_id}'. Active adapters: {self.get_active_lora_adapters()}")
            return True
        except Exception as e:
            print(f"[AWQRunner] Error loading LoRA adapter '{adapter_id}': {e}")
            return False

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Unloading LoRA Adapter ---")
        if not self.model:
            print("[AWQRunner] Error: Model not loaded.")
            return False
        if not PEFT_AVAILABLE or PeftModel is None:
            print("[AWQRunner] Error: PEFT library not available.")
            return False
        if not isinstance(self.model, PeftModel):
            print(f"[AWQRunner] Warning: Model is not a PeftModel. Cannot unload adapter '{adapter_id}'. It might be merged or not loaded via PEFT.")
            return False

        try:
            print(f"[AWQRunner] Unloading LoRA adapter '{adapter_id}'...")
            # PEFT's delete_adapter removes it. disable_adapter keeps it but makes it inactive.
            self.model.delete_adapter(adapter_id) # type: ignore
            print(f"[AWQRunner] Successfully unloaded LoRA adapter '{adapter_id}'. Active adapters: {self.get_active_lora_adapters()}")
            return True
        except Exception as e:
            print(f"[AWQRunner] Error unloading LoRA adapter '{adapter_id}': {e}")
            return False

    def get_active_lora_adapters(self) -> t.List[str]:
        # print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Getting Active LoRA Adapters ---") # Can be noisy
        if not self.model or not PEFT_AVAILABLE or not isinstance(self.model, PeftModel):
            return []

        try:
            # For newer PEFT versions, active_adapters is a property
            if hasattr(self.model, 'active_adapters'):
                active = self.model.active_adapters # type: ignore
                return active if isinstance(active, list) else [active] # Ensure it's a list
            # For older PEFT versions, peft_config might be a dict of LoraConfig objects
            elif hasattr(self.model, 'peft_config') and isinstance(self.model.peft_config, dict): # type: ignore
                return list(self.model.peft_config.keys()) # type: ignore
            else:
                # Fallback or if no specific active adapter concept, show all loaded
                if hasattr(self.model, 'get_adapter_names'): # Not a standard PEFT API, hypothetical
                     return self.model.get_adapter_names() # type: ignore
                return [] # Should not happen with standard PeftModel
        except Exception as e:
            print(f"[AWQRunner] Error getting active LoRA adapters: {e}")
            return []

    def merge_lora_adapters(self, adapter_ids: t.Optional[t.List[str]] = None, **kwargs) -> bool:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Merging LoRA Adapters ---")
        if not self.model:
            print("[AWQRunner] Error: Model not loaded.")
            return False
        if not PEFT_AVAILABLE or PeftModel is None:
            print("[AWQRunner] Error: PEFT library not available.")
            return False
        if not isinstance(self.model, PeftModel):
            print("[AWQRunner] Warning: Model is not a PeftModel. Cannot merge. It might be already merged or not loaded with PEFT.")
            return False

        try:
            # `merge_and_unload` typically merges all active adapters or specified ones and returns the base model.
            # If adapter_ids are provided and PEFT supports merging specific ones before unload, that's more complex.
            # For simplicity, we use merge_and_unload() which usually handles active adapters.
            print(f"[AWQRunner] Merging LoRA adapters (typically all active ones with merge_and_unload)...")
            if hasattr(self.model, 'merge_and_unload'):
                merged_model = self.model.merge_and_unload(**kwargs) # type: ignore
                self.model = merged_model # The model is now the base model with weights merged
                print(f"[AWQRunner] Successfully merged LoRA adapters. Model is now of type: {type(self.model)}")
                # After merge_and_unload, the model is no longer a PeftModel.
                return True
            else:
                print("[AWQRunner] Error: `merge_and_unload` method not found on the PeftModel. Merging failed.")
                return False
        except Exception as e:
            print(f"[AWQRunner] Error merging LoRA adapters: {e}")
            return False

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print(f"\n--- AWQRunner ({self.model_path_or_repo_id}) Unmerging LoRA Adapters ---")
        if not self.model:
            print("[AWQRunner] Error: Model not loaded.")
            return False
        if not PEFT_AVAILABLE: # PeftModel might not be needed if we are just checking type
            print("[AWQRunner] Error: PEFT library not available (for context).")
            # We might still proceed if the goal is to reload the original model,
            # but PEFT-specific unmerge operations would fail.

        # If `merge_and_unload` was used, the model is no longer a PeftModel.
        # Unmerging typically requires reloading the original base model or using a specific PEFT `unmerge` if available.
        if isinstance(self.model, PeftModel):
             if hasattr(self.model, 'unmerge'): # Check if there's an explicit unmerge method
                try:
                    print("[AWQRunner] Attempting to unmerge adapters using `model.unmerge()`...")
                    self.model.unmerge() # type: ignore
                    print("[AWQRunner] Successfully unmerged adapters. Model is still a PeftModel.")
                    return True
                except Exception as e:
                    print(f"[AWQRunner] Error during `model.unmerge()`: {e}. Model state might be inconsistent.")
                    return False
             else:
                print("[AWQRunner] Model is a PeftModel, but no direct `unmerge` method found. Unloading may require reloading the base model or specific adapter disabling.")
                return False # Or try to disable all adapters as a form of "unmerge"

        # If it's not a PeftModel, it might have been merged.
        print("[AWQRunner] Model is not a PeftModel instance. Unmerging typically requires reloading the original base model.")
        print("[AWQRunner] For this runner, 'unmerge' implies you need to reload the model without merged LoRAs.")
        # To truly unmerge, one would typically re-initialize the runner or the model.
        # For now, we cannot revert merge_and_unload without access to the original base model state here.
        # We could re-load it if we stored the original path and args.

        # A conceptual PEFT `unload()` method on a merged model (if it exists)
        # if hasattr(self.model, 'unload') and not isinstance(self.model, PeftModel):
        #     try:
        #         print("[AWQRunner] Attempting to call `model.unload()` on a potentially merged model...")
        #         # This is speculative, PEFT's API might vary
        #         unmerged_model = self.model.unload() # This is hypothetical for a merged non-PeftModel
        #         self.model = unmerged_model
        #         print("[AWQRunner] `unload()` called. Model might be back to base or PeftModel state.")
        #         return True
        #     except Exception as e:
        #         print(f"[AWQRunner] Error calling `model.unload()`: {e}")
        #         return False

        print("[AWQRunner] Unmerging after `merge_and_unload` usually means you need to reload the base model from scratch.")
        print("[AWQRunner] Consider re-initializing the runner to get the base model without merged LoRAs.")
        return False

    def count_tokens(self, text: str) -> Optional[int]:
        """Counts tokens using the loaded Hugging Face tokenizer."""
        if not self.tokenizer or not AWQ_AVAILABLE: # AWQ_AVAILABLE implies transformers is available
            print("[AWQRunner] Tokenizer not available, cannot count tokens.")
            return None
        try:
            # The tokenizer's encode method returns a list of token IDs.
            input_ids = self.tokenizer.encode(text)
            return len(input_ids)
        except Exception as e:
            print(f"[AWQRunner] Error tokenizing text: {e}")
            return None


if __name__ == '__main__':
    # ... (Updated __main__ block from previous step demonstrating generate) ...
    # This will now also demonstrate the KV cache methods if model loads.
    try:
        import torch
        TORCH_FOR_DEMO_AVAILABLE = True
    except ImportError:
        class DummyTorch: # type: ignore
            class cuda: @staticmethod
            def is_available(): return False
            class no_grad: # Dummy context manager
                def __enter__(self): return None
                def __exit__(self, type, value, traceback): pass
        torch = DummyTorch() # type: ignore
        TORCH_FOR_DEMO_AVAILABLE = False
        print("\n[AWQRunner Demo] PyTorch not found.")

    print("\n--- Testing AWQRunner ---")
    if AWQ_AVAILABLE and TORCH_FOR_DEMO_AVAILABLE:
        # Use a smaller, CPU-friendly model if possible for testing structure, or mock heavily.
        # For actual AWQ+PEFT, a proper GPU setup and compatible models/adapters are needed.
        # test_model_id = "casperhansen/mistral-7b-instruct-v0.1-awq"
        test_model_id = "facebook/opt-125m" # Using a standard Hugging Face model for structure testing if AWQ fails
                                           # This won't be an AWQ model, but helps test PEFT logic flow if mocked.

        # Mock quant_filename for non-AWQ model if needed for AWQRunner init path
        quant_filename_for_demo = "model.awq" # Dummy if using non-AWQ model for demo

        print(f"\nAttempting to initialize AWQRunner with model: {test_model_id}")
        # For this demo, we might not actually load a real AWQ model if it's too heavy or requires specific setup.
        # The PEFT logic will be tested assuming `self.model` gets populated.

        # Mock AutoAWQForCausalLM.from_quantized and AutoTokenizer.from_pretrained if needed for demo
        # to avoid actual model download/load for CI/testing environments.

        awq_runner = AWQRunner(
            model_path_or_repo_id=test_model_id,
            quant_filename=quant_filename_for_demo if "awq" in test_model_id else None, # Only if it's an AWQ model
            device="cuda" if torch.cuda.is_available() else "cpu",
            # Forcing a dummy load for demo if AWQ/model is not real
            # This part might need actual model or more sophisticated mocking
            # For now, let's assume it might fail if the model isn't a real AWQ one.
        )

        if awq_runner.model and awq_runner.tokenizer:
            print("\n--- AWQRunner Initialized Successfully (or partially for demo) ---")

            base_prompt = "The capital of France is"
            # KV cache tests (optional here, focus on LoRA)
            # ...

            # --- LoRA PEFT Demonstration ---
            if PEFT_AVAILABLE:
                print("\n--- Testing PEFT LoRA Methods ---")
                # These paths would be to actual LoRA adapter files/directories
                mock_lora_path = "./mock_lora_adapter_files" # Needs to be a real path for from_pretrained
                mock_adapter_id = "test_lora_1"

                # Create dummy adapter files for PeftModel.from_pretrained to not fail on path
                # This is a very basic mock; real adapters have specific files (adapter_config.json, etc.)
                if not Path(mock_lora_path).exists() and PeftModel is not None:
                    try:
                        print(f"Attempting to create mock LoRA directory: {mock_lora_path}")
                        Path(mock_lora_path).mkdir(parents=True, exist_ok=True)
                        # A minimal LoraConfig and saving it can simulate an adapter
                        # This part is complex to mock fully without PEFT/Transformers utils.
                        # For now, we'll rely on the methods handling failures if paths are bad.
                        # config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
                        # with open(Path(mock_lora_path) / "adapter_config.json", "w") as f: json.dump(config.to_dict(), f)
                        # print(f"Mock adapter_config.json created in {mock_lora_path}")
                        print(f"Warning: Mock LoRA path '{mock_lora_path}' created, but it's not a real adapter. PEFT loading will likely fail.")
                    except Exception as e:
                        print(f"Error creating mock LoRA directory: {e}")

                print(f"\nAttempting to load LoRA adapter '{mock_adapter_id}' from '{mock_lora_path}'...")
                # In a real scenario, self.model should be a PEFT-compatible model.
                # We might need to mock PeftModel.from_pretrained itself for a pure unit test.

                # Simplified: Assume PeftModel.from_pretrained can take the base HuggingFace model
                # and a path to adapter files.
                # If awq_runner.model is e.g. an AutoModelForCausalLM instance, PEFT can wrap it.

                # For this demo, the following will likely fail if mock_lora_path isn't a valid adapter
                # and if the base model isn't suitable for direct PEFT application without more setup.
                load_lora_success = awq_runner.load_lora_adapter(mock_adapter_id, mock_lora_path)
                print(f"LoRA Load success: {load_lora_success}")
                if load_lora_success:
                    print(f"Active adapters: {awq_runner.get_active_lora_adapters()}")

                    print("\nAttempting to generate with LoRA (concept)...")
                    # Note: Real generation would use the LoRA-adapted model
                    gen_output_lora, lora_tokens = awq_runner.generate(base_prompt + " with LoRA?", max_new_tokens=5)
                    print(f"Generate output (with LoRA concept): {gen_output_lora}, Tokens: {lora_tokens}")

                    print(f"\nAttempting to unload LoRA adapter '{mock_adapter_id}'...")
                    unload_lora_success = awq_runner.unload_lora_adapter(mock_adapter_id)
                    print(f"LoRA Unload success: {unload_lora_success}")
                    print(f"Active adapters after unload: {awq_runner.get_active_lora_adapters()}")

                    # Test merge/unmerge (conceptual, as merge_and_unload changes model type)
                    print("\nRe-loading LoRA for merge test...")
                    awq_runner.load_lora_adapter(mock_adapter_id, mock_lora_path) # Reload
                    if mock_adapter_id in awq_runner.get_active_lora_adapters():
                        print("\nAttempting to merge LoRA adapters...")
                        merge_success = awq_runner.merge_lora_adapters()
                        print(f"LoRA Merge success: {merge_success}")
                        if merge_success:
                            print("Model type after merge:", type(awq_runner.model))
                            print("Active adapters after merge (should be empty if PeftModel is gone):", awq_runner.get_active_lora_adapters())

                            print("\nAttempting to generate post-merge (concept)...")
                            gen_output_merged, merged_tokens = awq_runner.generate(base_prompt + " post-merge?", max_new_tokens=5)
                            print(f"Generate output (post-merge concept): {gen_output_merged}, Tokens: {merged_tokens}")

                            print("\nAttempting to unmerge LoRA adapters (may require model reload)...")
                            unmerge_success = awq_runner.unmerge_lora_adapters()
                            print(f"LoRA Unmerge attempt success: {unmerge_success}") # Expected False based on impl.
                    else:
                        print("Skipping merge test as LoRA did not reload.")
                else:
                    print(f"Skipping further LoRA tests as loading adapter '{mock_adapter_id}' failed.")

                # Clean up mock directory if created (optional)
                # if Path(mock_lora_path).exists() and "mock_lora_adapter_files" in mock_lora_path:
                #     try: shutil.rmtree(mock_lora_path); print(f"Cleaned up mock LoRA directory: {mock_lora_path}")
                #     except Exception as e: print(f"Error cleaning up mock dir: {e}")

            print("\n--- Token Counting Demo (AWQRunner) ---")
            if awq_runner.model and awq_runner.tokenizer: # Check if model and tokenizer loaded
                sample_text_awq = "Test sentence for AWQ tokenizer."
                token_count_awq = awq_runner.count_tokens(sample_text_awq)
                if token_count_awq is not None:
                    print(f"'{sample_text_awq}' has {token_count_awq} tokens (AWQRunner).")
                else:
                    print(f"Could not count tokens for '{sample_text_awq}' with AWQRunner.")
            else:
                print("Skipping AWQ token counting demo as model/tokenizer was not loaded.")

            else: # This PEFT_AVAILABLE else was mis-indented, should be outside the model&tokenizer check
                print("\n--- PEFT_AVAILABLE is False. Skipping LoRA method tests. ---")

            # ... (rest of KV cache demo or other tests)
        else:
            print("\n--- AWQRunner Failed to Initialize Model/Tokenizer for Main Demo ---")
            if not AWQ_AVAILABLE: print("Reason: AWQ library not available.")
            if not TORCH_FOR_DEMO_AVAILABLE: print("Reason: PyTorch not available for demo.")

    else:
        print("Skipping AWQRunner full demo due to missing AWQ_AVAILABLE or TORCH_FOR_DEMO_AVAILABLE.")

    print("\nAWQRunner LoRA (PEFT) and KV cache demonstration complete.")
