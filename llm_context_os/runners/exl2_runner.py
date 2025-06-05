# llm_context_os/runners/exl2_runner.py
import typing as t
import os
import torch # exllamav2 requires torch
from .base import BaseRunner

EXL2_AVAILABLE = False
try:
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Config,
        ExLlamaV2Cache,
        ExLlamaV2Cache_8bit, # Optional: For 8-bit KV cache
        ExLlamaV2Tokenizer,
        ExLlamaV2Generator,
        ExLlamaV2Lora # For LoRA support
    )
    from exllamav2.generator import ExLlamaV2Sampler # For settings
    EXL2_AVAILABLE = True
    print("Successfully imported exllamav2.")
except ImportError:
    print("Warning: exllamav2 library not found. EXL2Runner will not be functional.")
    ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Cache_8bit, ExLlamaV2Tokenizer, ExLlamaV2Generator, ExLlamaV2Sampler, ExLlamaV2Lora = (None,)*8


class EXL2Runner(BaseRunner):
    def __init__(self,
                 model_path: str,
                 gpu_split: t.Optional[str] = None, # e.g., "auto" or "20,24" in GB
                 max_seq_len: t.Optional[int] = None, # If None, will use model's default
                 use_8bit_cache: bool = False,
                 **kwargs: t.Any): # kwargs for ExLlamaV2Sampler.Settings

        super().__init__() # Initialize base class attributes like self.loras
        self.model_path = model_path
        self.gpu_split_str = gpu_split
        self.user_max_seq_len = max_seq_len
        self.use_8bit_cache = use_8bit_cache
        self.sampler_settings_kwargs = kwargs # Store for re-use in generate/stream

        self.config: t.Optional[ExLlamaV2Config] = None
        self.model: t.Optional[ExLlamaV2] = None
        self.tokenizer: t.Optional[ExLlamaV2Tokenizer] = None
        self.cache: t.Optional[t.Union[ExLlamaV2Cache, ExLlamaV2Cache_8bit]] = None
        self.generator: t.Optional[ExLlamaV2Generator] = None
        self._active_loras_objects: t.List[ExLlamaV2Lora] = [] # Store actual LoRA objects

        if not EXL2_AVAILABLE:
            print("Error: EXL2Runner cannot be initialized because exllamav2 library is not available.")
            return

        try:
            print(f"Initializing EXL2Runner for model directory: {self.model_path}")
            self.config = ExLlamaV2Config()
            self.config.model_dir = self.model_path
            self.config.prepare() # Detects model type, loads .json files

            if self.user_max_seq_len:
                self.config.max_seq_len = self.user_max_seq_len
            else:
                # Use model's default if not overridden by user
                self.user_max_seq_len = self.config.max_seq_len

            print(f"  Model config loaded. Max sequence length set to: {self.config.max_seq_len}")

            self.model = ExLlamaV2(self.config)

            # Cache selection
            CacheClass = ExLlamaV2Cache_8bit if self.use_8bit_cache else ExLlamaV2Cache
            self.cache = CacheClass(self.model, batch_size=1, max_seq_len=self.config.max_seq_len) # batch_size=1 for typical generation
            print(f"  Using {'8-bit' if self.use_8bit_cache else 'standard'} KV cache.")

            print("  Loading model weights...")
            if self.gpu_split_str:
                if self.gpu_split_str.lower() == "auto":
                    print("    Attempting auto GPU split.")
                    self.model.load_autosplit(self.cache)
                else:
                    try:
                        split_mb = [float(x.strip()) for x in self.gpu_split_str.split(",")]
                        print(f"    Attempting manual GPU split (MB per GPU): {split_mb}")
                        self.model.load(split_mb)
                    except ValueError:
                        print(f"    Error: Invalid gpu_split string '{self.gpu_split_str}'. Loading on single device.")
                        self.model.load_autosplit(self.cache) # Fallback or load on default
            else:
                print("    No GPU split specified, loading on default device (cuda:0 if available).")
                # For single device, load_autosplit is fine, or manually specify device if needed
                self.model.load_autosplit(self.cache)

            print("  Model weights loaded successfully.")

            self.tokenizer = ExLlamaV2Tokenizer(self.config)
            self.generator = ExLlamaV2Generator(self.model, self.cache, self.tokenizer)

            print("EXL2Runner initialized successfully.")

        except Exception as e:
            print(f"Error during EXL2Runner initialization: {e}")
            self.model = None # Ensure model is None if init fails
            # Potentially re-raise or handle more gracefully

    def _get_sampler_settings(self, **kwargs: t.Any) -> ExLlamaV2Sampler.Settings:
        settings = ExLlamaV2Sampler.Settings()

        # Apply defaults from init_kwargs first, then override with call-specific kwargs
        combined_kwargs = {**self.sampler_settings_kwargs, **kwargs}

        settings.temperature = combined_kwargs.get("temperature", 0.85)
        settings.top_k = combined_kwargs.get("top_k", 50)
        settings.top_p = combined_kwargs.get("top_p", 0.8)
        settings.token_repetition_penalty = combined_kwargs.get("token_repetition_penalty", 1.05)
        settings.disallow_tokens(self.tokenizer, combined_kwargs.get("disallow_tokens", [])) # Example of disallowing specific tokens

        # Add more settings as needed from ExLlamaV2Sampler.Settings
        # e.g., typical_p, min_p, top_a, mirostat, grammar_string
        if "mirostat" in combined_kwargs: settings.mirostat = combined_kwargs["mirostat"]
        if "mirostat_tau" in combined_kwargs: settings.mirostat_tau = combined_kwargs["mirostat_tau"]
        if "mirostat_eta" in combined_kwargs: settings.mirostat_eta = combined_kwargs["mirostat_eta"]

        return settings

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
        if not EXL2_AVAILABLE or not self.model or not self.generator or not self.tokenizer or not self.config:
            error_msg = "Error: EXL2Runner is not available or not properly initialized."
            print(f"[EXL2Runner] {error_msg}")
            return error_msg, {"prompt_tokens": 0, "completion_tokens": 0}

        # ... (logging as before) ...
        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Generating ---")
        print(f"Prompt: {prompt[:100]}...")
        if image_paths: print(f"  Image Paths: {image_paths} (Note: EXL2Runner base class does not support multimodal image input with text.)")
        if self._active_loras_objects: print(f"  Active LoRAs (applied): {list(self.loras.keys())}")


        settings = self._get_sampler_settings(**kwargs)
        # Default max_new_tokens, ensuring space for the prompt itself.
        # Tokenizer.encode() returns a tensor of shape (1, num_tokens).
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = prompt_ids.shape[-1]

        # Calculate max_new_tokens intelligently
        # Max new tokens should not exceed model's context length minus prompt tokens minus some buffer (e.g. 10 for safety)
        model_max_seq_len = self.config.max_seq_len
        max_possible_new_tokens = model_max_seq_len - prompt_tokens - 10

        # User can specify max_new_tokens or max_tokens (alias)
        user_max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_tokens'))

        if user_max_new_tokens is not None:
            max_new_tokens = min(user_max_new_tokens, max_possible_new_tokens)
        else: # No user value, use calculated max possible
            max_new_tokens = max_possible_new_tokens

        if max_new_tokens <= 0:
            error_msg = f"Error: Prompt length ({prompt_tokens}) is too close to or exceeds model max sequence length ({model_max_seq_len}). Cannot generate new tokens."
            print(f"[EXL2Runner] {error_msg}")
            return error_msg, {"prompt_tokens": prompt_tokens, "completion_tokens": 0}

        print(f"[EXL2Runner] Generating with settings: {settings}, max_new_tokens: {max_new_tokens}")

        full_text = self.generator.generate_simple(
            prompt=prompt, # generate_simple can take string directly
            gen_settings=settings,
            num_tokens=max_new_tokens, # generate_simple uses num_tokens for max_new_tokens
            seed=kwargs.get("seed")
        )

        generated_text_only = ""
        if full_text.startswith(prompt):
            generated_text_only = full_text[len(prompt):]
        else: # Fallback if prompt somehow not prepended (should not happen with generate_simple)
            generated_text_only = full_text
            print("[EXL2Runner] Warning: Full generated text did not start with the prompt.")

        completion_ids = self.tokenizer.encode(generated_text_only)
        completion_tokens = completion_ids.shape[-1]

        token_counts = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        print(f"  Tokens: prompt={prompt_tokens}, completion={completion_tokens}")
        return generated_text_only, token_counts


    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
        if not EXL2_AVAILABLE or not self.model or not self.generator or not self.tokenizer or not self.config:
            error_msg = "Error: EXL2Runner is not available or not properly initialized."
            print(f"[EXL2Runner] {error_msg}")
            yield {"prompt_tokens": 0}
            yield (error_msg, 0)
            return

        # ... (logging as before) ...
        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Streaming ---")
        print(f"Prompt: {prompt[:100]}...")
        if image_paths: print(f"  Image Paths: {image_paths} (Note: EXL2Runner base class does not support multimodal image input with text.)")
        if self._active_loras_objects: print(f"  Active LoRAs (applied): {list(self.loras.keys())}")

        settings = self._get_sampler_settings(**kwargs)

        input_ids = self.tokenizer.encode(prompt)
        prompt_tokens = input_ids.shape[-1]
        yield {"prompt_tokens": prompt_tokens}
        print(f"  Yielded prompt_tokens: {prompt_tokens}")

        # Calculate max_new_tokens for stream based on remaining context window
        model_max_seq_len = self.config.max_seq_len
        max_possible_new_tokens = model_max_seq_len - prompt_tokens - 10 # Buffer

        user_max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_tokens'))
        if user_max_new_tokens is not None:
            max_new_tokens_to_generate = min(user_max_new_tokens, max_possible_new_tokens)
        else:
            max_new_tokens_to_generate = max_possible_new_tokens

        if max_new_tokens_to_generate <= 0:
            yield (f"Error: Prompt length ({prompt_tokens}) near/exceeds model max sequence length ({model_max_seq_len}).", 0)
            return

        print(f"[EXL2Runner] Streaming with settings: {settings}, max_new_tokens_to_generate: {max_new_tokens_to_generate}")
        self.generator.begin_stream(input_ids, settings)

        generated_token_count_in_stream = 0
        try:
            while True:
                chunk, eos, _ = self.generator.stream() # _ is list of probabilities if enabled
                if chunk: # Can be empty if only BOS token, or if sampling produces empty string for a token
                    tokens_in_chunk = self.tokenizer.encode(chunk).shape[-1] if chunk else 0
                    yield (chunk, tokens_in_chunk)
                generated_token_count_in_stream += 1 # ExLlamaV2 streams token by token (text is decoded token)
                if eos or generated_token_count_in_stream >= max_new_tokens_to_generate:
                    break
        except Exception as e_stream:
            print(f"[EXL2Runner] Error during model streaming: {e_stream}")
            yield (f"[EXL2Runner] Error streaming response: {e_stream}", 0)
        finally:
            print("\nStreaming complete.")


    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        if not EXL2_AVAILABLE or not self.model or not self.cache or not self.tokenizer:
            print("Warning: EXL2Runner not initialized, cannot preload KV cache.")
            return

        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Preloading KV Cache ---")
        print(f"Prompt for KV: {prompt[:100]}...")

        input_ids = self.tokenizer.encode(prompt)
        # Ensure input_ids are not too long
        if input_ids.shape[-1] > self.config.max_seq_len:
            print(f"Warning: Prompt for KV preload ({input_ids.shape[-1]} tokens) exceeds max_seq_len ({self.config.max_seq_len}). Truncating.")
            input_ids = input_ids[:, :self.config.max_seq_len]

        # Reset cache or ensure it's in a state to be prefilled.
        # For ExLlamaV2Cache, it's stateful. Re-using the main self.cache is intended.
        # The `preprocess_only=True` will fill the cache without actual sampling.
        try:
            # Pass active LoRAs if any
            current_loras = self._active_loras_objects if self._active_loras_objects else None
            self.model.forward(input_ids, self.cache, preprocess_only=True, loras=current_loras)
            print(f"  KV Cache preloaded with {input_ids.shape[-1]} tokens.")
            # The cache is now populated. Subsequent generate/stream calls starting with a
            # continuation of this prompt (or the same prompt) will benefit.
        except Exception as e:
            print(f"  Error during KV cache preload: {e}")
        super().preload_kv(prompt, **kwargs) # Call base for logging

    def export_kv_cache(self) -> t.Any:
        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Exporting KV Cache ---")
        print("  NOTE: ExLlamaV2Cache objects are complex stateful objects tied to the model.")
        print("  Direct Python object serialization for transfer is not a standard feature.")
        print("  Returning the live cache object for potential in-process reuse if applicable, but this is limited.")
        if not EXL2_AVAILABLE or not self.cache:
            return None
        # Consider if specific state attributes can be extracted and returned if needed.
        # For now, this indicates that true "export" in a portable sense is not implemented.
        raise NotImplementedError("Exporting ExLlamaV2Cache state in a portable format is not implemented. The cache is stateful and model-dependent.")

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Importing KV Cache ---")
        print("  NOTE: Importing an ExLlamaV2Cache object or its state is non-trivial and model-specific.")
        # if not EXL2_AVAILABLE or not self.cache:
        #     print("  Error: Runner or cache not initialized.")
        #     return
        # if isinstance(cache_data, type(self.cache)):
        #     # This is a simplification. Real cache import would be much more complex,
        #     # potentially requiring model re-initialization with the cache or specific load methods.
        #     # Also, deepcopy might be needed if the cache_data is to be used elsewhere.
        #     print("  Warning: Attempting to assign cache_data. This is experimental and may not work as expected for ExLlamaV2Cache.")
        #     self.cache = cache_data # This is unlikely to be robust.
        # else:
        #     print(f"  Error: Provided cache_data is not a compatible ExLlamaV2Cache object. Type: {type(cache_data)}")
        raise NotImplementedError("Importing ExLlamaV2Cache state is not implemented due to its complexity and model-dependent nature.")

    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        if not EXL2_AVAILABLE or not self.model or not self.generator:
            print("Error: EXL2Runner not initialized, cannot load LoRA adapter.")
            return False

        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Loading LoRA Adapter ---")
        print(f"  ID/Name: {adapter_id}, Path: {adapter_path}")

        if adapter_id in self.loras:
            print(f"  Warning: LoRA adapter '{adapter_id}' already loaded.")
            return True # Or False if re-loading is an error

        try:
            lora = ExLlamaV2Lora.from_directory(self.model, adapter_path)
            self.loras[adapter_id] = lora # Store in the base class dict for tracking by ID
            self._active_loras_objects.append(lora) # Add to list for generator
            self.generator.set_loras(self._active_loras_objects) # Apply all currently active LoRAs
            print(f"  LoRA adapter '{adapter_id}' loaded and applied successfully.")
            return True
        except Exception as e:
            print(f"  Error loading LoRA adapter '{adapter_id}' from {adapter_path}: {e}")
            return False

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        if not EXL2_AVAILABLE or not self.generator:
            print("Error: EXL2Runner not initialized, cannot unload LoRA adapter.")
            return False

        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Unloading LoRA Adapter ---")
        print(f"  ID: {adapter_id}")

        if adapter_id not in self.loras:
            print(f"  Warning: LoRA adapter '{adapter_id}' not found or not loaded.")
            return False

        lora_to_remove = self.loras.pop(adapter_id) # Remove from tracking dict

        # Rebuild the list of active LoRA objects
        new_active_loras_objects = []
        for id_key, lora_obj in self.loras.items():
             new_active_loras_objects.append(lora_obj)
        self._active_loras_objects = new_active_loras_objects

        self.generator.set_loras(self._active_loras_objects if self._active_loras_objects else None)
        print(f"  LoRA adapter '{adapter_id}' unloaded. Active LoRAs updated.")
        return True

    def get_active_lora_adapters(self) -> t.List[str]:
        # Returns IDs of LoRAs that were successfully loaded and are intended to be active
        return super().get_active_lora_adapters()


    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Merging LoRA Adapters ---")
        raise NotImplementedError("Runtime merging of LoRA adapters is not a standard feature in ExLlamaV2. Merging is typically an offline model conversion process.")

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print(f"\n--- EXL2Runner ({os.path.basename(self.model_path)}) Unmerging LoRA Adapters ---")
        raise NotImplementedError("Runtime unmerging of LoRA adapters is not applicable as merging is offline.")

    def count_tokens(self, text: str) -> Optional[int]:
        """Counts tokens using the loaded ExLlamaV2Tokenizer."""
        if not self.tokenizer or not EXL2_AVAILABLE:
            print("[EXL2Runner] Tokenizer not available, cannot count tokens.")
            return None
        try:
            # ExLlamaV2Tokenizer.encode() returns a tensor of shape (1, num_tokens)
            input_ids = self.tokenizer.encode(text)
            return input_ids.shape[-1] # Get the last dimension for the token count
        except Exception as e:
            print(f"[EXL2Runner] Error tokenizing text: {e}")
            return None


if __name__ == '__main__':
    print(f"EXL2 Available for __main__ test: {EXL2_AVAILABLE}")
    if EXL2_AVAILABLE:
        # --- IMPORTANT ---
        # To run this example, you MUST set the environment variable:
        # 1. EXL2_MODEL_PATH: Full path to your EXL2 model directory.
        #    (e.g., "/mnt/models/exl2/MythoMax-L2-13B-EXL2_4.0bpw")
        #
        # Optional:
        # 2. EXL2_GPU_SPLIT: "auto" or comma-separated MBs (e.g., "20000,20000")
        # 3. EXL2_LORA_PATH: Path to a LoRA adapter directory compatible with the model.

        model_directory = os.getenv("EXL2_MODEL_PATH")
        gpu_split_config = os.getenv("EXL2_GPU_SPLIT") # e.g., "auto" or None
        lora_directory = os.getenv("EXL2_LORA_PATH")   # Optional

        if not model_directory:
            print("Error: EXL2_MODEL_PATH environment variable not set. Cannot run example.")
        else:
            print(f"Attempting to load model from: {model_directory}")
            if gpu_split_config: print(f"Using GPU split config: {gpu_split_config}")
            if lora_directory: print(f"Test LoRA path (optional): {lora_directory}")

            exl2_runner = EXL2Runner(
                model_path=model_directory,
                gpu_split=gpu_split_config,
                max_seq_len=4096, # Or adjust as needed
                use_8bit_cache=False, # Set to True to test 8-bit cache
                # Sampler settings can be passed here as defaults
                temperature=0.7,
                top_k=50,
                top_p=0.8
            )

            if exl2_runner.model: # Check if initialization was successful
                test_prompt = "Once upon a time, in a land of code and circuits,"

                # Test LoRA loading if path provided
                if lora_directory:
                    if os.path.exists(lora_directory):
                        print(f"\n--- Testing LoRA Loading ---")
                        load_success = exl2_runner.load_lora_adapter("my_test_lora", lora_directory)
                        print(f"LoRA 'my_test_lora' load attempt: {'Success' if load_success else 'Failed'}")
                        if load_success:
                             print(f"Active LoRAs: {exl2_runner.get_active_lora_adapters()}")
                    else:
                        print(f"LoRA directory not found: {lora_directory}, skipping LoRA load test.")

                # Test KV Preload
                print(f"\n--- Testing KV Cache Preload ---")
                exl2_runner.preload_kv(test_prompt[:50]) # Preload with a part of the prompt


                print(f"\n--- Testing generate() for model: {os.path.basename(exl2_runner.model_path)} ---")
                try:
                    # Override default sampler settings for this specific call if needed
                    response_text, token_counts = exl2_runner.generate(test_prompt, max_new_tokens=100, temperature=0.75)
                    print(f"\nGenerate call response:\n'{response_text}'")
                    print(f"Token counts from generate: {token_counts}")
                except Exception as e:
                    print(f"Error during generate(): {e}")

                print(f"\n--- Testing stream() for model: {os.path.basename(exl2_runner.model_path)} ---")
                try:
                    full_streamed_response_text = []
                    stream_completion_tokens = 0
                    print("Streamed response:")

                    stream_gen = exl2_runner.stream(test_prompt, max_new_tokens=120, top_p=0.5)
                    prompt_token_info = next(stream_gen)
                    print(f"\nPrompt token info from stream: {prompt_token_info}")

                    for item in stream_gen:
                        if isinstance(item, tuple):
                            text_chunk, tokens_in_chunk = item
                            print(text_chunk, end="", flush=True)
                            full_streamed_response_text.append(text_chunk)
                            stream_completion_tokens += tokens_in_chunk
                        else: # Should not happen with current implementation
                            print(f"\nUnexpected stream item: {item}")

                    print(f"\n\nFull streamed response collected: '{''.join(full_streamed_response_text)}'")
                    print(f"Total completion tokens from stream (calculated): {stream_completion_tokens}")
                except Exception as e:
                    print(f"Error during stream(): {e}")

                # Test unloading LoRA
                if lora_directory and "my_test_lora" in exl2_runner.get_active_lora_adapters():
                    print(f"\n--- Testing LoRA Unloading ---")
                    unload_success = exl2_runner.unload_lora_adapter("my_test_lora")
                    print(f"LoRA 'my_test_lora' unload attempt: {'Success' if unload_success else 'Failed'}")
                    print(f"Active LoRAs after unload: {exl2_runner.get_active_lora_adapters()}")

                print("\n--- Token Counting Demo (EXL2Runner) ---")
                if exl2_runner.model and exl2_runner.tokenizer: # Check if model and tokenizer loaded
                    sample_text_exl2 = "Test sentence for EXL2 tokenizer."
                    token_count_exl2 = exl2_runner.count_tokens(sample_text_exl2)
                    if token_count_exl2 is not None:
                        print(f"'{sample_text_exl2}' has {token_count_exl2} tokens (EXL2Runner).")
                    else:
                        print(f"Could not count tokens for '{sample_text_exl2}' with EXL2Runner.")
                else:
                    print("Skipping EXL2 token counting demo as model/tokenizer was not loaded.")
            else:
                print("EXL2Runner model initialization failed. Cannot run generation tests.")
    else:
        print("Skipping EXL2Runner __main__ example as exllamav2 library is not available.")

    print("\nEXL2Runner demonstration complete.")
