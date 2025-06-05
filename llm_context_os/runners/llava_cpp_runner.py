import time
import typing as t
from pathlib import Path
from .base import BaseRunner

LLAVA_CPP_AVAILABLE = False
LlavaLlama = None
LlavaImageEmbed = None

try:
    from llava_cpp import LlavaLlama, LlavaImageEmbed # type: ignore
    LLAVA_CPP_AVAILABLE = True
except ImportError:
    print("[LlavaCppRunner] llava-cpp-python not found. Please install it for LLaVA multimodal support.")
    # Keep LlavaLlama and LlavaImageEmbed as None


class LlavaCppRunner(BaseRunner):
    """
    Runner for LLaVA models using the llava-cpp-python library.
    Handles multimodal inputs (text and images).
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: str, # Path to the multimodal projector GGUF file
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        verbose: bool = False,
        # Add other llama.cpp specific params as needed
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.additional_params = kwargs
        self.model: t.Optional[LlavaLlama] = None
        self.image_embedder: t.Optional[LlavaImageEmbed] = None

        if not LLAVA_CPP_AVAILABLE:
            print(
                "[LlavaCppRunner] Warning: llava-cpp-python library not available. "
                "LlavaCppRunner will not be functional."
            )
            return

        try:
            print(f"[LlavaCppRunner] Initializing LlavaLlama model from: {self.model_path}")
            assert LlavaLlama is not None # Checked by LLAVA_CPP_AVAILABLE
            self.model = LlavaLlama.create_from_gguf(
                gguf_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
                # Pass only known Llama constructor params. Check llava_cpp.Llama docs.
                **{k: v for k, v in self.additional_params.items() if k in [
                    'seed', 'n_threads', 'n_batch', 'n_ubatch', 'logits_all',
                    'embedding', 'rope_freq_base', 'rope_freq_scale', 'yarn_ext_factor',
                    'yarn_attn_factor', 'yarn_beta_fast', 'yarn_beta_slow', 'yarn_orig_ctx',
                    'mul_mat_q', 'offload_kqv'
                    # Add other valid Llama/LlavaLlama params here
                ]}
            )
            print(f"[LlavaCppRunner] LlavaLlama model loaded. Now loading MM projector: {self.mmproj_path}")
            assert self.model is not None # Should be loaded if no exception
            assert LlavaImageEmbed is not None # Checked by LLAVA_CPP_AVAILABLE
            self.image_embedder = LlavaImageEmbed.load_from_gguf(
                gguf_path=self.mmproj_path,
                llama=self.model # Pass the initialized Llama object
            )
            print("[LlavaCppRunner] LlavaImageEmbed (multimodal projector) loaded successfully.")
        except Exception as e:
            print(f"[LlavaCppRunner] Error loading LLaVA model or multimodal projector: {e}")
            self.model = None
            self.image_embedder = None

    def _prepare_prompt_and_embeddings(
        self, prompt: str, image_paths: t.Optional[t.List[str]]
    ) -> tuple[str, t.Optional[list]]: # list will contain LlavaImageEmbed.EmbeddedImage instances
        """
        Prepares the prompt and embeds images if provided.
        LLaVA typically uses <image> placeholders in the prompt.
        This method assumes the prompt is already formatted with these placeholders.
        """
        if image_paths and self.image_embedder and self.model:
            print(f"[LlavaCppRunner] Embedding {len(image_paths)} images...")
            embedded_images = []
            try:
                for image_path_str in image_paths:
                    image_path_obj = Path(image_path_str)
                    if not image_path_obj.exists():
                        print(f"[LlavaCppRunner] Warning: Image path not found: {image_path_str}")
                        continue
                    assert self.image_embedder is not None
                    # embed_image returns LlavaImageEmbed.EmbeddedImage
                    img_embed = self.image_embedder.embed_image(image_path_obj)
                    embedded_images.append(img_embed)

                if not embedded_images: # If all image paths were invalid
                    return prompt, None
                return prompt, embedded_images
            except Exception as e:
                print(f"[LlavaCppRunner] Error during image embedding: {e}")
                return prompt, None
        return prompt, None

    def generate(
        self,
        prompt: str,
        image_paths: t.Optional[t.List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: t.Optional[t.List[str]] = None,
        **kwargs,
    ) -> str:
        if not LLAVA_CPP_AVAILABLE or not self.model:
            return "[LlavaCppRunner] Error: llava-cpp-python not available or model not loaded."
        if image_paths and (not self.image_embedder): # Check embedder specifically if images are passed
            return "[LlavaCppRunner] Error: Image paths provided, but multimodal projector not available/loaded."

        processed_prompt, embedded_images = self._prepare_prompt_and_embeddings(prompt, image_paths)

        if image_paths and embedded_images is None: # Check if embedding failed
             return "[LlavaCppRunner] Error: Image embedding failed. Cannot proceed with generation."

        # llava_cpp.LlavaLlama.generate params: text, image_embeds, temp, top_p, top_k, repeat_penalty, max_new_tokens, stop
        generation_params = {
            "temp": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "stop": stop or [],
            **{k: v for k, v in self.additional_params.items() if k in [
                'top_k', 'repeat_penalty', 'frequency_penalty', 'presence_penalty', 'tfs_z', 'mirostat_mode',
                'mirostat_tau', 'mirostat_eta'
            ]},
            **kwargs,
        }
        if generation_params["stop"] and None in generation_params["stop"]: # type: ignore
            generation_params["stop"] = [s for s in generation_params["stop"] if s is not None] # type: ignore


        print(f"[LlavaCppRunner] Generating text for prompt: '{processed_prompt[:50]}...' with {len(embedded_images or [])} images.")
        try:
            assert self.model is not None
            output_text = self.model.generate(
                text=processed_prompt,
                image_embeds=embedded_images, # list[LlavaImageEmbed.EmbeddedImage]
                **generation_params
            )
            return output_text
        except Exception as e:
            print(f"[LlavaCppRunner] Error during text generation: {e}")
            return f"[LlavaCppRunner] Error: {e}"

    def stream(
        self,
        prompt: str,
        image_paths: t.Optional[t.List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: t.Optional[t.List[str]] = None,
        **kwargs,
    ) -> t.Generator[str, None, None]:
        if not LLAVA_CPP_AVAILABLE or not self.model:
            yield "[LlavaCppRunner] Error: llava-cpp-python not available or model not loaded."
            return
        if image_paths and (not self.image_embedder):
            yield "[LlavaCppRunner] Error: Image paths provided, but multimodal projector not available/loaded."
            return

        processed_prompt, embedded_images = self._prepare_prompt_and_embeddings(prompt, image_paths)

        if image_paths and embedded_images is None:
            yield "[LlavaCppRunner] Error: Image embedding failed. Cannot proceed with streaming."
            return

        generation_params = {
            "temp": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "stop": stop or [],
            **{k: v for k, v in self.additional_params.items() if k in [
                'top_k', 'repeat_penalty', 'frequency_penalty', 'presence_penalty', 'tfs_z', 'mirostat_mode',
                'mirostat_tau', 'mirostat_eta'
            ]},
            **kwargs,
        }
        if generation_params["stop"] and None in generation_params["stop"]: # type: ignore
            generation_params["stop"] = [s for s in generation_params["stop"] if s is not None] # type: ignore

        print(f"[LlavaCppRunner] Streaming text for prompt: '{processed_prompt[:50]}...' with {len(embedded_images or [])} images.")
        try:
            assert self.model is not None
            # Check if self.model.generate has a 'stream' parameter
            import inspect
            sig = inspect.signature(self.model.generate)
            if 'stream' in sig.parameters: # pragma: no cover
                # This path is taken if llava-cpp-python's generate method supports a 'stream' flag
                generation_params['stream'] = True
                token_generator = self.model.generate(
                    text=processed_prompt,
                    image_embeds=embedded_images,
                    **generation_params
                )
                for token_chunk in token_generator: # type: ignore
                    if isinstance(token_chunk, dict) and 'text' in token_chunk:
                        yield token_chunk['text']
                    elif isinstance(token_chunk, str):
                         yield token_chunk
            else:
                # Fallback: If no 'stream' param, generate full response and yield as one chunk.
                # This is based on the observation that llava_cpp.LlavaLlama.generate returns str.
                print("[LlavaCppRunner] Warning: `llava_cpp.LlavaLlama.generate` may not support true streaming. Simulating.")
                full_response = self.model.generate(
                    text=processed_prompt,
                    image_embeds=embedded_images,
                    **generation_params # type: ignore
                )
                yield full_response

        except Exception as e:
            print(f"[LlavaCppRunner] Error during text streaming: {e}")
            yield f"[LlavaCppRunner] Error: {e}"

    def count_tokens(self, text: str, **kwargs) -> int:
        if not self.model:
            print("[LlavaCppRunner] Warning: Model not loaded, cannot count tokens accurately. Returning char count.")
            return len(text)
        try:
            tokenized = self.model.tokenize(text.encode("utf-8"))
            return len(tokenized)
        except Exception as e:
            print(f"[LlavaCppRunner] Error tokenizing text: {e}. Returning char count.")
            return len(text)

    # --- KV Cache and LoRA Methods ---
    # LlavaLlama inherits from llama_cpp.Llama.
    def get_kv_cache_info(self) -> t.Dict[str, t.Any]: # pragma: no cover
        if self.model and hasattr(self.model, "get_kv_cache_info"):
            try:
                info = self.model.get_kv_cache_info()
                return {"size": getattr(info, 'size', 0),
                        "used": getattr(info, 'used_cells', 0),
                        "sequence_length": getattr(info, 'seq_len', 0)}
            except Exception as e:
                print(f"[LlavaCppRunner] Error getting KV cache info: {e}")
                return {"error": f"Could not retrieve KV cache info: {e}"}
        return {"error": "Model not loaded or KV cache info not available."}

    def save_kv_cache_session(self, filepath: str) -> None: # pragma: no cover
        if self.model and hasattr(self.model, "save_session"):
            try:
                self.model.save_session(filepath.encode("utf-8"))
                print(f"[LlavaCppRunner] KV cache session saved to {filepath}.")
            except Exception as e:
                print(f"[LlavaCppRunner] Error saving KV cache session: {e}")
        else:
            print("[LlavaCppRunner] Model not loaded or does not support KV cache session saving.")

    def load_kv_cache_session(self, filepath: str) -> None: # pragma: no cover
        if self.model and hasattr(self.model, "load_session"):
            try:
                self.model.load_session(filepath.encode("utf-8"))
                print(f"[LlavaCppRunner] KV cache session loaded from {filepath}.")
            except Exception as e:
                print(f"[LlavaCppRunner] Error loading KV cache session: {e}")
        else:
            print("[LlavaCppRunner] Model not loaded or does not support KV cache session loading.")

    def list_available_loras(self) -> t.List[str]: # pragma: no cover
        if self.model and hasattr(self.model, "list_loras"):
            try:
                return self.model.list_loras()
            except Exception as e:
                print(f"[LlavaCppRunner] Error listing LoRAs: {e}")
                return ["Error listing LoRAs."]
        return ["LoRA support not available or model not loaded."]

    def apply_lora(
        self, lora_path: str, scale: float = 1.0, adapter_name: t.Optional[str] = None
    ) -> None: # pragma: no cover
        if self.model and hasattr(self.model, "apply_lora_from_file"):
            try:
                # llama-cpp-python's apply_lora_from_file:
                # (lora_path: str, scale: float = 1.0, lora_base: Optional[str] = None, n_threads: Optional[int] = None)
                self.model.apply_lora_from_file(lora_path=lora_path, scale=scale)
                print(f"[LlavaCppRunner] Applied LoRA from {lora_path} with scale {scale}.")
            except Exception as e:
                print(f"[LlavaCppRunner] Error applying LoRA from {lora_path}: {e}")
        else:
            print("[LlavaCppRunner] Model not loaded or does not support applying LoRAs.")

    def remove_lora(self, adapter_name: t.Optional[str] = None) -> None: # pragma: no cover
        if self.model and hasattr(self.model, "disable_lora"): # `disable_lora` is the method in llama-cpp-python
            try:
                self.model.disable_lora()
                print("[LlavaCppRunner] Disabled active LoRA.")
            except Exception as e:
                print(f"[LlavaCppRunner] Error disabling LoRA: {e}")
        else:
            print("[LlavaCppRunner] Model not loaded or does not support disabling LoRA.")


if __name__ == "__main__": # pragma: no cover
    MODEL_PATH_ENV = "LLAVA_MODEL_PATH"  # e.g., /path/to/llava-v1.5-7b-Q4_K_M.gguf
    MMPROJ_PATH_ENV = "LLAVA_MMPROJ_PATH" # e.g., /path/to/mmproj-model-f16.gguf
    IMAGE_PATH_ENV = "LLAVA_TEST_IMAGE_PATH" # e.g., /path/to/your/test_image.jpg

    import os
    model_path_main = os.getenv(MODEL_PATH_ENV, "dummy_llava_model.gguf")
    mmproj_path_main = os.getenv(MMPROJ_PATH_ENV, "dummy_mmproj.gguf")
    image_path_main = os.getenv(IMAGE_PATH_ENV, "dummy_image.jpg")

    # For local testing, you would set the environment variables or replace the defaults above.
    # Example:
    # model_path_main = "/Users/me/models/gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf"
    # mmproj_path_main = "/Users/me/models/gguf/mmproj-mistral7b-f16-q6_k.gguf" # Or appropriate for model
    # image_path_main = "/Users/me/Pictures/test_image.png"


    print(f"--- Testing LlavaCppRunner (LLAVA_CPP_AVAILABLE: {LLAVA_CPP_AVAILABLE}) ---")
    print(f"Model Path: {model_path_main}")
    print(f"MMProj Path: {mmproj_path_main}")
    print(f"Test Image Path: {image_path_main}")

    # Create dummy files if the library is not available or if paths are dummy paths
    # This allows the __main__ to run without failing on file existence for the LlavaLlama/LlavaImageEmbed loaders
    # if they were to check path existence before the LLAVA_CPP_AVAILABLE check.
    dummy_model_created = False
    dummy_mmproj_created = False
    dummy_image_created = False

    if not LLAVA_CPP_AVAILABLE or model_path_main == "dummy_llava_model.gguf":
        if not Path(model_path_main).exists():
            Path(model_path_main).touch()
            dummy_model_created = True
    if not LLAVA_CPP_AVAILABLE or mmproj_path_main == "dummy_mmproj.gguf":
        if not Path(mmproj_path_main).exists():
            Path(mmproj_path_main).touch()
            dummy_mmproj_created = True
    if image_path_main == "dummy_image.jpg":
        if not Path(image_path_main).exists():
            Path(image_path_main).write_text("dummy image content")
            dummy_image_created = True


    runner = LlavaCppRunner(
        model_path=model_path_main,
        mmproj_path=mmproj_path_main,
        n_gpu_layers=0, # Set >0 if you have GPU and compiled llava.cpp with GPU support
        n_ctx=2048,
        verbose=False, # True for more LLaMA details
    )

    if LLAVA_CPP_AVAILABLE and runner.model and runner.image_embedder:
        print("\n--- LlavaCppRunner Initialized with llava-cpp-python ---")

        # 1. Test generate with text only
        print("\n--- Testing generate (text only) ---")
        prompt_text_only = "USER: What is the capital of France?\nASSISTANT:"
        response_text_only = runner.generate(prompt_text_only, max_tokens=50)
        print(f"Response (text only): {response_text_only}")

        # 2. Test generate with text and image
        if Path(image_path_main).exists():
            print("\n--- Testing generate (text + image) ---")
            # LLaVA prompt format often includes <image> placeholder where image features are inserted.
            # The text prompt itself might also be structured, e.g. "USER: <image>\nDescribe this image.\nASSISTANT:"
            prompt_with_image_placeholder = "USER: <image>\nWhat is in this image?\nASSISTANT:"
            response_with_image = runner.generate(
                prompt_with_image_placeholder, image_paths=[image_path_main], max_tokens=100
            )
            print(f"Response (with image): {response_with_image}")
        else:
            print(f"Skipping generate with image test: Image file not found at {image_path_main}")

        # 3. Test stream with text only
        print("\n--- Testing stream (text only) ---")
        full_streamed_text = ""
        for chunk in runner.stream(prompt_text_only, max_tokens=50):
            # print(chunk, end="", flush=True) # For live printing
            full_streamed_text += chunk
        # print()
        print(f"Full streamed response (text only): {full_streamed_text}")


        # 4. Test stream with text and image
        if Path(image_path_main).exists():
            print("\n--- Testing stream (text + image) ---")
            prompt_with_image_placeholder_stream = "USER: <image>\nTell me a short story about this image.\nASSISTANT:"
            full_streamed_image_text = ""
            for chunk in runner.stream(
                prompt_with_image_placeholder_stream, image_paths=[image_path_main], max_tokens=100
            ):
                # print(chunk, end="", flush=True) # For live printing
                full_streamed_image_text += chunk
            # print()
            print(f"Full streamed response (with image): {full_streamed_image_text}")
        else:
            print(f"Skipping stream with image test: Image file not found at {image_path_main}")

        # 5. Test token counting
        print("\n--- Testing token counting ---")
        count = runner.count_tokens(prompt_text_only)
        print(f"Token count for '{prompt_text_only}': {count}")

    elif LLAVA_CPP_AVAILABLE:
        print("\n--- LlavaCppRunner Initialization Failed (llava-cpp-python available but model/projector load failed) ---")
        print("Please ensure valid GGUF model and multimodal projector paths are provided via environment variables or in the script.")
        print(f"  Tried Model: {model_path_main}")
        print(f"  Tried MMProj: {mmproj_path_main}")
        print("  And that llava-cpp-python is correctly installed and built for your system (CPU/GPU).")
    else:
        print("\n--- LlavaCppRunner Not Functional (llava-cpp-python not found) ---")
        print("Skipping functional tests. Basic instantiation checks passed.")
        print(f"  Model path stored: {runner.model_path}")
        print(f"  MMProj path stored: {runner.mmproj_path}")
        # Test basic placeholder calls (these will just print warnings/errors from the class)
        print(f"  Generate (placeholder call): {runner.generate('test prompt')}")
        print("  Stream (placeholder call): ", end="")
        for chunk in runner.stream("test prompt"): # Should yield error message
            print(chunk, end="")
        print()
        print(f"  Count tokens (placeholder call): {runner.count_tokens('test prompt')}")


    # Clean up dummy files
    if dummy_model_created: Path(model_path_main).unlink(missing_ok=True)
    if dummy_mmproj_created: Path(mmproj_path_main).unlink(missing_ok=True)
    if dummy_image_created: Path(image_path_main).unlink(missing_ok=True)
    if dummy_model_created or dummy_mmproj_created or dummy_image_created:
        print("\nDummy files cleaned up.")

    print("\n--- LlavaCppRunner Test Complete ---")
