# yawl/runners/vllm_runner.py
import typing as t
import json
import httpx # Added
from .base import BaseRunner

class VLLMRunner(BaseRunner):
    def __init__(self, model_name: str, api_url: str = "http://localhost:8000",
                 api_key: t.Optional[str] = None, **kwargs):
        self.model_name = model_name # This is the 'model' field in the OpenAI-compatible API payload
        self.api_url = api_url # This should be the base URL, e.g., http://localhost:8000
        self.api_key = api_key
        # For OpenAI-compatible vLLM, the endpoint is typically /v1/completions or /v1/chat/completions
        # Assuming /v1/completions for now as per original structure.
        # If using /v1/chat/completions, the payload structure in generate/stream would need to change.
        self.completion_endpoint = "/v1/completions"

        self.http_client = httpx.Client(
            base_url=self.api_url,
            timeout=kwargs.get("timeout", 60.0) # Allow timeout to be passed in kwargs
        )
        print(f"[VLLMRunner] Initialized for model '{self.model_name}'. HTTP client configured for API URL: {self.api_url}")
        if self.api_key:
            print(f"[VLLMRunner] API Key provided (ending with ...{self.api_key[-4:] if len(self.api_key) > 4 else '****'})")

    def _prepare_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
        if image_paths:
            print(f"[VLLMRunner] Warning: image_paths provided but standard vLLM OpenAI-compatible /v1/completions endpoint does not support them directly. Ignoring images: {image_paths}")

        headers = self._prepare_headers()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get('max_new_tokens', kwargs.get('max_tokens', 128)),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 1.0),
            "n": kwargs.get('n', 1),
            "stop": kwargs.get('stop', None),
            "stream": False,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        # Allow direct pass-through of other kwargs that might be vLLM specific for /completions
        payload.update({k:v for k,v in kwargs.items() if k not in payload})


        print(f"[VLLMRunner] Calling generate on {self.completion_endpoint} for model '{self.model_name}'")

        prompt_tokens = 0
        completion_tokens = 0
        generated_text = ""

        try:
            response = self.http_client.post(
                self.completion_endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            response_data = response.json()

            generated_text = response_data.get("choices", [{}])[0].get("text", "").strip()

            if "usage" in response_data:
                prompt_tokens = response_data["usage"].get("prompt_tokens", 0)
                completion_tokens = response_data["usage"].get("completion_tokens", 0)
            else: # Fallback if no usage field
                if hasattr(self, 'count_tokens') and callable(self.count_tokens): # Check if count_tokens is available
                    prompt_tokens_est = self.count_tokens(prompt)
                    prompt_tokens = prompt_tokens_est if prompt_tokens_est is not None else 0

                    completion_tokens_est = self.count_tokens(generated_text)
                    completion_tokens = completion_tokens_est if completion_tokens_est is not None else 0
                else: # Absolute fallback
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(generated_text.split())

            return generated_text, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP error {e.response.status_code} from vLLM server: {e.response.text}"
            print(f"[VLLMRunner] {error_detail}")
            return f"[VLLMRunner] Error: {error_detail}", {"prompt_tokens": prompt_tokens, "completion_tokens": 0}
        except httpx.RequestError as e:
            error_detail = f"Request error connecting to vLLM server: {str(e)}"
            print(f"[VLLMRunner] {error_detail}")
            return f"[VLLMRunner] Error: {error_detail}", {"prompt_tokens": prompt_tokens, "completion_tokens": 0}
        except json.JSONDecodeError as e:
            error_detail = f"Failed to decode JSON response from vLLM server: {str(e)}"
            print(f"[VLLMRunner] {error_detail}")
            return f"[VLLMRunner] Error: {error_detail}", {"prompt_tokens": prompt_tokens, "completion_tokens": 0}
        except Exception as e:
            error_detail = f"Unexpected error processing vLLM response: {str(e)}"
            print(f"[VLLMRunner] {error_detail}")
            return f"[VLLMRunner] Error: {error_detail}", {"prompt_tokens": prompt_tokens, "completion_tokens": 0}

    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
        if image_paths:
            print(f"[VLLMRunner] Warning: image_paths provided but standard vLLM OpenAI-compatible /v1/completions endpoint does not support them directly. Ignoring images: {image_paths}")

        headers = self._prepare_headers()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get('max_new_tokens', kwargs.get('max_tokens', 128)),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 1.0),
            "n": kwargs.get('n', 1),
            "stop": kwargs.get('stop', None),
            "stream": True,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        payload.update({k:v for k,v in kwargs.items() if k not in payload})


        print(f"[VLLMRunner] Calling stream on {self.completion_endpoint} for model '{self.model_name}'")

        prompt_tokens = 0
        if hasattr(self, 'count_tokens') and callable(self.count_tokens):
            pt_count = self.count_tokens(prompt)
            prompt_tokens = pt_count if pt_count is not None else 0
        else:
            prompt_tokens = len(prompt.split())
        yield {"prompt_tokens": prompt_tokens}


        try:
            with self.http_client.stream("POST", self.completion_endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: [DONE]"):
                        print("[VLLMRunner] Stream DONE marker received.")
                        break
                    if line.startswith("data: "):
                        try:
                            data_json_str = line.split("data: ", 1)[1]
                            data_json = json.loads(data_json_str)
                            chunk_text = data_json.get("choices", [{}])[0].get("text", "")
                            if chunk_text:
                                # Estimating tokens for the chunk. This is an approximation.
                                tokens_in_chunk = 0
                                if hasattr(self, 'count_tokens') and callable(self.count_tokens):
                                     tok_est = self.count_tokens(chunk_text)
                                     tokens_in_chunk = tok_est if tok_est is not None else 0
                                else:
                                     tokens_in_chunk = len(chunk_text.split()) # Fallback
                                yield (chunk_text, tokens_in_chunk)
                        except json.JSONDecodeError:
                            print(f"[VLLMRunner] Warning: Could not decode JSON from stream line: {line}")
                        except Exception as e_chunk:
                            print(f"[VLLMRunner] Warning: Error processing stream chunk '{line}': {e_chunk}")
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP error {e.response.status_code} starting vLLM stream: {e.response.text}"
            print(f"[VLLMRunner] {error_detail}")
            yield (f"[VLLMRunner] Error: {error_detail}", 0)
        except httpx.RequestError as e:
            error_detail = f"Request error connecting to vLLM server for stream: {str(e)}"
            print(f"[VLLMRunner] {error_detail}")
            yield (f"[VLLMRunner] Error: {error_detail}", 0)
        except Exception as e:
            error_detail = f"Unexpected error during vLLM stream: {str(e)}"
            print(f"[VLLMRunner] {error_detail}")
            yield (f"[VLLMRunner] Error: {error_detail}", 0)
        finally:
            print("[VLLMRunner] Stream finished or aborted.")

    # --- KV Cache and LoRA methods remain the same (mostly no-ops for typical vLLM API client) ---
    def export_kv_cache(self) -> t.Any:
        print(f"[VLLMRunner] export_kv_cache called for model '{self.model_name}'. No-op.")
        return None

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print(f"[VLLMRunner] import_kv_cache called for model '{self.model_name}'. No-op.")

    def preload_kv(self, prompt: str, **kwargs) -> None:
        print(f"[VLLMRunner] preload_kv called for model '{self.model_name}'. Prompt: '{prompt[:50]}...'. No-op for client.")

    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"\n--- VLLMRunner ({self.model_name}) Loading LoRA Adapter ---")
        print(f"  ID: {adapter_id}, Path/Name: {adapter_path}, Params: {kwargs}")
        print("  For vLLM, LoRAs are typically configured on the server or passed via API request params (e.g., 'lora_request'). Client-side load call is a placeholder.")
        return False

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"\n--- VLLMRunner ({self.model_name}) Unloading LoRA Adapter ---")
        print("  Server-dependent operation. Client cannot directly unload server-side LoRAs via this generic interface.")
        return False

    def get_active_lora_adapters(self) -> t.List[str]:
        print(f"\n--- VLLMRunner ({self.model_name}) Getting Active LoRA Adapters ---")
        print("  Client cannot typically query active server-side LoRAs via standard API. Returning empty list.")
        return []

    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"\n--- VLLMRunner ({self.model_name}) Merging LoRA Adapters ---")
        print("  Merging LoRAs is a server-side or offline operation for vLLM. Not client controllable.")
        return False

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print(f"\n--- VLLMRunner ({self.model_name}) Unmerging LoRA Adapters ---")
        print("  Unmerging LoRAs is server-side or offline for vLLM. Not client controllable.")
        return False

    def count_tokens(self, text: str) -> t.Optional[int]:
        # vLLM /v1/completions does not have a standard /v1/tokenize endpoint.
        # This is a placeholder. A real implementation might use a local tokenizer
        # known to be compatible with the vLLM model, or this method could be removed
        # if token counting is not expected from this client-side runner.
        print(f"[VLLMRunner] count_tokens called for text: '{text[:50]}...'. This is a placeholder and uses word count.")
        return len(text.split())


if __name__ == '__main__':
    import os
    print("--- Testing VLLMRunner ---")
    # For this demo to work, ensure a vLLM server with an OpenAI-compatible endpoint is running.
    # Example vLLM server startup:
    # python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.1 --host 0.0.0.0 --port 8000

    vllm_api_url = os.environ.get("VLLM_API_URL", "http://localhost:8000") # Base URL, e.g., http://localhost:8000
    # The runner will append "/v1/completions" or similar.

    # The model name here should match the model loaded by your vLLM server.
    # For vLLM's OpenAI API, this is often the Hugging Face repo ID used to launch the server.
    vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")

    # Optional: API Key if your vLLM endpoint is protected
    vllm_api_key = os.environ.get("VLLM_API_KEY", None)

    print(f"Using VLLM API URL: {vllm_api_url}")
    print(f"Using VLLM Model Name: {vllm_model_name}")
    if vllm_api_key:
        print("Using VLLM API Key.")
    else:
        print("No VLLM API Key provided (assuming endpoint is open or key is not needed).")

    vllm_runner = VLLMRunner(
        model_name=vllm_model_name,
        api_url=vllm_api_url,
        api_key=vllm_api_key
    )

    print("\n--- Generate Demo ---")
    gen_prompt = "What is the capital of France?"
    # Note: image_paths are currently ignored by this runner's generate/stream methods
    # as standard /v1/completions doesn't take them.
    # example_image_paths_vllm = ["http://images.cocodataset.org/val2017/000000039769.jpg"]

    gen_params = {"max_tokens": 50, "temperature": 0.5} # Using max_tokens as per OpenAI /v1/completions
    print(f"Sending to generate: '{gen_prompt}' with params: {gen_params}")
    gen_output, gen_tokens = vllm_runner.generate(gen_prompt, **gen_params)
    print(f"Generate Output:\n{gen_output}")
    print(f"Token counts: {gen_tokens}")


    print("\n--- Stream Demo ---")
    stream_prompt = "Tell me a short joke."

    stream_params = {"temperature": 0.7, "max_tokens": 60}
    print(f"Sending to stream: '{stream_prompt}' with params: {stream_params}")
    full_streamed_response = []
    stream_completion_tokens_estimate = 0
    print("Streamed Output:")

    stream_generator = vllm_runner.stream(stream_prompt, **stream_params)

    first_item = next(stream_generator)
    if isinstance(first_item, dict) and "prompt_tokens" in first_item:
        print(f"(Prompt tokens: {first_item['prompt_tokens']})")

    for item in stream_generator:
        if isinstance(item, tuple):
            chunk_text, chunk_tokens = item
            print(chunk_text, end='', flush=True)
            full_streamed_response.append(chunk_text)
            stream_completion_tokens_estimate += chunk_tokens
        else: # Should not happen
            print(f"Unexpected item: {item}")

    print("\nFull streamed response collected:", "".join(full_streamed_response))
    print(f"Estimated completion tokens from stream: {stream_completion_tokens_estimate}")
    print("\n")

    print("\nVLLMRunner functional demonstration complete.")
