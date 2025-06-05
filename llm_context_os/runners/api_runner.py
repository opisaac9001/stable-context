# llm_context_os/runners/api_runner.py
import typing as t
import httpx
import json
from .base import BaseRunner

# Attempt to import tiktoken for fallback token counting
TIKTOKEN_AVAILABLE = False
tiktoken_encoding = None
try:
    import tiktoken
    # Using cl100k_base as it's common for OpenAI models.
    # Other models might need different encodings.
    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
    print("tiktoken library found, will be used for fallback token counting if API doesn't provide usage stats.")
except ImportError:
    print("Warning: tiktoken library not found. Fallback token counting will not be available for APIRunner.")
except Exception as e:
    print(f"Warning: Error initializing tiktoken, fallback token counting may not work: {e}")


class APIRunner(BaseRunner):
    def __init__(self, model_name: str, api_url: str, api_key: t.Optional[str] = None, **kwargs: t.Any):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        timeout = kwargs.pop("timeout", 60.0)
        self.http_client = httpx.Client(timeout=timeout, **kwargs)

        self.tokenizer = tiktoken_encoding if TIKTOKEN_AVAILABLE else None # Store for fallback

        print(f"APIRunner initialized for model '{self.model_name}' at URL: {self.api_url}")
        if self.api_key:
            print(f"API Key: {'*' * (len(self.api_key) - 4) + self.api_key[-4:] if len(self.api_key) > 4 else '****'}")
        else:
            print("API Key: Not provided")

    def _prepare_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_payload(self, prompt: str, stream: bool = False, **kwargs: t.Any) -> dict:
        # OpenAI-compatible payload structure
        payload = {
            "model": self.model_name,
            # Assuming "messages" format is preferred.
            # If a raw prompt is more common for some APIs, this might need adjustment
            # or a check on self.api_url to determine format.
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": stream,
        }
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs: # stop sequences
            payload["stop"] = kwargs["stop"]
        # Add other common parameters as needed, filtering from kwargs
        # e.g. presence_penalty, frequency_penalty
        return payload

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
        print(f"\n--- APIRunner ({self.model_name}) Generating ---")
        # ... (logging prompt, image_paths as before) ...

        headers = self._prepare_headers()
        payload = self._prepare_payload(prompt, stream=False, **kwargs)
        print(f"Request Payload: {json.dumps(payload, indent=2)}")

        generated_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            response = self.http_client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            print(f"Full API Response Data: {json.dumps(response_data, indent=2)}")

            if response_data.get("choices"):
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    generated_text = choice["message"]["content"]
                elif "text" in choice:
                    generated_text = choice["text"]

            if not generated_text: # If no text found via common paths
                 raise ValueError(f"Unexpected API response structure, could not find generated text: {response_data}")

            # Token counts from API response
            if "usage" in response_data:
                usage_data = response_data["usage"]
                prompt_tokens = usage_data.get("prompt_tokens", 0)
                completion_tokens = usage_data.get("completion_tokens", 0)
                print(f"  Token counts from API: prompt={prompt_tokens}, completion={completion_tokens}")

            # Fallback token counting if not in API response
            if not prompt_tokens and not completion_tokens and self.tokenizer:
                print("  Warning: Token usage data not found in API response. Using local tiktoken for estimation.")
                prompt_tokens = len(self.tokenizer.encode(prompt))
                completion_tokens = len(self.tokenizer.encode(generated_text))
                print(f"  Token counts from tiktoken: prompt={prompt_tokens}, completion={completion_tokens}")

            return generated_text, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API request failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e: # Catch other errors like RequestError, JSONDecodeError, ValueError
            print(f"Error during API generate call: {e}")
            # Return empty/error state for tokens as well
            return str(e), {"prompt_tokens": 0, "completion_tokens": 0}


    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
        print(f"\n--- APIRunner ({self.model_name}) Streaming ---")
        # ... (logging as before) ...

        headers = self._prepare_headers()
        payload = self._prepare_payload(prompt, stream=True, **kwargs)
        print(f"Request Payload: {json.dumps(payload, indent=2)}")

        # Yield prompt tokens first
        prompt_tokens = 0
        if self.tokenizer:
            try:
                prompt_tokens = len(self.tokenizer.encode(prompt))
            except Exception as e_tok:
                print(f"Warning: Error encoding prompt with tiktoken: {e_tok}. Prompt tokens will be 0.")
        yield {"prompt_tokens": prompt_tokens}
        print(f"  Yielded initial prompt_tokens: {prompt_tokens}")

        try:
            with self.http_client.stream("POST", self.api_url, headers=headers, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        line_data = line[len("data: "):]
                        if line_data.strip() == "[DONE]":
                            print("Stream finished with [DONE] signal.")
                            break
                        try:
                            data_json = json.loads(line_data)
                            if data_json.get("choices"):
                                delta = data_json["choices"][0].get("delta", {})
                                content_chunk = delta.get("content")
                                if content_chunk: # Can be None or empty string
                                    tokens_in_chunk = 0
                                    if self.tokenizer:
                                        try: tokens_in_chunk = len(self.tokenizer.encode(content_chunk))
                                        except Exception: pass # Ignore if chunk is not valid for tokenizer
                                    yield (content_chunk, tokens_in_chunk)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from stream line: {line_data}")
                            # Yield problematic line as raw text chunk with 0 tokens? Or skip?
                            # For now, skipping non-JSON data lines.
                            continue
                    # else: print(f"Unrecognized stream line: {line}") # Optional: log non-data lines
            print("Streaming complete.")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error during stream: {e.response.status_code} - {e.response.text}")
            yield (f"Error: API stream request failed with status {e.response.status_code}", 0)
        except httpx.RequestError as e:
            print(f"Request error occurred during stream: {e}")
            yield (f"Error: API stream request failed due to a network or request error", 0)
        except Exception as e_gen: # Catch any other general error during streaming
            print(f"Generic error during stream processing: {e_gen}")
            yield (f"Error: {str(e_gen)}", 0)


    def preload_kv(self, prompt: str, **kwargs: t.Any) -> None:
        print(f"\n--- APIRunner ({self.model_name}) Preloading KV Cache ---")
        print(f"Prompt for KV: {prompt}")
        print("[APIRunner] preload_kv called. Typically, KV caching is managed by the remote API endpoint or not applicable for stateless API calls.")
        super().preload_kv(prompt, **kwargs)

    def export_kv_cache(self) -> t.Any:
        print(f"\n--- APIRunner ({self.model_name}) Exporting KV Cache ---")
        print("API runners typically do not manage or expose exportable KV cache directly from the client side.")
        return None

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print(f"\n--- APIRunner ({self.model_name}) Importing KV Cache ---")
        print("Importing KV cache is generally not applicable to stateless API runners from the client side.")
        if cache_data:
            print(f"  (Received cache_data of type: {type(cache_data)}, but it will not be used.)")

    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Loading LoRA Adapter ---")
        print("  LoRA management is typically server-side for API-based models. This call is a no-op for APIRunner.")
        return False

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Unloading LoRA Adapter ---")
        print("  LoRA management is typically server-side for API-based models. This call is a no-op for APIRunner.")
        return False

    def get_active_lora_adapters(self) -> t.List[str]:
        print(f"\n--- APIRunner ({self.model_name}) Getting Active LoRA Adapters ---")
        print("  APIRunner does not manage LoRA adapters on the client-side.")
        return []

    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Merging LoRA Adapters ---")
        print("  Merging LoRAs is a server-side operation for API-based models. This call is a no-op for APIRunner.")
        return False

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print(f"\n--- APIRunner ({self.model_name}) Unmerging LoRA Adapters ---")
        print("  Unmerging LoRAs is a server-side operation for API-based models. This call is a no-op for APIRunner.")
        return False

    def count_tokens(self, text: str) -> Optional[int]:
        """
        Estimates the number of tokens in the given text.
        Uses the loaded tiktoken tokenizer if available (typically for OpenAI models).
        Falls back to word count if tiktoken is not available or fails.
        Note: Actual tokenization is server-side and can vary.
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                print(f"Warning: Error using tiktoken for count_tokens: {e}. Falling back to word count.")
                # Fall through to word count

        print("[APIRunner] Warning: count_tokens is using basic word count as a rough estimate because tiktoken is unavailable or failed.")
        return len(text.split())


if __name__ == '__main__':
    # --- IMPORTANT ---
    # To run this example, you need to set the following environment variables or replace the placeholders:
    # 1. YOUR_API_BASE_URL: The base URL of your OpenAI-compatible API.
    #    (e.g., for a local server: "http://localhost:8000/v1")
    # 2. YOUR_API_KEY: Your API key (if required by the endpoint). Use "None" or an empty string if not needed.
    # 3. YOUR_MODEL_NAME: The name of the model you want to use via the API.

    # Example using environment variables (recommended for sensitive data like API keys)
    import os
    api_base_url = os.getenv("YOUR_API_BASE_URL", "http://localhost:8000/v1") # Default to a common local setup
    api_key = os.getenv("YOUR_API_KEY", "sk-your_api_key_here_or_empty") # Example placeholder
    model_name_on_api = os.getenv("YOUR_MODEL_NAME", "Mistral-7B-Instruct-v0.1") # Example model

    # Ensure the URL points to the correct completions endpoint
    # For OpenAI-compatible APIs, this is often /v1/chat/completions
    # If api_base_url is "http://localhost:8000/v1", then api_full_url should be "http://localhost:8000/v1/chat/completions"
    if not api_base_url.endswith("/"):
        api_base_url += "/"
    api_full_url = f"{api_base_url}chat/completions"


    print(f"--- APIRunner Example ---")
    print(f"Using API URL: {api_full_url}")
    print(f"Using Model: {model_name_on_api}")
    print("If this is your first time, make sure your API server is running and the variables above are correctly set.")
    print("---------------------------\n")

    # Check if the user has provided a placeholder key or an obviously fake one
    if not api_key or "your_api_key_here" in api_key or not api_base_url.startswith("http"):
        print("WARNING: API Key or Base URL seems to be a placeholder or missing.")
        print("The API calls will likely fail. Please set them to valid values to test.")

    api_model = APIRunner(
        model_name=model_name_on_api,
        api_url=api_full_url,
        api_key=api_key if api_key and api_key.lower() != "none" else None
    )

    generation_params = {"temperature": 0.7, "max_tokens": 50}
    test_prompt = "What is the capital of France? Explain in a short sentence."

    print(f"\n--- Testing generate() for model: {api_model.model_name} ---")
    try:
        response_text, token_counts = api_model.generate(test_prompt, **generation_params)
        print(f"\nGenerate call response:\n'{response_text}'")
        print(f"Token counts: {token_counts}")
    except Exception as e:
        print(f"Error during generate(): {e}")

    print(f"\n--- Testing stream() for model: {api_model.model_name} ---")
    try:
        full_api_streamed_response_text = []
        total_completion_tokens_in_stream = 0
        print("Streamed response:")

        stream_generator = api_model.stream(test_prompt, **generation_params)

        # First item is prompt_tokens dict
        prompt_token_info = next(stream_generator)
        print(f"\nPrompt token info: {prompt_token_info}")

        # Subsequent items are (text_chunk, tokens_in_chunk)
        for item in stream_generator:
            if isinstance(item, tuple): # Should be (text_chunk, tokens_in_chunk)
                text_chunk, tokens_in_chunk = item
                print(text_chunk, end="", flush=True)
                full_api_streamed_response_text.append(text_chunk)
                total_completion_tokens_in_stream += tokens_in_chunk if tokens_in_chunk else 0
            else: # Should not happen with current APIRunner.stream logic
                 print(f"\nUnexpected stream item: {item}")

        print(f"\n\nFull API streamed response collected: '{''.join(full_api_streamed_response_text)}'")
        print(f"Total completion tokens from stream (calculated locally): {total_completion_tokens_in_stream}")
    except Exception as e:
        print(f"Error during stream(): {e}")

    print("\n--- Testing count_tokens() ---")
    sample_text_for_counting = "This is a sample text for the new count_tokens method."
    estimated_tokens = api_model.count_tokens(sample_text_for_counting)
    if estimated_tokens is not None:
        print(f"Estimated tokens for '{sample_text_for_counting}': {estimated_tokens}")
    else:
        print(f"Token counting not available/supported for APIRunner with text: '{sample_text_for_counting}'")

    print("\nAPIRunner functional demonstration complete.")
