# llm_context_os/runners/api_runner.py
import typing as t
import httpx
import json
from .base import BaseRunner

class APIRunner(BaseRunner):
    def __init__(self, model_name: str, api_url: str, api_key: t.Optional[str] = None, **kwargs: t.Any):
        self.model_name = model_name
        self.api_url = api_url # This should be the full URL to the completions/chat_completions endpoint
        self.api_key = api_key
        # Default timeout is 5 seconds, which might be too short for model responses.
        # Allow overriding via kwargs or set a higher default.
        timeout = kwargs.pop("timeout", 60.0)
        self.http_client = httpx.Client(timeout=timeout, **kwargs)

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

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> str:
        print(f"\n--- APIRunner ({self.model_name}) Generating ---")
        print(f"Target URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: Standard OpenAI-compatible API for text models may not support images directly in this field. This runner currently ignores them in the request.)")

        headers = self._prepare_headers()
        payload = self._prepare_payload(prompt, stream=False, **kwargs)

        print(f"Request Payload: {json.dumps(payload, indent=2)}")

        try:
            response = self.http_client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raises an HTTPStatusError for 4xx/5xx responses

            response_data = response.json()
            print(f"Full API Response Data: {json.dumps(response_data, indent=2)}")

            # Extract text based on common OpenAI structures
            if response_data.get("choices"):
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                elif "text" in choice: # For older completion APIs
                    return choice["text"]

            # Fallback or error if structure is unexpected
            raise ValueError(f"Unexpected API response structure: {response_data}")

        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            # You might want to re-raise a custom exception or return an error message
            raise Exception(f"API request failed with status {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            raise Exception(f"API request failed due to a network or request error: {e}") from e
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response: {e}")
            raise Exception(f"Could not parse JSON response from API: {e}") from e


    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[str, None, None]:
        print(f"\n--- APIRunner ({self.model_name}) Streaming ---")
        print(f"Target URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: Standard OpenAI-compatible API for text models may not support images directly in this field. This runner currently ignores them in the request.)")
        print(f"Streaming Config: {kwargs}")

        headers = self._prepare_headers()
        payload = self._prepare_payload(prompt, stream=True, **kwargs)

        print(f"Request Payload: {json.dumps(payload, indent=2)}")

        try:
            with self.http_client.stream("POST", self.api_url, headers=headers, json=payload) as response:
                response.raise_for_status() # Check for HTTP errors before starting to iterate

                for line in response.iter_lines():
                    if not line: # Skip empty keep-alive lines
                        continue
                    if line.startswith("data: "):
                        line_data = line[len("data: "):]
                        if line_data.strip() == "[DONE]":
                            print("Stream finished with [DONE] signal.")
                            break
                        try:
                            data_json = json.loads(line_data)
                            if data_json.get("choices"):
                                delta = data_json["choices"][0].get("delta", {})
                                content = delta.get("content")
                                if content: # Ensure content is not None or empty string if you want to skip those
                                    yield content
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from stream line: {line_data}")
                            continue # Or handle error more strictly
                    else:
                        print(f"Unrecognized stream line: {line}")
            print("Streaming complete.")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred during stream: {e.response.status_code} - {e.response.text}")
            # Depending on when this happens, part of the stream might have been yielded.
            # Consider how to signal this error to the consumer.
            # For now, just printing and re-raising.
            raise Exception(f"API stream request failed with status {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            print(f"Request error occurred during stream: {e}")
            raise Exception(f"API stream request failed due to a network or request error: {e}") from e


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
        # You could exit here, or let it try and fail. For now, let it try.

    api_model = APIRunner(
        model_name=model_name_on_api,
        api_url=api_full_url,
        api_key=api_key if api_key and api_key.lower() != "none" else None
    )

    generation_params = {"temperature": 0.7, "max_tokens": 150}
    # Image paths are not used by standard OpenAI text completion APIs via this runner
    # example_image_paths = ["/path/to/image1.jpg", "/path/to/image2.png"]

    test_prompt = "What is the capital of France?"

    print(f"\n--- Testing generate() for model: {api_model.model_name} ---")
    try:
        response_text = api_model.generate(test_prompt, **generation_params)
        print(f"\nGenerate call response:\n'{response_text}'")
    except Exception as e:
        print(f"Error during generate(): {e}")

    print(f"\n--- Testing stream() for model: {api_model.model_name} ---")
    try:
        full_api_streamed_response = []
        print("Streamed response:")
        for chunk in api_model.stream(test_prompt, **generation_params):
            print(chunk, end="", flush=True)
            full_api_streamed_response.append(chunk)
        print(f"\n\nFull API streamed response collected: '{''.join(full_api_streamed_response)}'")
    except Exception as e:
        print(f"Error during stream(): {e}")

    print("\nAPIRunner functional demonstration complete.")
