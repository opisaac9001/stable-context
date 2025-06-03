# llm_context_os/tests/test_multimodal_integration.py
import unittest
from fastapi.testclient import TestClient
from pathlib import Path # For potential cleanup if KVCacheManager makes dirs
import shutil # For cleanup
import io # For capturing stdout
import sys # For capturing stdout
from contextlib import redirect_stdout # For capturing stdout

from llm_context_os.api.main import app, model_mgr, ctx_mgr # Import app and managers
from llm_context_os.api.schemas import ChatRequest, GenerationParams # Schemas used by client
from llm_context_os.runners.llava_cpp_runner import LlavaCppRunner # To check instance type
from llm_context_os.caching.kv_cache_manager import KVCacheManager # To manage test cache dir

# Helper to capture stdout for tests that print a lot
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class TestMultimodalIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Ensure ModelManager uses a clean, test-specific KV cache directory
        self.test_cache_super_dir = Path("data_test_multimodal_integration")
        self.test_kv_cache_dir = self.test_cache_super_dir / "kv_cache_manager_multimodal_tests"

        if self.test_cache_super_dir.exists():
            shutil.rmtree(self.test_cache_super_dir)
        self.test_kv_cache_dir.mkdir(parents=True, exist_ok=True)

        # Override the manager's KVCacheManager instance
        # Suppress print from KVCacheManager init for cleaner test logs
        with Capturing():
            model_mgr.kv_cache_mgr = KVCacheManager(cache_dir=str(self.test_kv_cache_dir))

        # Ensure no model is loaded initially and context is clear
        model_mgr.unload()
        ctx_mgr._messages = []
        ctx_mgr._start = 0

        # Load a LlavaCppRunner placeholder for these tests
        load_payload = {
            "model_type": "llava_cpp",
            "model_path_or_name": "test/llava-model.gguf", # Using a distinct name for this test
            "runner_params": {"mmproj_path": "test/llava-mmproj.gguf", "n_gpu_layers": 0, "verbose": False}
        }
        with Capturing(): # Suppress prints from load operation
            response = self.client.post("/load_model", json=load_payload)

        self.assertEqual(response.status_code, 200, f"Model loading failed: {response.text}")
        self.assertTrue(isinstance(model_mgr.current_runner, LlavaCppRunner),
                        f"LlavaCppRunner was not loaded. Got: {type(model_mgr.current_runner)}")
        self.assertIsNotNone(model_mgr.current_runner, "Failed to load LlavaCppRunner for multimodal tests.")


    def tearDown(self):
        # Clean up the test-specific cache directory
        if self.test_cache_super_dir.exists():
            shutil.rmtree(self.test_cache_super_dir)
        # Optional: reset global managers to their default states if necessary
        # model_mgr.kv_cache_mgr = KVCacheManager() # Reset to default path

    def test_chat_with_image_path(self):
        test_message = "Describe this image for me"
        test_image_paths = ["path/to/a_test_image.jpg"]

        chat_payload_dict = {
            "message": test_message,
            "image_paths": test_image_paths,
            "generation_params": {"max_new_tokens": 10} # Ensure it's a dict
        }

        with Capturing() as api_call_logs: # Capture logs from API call
            response = self.client.post("/chat", json=chat_payload_dict)

        self.assertEqual(response.status_code, 200, f"Chat request failed: {response.text}")
        json_response = response.json()
        self.assertIn("reply", json_response)

        # Check if placeholder for image processing is in reply (from LlavaCppRunner placeholder)
        # LlavaCppRunner's generate: f"[LLaVA-CPP Response to: {prompt[:30]}... (Processed {len(image_paths)} image(s) like '{image_paths[0]}'). More text...]"
        self.assertIn(f"(Processed {len(test_image_paths)} image(s) like '{test_image_paths[0]}')", json_response['reply'])

        # Check if context manager included the image placeholder in its prompt build
        # This requires inspecting the prompt that was built.
        # The API passes only the first image to ctx_mgr.add.
        # The prompt built by ctx_mgr will have the [IMAGE: path] placeholder.
        # The runner (LlavaCppRunner) will then receive this prompt.
        # We can check the logs captured from the API call where main.py prints the built prompt.

        # Example of checking logs for the built prompt:
        # This is a bit fragile as it depends on exact log messages from main.py's /chat endpoint.
        # The logged prompt includes system prompt and role prefixes.
        # A simpler check: ensure the image path and the message text are present in the logs.
        logged_output_str = "".join(api_call_logs)
        self.assertIn(test_image_paths[0], logged_output_str,
                      f"Image path '{test_image_paths[0]}' not found in API call logs.")
        self.assertIn(test_message, logged_output_str,
                      f"Message text '{test_message}' not found in API call logs for built prompt.")
        # More specific check for the placeholder format if needed:
        self.assertIn(f"user: [IMAGE: {test_image_paths[0]}] {test_message}", logged_output_str)


    def test_chat_without_image_path(self):
        chat_payload_dict = {
            "message": "Just a regular text query.",
            "generation_params": {"max_new_tokens": 10}
        }
        with Capturing() as api_call_logs:
            response = self.client.post("/chat", json=chat_payload_dict)

        self.assertEqual(response.status_code, 200, f"Chat request failed: {response.text}")
        json_response = response.json()
        self.assertIn("reply", json_response)

        # Check that "Processed image data" or similar is NOT in reply
        self.assertNotIn("Processed image", json_response['reply'])
        self.assertNotIn("(images:", json_response['reply'])

        # Check that context manager did not add an image placeholder
        no_image_placeholder_in_prompt = True
        for line in api_call_logs:
            if "Built prompt for model" in line and "[IMAGE:" in line :
                no_image_placeholder_in_prompt = False
                break
        self.assertTrue(no_image_placeholder_in_prompt,
                        f"Image placeholder unexpectedly found in logged built prompt for text-only chat. Logs: {''.join(api_call_logs)}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
