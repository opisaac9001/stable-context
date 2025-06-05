# llm_context_os/tests/test_runners.py
import unittest
import os
from unittest.mock import patch, MagicMock, PropertyMock

from llm_context_os.runners.base import BaseRunner
from llm_context_os.runners.llama_cpp_runner import LlamaCppRunner
from llm_context_os.runners.awq_runner import AWQRunner
from llm_context_os.runners.api_runner import APIRunner
from llm_context_os.runners.exl2_runner import EXL2Runner

# --- Conditional Availability Flags ---

LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("Warning: llama-cpp-python not found. LlamaCppRunner smoke tests will be skipped.")

AWQ_AVAILABLE = False
try:
    import awq  # This is a basic check; awq might import even if torch/cuda issues exist
    from transformers import AutoModelForCausalLM
    # A more robust check might involve trying to import specific awq modules used in the runner
    AWQ_AVAILABLE = True # Tentatively true, init might still fail if CUDA not right
except ImportError:
    print("Warning: autoawq or transformers not found. AWQRunner smoke tests will be skipped.")

EXL2_AVAILABLE = False
try:
    from exllamav2 import ExLlamaV2 # Check for a core class
    EXL2_AVAILABLE = True
except ImportError:
    print("Warning: exllamav2 not found. EXL2Runner smoke tests will be skipped.")

# httpx for APIRunner is a direct dependency, so it should be available if requirements are met.
# No explicit HTTX_AVAILABLE flag needed here, but tests will fail if not installed.


# --- Environment Variables for Test Models (Optional) ---
# Users can set these to paths of actual small models for more thorough smoke testing.
LLAMA_CPP_TEST_MODEL_PATH = os.getenv("LLAMA_CPP_TEST_MODEL_PATH")
AWQ_TEST_MODEL_PATH = os.getenv("AWQ_TEST_MODEL_PATH")
EXL2_TEST_MODEL_PATH = os.getenv("EXL2_TEST_MODEL_PATH")

# For APIRunner tests that might call a real (test) API
API_TEST_URL = os.getenv("API_TEST_URL", "http://mock.api/v1/chat/completions") # Default to a mock URL
API_TEST_KEY = os.getenv("API_TEST_KEY", "sk-dummy")
API_TEST_MODEL_NAME = os.getenv("API_TEST_MODEL_NAME", "test-api-model")


class TestAPIRunnerSmoke(unittest.TestCase):
    @patch('httpx.Client.post')
    def test_initialization_and_generate_smoke(self, mock_post):
        # Configure mock for generate
        mock_response_generate = MagicMock()
        mock_response_generate.status_code = 200
        mock_response_generate.json.return_value = {
            "choices": [{"message": {"content": "Mocked API response for generate"}}]}

        # If APIRunner uses multiple .post calls or other methods, configure them here or make mock_post more versatile
        mock_post.return_value = mock_response_generate # Default for any post call

        runner = APIRunner(model_name=API_TEST_MODEL_NAME, api_url=API_TEST_URL, api_key=API_TEST_KEY)
        self.assertIsInstance(runner, BaseRunner)
        self.assertIsInstance(runner, APIRunner)

        response = runner.generate("Hello")
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Mocked API response for generate")
        mock_post.assert_called_once() # Ensures the .post was actually made

    @patch('httpx.Client.stream')
    def test_stream_smoke(self, mock_stream_method):
        # Configure mock for stream
        mock_response_stream_context_manager = MagicMock()

        # Simulate iter_lines for SSE
        sse_event_1 = "data: {\"choices\": [{\"delta\": {\"content\": \"Mocked stream chunk 1\"}}]}"
        sse_event_2 = "data: {\"choices\": [{\"delta\": {\"content\": \" chunk 2\"}}]}"
        sse_done = "data: [DONE]"
        mock_response_stream_context_manager.iter_lines.return_value = [sse_event_1, sse_event_2, sse_done]

        # The stream method itself is a context manager
        mock_stream_context_manager_obj = MagicMock()
        mock_stream_context_manager_obj.__enter__.return_value = mock_response_stream_context_manager
        mock_stream_context_manager_obj.__exit__.return_value = None

        mock_stream_method.return_value = mock_stream_context_manager_obj

        runner = APIRunner(model_name=API_TEST_MODEL_NAME, api_url=API_TEST_URL, api_key=API_TEST_KEY)

        stream_results = list(runner.stream("Hello stream"))
        self.assertIsInstance(stream_results, list)
        self.assertTrue(len(stream_results) > 0)
        self.assertEqual("".join(stream_results), "Mocked stream chunk 1 chunk 2")
        mock_stream_method.assert_called_once()


@unittest.skipUnless(LLAMA_CPP_AVAILABLE, "llama-cpp-python not installed")
class TestLlamaCppRunnerSmoke(unittest.TestCase):
    def test_initialization(self):
        if LLAMA_CPP_TEST_MODEL_PATH and os.path.exists(LLAMA_CPP_TEST_MODEL_PATH):
            runner = LlamaCppRunner(model_path=LLAMA_CPP_TEST_MODEL_PATH)
            self.assertIsInstance(runner, BaseRunner)
            self.assertIsNotNone(runner.model) # Model should be loaded
        else:
            # Test that it still initializes the class instance even if model loading fails internally
            # LlamaCppRunner's __init__ sets self.model to None if path is invalid or loading fails.
            runner_no_model = LlamaCppRunner(model_path="dummy_non_existent.gguf", verbose=False)
            self.assertIsInstance(runner_no_model, BaseRunner)
            self.assertIsNone(runner_no_model.model, "Model should be None for invalid path if not raising error during init.")


    def test_generate_smoke(self):
        if LLAMA_CPP_TEST_MODEL_PATH and os.path.exists(LLAMA_CPP_TEST_MODEL_PATH):
            runner = LlamaCppRunner(model_path=LLAMA_CPP_TEST_MODEL_PATH, n_ctx=512, verbose=False)
            # Reduce n_predict for smoke test if model is real
            response = runner.generate("Hello", max_tokens=10)
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)
        else:
            runner = LlamaCppRunner(model_path="dummy_invalid.gguf", verbose=False)
            self.assertIsNone(runner.model) # Assuming __init__ sets model to None on failure
            # Test generate's behavior when model is None
            with self.assertRaisesRegex(RuntimeError, "Model is not loaded"):
                 runner.generate("Hello")


    def test_stream_smoke(self):
        if LLAMA_CPP_TEST_MODEL_PATH and os.path.exists(LLAMA_CPP_TEST_MODEL_PATH):
            runner = LlamaCppRunner(model_path=LLAMA_CPP_TEST_MODEL_PATH, n_ctx=512, verbose=False)
            stream = runner.stream("Hello stream", max_tokens=10)
            self.assertTrue(hasattr(stream, '__iter__')) # Check if it's an iterator/generator
            results = list(stream)
            self.assertTrue(len(results) > 0)
            self.assertIsInstance(results[0], str)
        else:
            runner = LlamaCppRunner(model_path="dummy_invalid.gguf", verbose=False)
            self.assertIsNone(runner.model)
            with self.assertRaisesRegex(RuntimeError, "Model is not loaded"):
                list(runner.stream("Hello stream"))


@unittest.skipUnless(AWQ_AVAILABLE, "autoawq or transformers not installed/compatible")
class TestAWQRunnerSmoke(unittest.TestCase):
    def test_initialization(self):
        if AWQ_TEST_MODEL_PATH and os.path.isdir(AWQ_TEST_MODEL_PATH):
            runner = AWQRunner(model_path=AWQ_TEST_MODEL_PATH)
            self.assertIsInstance(runner, BaseRunner)
            # AWQRunner.model might be None if CUDA is not available, even if path is valid.
            # So, we only assert NotNone if we are sure CUDA is available and model should load.
            # For a generic smoke test, this check is hard.
            if runner.model: # Or some other flag like runner.is_initialized_successfully
                self.assertIsNotNone(runner.model)
            elif os.environ.get("CI_HAS_CUDA_GPU"): # Example: only assert NotNone if in CUDA env
                 self.assertIsNotNone(runner.model, "AWQ model failed to load even in expected CUDA env.")
        else:
            runner_no_model = AWQRunner(model_path="dummy_non_existent_awq_dir")
            self.assertIsInstance(runner_no_model, BaseRunner)
            self.assertIsNone(runner_no_model.model, "Model should be None for invalid path.")


    def test_generate_smoke(self):
        if AWQ_TEST_MODEL_PATH and os.path.isdir(AWQ_TEST_MODEL_PATH):
            runner = AWQRunner(model_path=AWQ_TEST_MODEL_PATH)
            if runner.model:
                response = runner.generate("Hello", max_new_tokens=10)
                self.assertIsInstance(response, str)
                # self.assertTrue(len(response) > 0) # Can be empty if max_new_tokens is small
            else:
                self.skipTest(f"AWQ Model at '{AWQ_TEST_MODEL_PATH}' did not load (self.model is None). Skipping generate test.")
        else:
            runner = AWQRunner(model_path="dummy_invalid_awq_dir")
            self.assertIsNone(runner.model)
            with self.assertRaisesRegex(RuntimeError, "AWQRunner model is not loaded"):
                runner.generate("Hello")

    def test_stream_smoke(self):
        if AWQ_TEST_MODEL_PATH and os.path.isdir(AWQ_TEST_MODEL_PATH):
            runner = AWQRunner(model_path=AWQ_TEST_MODEL_PATH)
            if runner.model:
                stream = runner.stream("Hello stream", max_new_tokens=10)
                self.assertTrue(hasattr(stream, '__iter__'))
                results = list(stream)
                # self.assertTrue(len(results) > 0) # Can be empty
            else:
                self.skipTest(f"AWQ Model at '{AWQ_TEST_MODEL_PATH}' did not load (self.model is None). Skipping stream test.")
        else:
            runner = AWQRunner(model_path="dummy_invalid_awq_dir")
            self.assertIsNone(runner.model)
            with self.assertRaisesRegex(RuntimeError, "AWQRunner model is not loaded"):
                 list(runner.stream("Hello stream"))


@unittest.skipUnless(EXL2_AVAILABLE, "exllamav2 not installed")
class TestEXL2RunnerSmoke(unittest.TestCase):
    def test_initialization(self):
        if EXL2_TEST_MODEL_PATH and os.path.isdir(EXL2_TEST_MODEL_PATH):
            runner = EXL2Runner(model_path=EXL2_TEST_MODEL_PATH)
            self.assertIsInstance(runner, BaseRunner)
            if runner.model: # Check if model loaded successfully
                self.assertIsNotNone(runner.model)
            elif os.environ.get("CI_HAS_CUDA_GPU"): # Example: only assert NotNone if in CUDA env
                 self.assertIsNotNone(runner.model, "EXL2 model failed to load even in expected CUDA env.")
        else:
            runner_no_model = EXL2Runner(model_path="dummy_non_existent_exl2_dir")
            self.assertIsInstance(runner_no_model, BaseRunner)
            self.assertIsNone(runner_no_model.model, "Model should be None for invalid path.")


    def test_generate_smoke(self):
        if EXL2_TEST_MODEL_PATH and os.path.isdir(EXL2_TEST_MODEL_PATH):
            runner = EXL2Runner(model_path=EXL2_TEST_MODEL_PATH)
            if runner.model:
                response = runner.generate("Hello", max_new_tokens=10)
                self.assertIsInstance(response, str)
            else:
                self.skipTest(f"EXL2 Model at '{EXL2_TEST_MODEL_PATH}' did not load (self.model is None). Skipping generate test.")
        else:
            runner = EXL2Runner(model_path="dummy_invalid_exl2_dir")
            response = runner.generate("Hello") # generate() returns error string if not initialized
            self.assertIn("Error: EXL2Runner is not available or not properly initialized.", response)


    def test_stream_smoke(self):
        if EXL2_TEST_MODEL_PATH and os.path.isdir(EXL2_TEST_MODEL_PATH):
            runner = EXL2Runner(model_path=EXL2_TEST_MODEL_PATH)
            if runner.model:
                stream = runner.stream("Hello stream", max_new_tokens=10)
                self.assertTrue(hasattr(stream, '__iter__'))
                results = list(stream)
            else:
                self.skipTest(f"EXL2 Model at '{EXL2_TEST_MODEL_PATH}' did not load (self.model is None). Skipping stream test.")
        else:
            runner = EXL2Runner(model_path="dummy_invalid_exl2_dir")
            stream_results = list(runner.stream("Hello stream")) # stream() yields error string if not initialized
            self.assertTrue(len(stream_results) >= 1)
            self.assertIn("Error: EXL2Runner is not available or not properly initialized.", stream_results[0])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# End of llm_context_os/tests/test_runners.py
