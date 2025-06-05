# llm_context_os/tests/test_runners.py
import unittest
import os
from unittest.mock import patch, MagicMock, PropertyMock

from llm_context_os.runners.base import BaseRunner
from llm_context_os.runners.llama_cpp_runner import LlamaCppRunner
from llm_context_os.runners.awq_runner import AWQRunner
from llm_context_os.runners.api_runner import APIRunner
from llm_context_os.runners.exl2_runner import EXL2Runner
from llm_context_os.runners.llava_cpp_runner import LlavaCppRunner, LLAVA_CPP_AVAILABLE as LLAVA_RUNNER_FLAG # Alias to avoid clash
from pathlib import Path


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

# LLAVA_CPP_AVAILABLE is imported as LLAVA_RUNNER_FLAG from llava_cpp_runner.py
# No need to redefine it here, just use LLAVA_RUNNER_FLAG

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

    def test_count_tokens(self):
        runner = APIRunner(model_name=API_TEST_MODEL_NAME, api_url=API_TEST_URL, api_key=API_TEST_KEY)
        self.assertIsInstance(runner, APIRunner)

        text = "Hello, world!"
        count = runner.count_tokens(text)
        self.assertIsInstance(count, int)

        # APIRunner's count_tokens uses tiktoken if available, else word split.
        # This checks if tiktoken was likely used (more than word count for this specific phrase)
        # or falls back to checking exact word count.
        # "Hello," (1), " world" (1), "!" (1) -> 3 tokens for cl100k_base (used by APIRunner's tiktoken)
        expected_tiktoken_count = 3
        if runner.tokenizer is not None: # Indicates tiktoken was available and loaded
             self.assertEqual(count, expected_tiktoken_count)
        else: # Fallback to word count
            self.assertEqual(count, len(text.split()))

        self.assertEqual(runner.count_tokens(""), 0)


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

    def test_count_tokens(self):
        if LLAMA_CPP_TEST_MODEL_PATH and os.path.exists(LLAMA_CPP_TEST_MODEL_PATH):
            runner = LlamaCppRunner(model_path=LLAMA_CPP_TEST_MODEL_PATH, verbose=False)
            if runner.model:
                self.assertIsInstance(runner.count_tokens("Hello world"), int)
                # For Llama.cpp, "" might be tokenized to a BOS token if add_bos_token is true (default false)
                # or empty list. If BOS is added, count is 1. If not, 0.
                # Assuming default (no auto BOS on tokenize method):
                self.assertEqual(runner.count_tokens(""), 0)
            else:
                self.skipTest(f"LlamaCPP Model at '{LLAMA_CPP_TEST_MODEL_PATH}' did not load. Skipping count_tokens test.")
        else:
            runner = LlamaCppRunner(model_path="dummy_invalid.gguf", verbose=False)
            self.assertIsNone(runner.model)
            self.assertIsNone(runner.count_tokens("Hello world")) # No model, should return None


@unittest.skipUnless(AWQ_AVAILABLE, "autoawq or transformers not installed/compatible")
class TestAWQRunnerSmoke(unittest.TestCase):
    def test_initialization(self):
        if AWQ_TEST_MODEL_PATH and os.path.isdir(AWQ_TEST_MODEL_PATH):
            runner = AWQRunner(model_path=AWQ_TEST_MODEL_PATH)
            self.assertIsInstance(runner, BaseRunner)
            if runner.model:
                self.assertIsNotNone(runner.model)
            elif os.environ.get("CI_HAS_CUDA_GPU"):
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

    def test_count_tokens(self):
        if AWQ_TEST_MODEL_PATH and os.path.isdir(AWQ_TEST_MODEL_PATH):
            runner = AWQRunner(model_path=AWQ_TEST_MODEL_PATH)
            if runner.tokenizer: # AWQRunner initialization sets self.tokenizer
                self.assertIsInstance(runner.count_tokens("Hello world"), int)
                # Most Hugging Face tokenizers return 0 for empty string if no special tokens are added.
                # Some might add BOS/EOS by default in encode(), so this might vary.
                # For a generic test, checking type is safer if specific count isn't known for ""
                self.assertGreaterEqual(runner.count_tokens(""), 0)
            else:
                self.skipTest(f"AWQ Tokenizer for '{AWQ_TEST_MODEL_PATH}' did not load. Skipping count_tokens test.")
        else:
            runner = AWQRunner(model_path="dummy_invalid_awq_dir")
            self.assertIsNone(runner.tokenizer)
            self.assertIsNone(runner.count_tokens("Hello world"))


@unittest.skipUnless(EXL2_AVAILABLE, "exllamav2 not installed")
class TestEXL2RunnerSmoke(unittest.TestCase):
    def test_initialization(self):
        if EXL2_TEST_MODEL_PATH and os.path.isdir(EXL2_TEST_MODEL_PATH):
            runner = EXL2Runner(model_path=EXL2_TEST_MODEL_PATH)
            self.assertIsInstance(runner, BaseRunner)
            if runner.model:
                self.assertIsNotNone(runner.model)
            elif os.environ.get("CI_HAS_CUDA_GPU"):
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

    def test_count_tokens(self):
        if EXL2_TEST_MODEL_PATH and os.path.isdir(EXL2_TEST_MODEL_PATH):
            runner = EXL2Runner(model_path=EXL2_TEST_MODEL_PATH)
            if runner.tokenizer: # EXL2Runner initialization sets self.tokenizer
                self.assertIsInstance(runner.count_tokens("Hello world"), int)
                # ExLlamaV2Tokenizer.encode("") often returns a tensor like tensor([[2]]) (EOS token), so length 1.
                self.assertEqual(runner.count_tokens(""), 1)
            else:
                self.skipTest(f"EXL2 Tokenizer for '{EXL2_TEST_MODEL_PATH}' did not load. Skipping count_tokens test.")
        else:
            runner = EXL2Runner(model_path="dummy_invalid_exl2_dir")
            self.assertIsNone(runner.tokenizer) # If init fails, tokenizer should be None
            self.assertIsNone(runner.count_tokens("Hello world"))


@unittest.skipUnless(LLAVA_RUNNER_FLAG, "llava-cpp-python library not available or failed to import.")
@patch('llm_context_os.runners.llava_cpp_runner.LlavaImageEmbed')
@patch('llm_context_os.runners.llava_cpp_runner.LlavaLlama')
class TestLlavaCppRunnerSmoke(unittest.TestCase):

    def setUp(self, MockLlavaLlama, MockLlavaImageEmbed): # Mocks are passed by class decorators order
        # Store mock classes themselves if needed, or their instances
        self.MockLlavaLlamaClass = MockLlavaLlama
        self.MockLlavaImageEmbedClass = MockLlavaImageEmbed

        self.mock_llava_model_instance = MockLlavaLlama.return_value
        self.mock_image_embedder_instance = MockLlavaImageEmbed.return_value

        # Configure default behaviors for successful loading
        MockLlavaLlama.create_from_gguf.return_value = self.mock_llava_model_instance
        MockLlavaImageEmbed.load_from_gguf.return_value = self.mock_image_embedder_instance

        self.mock_llava_model_instance.generate.return_value = "Mocked LLaVA generation"
        self.mock_llava_model_instance.tokenize.return_value = [1,2,3,4,5] # For count_tokens

        # Default runner instance for tests
        # We pass mock classes to individual tests that need to assert calls on the class itself (like create_from_gguf)
        self.runner = LlavaCppRunner(model_path="fake/llava.gguf", mmproj_path="fake/mmproj.gguf")
        # Check if mocks were correctly assigned during init by the runner
        self.assertEqual(self.runner.model, self.mock_llava_model_instance)
        self.assertEqual(self.runner.image_embedder, self.mock_image_embedder_instance)

    def test_llava_runner_initialization_success(self, MockLlavaLlama, MockLlavaImageEmbed):
        # Runner is already initialized in setUp, so we check calls based on that
        MockLlavaLlama.create_from_gguf.assert_called_once_with(
            gguf_path="fake/llava.gguf",
            n_gpu_layers=0,
            n_ctx=2048,
            verbose=False,
            # Kwargs for llama.cpp specific params would be checked here if passed in init
            # Using **{} for the additional_params part of the call check.
            **{k:v for k,v in {} if k in [
                    'seed', 'n_threads', 'n_batch', 'n_ubatch', 'logits_all',
                    'embedding', 'rope_freq_base', 'rope_freq_scale', 'yarn_ext_factor',
                    'yarn_attn_factor', 'yarn_beta_fast', 'yarn_beta_slow', 'yarn_orig_ctx',
                    'mul_mat_q', 'offload_kqv']}
        )
        MockLlavaImageEmbed.load_from_gguf.assert_called_once_with(
            gguf_path="fake/mmproj.gguf",
            llama=self.mock_llava_model_instance
        )
        self.assertIsNotNone(self.runner.model)
        self.assertIsNotNone(self.runner.image_embedder)

    def test_llava_runner_initialization_model_load_failure(self, MockLlavaLlama, MockLlavaImageEmbed):
        MockLlavaLlama.create_from_gguf.side_effect = Exception("Model load error")
        runner = LlavaCppRunner(model_path="fake/llava.gguf", mmproj_path="fake/mmproj.gguf")
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.image_embedder) # Projector loading shouldn't be attempted if model fails

    def test_llava_runner_initialization_projector_load_failure(self, MockLlavaLlama, MockLlavaImageEmbed):
        # Reset model loading mock to success for this test case
        MockLlavaLlama.create_from_gguf.side_effect = None
        MockLlavaLlama.create_from_gguf.return_value = self.mock_llava_model_instance

        MockLlavaImageEmbed.load_from_gguf.side_effect = Exception("Projector load error")
        runner = LlavaCppRunner(model_path="fake/llava.gguf", mmproj_path="fake/mmproj.gguf")

        self.assertIsNotNone(runner.model, "Model should be loaded before projector loading is attempted")
        self.assertIsNone(runner.image_embedder, "Image embedder should be None after projector load failure")


    def test_llava_runner_generate_text_only(self, MockLlavaLlama, MockLlavaImageEmbed):
        response = self.runner.generate("Text prompt")
        self.mock_llava_model_instance.generate.assert_called_once_with(
            text="Text prompt",
            image_embeds=None, # No images passed
            temp=0.8, top_p=0.95, max_new_tokens=512, stop=[]
            # Other default generation params from self.additional_params would be here
        )
        self.assertEqual(response, "Mocked LLaVA generation")

    @patch('llm_context_os.runners.llava_cpp_runner.Path')
    def test_llava_runner_generate_with_images(self, MockPath, MockLlavaLlama, MockLlavaImageEmbed):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True # Simulate image file exists

        mock_embedded_image = MagicMock(spec_set=True) # Use spec_set for stricter mocking if type is known
        self.mock_image_embedder_instance.embed_image.return_value = mock_embedded_image

        mock_image_file = "dummy_image.png"
        response = self.runner.generate("Describe <image>", image_paths=[mock_image_file])

        MockPath.assert_called_with(mock_image_file) # Check Path was instantiated with the file
        mock_path_instance.exists.assert_called_once()
        self.mock_image_embedder_instance.embed_image.assert_called_with(mock_path_instance)
        self.mock_llava_model_instance.generate.assert_called_once_with(
            text="Describe <image>",
            image_embeds=[mock_embedded_image],
            temp=0.8, top_p=0.95, max_new_tokens=512, stop=[]
        )
        self.assertEqual(response, "Mocked LLaVA generation")

    def test_llava_runner_stream_text_only(self, MockLlavaLlama, MockLlavaImageEmbed):
        # Current LlavaCppRunner simulates stream by calling generate if model.generate doesn't have 'stream' param
        self.mock_llava_model_instance.generate.return_value = "Full streamed text"

        # Simulate that self.model.generate does not have 'stream' in its signature
        # to test the fallback path in LlavaCppRunner.stream()
        del self.mock_llava_model_instance.generate._mock_extra_kwargs['stream'] # If it was added by mistake

        # To properly test the inspect.signature part, we'd need a more complex mock for generate itself
        # or ensure the default mock_llava_model_instance.generate doesn't have 'stream' in its signature.
        # For simplicity, we assume the fallback is hit if 'stream' is not in sig.

        with patch('inspect.signature') as mock_inspect_signature:
            mock_signature = MagicMock()
            mock_signature.parameters = {} # Empty dict means no 'stream' parameter
            mock_inspect_signature.return_value = mock_signature

            chunks = list(self.runner.stream("Stream prompt"))

            self.mock_llava_model_instance.generate.assert_called_once_with(
                text="Stream prompt",
                image_embeds=None,
                temp=0.8, top_p=0.95, max_new_tokens=512, stop=[]
            )
            self.assertEqual(chunks, ["Full streamed text"])

    @patch('llm_context_os.runners.llava_cpp_runner.Path')
    def test_llava_runner_stream_with_images(self, MockPath, MockLlavaLlama, MockLlavaImageEmbed):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_embedded_image = MagicMock(spec_set=True)
        self.mock_image_embedder_instance.embed_image.return_value = mock_embedded_image

        self.mock_llava_model_instance.generate.return_value = "Full streamed text with image"

        with patch('inspect.signature') as mock_inspect_signature:
            mock_signature = MagicMock()
            mock_signature.parameters = {} # No 'stream' parameter
            mock_inspect_signature.return_value = mock_signature

            mock_image_file = "dummy_image.png"
            chunks = list(self.runner.stream("Stream <image>", image_paths=[mock_image_file]))

            self.mock_image_embedder_instance.embed_image.assert_called_with(mock_path_instance)
            self.mock_llava_model_instance.generate.assert_called_once_with(
                text="Stream <image>",
                image_embeds=[mock_embedded_image],
                temp=0.8, top_p=0.95, max_new_tokens=512, stop=[]
            )
            self.assertEqual(chunks, ["Full streamed text with image"])


    def test_llava_runner_count_tokens(self, MockLlavaLlama, MockLlavaImageEmbed):
        count = self.runner.count_tokens("Count this text")
        # The runner encodes with utf-8. llava-cpp-python's tokenize expects bytes.
        self.mock_llava_model_instance.tokenize.assert_called_with("Count this text".encode('utf-8'))
        self.assertEqual(count, 5) # Based on setUp mock: [1,2,3,4,5]

    def test_llava_runner_kv_cache_save_session(self, MockLlavaLlama, MockLlavaImageEmbed):
        self.runner.save_kv_cache_session("test_session.bin")
        self.mock_llava_model_instance.save_session.assert_called_once_with("test_session.bin".encode('utf-8'))

    def test_llava_runner_kv_cache_load_session(self, MockLlavaLlama, MockLlavaImageEmbed):
        self.runner.load_kv_cache_session("test_session.bin")
        self.mock_llava_model_instance.load_session.assert_called_once_with("test_session.bin".encode('utf-8'))

    def test_llava_runner_apply_lora(self, MockLlavaLlama, MockLlavaImageEmbed):
        self.runner.apply_lora("lora_path.gguf", scale=0.8)
        self.mock_llava_model_instance.apply_lora_from_file.assert_called_once_with(
            lora_path="lora_path.gguf", scale=0.8
        )

    def test_llava_runner_remove_lora(self, MockLlavaLlama, MockLlavaImageEmbed):
        self.runner.remove_lora()
        self.mock_llava_model_instance.disable_lora.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# End of llm_context_os/tests/test_runners.py
