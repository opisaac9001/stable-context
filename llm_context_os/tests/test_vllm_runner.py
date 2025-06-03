# llm_context_os/tests/test_vllm_runner.py
import unittest
import typing as t
from io import StringIO # To capture print statements if needed, though not strictly for these tests
import sys # For print capturing if used
from unittest.mock import patch # If we wanted to assert prints

from llm_context_os.runners.vllm_runner import VLLMRunner
from llm_context_os.runners.base import BaseRunner # For type checking if needed

class TestVLLMRunner(unittest.TestCase):
    def setUp(self):
        self.model_name = "test-vllm-model/some-variant"
        self.api_url = "http://localhost:23456/v1" # Example, not actually called

        # Suppress print statements from VLLMRunner's __init__ during test setup
        # to keep test output clean, unless we want to assert them.
        with patch('sys.stdout', new_callable=StringIO):
            self.runner = VLLMRunner(model_name=self.model_name, api_url=self.api_url, api_key="sk-test")

    def test_instantiation(self):
        self.assertIsNotNone(self.runner)
        self.assertIsInstance(self.runner, BaseRunner)
        self.assertEqual(self.runner.model_name, self.model_name)
        self.assertEqual(self.runner.api_url, self.api_url)
        self.assertEqual(self.runner.api_key, "sk-test")

    def test_generate_placeholder(self):
        test_prompt = "Hello vLLM world, this is a test prompt."
        # Expected response from placeholder: f"[vLLM gen for {self.model_name}: {prompt[:30]}...]"
        # prompt[:30] for the above is "Hello vLLM world, this is a t"
        expected_response = f"[vLLM gen for {self.model_name}: {test_prompt[:30]}...]"

        # Capture prints from the generate method if we want to assert them
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            response = self.runner.generate(test_prompt, max_new_tokens=10, temperature=0.1)

        self.assertEqual(response, expected_response)
        # Example: Asserting specific prints (optional)
        # log_output = mock_stdout.getvalue()
        # self.assertIn(f"[VLLMRunner] generate called for model '{self.model_name}'", log_output)
        # self.assertIn(f'"prompt": "{test_prompt}"', log_output) # Check if full prompt in logged payload

    def test_stream_placeholder(self):
        test_prompt = "Stream this vLLM test prompt for placeholder."
        # Expected chunks from VLLMRunner placeholder (as of its current implementation):
        # yield "[vLLM_stream_chunk1_begin] "
        # yield f"data: {{\"id\": \"cmpl-xxxxxxxxxxxx\", ..., \"text\": \"{prompt[:10]}...\", ...}}\n\n"
        # yield "[vLLM_stream_chunk2_more_text] "
        # yield f"data: {{\"id\": \"cmpl-xxxxxxxxxxxx\", ..., \"text\": \" more text...\", ...}}\n\n"
        # yield "data: [DONE]\n\n"

        expected_chunks = [
            "[vLLM_stream_chunk1_begin] ",
            f"data: {{\"id\": \"cmpl-xxxxxxxxxxxx\", \"object\": \"text_completion.chunk\", \"created\": 12345, \"model\": \"{self.model_name}\", \"choices\": [{{\"text\": \"{test_prompt[:10]}...\", \"index\": 0, \"logprobs\": null, \"finish_reason\": null}}]}}\n\n",
            "[vLLM_stream_chunk2_more_text] ",
            f"data: {{\"id\": \"cmpl-xxxxxxxxxxxx\", \"object\": \"text_completion.chunk\", \"created\": 12345, \"model\": \"{self.model_name}\", \"choices\": [{{\"text\": \" more text...\", \"index\": 0, \"logprobs\": null, \"finish_reason\": null}}]}}\n\n",
            "data: [DONE]\n\n"
        ]

        with patch('sys.stdout', new_callable=StringIO): # Capture prints
            stream_iter = self.runner.stream(test_prompt, temperature=0.1)

        self.assertTrue(hasattr(stream_iter, '__iter__') and hasattr(stream_iter, '__next__'),
                        "Stream method did not return an iterator.")

        collected_chunks = list(stream_iter)
        self.assertEqual(collected_chunks, expected_chunks)

    def test_kv_cache_methods_placeholders(self):
        # These methods in the VLLMRunner placeholder primarily print messages.
        # We test that they can be called without error and return expected values.

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout_export:
            exported_cache = self.runner.export_kv_cache()
        self.assertIsNone(exported_cache, "export_kv_cache should return None for VLLMRunner placeholder.")
        self.assertIn("vLLM manages its KV cache internally", mock_stdout_export.getvalue())

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout_import:
            self.runner.import_kv_cache({'dummy_key': 'dummy_value'})
        self.assertIn("Not typically applicable for vLLM", mock_stdout_import.getvalue())

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout_preload:
            self.runner.preload_kv("Sample prefix for preload.")
        self.assertIn("vLLM's continuous batching handles prefix caching implicitly", mock_stdout_preload.getvalue())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
