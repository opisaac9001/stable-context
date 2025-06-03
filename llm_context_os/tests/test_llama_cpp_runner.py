# llm_context_os/tests/test_llama_cpp_runner.py
import unittest
from unittest.mock import patch, MagicMock, mock_open, ANY
import typing as t
from pathlib import Path
import tempfile
import os
from io import StringIO # Correct import for StringIO

from llm_context_os.runners.llama_cpp_runner import LlamaCppRunner

DUMMY_MODEL_PATH = "test_model.gguf"

# Class-level patch removed, will be handled in setUp for more control
class TestLlamaCppRunner(unittest.TestCase):

    def setUp(self):
        self.patcher = patch('llm_context_os.runners.llama_cpp_runner.Llama')
        self.MockLlamaClass = self.patcher.start() # This is the MagicMock class
        self.addCleanup(self.patcher.stop) # Ensures patch is stopped after each test

        # Configure the instance that self.MockLlamaClass() will return by default
        self.mock_llama_instance_default_config = self.MockLlamaClass.return_value
        self.mock_llama_instance_default_config.n_ctx_train.return_value = 2048
        self.mock_llama_instance_default_config.n_ctx.return_value = 2048

        # Default side_effect for create_completion for stream/non-stream differentiation
        # This will apply to the mock instance returned by MockLlamaClass()
        def create_completion_side_effect(*args, **kwargs_call):
            if kwargs_call.get("stream") is True: # Check if stream=True was passed
                return iter([{'choices': [{'text': 'default stream chunk '}]}])
            return {'choices': [{'text': 'default non-stream response'}]}
        self.mock_llama_instance_default_config.create_completion.side_effect = create_completion_side_effect

        # Now instantiate the runner; its self.model will be self.mock_llama_instance_default_config
        with patch('sys.stdout', new_callable=StringIO): # Suppress LlamaCppRunner's init prints
            self.runner = LlamaCppRunner(
                model_path=DUMMY_MODEL_PATH,
                n_gpu_layers=0,
                n_ctx=2048,
                verbose=False
            )
        # For most tests, self.runner.model can be used directly.
        # It's an instance of MagicMock (MockLlamaClass.return_value)
        # We can reset its specific method mocks (like create_completion) if needed per test.
        if self.runner.model: # Should be the mock instance now
            self.runner.model.reset_mock() # Reset calls, but not return_values/side_effects set on instance
            # Re-apply side_effect because reset_mock clears it.
            self.runner.model.create_completion.side_effect = create_completion_side_effect
            self.runner.model.n_ctx_train.return_value = 2048 # Ensure these are set on the instance
            self.runner.model.n_ctx.return_value = 2048


    def test_init_successful_model_load(self):
        # self.MockLlamaClass is available from setUp's patcher.start()
        # self.runner was already created in setUp. We check the call to its __init__.
        self.MockLlamaClass.assert_called_once_with(
            model_path=DUMMY_MODEL_PATH,
            n_gpu_layers=0, n_ctx=2048, verbose=False, seed=ANY, n_threads=None,
            n_batch=512, logits_all=False, embedding=False
        )
        self.assertIsNotNone(self.runner.model, "Model should be loaded (mocked).")
        self.assertEqual(self.runner.model, self.mock_llama_instance_default_config)

    def test_init_model_load_failure(self):
        self.MockLlamaClass.side_effect = Exception("Test Llama load error")
        with patch('sys.stdout', new_callable=StringIO):
            runner = LlamaCppRunner(model_path="fail_model.gguf")
        self.assertIsNone(runner.model, "Model should be None after a load failure.")
        self.MockLlamaClass.side_effect = None # Reset for other tests

    def test_generate_calls_model_create_completion(self):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        test_prompt = "Generate test prompt"
        expected_completion_text = 'Generated text for test'

        self.runner.model.create_completion.side_effect = None # Clear stream side_effect
        self.runner.model.create_completion.return_value = {'choices': [{'text': expected_completion_text}]}

        response = self.runner.generate(test_prompt, max_new_tokens=50, temperature=0.5)

        self.runner.model.create_completion.assert_called_once()
        called_args_kwargs = self.runner.model.create_completion.call_args.kwargs
        self.assertEqual(called_args_kwargs['prompt'], test_prompt)
        self.assertEqual(called_args_kwargs['max_tokens'], 50)
        self.assertEqual(called_args_kwargs['temperature'], 0.5)
        self.assertFalse(called_args_kwargs.get('stream', False))
        self.assertEqual(response, expected_completion_text)

    def test_generate_no_model(self):
        with patch('sys.stdout', new_callable=StringIO):
            self.runner.model = None
            response = self.runner.generate("Prompt")
        self.assertEqual(response, "[LlamaCppRunner] Error: Model not loaded.")

    def test_stream_calls_model_create_completion_stream(self):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        test_prompt = "Stream test prompt"
        mock_stream_chunks = [{'choices': [{'text': 'Streamed '}]}, {'choices': [{'text': 'text.'}]}]

        self.runner.model.create_completion.side_effect = lambda *args, **kwargs_call: iter(mock_stream_chunks) if kwargs_call.get("stream") else None

        with patch('sys.stdout', new_callable=StringIO):
            collected_chunks = list(self.runner.stream(test_prompt, max_new_tokens=60, top_p=0.9))

        self.runner.model.create_completion.assert_called_once()
        called_args_kwargs = self.runner.model.create_completion.call_args.kwargs
        self.assertEqual(called_args_kwargs['prompt'], test_prompt)
        self.assertEqual(called_args_kwargs['max_tokens'], 60)
        self.assertEqual(called_args_kwargs['top_p'], 0.9)
        self.assertTrue(called_args_kwargs['stream'])
        self.assertEqual(collected_chunks, ['Streamed ', 'text.'])

    def test_stream_no_model(self):
        with patch('sys.stdout', new_callable=StringIO):
            self.runner.model = None
            response_generator = self.runner.stream("Prompt")
            self.assertEqual(next(response_generator), "[LlamaCppRunner] Error: Model not loaded.")
        with self.assertRaises(StopIteration):
            next(response_generator)

    def test_preload_kv_calls_tokenize_eval(self):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        test_prompt = "Preload this context."
        mock_tokens = [10, 20, 30]
        self.runner.model.tokenize.return_value = mock_tokens

        with patch('sys.stdout', new_callable=StringIO):
            self.runner.preload_kv(test_prompt)

        self.runner.model.tokenize.assert_called_once_with(test_prompt.encode('utf-8', errors='ignore'))
        self.runner.model.eval.assert_called_once_with(mock_tokens)

    @patch('tempfile.NamedTemporaryFile')
    def test_export_kv_cache_calls_save_session(self, mock_named_temp_file):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        mock_temp_file_obj = MagicMock()
        mock_temp_file_obj.name = "dummy_temp_path.kst"
        mock_named_temp_file.return_value.__enter__.return_value = mock_temp_file_obj

        with patch('sys.stdout', new_callable=StringIO):
            returned_path = self.runner.export_kv_cache()

        self.runner.model.save_session_file.assert_called_once_with("dummy_temp_path.kst")
        self.assertEqual(returned_path, "dummy_temp_path.kst")

    @patch('os.path.exists')
    def test_import_kv_cache_calls_load_session(self, mock_os_path_exists):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        mock_os_path_exists.return_value = True
        dummy_cache_path = "imported_cache.kst"

        with patch('sys.stdout', new_callable=StringIO):
            self.runner.import_kv_cache(dummy_cache_path)

        self.runner.model.load_session_file.assert_called_once_with(dummy_cache_path)
        mock_os_path_exists.assert_called_once_with(dummy_cache_path)

    def test_load_unload_lora_updates_active_list(self):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        lora_id = "test_lora1"
        lora_path = "/path/to/lora1.bin"
        lora_scale = 0.75

        with patch('sys.stdout', new_callable=StringIO):
            load_success = self.runner.load_lora_adapter(lora_id, lora_path, adapter_scale=lora_scale)
        self.assertTrue(load_success)
        self.assertIn(lora_id, self.runner.active_loras)
        self.assertEqual(self.runner.active_loras[lora_id], (lora_path, lora_scale))

        with patch('sys.stdout', new_callable=StringIO):
            unload_success = self.runner.unload_lora_adapter(lora_id)
        self.assertTrue(unload_success)
        self.assertNotIn(lora_id, self.runner.active_loras)

    def test_generate_with_active_lora(self):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")

        lora_id = "test_lora_gen"
        lora_path = "/path/to/lora_gen.bin"
        lora_scale = 0.6
        with patch('sys.stdout', new_callable=StringIO):
            self.runner.load_lora_adapter(lora_id, lora_path, adapter_scale=lora_scale)

        self.runner.model.create_completion.side_effect = None
        self.runner.model.create_completion.return_value = {'choices': [{'text': 'LoRA generated text'}]}

        with patch('sys.stdout', new_callable=StringIO):
            self.runner.generate("Prompt for LoRA generation")

        self.runner.model.create_completion.assert_called_once()
        called_args_kwargs = self.runner.model.create_completion.call_args.kwargs
        self.assertEqual(called_args_kwargs.get('lora_path'), lora_path)
        self.assertEqual(called_args_kwargs.get('lora_scale'), lora_scale)

    def test_merge_unmerge_lora_placeholders(self):
        if not self.runner.model: self.skipTest("Model not loaded in setUp correctly")
        with patch('sys.stdout', new_callable=StringIO):
            self.assertFalse(self.runner.merge_lora_adapters(["some_lora"]))
            self.assertFalse(self.runner.unmerge_lora_adapters())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
