# llm_context_os/tests/test_runners.py
import unittest
import io
import sys
from contextlib import redirect_stdout
import typing as t # Added for t.Any
from pathlib import Path # Added import for Path
import shutil # Added import for shutil

from llm_context_os.runners.base import BaseRunner
from llm_context_os.runners.api_runner import APIRunner
from llm_context_os.runners.llama_cpp_runner import LlamaCppRunner
from llm_context_os.runners.awq_runner import AWQRunner
from llm_context_os.runners.exl2_runner import EXL2Runner
from llm_context_os.runners.manager import ModelManager

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

class TestApiRunner(unittest.TestCase):
    def setUp(self):
        with Capturing(): # Suppress init prints during test setup
            self.runner = APIRunner(model_name="test-api-model", api_url="http://dummy.api/v1", api_key="testkey")

    def test_instantiation_and_methods(self):
        self.assertIsInstance(self.runner, APIRunner)
        # Check if init print happened (though captured in setUp, we can check runner attributes)
        self.assertEqual(self.runner.model_name, "test-api-model")

        with Capturing() as output_gen:
            response = self.runner.generate(prompt="test api prompt")
        self.assertIn("[API Response from test-api-model to: test api prompt...]", response)
        self.assertIn("--- APIRunner (test-api-model) Generating ---", "\n".join(output_gen))

        stream_output_content = []
        with Capturing() as output_stream:
            for chunk in self.runner.stream(prompt="test api stream"):
                stream_output_content.append(chunk)
        self.assertTrue(any("[Chunk 1 from test-api-model" in chunk for chunk in stream_output_content))
        self.assertTrue(any("[End of stream from test-api-model]" in chunk for chunk in stream_output_content))
        self.assertIn("--- APIRunner (test-api-model) Streaming ---", "\n".join(output_stream))

    def test_kv_cache_export_import(self):
        mock_data = {'key': 'value_api'}
        with Capturing() as import_output:
            self.runner.import_kv_cache(mock_data)
        self.assertTrue(any("Importing KV cache is generally not applicable" in line for line in import_output))

        with Capturing() as export_output:
            exported_data = self.runner.export_kv_cache()
        self.assertIsNone(exported_data) # APIRunner returns None
        self.assertTrue(any("API runners typically do not manage or expose exportable KV cache" in line for line in export_output))

        with Capturing() as preload_output:
            self.runner.preload_kv('some prefix')
        self.assertTrue(any("preload_kv called. Typically, KV caching is managed by the remote API endpoint" in line for line in preload_output))


class TestLlamaCppRunner(unittest.TestCase):
    def setUp(self):
        with Capturing():
            self.runner = LlamaCppRunner(model_path="dummy.gguf", n_gpu_layers=0)

    def test_instantiation_and_methods(self):
        self.assertIsInstance(self.runner, LlamaCppRunner)
        self.assertEqual(self.runner.model_path, "dummy.gguf")

        with Capturing():
            response = self.runner.generate(prompt="test llama prompt")
        self.assertIn("[LlamaCpp Response from dummy.gguf to: test llama prompt...]", response)

        stream_output = []
        with Capturing():
            for chunk in self.runner.stream(prompt="test llama stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[LlamaCpp Chunk 1" in chunk for chunk in stream_output))

    def test_kv_cache_export_import(self):
        mock_data_in = {'status': 'imported_test_data_llama', 'data': [1, 2, 3]}
        with Capturing():
            self.runner.import_kv_cache(mock_data_in)
        self.assertEqual(self.runner.mock_kv_cache_data, mock_data_in)

        with Capturing():
            exported_data = self.runner.export_kv_cache()
        self.assertEqual(exported_data, mock_data_in)

        prefix_text = 'test prefix for llama'
        with Capturing():
            self.runner.preload_kv(prefix_text)
        # Check if mock_kv_cache_data was updated by preload_kv's placeholder logic
        self.assertIsNotNone(self.runner.mock_kv_cache_data)
        self.assertEqual(self.runner.mock_kv_cache_data.get("status"), "preloaded")
        self.assertEqual(self.runner.mock_kv_cache_data.get("preloaded_prompt_prefix"), prefix_text[:50])

        exported_after_preload = self.runner.export_kv_cache()
        self.assertEqual(exported_after_preload, self.runner.mock_kv_cache_data)


class TestAWQRunner(unittest.TestCase):
    def setUp(self):
        with Capturing():
            self.runner = AWQRunner(model_path_or_repo_id="dummy/awq-model")

    def test_instantiation_and_methods(self):
        self.assertIsInstance(self.runner, AWQRunner)
        self.assertEqual(self.runner.model_path_or_repo_id, "dummy/awq-model")

        with Capturing():
            response = self.runner.generate(prompt="test awq prompt")
        self.assertIn("[AWQ Response from dummy/awq-model to: test awq prompt...]", response)

        stream_output = []
        with Capturing():
            for chunk in self.runner.stream(prompt="test awq stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[AWQ Chunk 1" in chunk for chunk in stream_output))

    def test_kv_cache_export_import(self):
        mock_data_in = {'status': 'imported_test_data_awq', 'past_key_values_mock': ((1,), (2,))}
        with Capturing():
            self.runner.import_kv_cache(mock_data_in)
        self.assertEqual(self.runner.mock_kv_cache_data, mock_data_in)

        with Capturing():
            exported_data = self.runner.export_kv_cache()
        self.assertEqual(exported_data, mock_data_in)

        prefix_text = 'test prefix for awq'
        with Capturing():
            self.runner.preload_kv(prefix_text)
        self.assertIsNotNone(self.runner.mock_kv_cache_data)
        self.assertEqual(self.runner.mock_kv_cache_data.get("status"), "preloaded_awq")
        self.assertEqual(self.runner.mock_kv_cache_data.get("preloaded_prompt_hash"), hash(prefix_text))

        exported_after_preload = self.runner.export_kv_cache()
        self.assertEqual(exported_after_preload, self.runner.mock_kv_cache_data)


class TestEXL2Runner(unittest.TestCase):
    def setUp(self):
        with Capturing():
            self.runner = EXL2Runner(model_path="dummy/exl2-model-dir")

    def test_instantiation_and_methods(self):
        self.assertIsInstance(self.runner, EXL2Runner)
        self.assertEqual(self.runner.model_path, "dummy/exl2-model-dir")

        with Capturing():
            response = self.runner.generate(prompt="test exl2 prompt")
        self.assertIn("[EXL2 Response from exl2-model-dir to: test exl2 prompt...]", response)

        stream_output = []
        with Capturing():
            for chunk in self.runner.stream(prompt="test exl2 stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[EXL2 Chunk 1" in chunk for chunk in stream_output))

    def test_kv_cache_export_import(self):
        mock_data_in = {'status': 'imported_test_data_exl2', 'cache_object_state_mock': "some_state"}
        with Capturing():
            self.runner.import_kv_cache(mock_data_in)
        self.assertEqual(self.runner.mock_kv_cache_object, mock_data_in)

        with Capturing():
            exported_data = self.runner.export_kv_cache()
        self.assertEqual(exported_data, mock_data_in)

        prefix_text = 'test prefix for exl2'
        with Capturing():
            self.runner.preload_kv(prefix_text)
        self.assertIsNotNone(self.runner.mock_kv_cache_object)
        self.assertEqual(self.runner.mock_kv_cache_object.get("status"), "preloaded_exl2")
        self.assertEqual(self.runner.mock_kv_cache_object.get("preloaded_prompt_len"), len(prefix_text))

        exported_after_preload = self.runner.export_kv_cache()
        self.assertEqual(exported_after_preload, self.runner.mock_kv_cache_object)


class TestModelManager(unittest.TestCase):
    def setUp(self):
        # For ModelManager tests, ensure KVCacheManager uses a specific test directory
        # that can be cleaned up if needed. This is more important for manager tests
        # as manager interacts with KVCacheManager.
        self.test_cache_dir = Path("data/kv_cache_test_manager")
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
        # KVCacheManager in ModelManager's __init__ will create its default if not overridden.
        # For isolated tests, it's better to control it.
        self.manager = ModelManager() # This will print its default cache dir
        self.manager.kv_cache_mgr.cache_dir = self.test_cache_dir # Override cache_dir
        self.test_cache_dir.mkdir(parents=True, exist_ok=True) # Ensure test dir exists after override

    def tearDown(self):
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
        # Reset ModelManager's kv_cache_mgr to default to avoid interference if tests run in same process
        # This is less critical if each test run is a fresh Python process.
        # self.manager.kv_cache_mgr = KVCacheManager() # Resets to default "data/kv_cache"

    def test_manager_initialization(self):
        self.assertIsNone(self.manager.current_runner)
        self.assertTrue(str(self.test_cache_dir.name) in str(self.manager.kv_cache_mgr.cache_dir.name))


    def test_get_runner_when_none_loaded(self):
        with Capturing(): # Suppress "Error: No model is currently loaded." from manager.get()
            runner = self.manager.get()
        self.assertIsNone(runner)

    def test_load_api_runner(self):
        with Capturing():
            self.manager.load(model_type='api', model_path_or_name='test-api', api_url='http://dummy.api')
        runner = self.manager.get()
        self.assertIsInstance(runner, APIRunner)
        self.assertEqual(runner.model_name, 'test-api')

    def test_load_gguf_runner(self):
        with Capturing():
            self.manager.load(model_type='gguf', model_path_or_name='test.gguf', n_gpu_layers=10)
        runner = self.manager.get()
        self.assertIsInstance(runner, LlamaCppRunner)
        self.assertEqual(runner.model_path, 'test.gguf')

    def test_load_awq_runner(self):
        with Capturing():
            self.manager.load('awq', 'test/awq-model', device='cpu')
        runner = self.manager.get()
        self.assertIsInstance(runner, AWQRunner)
        self.assertEqual(runner.model_path_or_repo_id, 'test/awq-model')

    def test_load_exl2_runner(self):
        with Capturing():
            self.manager.load(model_type='exl2', model_path_or_name='test/exl2-model', gpu_split="auto")
        runner = self.manager.get()
        self.assertIsInstance(runner, EXL2Runner)
        self.assertEqual(runner.model_path, 'test/exl2-model')

    def test_load_unloads_previous_runner(self):
        with Capturing():
            self.manager.load(model_type='api', model_path_or_name='first-model', api_url='http://dummy.api')
        first_runner = self.manager.get()

        with Capturing():
            self.manager.load(model_type='gguf', model_path_or_name='second-model.gguf')
        second_runner = self.manager.get()

        self.assertIsNotNone(first_runner)
        self.assertIsNotNone(second_runner)
        self.assertIsInstance(self.manager.current_runner, LlamaCppRunner)
        self.assertNotEqual(id(first_runner), id(second_runner))

    def test_load_invalid_model_type(self):
        with Capturing():
            self.manager.load(model_type='api', model_path_or_name='initial-model', api_url='http://dummy.api')
        self.assertIsInstance(self.manager.current_runner, APIRunner)

        with Capturing() as output_invalid_load:
            self.manager.load(model_type='INVALID_TYPE', model_path_or_name='some/path')

        self.assertTrue(any("Error: Unknown model type 'INVALID_TYPE'" in line for line in output_invalid_load))
        self.assertIsNone(self.manager.current_runner)
        self.assertIsNone(self.manager.current_model_type)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
