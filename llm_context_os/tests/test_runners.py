# llm_context_os/tests/test_runners.py
import unittest
import io
import sys
from contextlib import redirect_stdout

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

class TestRunners(unittest.TestCase):

    def test_api_runner_instantiation_and_methods(self):
        with Capturing() as output: # Capture prints during instantiation
            runner = APIRunner(model_name="test-api-model", api_url="http://dummy.api/v1", api_key="testkey")
        self.assertIsInstance(runner, APIRunner)
        self.assertIn("APIRunner initialized for model 'test-api-model'", "\n".join(output))

        with Capturing() as output:
            response = runner.generate(prompt="test api prompt")
        self.assertIn("[API Response from test-api-model to: test api prompt...]", response)
        self.assertIn("--- APIRunner (test-api-model) Generating ---", "\n".join(output))

        stream_output = []
        with Capturing() as output:
            for chunk in runner.stream(prompt="test api stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[Chunk 1 from test-api-model" in chunk for chunk in stream_output))
        self.assertTrue(any("[End of stream from test-api-model]" in chunk for chunk in stream_output))
        self.assertIn("--- APIRunner (test-api-model) Streaming ---", "\n".join(output))

    def test_llama_cpp_runner_instantiation_and_methods(self):
        with Capturing(): # Suppress prints for this test's instantiation
            runner = LlamaCppRunner(model_path="dummy.gguf", n_gpu_layers=0)
        self.assertIsInstance(runner, LlamaCppRunner)

        with Capturing():
            response = runner.generate(prompt="test llama prompt")
        self.assertIn("[LlamaCpp Response from dummy.gguf to: test llama prompt...]", response)

        stream_output = []
        with Capturing():
            for chunk in runner.stream(prompt="test llama stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[LlamaCpp Chunk 1" in chunk for chunk in stream_output))
        self.assertTrue(any("[LlamaCpp End of Stream]" in chunk for chunk in stream_output))

    def test_awq_runner_instantiation_and_methods(self):
        with Capturing():
            runner = AWQRunner(model_path_or_repo_id="dummy/awq-model")
        self.assertIsInstance(runner, AWQRunner)

        with Capturing():
            response = runner.generate(prompt="test awq prompt")
        self.assertIn("[AWQ Response from dummy/awq-model to: test awq prompt...]", response)

        stream_output = []
        with Capturing():
            for chunk in runner.stream(prompt="test awq stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[AWQ Chunk 1" in chunk for chunk in stream_output))
        self.assertTrue(any("[AWQ End of Stream]" in chunk for chunk in stream_output))

    def test_exl2_runner_instantiation_and_methods(self):
        with Capturing():
            runner = EXL2Runner(model_path="dummy/exl2-model-dir")
        self.assertIsInstance(runner, EXL2Runner)

        with Capturing():
            response = runner.generate(prompt="test exl2 prompt")
        self.assertIn("[EXL2 Response from exl2-model-dir to: test exl2 prompt...]", response)

        stream_output = []
        with Capturing():
            for chunk in runner.stream(prompt="test exl2 stream"):
                stream_output.append(chunk)
        self.assertTrue(any("[EXL2 Chunk 1" in chunk for chunk in stream_output))
        self.assertTrue(any("[EXL2 End of Stream]" in chunk for chunk in stream_output))


class TestModelManager(unittest.TestCase):

    def setUp(self):
        # Suppress prints from manager itself for cleaner test logs if needed
        # For now, let them pass to see manager's logging during tests.
        self.manager = ModelManager()

    def test_manager_initialization(self):
        self.assertIsNone(self.manager.current_runner)

    def test_get_runner_when_none_loaded(self):
        with Capturing() as output: # Capture "Error: No model is currently loaded."
            runner = self.manager.get()
        self.assertIsNone(runner)
        self.assertTrue(any("Error: No model is currently loaded." in line for line in output))


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
        self.assertEqual(runner.n_gpu_layers, 10)

    def test_load_awq_runner(self):
        with Capturing():
            self.manager.load('awq', 'test/awq-model', device='cpu')
        runner = self.manager.get()
        self.assertIsInstance(runner, AWQRunner)
        self.assertEqual(runner.model_path_or_repo_id, 'test/awq-model')
        self.assertEqual(runner.device, 'cpu')

    def test_load_exl2_runner(self):
        with Capturing():
            self.manager.load(model_type='exl2', model_path_or_name='test/exl2-model', gpu_split="auto")
        runner = self.manager.get()
        self.assertIsInstance(runner, EXL2Runner)
        self.assertEqual(runner.model_path, 'test/exl2-model')
        self.assertEqual(runner.gpu_split_str, "auto")

    def test_load_unloads_previous_runner(self):
        with Capturing():
            self.manager.load(model_type='api', model_path_or_name='first-model', api_url='http://dummy.api')
        first_runner = self.manager.get() # Get the actual runner object

        with Capturing():
            self.manager.load(model_type='gguf', model_path_or_name='second-model.gguf')
        second_runner = self.manager.get() # Get the actual runner object

        self.assertIsNotNone(first_runner, "First runner should not be None")
        self.assertIsNotNone(second_runner, "Second runner should not be None")
        self.assertIsInstance(self.manager.current_runner, LlamaCppRunner) # Check current type
        self.assertNotEqual(id(first_runner), id(second_runner), "Manager should have loaded a new runner instance.")


    def test_load_invalid_model_type(self):
        with Capturing(): # Capture prints from load
            self.manager.load(model_type='api', model_path_or_name='initial-model', api_url='http://dummy.api')
        initial_runner = self.manager.get() # APIRunner instance
        self.assertIsInstance(initial_runner, APIRunner)

        with Capturing() as output_invalid_load:
            self.manager.load(model_type='INVALID_TYPE', model_path_or_name='some/path')

        # Check that an error was printed
        self.assertTrue(any("Error: Unknown model type 'INVALID_TYPE'" in line for line in output_invalid_load))

        # Check that current_runner is now None because the load failed after unloading previous
        final_runner = self.manager.get()
        self.assertIsNone(final_runner, "Runner should be None after failed load of invalid type.")
        self.assertIsNone(self.manager.current_model_type)
        self.assertIsNone(self.manager.current_model_identifier)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
