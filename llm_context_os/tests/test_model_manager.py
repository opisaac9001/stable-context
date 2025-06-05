# llm_context_os/tests/test_model_manager.py
import unittest
from unittest.mock import patch, MagicMock, call
import time # For mocking time

from llm_context_os.runners.manager import ModelManager
from llm_context_os.runners.base import BaseRunner
from llm_context_os.runners.api_runner import APIRunner
from llm_context_os.runners.exl2_runner import EXL2Runner, EXL2_AVAILABLE
# KVCacheManager and AutoTuner are imported by ModelManager, so we patch their paths there.

# Mock instances to be configured and used
mock_kv_cache_mgr_instance = MagicMock()
mock_auto_tuner_instance = MagicMock()

@patch('llm_context_os.runners.manager.AutoTuner', MagicMock(return_value=mock_auto_tuner_instance))
@patch('llm_context_os.runners.manager.KVCacheManager', MagicMock(return_value=mock_kv_cache_mgr_instance))
class TestModelManagerIntegration(unittest.TestCase):

    def setUp(self):
        # Reset mocks for each test to ensure clean state
        mock_kv_cache_mgr_instance.reset_mock()
        mock_auto_tuner_instance.reset_mock()

        # Configure default mock behaviors (can be overridden in specific tests)
        mock_kv_cache_mgr_instance.load_kv_cache.return_value = None
        mock_kv_cache_mgr_instance.save_kv_cache.return_value = None
        mock_auto_tuner_instance.get_optimal_settings.return_value = {}
        mock_auto_tuner_instance.needs_tuning.return_value = False

        self.manager = ModelManager()
        # Ensure the manager is indeed using our mocked instances by checking a call during init
        # ModelManager initializes KVCacheManager with a root_path.
        # We can check if our mock_kv_cache_mgr_constructor was called.
        # The patching at class level should ensure ModelManager() uses the MagicMock constructors.


    @patch('llm_context_os.runners.manager.APIRunner')
    def test_load_get_unload_api_runner(self, MockAPIRunner):
        mock_api_runner_instance = MagicMock(spec=APIRunner)
        mock_api_runner_instance.model_name = "test-api-model" # Needed for some internal manager logic
        MockAPIRunner.return_value = mock_api_runner_instance

        load_params = {
            'model_type': 'api',
            'model_path_or_name': 'test-api-model',
            'alias': 'my_api_model',
            'api_url': 'http://fake.api/v1/chat/completions',
            'api_key': 'fake_key',
            'default_prefix_text': 'System: You are helpful.' # Test KV cache load call path
        }
        self.manager.load(**load_params)

        MockAPIRunner.assert_called_once_with(
            model_name='test-api-model',
            api_url='http://fake.api/v1/chat/completions',
            api_key='fake_key'
            # Other params from load_params might be passed as **kwargs to APIRunner init
        )
        # Check if kv_cache_mgr.load_kv_cache was called due to default_prefix_text
        mock_kv_cache_mgr_instance.load_kv_cache.assert_called_with(
            model_id='test-api-model', # APIRunner uses model_name as its id
            prefix_text=load_params['default_prefix_text']
        )
        # As APIRunner.preload_kv is a simple print, this mock doesn't need specific setup for it for now.
        # If it were complex, we'd mock mock_api_runner_instance.preload_kv

        runner = self.manager.get()
        self.assertIs(runner, mock_api_runner_instance)
        self.assertTrue(self.manager.is_model_loaded())
        self.assertEqual(self.manager.get_loaded_model_id(), 'test-api-model') # APIRunner's model_name

        runner.generate("test prompt")
        mock_api_runner_instance.generate.assert_called_once_with("test prompt")

        list(runner.stream("test stream prompt"))
        mock_api_runner_instance.stream.assert_called_once_with("test stream prompt")

        self.manager.unload()
        self.assertIsNone(self.manager.get())
        self.assertFalse(self.manager.is_model_loaded())
        # Check if kv_cache_mgr.save_kv_cache was called on unload
        mock_kv_cache_mgr_instance.save_kv_cache.assert_called_with(
            model_id='test-api-model', # APIRunner uses model_name as its id
            cache_object=mock_api_runner_instance.export_kv_cache.return_value # export_kv_cache is also a mock
        )
        # Check if the runner's internal cleanup (if any) like http_client.close() is called
        # For a MagicMock spec'd object, methods like close() would exist if part of APIRunner spec.
        # If APIRunner's __del__ or a specific close method exists and is important, test its call.
        # For now, assume ModelManager.unload() handles what's needed for the runner object itself.


    @patch('llm_context_os.runners.manager.EXL2Runner')
    @unittest.skipUnless(EXL2_AVAILABLE, "EXL2 dependencies not available, skipping EXL2 integration test.")
    def test_load_get_unload_exl2_runner(self, MockEXL2Runner):
        mock_exl2_runner_instance = MagicMock(spec=EXL2Runner)
        # EXL2Runner's __init__ sets up model, tokenizer, etc. Mock them if manager interacts.
        # For this test, we mostly care about the runner instance being handled correctly.
        # EXL2Runner's internal model_path is used as its ID by default in the manager
        mock_exl2_runner_instance.model_path = 'fake/exl2/model'
        MockEXL2Runner.return_value = mock_exl2_runner_instance

        # Mock the return of export_kv_cache for the EXL2 runner instance
        # EXL2Runner's export_kv_cache raises NotImplementedError.
        # The manager should catch this and not save KV cache.
        mock_exl2_runner_instance.export_kv_cache.side_effect = NotImplementedError("EXL2 KV export not implemented")


        load_params = {
            'model_type': 'exl2',
            'model_path_or_name': 'fake/exl2/model',
            'alias': 'my_exl2_model',
            'gpu_split': 'auto',
            'default_prefix_text': 'System: EXL2 model ready.'
        }
        self.manager.load(**load_params)

        MockEXL2Runner.assert_called_once_with(
            model_path='fake/exl2/model',
            gpu_split='auto'
            # Other params from load_params might be passed as **kwargs to EXL2Runner init
        )
        mock_kv_cache_mgr_instance.load_kv_cache.assert_called_with(
            model_id='fake/exl2/model', # EXL2Runner uses model_path as its id
            prefix_text=load_params['default_prefix_text']
        )


        runner = self.manager.get()
        self.assertIs(runner, mock_exl2_runner_instance)
        self.assertTrue(self.manager.is_model_loaded())
        self.assertEqual(self.manager.get_loaded_model_id(), 'fake/exl2/model')

        runner.generate("exl2 test prompt")
        mock_exl2_runner_instance.generate.assert_called_once_with("exl2 test prompt")

        list(runner.stream("exl2 test stream prompt"))
        mock_exl2_runner_instance.stream.assert_called_once_with("exl2 test stream prompt")

        self.manager.unload()
        self.assertIsNone(self.manager.get())
        self.assertFalse(self.manager.is_model_loaded())
        # Check that save_kv_cache was NOT called for EXL2 due to NotImplementedError from export_kv_cache
        mock_kv_cache_mgr_instance.save_kv_cache.assert_not_called()


    @patch('llm_context_os.runners.manager.APIRunner')
    @patch('time.time') # Patch time globally for this test method
    def test_idle_unload_api_runner(self, mock_time, MockAPIRunner):
        mock_api_runner_instance = MagicMock(spec=APIRunner)
        mock_api_runner_instance.model_name = "idle-test-api-model"
        MockAPIRunner.return_value = mock_api_runner_instance

        mock_time.return_value = 1000.0 # Initial time

        load_params = {
            'model_type': 'api',
            'model_path_or_name': 'idle-test-api-model',
            'api_url': 'http://fake.api/idle',
            'idle_unload_sec': 10 # Unload after 10 seconds of inactivity
        }
        self.manager.load(**load_params)
        self.assertIsNotNone(self.manager.get(), "Model should be loaded initially.")
        self.assertEqual(self.manager.last_accessed_time, 1000.0)

        # Simulate time passing (5 seconds), access the model
        mock_time.return_value = 1005.0
        self.manager.get() # Access resets idle timer
        self.assertEqual(self.manager.last_accessed_time, 1005.0, "Last accessed time should update on get()")
        self.manager.check_idle() # Should not unload
        self.assertIsNotNone(self.manager.get(), "Model should still be loaded after 5s and access.")

        # Simulate time passing beyond idle_unload_sec (6 seconds after last access at 1005.0)
        mock_time.return_value = 1011.0
        self.manager.check_idle() # Should unload now
        self.assertIsNone(self.manager.get(), "Model should be unloaded after idle timeout.")
        self.assertFalse(self.manager.is_model_loaded())

        # Verify save_kv_cache was called upon idle unload
        mock_kv_cache_mgr_instance.save_kv_cache.assert_called_with(
            model_id='idle-test-api-model',
            cache_object=mock_api_runner_instance.export_kv_cache.return_value
        )

    @patch('llm_context_os.runners.manager.EXL2Runner')
    @unittest.skipUnless(EXL2_AVAILABLE, "EXL2 dependencies not available.")
    def test_load_exl2_with_kv_preload_not_implemented(self, MockEXL2Runner):
        mock_exl2_runner_instance = MagicMock(spec=EXL2Runner)
        mock_exl2_runner_instance.model_path = 'fake/exl2/kv_error_model'
        # Configure preload_kv to raise NotImplementedError
        mock_exl2_runner_instance.preload_kv.side_effect = NotImplementedError("EXL2 KV preload not implemented")
        MockEXL2Runner.return_value = mock_exl2_runner_instance

        load_params = {
            'model_type': 'exl2',
            'model_path_or_name': 'fake/exl2/kv_error_model',
            'default_prefix_text': 'Warmup prompt for KV cache'
        }

        # We expect the manager to catch the NotImplementedError from preload_kv and continue loading
        # It should log an error (not easily testable without patching logging) but not crash.
        try:
            self.manager.load(**load_params)
        except Exception as e:
            self.fail(f"Manager.load() crashed with unexpected exception: {e} when runner's preload_kv raised NotImplementedError.")

        self.assertIsNotNone(self.manager.get(), "Model should still be loaded even if preload_kv fails with NotImplementedError.")
        self.assertIs(self.manager.get(), mock_exl2_runner_instance)

        # Verify that load_kv_cache was still attempted on the KVCacheManager
        mock_kv_cache_mgr_instance.load_kv_cache.assert_called_with(
            model_id='fake/exl2/kv_error_model',
            prefix_text=load_params['default_prefix_text']
        )
        # Verify that the runner's preload_kv was indeed called
        mock_exl2_runner_instance.preload_kv.assert_called_once_with(
            load_params['default_prefix_text'],
            cache_to_use=mock_kv_cache_mgr_instance.load_kv_cache.return_value
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
