# llm_context_os/tests/test_auto_tuner.py
import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import shutil
import random # For mocking
import typing as t # For t.Any, t.List, t.Dict
from io import StringIO # Import for new_callable

from llm_context_os.tuning.auto_tuner import AutoTuner
from llm_context_os.runners.manager import ModelManager
from llm_context_os.runners.llama_cpp_runner import LlamaCppRunner
from llm_context_os.caching.kv_cache_manager import KVCacheManager

class TestAutoTuner(unittest.TestCase):
    def setUp(self):
        self.test_results_dir_root = Path("data_test_auto_tuner_module") # Unique root for this test class
        self.test_results_file = self.test_results_dir_root / "tuning_results_unittest.json"

        # Clean up before each test method to ensure isolation
        if self.test_results_dir_root.exists():
            shutil.rmtree(self.test_results_dir_root)
        self.test_results_dir_root.mkdir(parents=True, exist_ok=True) # AutoTuner's init expects parent dir to exist

        self.tuner = AutoTuner(results_path=str(self.test_results_file))

    def tearDown(self):
        if self.test_results_dir_root.exists():
            shutil.rmtree(self.test_results_dir_root)

    def test_instantiation(self):
        self.assertIsNotNone(self.tuner)
        self.assertEqual(self.tuner.results_path, self.test_results_file)
        # AutoTuner's __init__ ensures its parent dir exists, so self.test_results_file.parent should exist
        self.assertTrue(self.test_results_file.parent.exists())

    def test_benchmark_settings_returns_best(self):
        settings_list = [
            {'param': 'a', 'value': 1, 'n_gpu_layers': 10}, # To match expected keys by some runners
            {'param': 'b', 'value': 2, 'n_gpu_layers': 20}
        ]
        # Mock random.uniform to control scores; first call gets 70.0, second 90.0
        with patch('random.uniform', side_effect=[70.0, 90.0]) as mock_rand_uniform:
            # Suppress prints from benchmark_settings
            with patch('sys.stdout', new_callable=StringIO):
                best = self.tuner.benchmark_settings("test/model", "gguf", settings_list)
            self.assertEqual(best, settings_list[1]) # Second one should have higher score
            self.assertEqual(mock_rand_uniform.call_count, 2)

    def test_get_optimal_settings_no_saved_file(self):
        self.assertFalse(self.test_results_file.exists())
        expected_optimal = {'n_gpu_layers': 35, 'attn_implementation': 'native'} # Example from gguf options

        # Mock benchmark_settings to return a predictable value
        with patch.object(self.tuner, 'benchmark_settings', return_value=expected_optimal) as mock_benchmark:
            with patch('sys.stdout', new_callable=StringIO): # Suppress prints
                optimal = self.tuner.get_optimal_settings("test/model.gguf", "gguf")

        self.assertEqual(optimal, expected_optimal)
        mock_benchmark.assert_called_once() # Check that benchmark_settings was called

        # Check if results file was created and contains the optimal setting
        self.assertTrue(self.test_results_file.exists())
        with open(self.test_results_file, 'r') as f:
            saved_results = json.load(f)
        # AutoTuner creates key like "gguf_test/model.gguf" by replacing '/' with '_'
        expected_key = "gguf_test_model.gguf"
        self.assertEqual(saved_results.get(expected_key), expected_optimal)

    def test_get_optimal_settings_with_saved_file_placeholder_rebenchmarks(self):
        initial_optimal = {'n_gpu_layers': 20, 'attn_implementation': 'native'}
        model_id = "test/model_v2.gguf"
        model_type = "gguf"
        expected_key = f"{model_type}_{model_id}".replace('/', '_')

        with open(self.test_results_file, 'w') as f:
            json.dump({expected_key: initial_optimal}, f)

        new_benchmark_result = {'n_gpu_layers': 33, 'attn_implementation': 'flash_attn'}
        with patch.object(self.tuner, 'benchmark_settings', return_value=new_benchmark_result) as mock_benchmark:
            with patch('sys.stdout', new_callable=StringIO): # Suppress prints
                optimal = self.tuner.get_optimal_settings(model_id, model_type)

        mock_benchmark.assert_called_once() # Placeholder always re-benchmarks
        self.assertEqual(optimal, new_benchmark_result)

        with open(self.test_results_file, 'r') as f: # Verify file was updated
            saved_results = json.load(f)
        self.assertEqual(saved_results.get(expected_key), new_benchmark_result)

    def test_get_optimal_settings_unsupported_type(self):
        with patch('sys.stdout', new_callable=StringIO): # Suppress prints
            optimal = self.tuner.get_optimal_settings("some_api_model", "api") # 'api' is unsupported by placeholder tuner
        self.assertIsNone(optimal)

class TestModelManagerWithAutoTuner(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager() # Instantiates its own AutoTuner and KVCacheManager

        self.test_data_root = Path("data_test_manager_autotune_integration")
        # Setup specific paths for manager's internal components for this test suite
        self.test_tuner_results_file = self.test_data_root / "tuning_results.json"
        self.test_kv_cache_dir = self.test_data_root / "kv_cache"

        if self.test_data_root.exists():
            shutil.rmtree(self.test_data_root)
        self.test_data_root.mkdir(parents=True, exist_ok=True)

        # Override the manager's tuner and kv_cache_mgr instances
        with patch('sys.stdout', new_callable=StringIO): # Suppress init prints
            self.manager.auto_tuner = AutoTuner(results_path=str(self.test_tuner_results_file))
            self.manager.kv_cache_mgr = KVCacheManager(cache_dir=str(self.test_kv_cache_dir))

        self.manager.unload()

    def tearDown(self):
        self.manager.unload()
        if self.test_data_root.exists():
            shutil.rmtree(self.test_data_root)

    @patch.object(AutoTuner, 'get_optimal_settings')
    def test_load_with_auto_tune_enabled_applies_settings(self, mock_get_optimal_settings):
        mock_optimal_params = {'n_gpu_layers': 99, 'attn_implementation': 'mock_tuned_for_gguf'}
        mock_get_optimal_settings.return_value = mock_optimal_params

        initial_runner_params = {'n_gpu_layers': 10, 'n_ctx': 2048, 'some_other_param': 'value'}

        with patch('llm_context_os.runners.llama_cpp_runner.LlamaCppRunner.__init__', return_value=None) as mock_runner_init:
            with patch('sys.stdout', new_callable=StringIO): # Suppress manager load prints
                self.manager.load(
                    model_type='gguf',
                    model_path_or_name='test/model_for_tuning.gguf',
                    auto_tune=True,
                    **initial_runner_params
                )

        mock_get_optimal_settings.assert_called_once_with('test/model_for_tuning.gguf', 'gguf')

        mock_runner_init.assert_called_once()
        # All args are passed as kwargs in ModelManager.load to the runner constructor after model_path_or_name
        called_kwargs = mock_runner_init.call_args.kwargs
        self.assertEqual(called_kwargs.get('model_path'), 'test/model_for_tuning.gguf') # model_path
        self.assertEqual(called_kwargs.get('n_gpu_layers'), mock_optimal_params['n_gpu_layers']) # Overridden by tuner
        self.assertEqual(called_kwargs.get('attn_implementation'), mock_optimal_params['attn_implementation']) # Added by tuner
        self.assertEqual(called_kwargs.get('n_ctx'), initial_runner_params['n_ctx']) # Original param preserved
        self.assertEqual(called_kwargs.get('some_other_param'), initial_runner_params['some_other_param'])


    @patch.object(AutoTuner, 'get_optimal_settings')
    def test_load_with_auto_tune_disabled(self, mock_get_optimal_settings):
        initial_params = {'n_gpu_layers': 5}
        with patch('llm_context_os.runners.llama_cpp_runner.LlamaCppRunner.__init__', return_value=None) as mock_runner_init:
            with patch('sys.stdout', new_callable=StringIO):
                self.manager.load(model_type='gguf', model_path_or_name='test/model.gguf', auto_tune=False, **initial_params)

        mock_get_optimal_settings.assert_not_called()

        # Check that LlamaCppRunner constructor was called with the original n_gpu_layers=5
        mock_runner_init.assert_called_once()
        called_kwargs_disabled_tune = mock_runner_init.call_args.kwargs
        self.assertEqual(called_kwargs_disabled_tune.get('n_gpu_layers'), 5)


    @patch.object(AutoTuner, 'get_optimal_settings')
    def test_load_with_auto_tune_unsupported_type(self, mock_get_optimal_settings):
        with patch('sys.stdout', new_callable=StringIO): # Suppress ModelManager/APIRunner prints
            self.manager.load(model_type='api', model_path_or_name='test_api_model', auto_tune=True, api_url="http://dummy")
        # AutoTuner's get_optimal_settings is not called by ModelManager if type is not in its supported list
        mock_get_optimal_settings.assert_not_called()
        self.assertIsNotNone(self.manager.current_runner) # APIRunner should still load

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
