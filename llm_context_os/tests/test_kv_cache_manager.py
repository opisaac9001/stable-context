# llm_context_os/tests/test_kv_cache_manager.py
import unittest
import shutil
from pathlib import Path
import typing as t # Ensure t is imported for t.Any if used more broadly

from llm_context_os.caching.kv_cache_manager import KVCacheManager

class TestKVCacheManager(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir_str = "data/kv_cache_test_kv_manager"
        self.test_cache_dir = Path(self.test_cache_dir_str)
        # Clean up before each test
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
        self.test_cache_dir.mkdir(parents=True, exist_ok=True)

        self.kv_manager = KVCacheManager(cache_dir=self.test_cache_dir_str)
        self.mock_model_id = "test_model/v1.0_gguf" # More complex ID
        self.mock_prefix_text = "This is a test prefix that will be hashed."
        self.mock_cache_data = [
            {'key_cache_layer_0': [[0.1, 0.2, 0.3]], 'value_cache_layer_0': [[0.4, 0.5, 0.6]]},
            {'key_cache_layer_1': [[0.7, 0.8]], 'value_cache_layer_1': [[0.9, 1.0]]}
        ]

    def tearDown(self):
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)

    def test_instantiation_creates_directory(self):
        """Test if the cache directory is created on instantiation."""
        temp_dir = Path("data/temp_kv_cache_instantiation_test")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        self.assertFalse(temp_dir.exists())
        KVCacheManager(cache_dir=str(temp_dir))
        self.assertTrue(temp_dir.exists())
        shutil.rmtree(temp_dir)

    def test_save_and_load_kv_cache(self):
        # Test saving
        save_success = self.kv_manager.save_kv_cache(
            self.mock_cache_data, self.mock_model_id, self.mock_prefix_text
        )
        self.assertTrue(save_success, "Failed to save KV cache.")

        # Verify file was created
        prefix_hash = self.kv_manager._get_prefix_hash(self.mock_prefix_text)
        # Use the manager's internal sanitization for checking filename
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        expected_filename = f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl"
        expected_filepath = self.test_cache_dir / expected_filename
        self.assertTrue(expected_filepath.exists(), f"Cache file was not created at {expected_filepath}.")

        # Test loading
        loaded_data = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text)
        self.assertIsNotNone(loaded_data, "Failed to load KV cache.")
        self.assertEqual(loaded_data, self.mock_cache_data, "Loaded data does not match saved data.")

    def test_load_non_existent_cache(self):
        loaded_data = self.kv_manager.load_kv_cache(self.mock_model_id, "a_completely_non_existent_prefix_text")
        self.assertIsNone(loaded_data, "Should return None for non-existent cache.")

    def test_save_none_data(self):
        save_success = self.kv_manager.save_kv_cache(None, self.mock_model_id, "prefix_for_none_data_test")
        self.assertFalse(save_success, "Should return False when trying to save None data.")

        # Verify file was not created
        prefix_hash = self.kv_manager._get_prefix_hash("prefix_for_none_data_test")
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        expected_filename = f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl"
        expected_filepath = self.test_cache_dir / expected_filename
        self.assertFalse(expected_filepath.exists(), "Cache file should not have been created for None data.")

    def test_sanitization_in_filename(self):
        problematic_model_id = "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
        save_success = self.kv_manager.save_kv_cache(
            self.mock_cache_data, problematic_model_id, self.mock_prefix_text
        )
        self.assertTrue(save_success)

        prefix_hash = self.kv_manager._get_prefix_hash(self.mock_prefix_text)
        # Check what the sanitized name should be based on the manager's logic
        sanitized_id = self.kv_manager._sanitize_identifier(problematic_model_id)
        expected_filepath = self.test_cache_dir / f"{sanitized_id}_{prefix_hash}.kvcache.pkl"
        self.assertTrue(expected_filepath.exists(), f"File not found at expected sanitized path: {expected_filepath}")


if __name__ == '__main__':
    unittest.main()
