# llm_context_os/tests/test_kv_cache_manager.py
import unittest
import shutil
from pathlib import Path
import typing as t
import pickle # For dummy file content in new test

from llm_context_os.caching.kv_cache_manager import KVCacheManager

class TestKVCacheManager(unittest.TestCase):
    def setUp(self):
        self.test_data_dir_root = Path("data_test_kv_cache_manager_module") # Main test dir for this class
        self.test_cache_dir = self.test_data_dir_root / "kv_cache_storage" # Cache storage for KVCacheManager
        self.source_files_temp_dir = self.test_data_dir_root / "temp_source_files" # For creating dummy source files

        # Clean up and create fresh directories for each test method
        if self.test_data_dir_root.exists():
            shutil.rmtree(self.test_data_dir_root)
        self.test_cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_files_temp_dir.mkdir(parents=True, exist_ok=True)

        self.kv_manager = KVCacheManager(cache_dir=str(self.test_cache_dir))

        self.mock_model_id = "test_model/v1.0_gguf"
        self.mock_prefix_text = "This is a test prefix that will be hashed."
        self.mock_pickle_data = [
            {'key_cache_layer_0': [[0.1, 0.2, 0.3]], 'value_cache_layer_0': [[0.4, 0.5, 0.6]]},
            {'key_cache_layer_1': [[0.7, 0.8]], 'value_cache_layer_1': [[0.9, 1.0]]}
        ]

        # Create a dummy source file for file copy tests
        self.dummy_source_file_path = self.source_files_temp_dir / "source_session.kst"
        self.dummy_source_file_content = {"session_data": "this is a dummy KST file content", "version": 1}
        with open(self.dummy_source_file_path, 'wb') as f:
            pickle.dump(self.dummy_source_file_content, f) # Using pickle for dummy content

    def tearDown(self):
        if self.test_data_dir_root.exists():
            shutil.rmtree(self.test_data_dir_root)

    def test_instantiation_creates_directory(self):
        temp_dir_for_inst_test = self.test_data_dir_root / "temp_instantiation_test_cache"
        if temp_dir_for_inst_test.exists(): # Should not exist due to root cleanup in setUp
            shutil.rmtree(temp_dir_for_inst_test)
        self.assertFalse(temp_dir_for_inst_test.exists())
        KVCacheManager(cache_dir=str(temp_dir_for_inst_test)) # This should create it
        self.assertTrue(temp_dir_for_inst_test.exists())

    def test_save_and_load_pickle_data(self): # Renamed
        # Test saving pickled data
        save_success = self.kv_manager.save_kv_cache(
            self.mock_pickle_data, self.mock_model_id, self.mock_prefix_text
        )
        self.assertTrue(save_success, "Failed to save KV cache (pickled data).")

        prefix_hash = self.kv_manager._get_prefix_hash(self.mock_prefix_text)
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        expected_filename = f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl" # Default pickle extension
        expected_filepath = self.test_cache_dir / expected_filename
        self.assertTrue(expected_filepath.exists(), f"Pickled cache file was not created at {expected_filepath}.")

        # Test loading pickled data - raw_file_type_suffix should be None or omitted for pickled objects
        loaded_data = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text, raw_file_type_suffix=None)
        self.assertIsNotNone(loaded_data, "Failed to load KV cache (pickled data).")
        self.assertEqual(loaded_data, self.mock_pickle_data, "Loaded pickled data does not match saved data.")

    def test_save_and_load_file_copy_data(self):
        # Test saving a file (by copying)
        save_success = self.kv_manager.save_kv_cache(
            str(self.dummy_source_file_path), self.mock_model_id, self.mock_prefix_text
        )
        self.assertTrue(save_success, "Failed to save KV cache (file copy).")

        prefix_hash = self.kv_manager._get_prefix_hash(self.mock_prefix_text)
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        # Expected filename in cache dir will have the original suffix
        expected_cached_filename = f"{sanitized_model_id}_{prefix_hash}{self.dummy_source_file_path.suffix}"
        expected_cached_filepath = self.test_cache_dir / expected_cached_filename
        self.assertTrue(expected_cached_filepath.exists(), f"Copied cache file was not created at {expected_cached_filepath}.")

        # Verify content of the copied file
        with open(expected_cached_filepath, 'rb') as f:
            copied_content = pickle.load(f) # We know original dummy was pickled
        self.assertEqual(copied_content, self.dummy_source_file_content, "Content of copied cache file differs from source.")

        # Test loading the path to the copied file
        loaded_path_str = self.kv_manager.load_kv_cache(
            self.mock_model_id, self.mock_prefix_text, raw_file_type_suffix=self.dummy_source_file_path.suffix
        )
        self.assertIsNotNone(loaded_path_str, "Failed to load KV cache (file copy path).")
        self.assertIsInstance(loaded_path_str, str, "load_kv_cache should return a string path for file copies.")
        self.assertEqual(Path(loaded_path_str), expected_cached_filepath.resolve(), "Loaded path does not match expected cached file path.")

        # Verify content again from loaded_path_str
        with open(loaded_path_str, 'rb') as f:
            content_from_loaded_path = pickle.load(f)
        self.assertEqual(content_from_loaded_path, self.dummy_source_file_content)


    def test_load_priority_file_copy_over_pickle(self):
        # 1. Save data via pickling for a specific model/prefix
        self.kv_manager.save_kv_cache(self.mock_pickle_data, self.mock_model_id, self.mock_prefix_text)

        # 2. Save a file copy for the *same* model/prefix (KVCacheManager should allow this, creating a different filename due to suffix)
        # This assumes KVCacheManager differentiates by suffix, which it does.
        # The save_kv_cache for file copy will create modelid_hash.kst
        # The save_kv_cache for pickle will create modelid_hash.kvcache.pkl
        self.kv_manager.save_kv_cache(str(self.dummy_source_file_path), self.mock_model_id, self.mock_prefix_text)

        # 3. Call load_kv_cache expecting the file copy path (.kst)
        loaded_path = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text, raw_file_type_suffix=".kst")
        self.assertIsInstance(loaded_path, str, "Should have loaded the path to the .kst file.")
        with open(loaded_path, 'rb') as f:
            copied_content = pickle.load(f)
        self.assertEqual(copied_content, self.dummy_source_file_content)

        # 4. Call load_kv_cache expecting the pickled data (by passing raw_file_type_suffix=None or omitting it)
        loaded_object = self.kv_manager.load_kv_cache(self.mock_model_id, self.mock_prefix_text)
        self.assertEqual(loaded_object, self.mock_pickle_data, "Should have loaded the pickled object.")


    def test_load_non_existent_cache(self):
        # Test with raw_file_type_suffix=None (or omitted) for a non-existent pickled object
        loaded_data = self.kv_manager.load_kv_cache(self.mock_model_id, "a_completely_non_existent_prefix_text")
        self.assertIsNone(loaded_data, "Should return None for non-existent cache (pickled object).")

        # Test with a specific raw_file_type_suffix for a non-existent raw file
        loaded_path = self.kv_manager.load_kv_cache(self.mock_model_id, "another_non_existent_prefix", raw_file_type_suffix=".kst")
        self.assertIsNone(loaded_path, "Should return None for non-existent .kst cache.")


    def test_save_none_data(self):
        save_success = self.kv_manager.save_kv_cache(None, self.mock_model_id, "prefix_for_none_data_test")
        self.assertFalse(save_success, "Should return False when trying to save None data.")
        # Verify no file was created (neither .pkl nor any other type)
        prefix_hash = self.kv_manager._get_prefix_hash("prefix_for_none_data_test")
        sanitized_model_id = self.kv_manager._sanitize_identifier(self.mock_model_id)
        expected_pkl_filepath = self.test_cache_dir / f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl"
        self.assertFalse(expected_pkl_filepath.exists(), "Pickle cache file should not have been created for None data.")
        # Check for a common copied file extension too just in case
        expected_kst_filepath = self.test_cache_dir / f"{sanitized_model_id}_{prefix_hash}.kst"
        self.assertFalse(expected_kst_filepath.exists(), "Copied cache file should not have been created for None data.")


    def test_sanitization_in_filename(self):
        problematic_model_id = "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
        # Test with pickling
        self.kv_manager.save_kv_cache(self.mock_pickle_data, problematic_model_id, "prefix_pickle")
        prefix_hash_pickle = self.kv_manager._get_prefix_hash("prefix_pickle")
        sanitized_id = self.kv_manager._sanitize_identifier(problematic_model_id)
        expected_pickle_path = self.test_cache_dir / f"{sanitized_id}_{prefix_hash_pickle}.kvcache.pkl"
        self.assertTrue(expected_pickle_path.exists())

        # Test with file copy
        self.kv_manager.save_kv_cache(str(self.dummy_source_file_path), problematic_model_id, "prefix_filecopy")
        prefix_hash_filecopy = self.kv_manager._get_prefix_hash("prefix_filecopy")
        expected_filecopy_path = self.test_cache_dir / f"{sanitized_id}_{prefix_hash_filecopy}{self.dummy_source_file_path.suffix}"
        self.assertTrue(expected_filecopy_path.exists())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
