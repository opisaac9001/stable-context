# llm_context_os/caching/kv_cache_manager.py
import os
import pickle
import hashlib
from pathlib import Path
import typing as t
import re
import shutil

class KVCacheManager:
    """
    Manages saving and loading of KV cache data for different models and prefixes.
    If cache_data is a file path, it copies the file. Otherwise, it pickles the data.
    For real tensor data, consider using a more appropriate serialization format
    like safetensors or framework-specific methods if not dealing with file paths.
    """

    def __init__(self, cache_dir: str = "data/kv_cache"):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # print(f"[KVCacheManager] Initialized. Cache directory: {self.cache_dir.resolve()}") # Less verbose
        except Exception as e:
            print(f"[KVCacheManager] Error creating cache directory '{self.cache_dir}': {e}")

    def _sanitize_identifier(self, identifier: str) -> str:
        identifier = identifier.replace('/', '_').replace('\\', '_')
        identifier = re.sub(r'[^a-zA-Z0-9_.-]+', '', identifier)
        return identifier[:100]

    def _get_prefix_hash(self, prefix_text: str) -> str:
        return hashlib.sha256(prefix_text.encode('utf-8')).hexdigest()

    def _get_pickle_filepath(self, model_identifier: str, prefix_hash: str) -> Path:
        sanitized_model_id = self._sanitize_identifier(model_identifier)
        return self.cache_dir / f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl"

    def _get_raw_file_cache_path(self, model_identifier: str, prefix_hash: str, original_suffix: str) -> Path:
        sanitized_model_id = self._sanitize_identifier(model_identifier)
        # Ensure suffix starts with a dot if not already
        suffix_to_use = original_suffix if original_suffix.startswith('.') else '.' + original_suffix
        return self.cache_dir / f"{sanitized_model_id}_{prefix_hash}{suffix_to_use}"

    def save_kv_cache(self, cache_data: t.Any, model_identifier: str, prefix_text: str) -> bool:
        if cache_data is None:
            print("[KVCacheManager] No cache data provided to save.")
            return False

        prefix_hash = self._get_prefix_hash(prefix_text)

        if isinstance(cache_data, (str, Path)) and Path(cache_data).is_file():
            source_file = Path(cache_data)
            target_filepath = self._get_raw_file_cache_path(model_identifier, prefix_hash, source_file.suffix)

            print(f"[KVCacheManager] Cache data is a file path. Copying from '{source_file}' to '{target_filepath}'")
            try:
                target_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure target dir exists
                shutil.copy2(source_file, target_filepath)
                print(f"[KVCacheManager] Successfully copied cache file to {target_filepath}")
                return True
            except Exception as e:
                print(f"[KVCacheManager] Error copying cache file from {source_file} to {target_filepath}: {e}")
                return False
        else:
            pickle_filepath = self._get_pickle_filepath(model_identifier, prefix_hash)
            print(f"[KVCacheManager] Saving KV cache (pickling data) for model '{model_identifier}' to {pickle_filepath}")
            # TODO: For complex objects like Transformers past_key_values (tuples of tensors),
            # consider using torch.save/torch.load or safetensors for better efficiency,
            # portability, and to avoid potential issues with pickling tensors directly,
            # especially across different PyTorch versions or environments.
            # For now, using pickle as a general fallback.
            try:
                pickle_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure target dir exists
                with open(pickle_filepath, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"[KVCacheManager] KV cache (pickled data) saved successfully.")
                return True
            except Exception as e: # Broader exception for pickling/IO
                print(f"[KVCacheManager] Error saving/pickling KV cache to '{pickle_filepath}': {e}")
            return False

    def load_kv_cache(self, model_identifier: str, prefix_text: str,
                        raw_file_type_suffix: t.Optional[str] = None) -> t.Any:
        prefix_hash = self._get_prefix_hash(prefix_text)

        if raw_file_type_suffix:
            # Ensure suffix starts with a dot if not already
            suffix_to_check = raw_file_type_suffix if raw_file_type_suffix.startswith('.') else '.' + raw_file_type_suffix

            raw_filepath = self._get_raw_file_cache_path(model_identifier, prefix_hash, suffix_to_check)
            if raw_filepath.exists() and raw_filepath.is_file():
                print(f"[KVCacheManager] Found raw cache file copy at {raw_filepath}")
                return str(raw_filepath.resolve()) # Return the absolute path to the copied file

        # If not expecting a specific raw file type, or if raw file not found, try loading standard pickled data
        pickle_filepath = self._get_pickle_filepath(model_identifier, prefix_hash)
        if pickle_filepath.exists() and pickle_filepath.is_file():
            print(f"[KVCacheManager] Loading pickled KV cache from {pickle_filepath}")
            # TODO: If this .pkl file contains complex objects like Transformers past_key_values, consider if torch.load or safetensors would be more appropriate if issues arise with pickle.
            try:
                with open(pickle_filepath, 'rb') as f:
                    loaded_data = pickle.load(f)
                print(f"[KVCacheManager] KV cache (pickled data) loaded successfully.")
                return loaded_data
            except Exception as e:
                print(f"[KVCacheManager] Error loading/unpickling from '{pickle_filepath}': {e}")
                return None

        # print(f"[KVCacheManager] No cache found for model '{model_identifier}', prefix '{prefix_text[:20]}...' "
        #       f"(raw_suffix_tried: {raw_file_type_suffix}, pickle_path_tried: {pickle_filepath})")
        return None

if __name__ == '__main__':
    print("--- KVCacheManager Demo ---")
    test_cache_dir_main = Path("data/kv_cache_test_main_kvc")
    if test_cache_dir_main.exists():
        shutil.rmtree(test_cache_dir_main)

    # KVCacheManager's __init__ creates the directory
    cache_manager = KVCacheManager(cache_dir=str(test_cache_dir_main))

    model_id_test = "test_model/v1.0_gguf" # Contains slash, tests sanitization
    prefix1_text = "Prefix for pickling data."
    prefix2_text = "Prefix for file copy."

    mock_object_cache_data = {"layer_0": {"k": [0.1], "v": [0.2]}, "metadata":"pickled_obj"}

    temp_source_files_dir = test_cache_dir_main / "temp_source_files" # Create temp source inside test_cache_dir_main for easy cleanup
    temp_source_files_dir.mkdir(parents=True, exist_ok=True)
    dummy_session_file_path = temp_source_files_dir / "dummy_session.kst"
    dummy_session_content = {"session_key": "session_value_kst", "version": 1.0}
    with open(dummy_session_file_path, 'wb') as f:
        pickle.dump(dummy_session_content, f)

    print(f"\n--- Saving cache (pickling object) for prefix 1 ---")
    save_success_p1 = cache_manager.save_kv_cache(mock_object_cache_data, model_id_test, prefix1_text)
    print(f"Save successful for p1: {save_success_p1}")
    assert save_success_p1

    print(f"\n--- Saving cache (copying file) for prefix 2 ---")
    save_success_p2 = cache_manager.save_kv_cache(str(dummy_session_file_path), model_id_test, prefix2_text)
    print(f"Save successful for p2: {save_success_p2}")
    assert save_success_p2

    print("\n--- Loading cache for prefix 1 (expecting unpickled object) ---")
    loaded_data_p1 = cache_manager.load_kv_cache(model_id_test, prefix1_text, raw_file_type_suffix=None)
    assert loaded_data_p1 == mock_object_cache_data, f"Pickled data mismatch. Got: {loaded_data_p1}"
    print(f"  Pickled data loaded correctly: {loaded_data_p1}")

    print("\n--- Loading cache for prefix 2 (expecting path to copied .kst file) ---")
    loaded_data_p2_path_str = cache_manager.load_kv_cache(model_id_test, prefix2_text, raw_file_type_suffix=".kst")
    assert isinstance(loaded_data_p2_path_str, str), f"Expected path string, got {type(loaded_data_p2_path_str)}"
    loaded_data_p2_path = Path(loaded_data_p2_path_str)
    assert loaded_data_p2_path.exists(), f"Copied file path does not exist: {loaded_data_p2_path}"

    # Construct expected path for verification
    expected_sanitized_id = cache_manager._sanitize_identifier(model_id_test)
    expected_prefix_hash = cache_manager._get_prefix_hash(prefix2_text)
    expected_copied_file_path = cache_manager.cache_dir / f"{expected_sanitized_id}_{expected_prefix_hash}.kst"
    assert loaded_data_p2_path.resolve() == expected_copied_file_path.resolve(), \
        f"Loaded path {loaded_data_p2_path.resolve()} != expected {expected_copied_file_path.resolve()}"

    with open(loaded_data_p2_path, 'rb') as f:
        copied_content = pickle.load(f)
    assert copied_content == dummy_session_content, "Content of copied .kst file is incorrect."
    print(f"  File copy path loaded correctly: {loaded_data_p2_path}, content verified.")

    print("\n--- Test loading non-existent ---")
    assert cache_manager.load_kv_cache("m", "nonexist_raw", raw_file_type_suffix=".kst") is None
    assert cache_manager.load_kv_cache("m", "nonexist_pkl") is None # Checks for .pkl by default
    print("  Non-existent loads correctly returned None.")

    # Test load_priority (conceptual, as load_kv_cache now takes a hint)
    print("\n--- Test loading with explicit suffix preference ---")
    # Create both .kst and .pkl for same model/prefix
    prefix_both = "prefix_for_both_types"
    cache_manager.save_kv_cache(str(dummy_session_file_path), model_id_test, prefix_both) # Saves as .kst
    cache_manager.save_kv_cache(mock_object_cache_data, model_id_test, prefix_both)    # Saves as .kvcache.pkl

    # Load requesting .kst
    loaded_kst_path = cache_manager.load_kv_cache(model_id_test, prefix_both, raw_file_type_suffix=".kst")
    assert isinstance(loaded_kst_path, str) and ".kst" in loaded_kst_path
    print(f"  Loaded .kst file path: {loaded_kst_path}")

    # Load requesting pickled (by not specifying raw_file_type_suffix or by .pkl if that was the convention)
    loaded_pickle_obj = cache_manager.load_kv_cache(model_id_test, prefix_both) # Default tries .pkl
    assert loaded_pickle_obj == mock_object_cache_data
    print(f"  Loaded pickled object: {loaded_pickle_obj}")


    print("\nKVCacheManager demo complete.")
    if test_cache_dir_main.exists():
        shutil.rmtree(test_cache_dir_main)
    print(f"Cleaned up: {test_cache_dir_main}")
