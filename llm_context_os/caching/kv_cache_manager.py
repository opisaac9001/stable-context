# llm_context_os/caching/kv_cache_manager.py
import os
import pickle
import hashlib
from pathlib import Path
import typing as t
import re

class KVCacheManager:
    """
    Manages saving and loading of KV cache data for different models and prefixes.
    Uses pickle for serialization in this placeholder version.
    For real tensor data, consider using a more appropriate serialization format
    like safetensors or framework-specific methods.
    """

    def __init__(self, cache_dir: str = "data/kv_cache"):
        """
        Initializes the KVCacheManager.

        Args:
            cache_dir (str): The directory where KV cache files will be stored.
                             Defaults to "data/kv_cache".
        """
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[KVCacheManager] Initialized. Cache directory: {self.cache_dir.resolve()}")
        except Exception as e:
            print(f"[KVCacheManager] Error creating cache directory '{self.cache_dir}': {e}")
            # Depending on requirements, might want to raise this or handle more gracefully.

    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitizes an identifier to be filesystem-friendly."""
        # Replace slashes and other common problematic characters
        identifier = identifier.replace('/', '_').replace('\\', '_')
        # Remove any characters not suitable for filenames
        identifier = re.sub(r'[^a-zA-Z0-9_.-]+', '', identifier)
        # Ensure it's not excessively long (optional)
        return identifier[:100] # Max length of 100 chars for sanitized part

    def _get_prefix_hash(self, prefix_text: str) -> str:
        """
        Generates a SHA256 hash for the given prefix text.
        """
        return hashlib.sha256(prefix_text.encode('utf-8')).hexdigest()

    def _get_cache_filepath(self, model_identifier: str, prefix_hash: str) -> Path:
        """
        Constructs the filepath for a KV cache file.

        Args:
            model_identifier (str): A unique identifier for the model
                                    (e.g., model path or name).
            prefix_hash (str): The hash of the prefix text.

        Returns:
            Path: The full path to the cache file.
        """
        sanitized_model_id = self._sanitize_identifier(model_identifier)
        filename = f"{sanitized_model_id}_{prefix_hash}.kvcache.pkl"
        return self.cache_dir / filename

    def save_kv_cache(self, cache_data: t.Any, model_identifier: str, prefix_text: str) -> bool:
        """
        Saves KV cache data to a file.

        Args:
            cache_data (t.Any): The KV cache data to save.
            model_identifier (str): Identifier for the model.
            prefix_text (str): The text prefix associated with this KV cache.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if cache_data is None:
            print("[KVCacheManager] No cache data provided to save.")
            return False

        prefix_hash = self._get_prefix_hash(prefix_text)
        filepath = self._get_cache_filepath(model_identifier, prefix_hash)

        print(f"[KVCacheManager] Saving KV cache for model '{model_identifier}' and prefix hash '{prefix_hash[:8]}...' to {filepath}")

        # TODO: For LlamaCppRunner, 'cache_data' might be a filepath to a .kst session file.
        # In that case, instead of pickling the path string, this method should copy the actual file
        # from cache_data (the path) to self._get_cache_filepath(...).
        # For now, it pickles the path string if cache_data is a string.
        if isinstance(cache_data, str) and Path(cache_data).exists():
            print(f"[KVCacheManager] Warning: Pickling a filepath string '{cache_data}' as cache data. Consider copying file content for portability.")

        print("  (Note: Using pickle for serialization. For real tensor data from runners like LlamaCppRunner (session files) or HF/EXL2 (tensors), a direct file copy or safetensors/torch.save would be more appropriate than pickling the data/path itself.)")

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[KVCacheManager] KV cache saved successfully.")
            return True
        except pickle.PicklingError as e:
            print(f"[KVCacheManager] Error pickling KV cache data: {e}")
        except IOError as e:
            print(f"[KVCacheManager] Error writing KV cache file '{filepath}': {e}")
        except Exception as e:
            print(f"[KVCacheManager] Unexpected error saving KV cache: {e}")
        return False

    def load_kv_cache(self, model_identifier: str, prefix_text: str) -> t.Any:
        """
        Loads KV cache data from a file.

        Args:
            model_identifier (str): Identifier for the model.
            prefix_text (str): The text prefix associated with this KV cache.

        Returns:
            t.Any: The loaded KV cache data, or None if not found or an error occurs.
        """
        prefix_hash = self._get_prefix_hash(prefix_text)
        filepath = self._get_cache_filepath(model_identifier, prefix_hash)

        if filepath.exists() and filepath.is_file():
            print(f"[KVCacheManager] Loading KV cache for model '{model_identifier}' and prefix hash '{prefix_hash[:8]}...' from {filepath}")
            # TODO: If the saved cache for LlamaCppRunner was a file copy (as per future save_kv_cache logic),
            # this should return the path to that copied file, or handle it accordingly.
            # Currently, if a path string was pickled (due to current save_kv_cache logic), it unpickles that path string.
            print("  (Note: Using pickle for deserialization. Ensure data was saved compatibly.)")
            try:
                with open(filepath, 'rb') as f:
                    cache_data = pickle.load(f)
                print(f"[KVCacheManager] KV cache loaded successfully.")
                return cache_data
            except pickle.UnpicklingError as e:
                print(f"[KVCacheManager] Error unpickling KV cache data from '{filepath}': {e}")
            except IOError as e:
                print(f"[KVCacheManager] Error reading KV cache file '{filepath}': {e}")
            except Exception as e:
                print(f"[KVCacheManager] Unexpected error loading KV cache: {e}")
            return None
        else:
            print(f"[KVCacheManager] No KV cache file found at {filepath} for model '{model_identifier}' and prefix.")
            return None

if __name__ == '__main__':
    print("--- KVCacheManager Demo ---")
    # Ensure the cache directory is created for the demo if it doesn't exist
    # The constructor handles this, but good to be aware for manual tests.
    # Path("data/kv_cache").mkdir(parents=True, exist_ok=True)

    cache_manager = KVCacheManager(cache_dir="data/kv_cache_test") # Use a test-specific dir

    model_id_test = "test_model/v1.0_gguf"
    prefix1_text = "Once upon a time in a land far, far away,"
    prefix2_text = "The quick brown fox jumps over the lazy dog."

    # Mock KV cache data (structure depends on the runner)
    mock_cache_data_p1 = {
        "layer_0": {"k": [0.1, 0.2], "v": [0.3, 0.4]},
        "layer_1": {"k": [0.5, 0.6], "v": [0.7, 0.8]},
        "metadata": {"prompt_len": 10, "model_type": "gguf_mock"}
    }
    mock_cache_data_p2 = {
        "past_key_values": (([1.1, 1.2], [1.3, 1.4]), ([1.5, 1.6], [1.7, 1.8])),
        "metadata": {"prompt_len": 9, "model_type": "hf_mock"}
    }

    print(f"\n--- Saving cache for prefix 1 ('{prefix1_text[:20]}...') ---")
    save_success_p1 = cache_manager.save_kv_cache(mock_cache_data_p1, model_id_test, prefix1_text)
    print(f"Save successful for p1: {save_success_p1}")
    assert save_success_p1

    print(f"\n--- Saving cache for prefix 2 ('{prefix2_text[:20]}...') ---")
    save_success_p2 = cache_manager.save_kv_cache(mock_cache_data_p2, model_id_test, prefix2_text)
    print(f"Save successful for p2: {save_success_p2}")
    assert save_success_p2

    print("\n--- Loading cache for prefix 1 ---")
    loaded_data_p1 = cache_manager.load_kv_cache(model_id_test, prefix1_text)
    if loaded_data_p1:
        print(f"Loaded data for p1 matches original: {loaded_data_p1 == mock_cache_data_p1}")
        assert loaded_data_p1 == mock_cache_data_p1
        print(f"  Content (metadata): {loaded_data_p1.get('metadata')}")
    else:
        print("  Failed to load data for p1.")
        assert False, "Loading p1 failed"

    print("\n--- Loading cache for prefix 2 ---")
    loaded_data_p2 = cache_manager.load_kv_cache(model_id_test, prefix2_text)
    if loaded_data_p2:
        print(f"Loaded data for p2 matches original: {loaded_data_p2 == mock_cache_data_p2}")
        assert loaded_data_p2 == mock_cache_data_p2
        print(f"  Content (metadata): {loaded_data_p2.get('metadata')}")

    else:
        print("  Failed to load data for p2.")
        assert False, "Loading p2 failed"

    print("\n--- Attempting to load non-existent cache (different prefix) ---")
    non_existent_prefix = "This prefix does not have a cache."
    loaded_data_non_existent = cache_manager.load_kv_cache(model_id_test, non_existent_prefix)
    if loaded_data_non_existent is None:
        print("  Correctly returned None for non-existent cache.")
        assert True
    else:
        print(f"  Incorrectly found data for non-existent cache: {loaded_data_non_existent}")
        assert False, "Should not have found data"

    print("\n--- Attempting to load for a different model ID ---")
    loaded_data_other_model = cache_manager.load_kv_cache("other_model/v1", prefix1_text)
    if loaded_data_other_model is None:
        print("  Correctly returned None for different model ID.")
        assert True
    else:
        print(f"  Incorrectly found data for other model ID: {loaded_data_other_model}")
        assert False, "Should not have found data for other model"

    print("\n--- Testing saving None (should not save) ---")
    save_none_success = cache_manager.save_kv_cache(None, "none_model", "none_prefix")
    print(f"Save 'None' successful: {save_none_success}")
    assert not save_none_success
    loaded_none = cache_manager.load_kv_cache("none_model", "none_prefix")
    assert loaded_none is None
    print("  Correctly did not save or load None.")


    print("\nKVCacheManager demo complete.")
    # Consider cleaning up the "data/kv_cache_test" directory after demo if needed
    # import shutil
    # shutil.rmtree(cache_manager.cache_dir)
    # print(f"Cleaned up test cache directory: {cache_manager.cache_dir}")
