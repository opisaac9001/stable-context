# yawl/tests/test_lora_integration.py
import unittest
from pathlib import Path
import shutil
import typing as t

from yawl.runners.manager import ModelManager
from yawl.runners import LlamaCppRunner, APIRunner # Import concrete runners for testing
from yawl.caching.kv_cache_manager import KVCacheManager # For controlling cache path in tests

class TestLoraIntegration(unittest.TestCase):

    def setUp(self):
        self.manager = ModelManager()
        # Ensure ModelManager uses a clean, test-specific KV cache directory
        # This is important as ModelManager instantiates KVCacheManager itself.
        self.test_cache_super_dir = Path("data_test_lora_integration") # Parent for all test caches
        self.test_cache_dir = self.test_cache_super_dir / "kv_cache_manager_lora_tests"

        if self.test_cache_super_dir.exists(): # Clean up if anything left from previous failed run
            shutil.rmtree(self.test_cache_super_dir)
        self.test_cache_dir.mkdir(parents=True, exist_ok=True)

        # Override the manager's KVCacheManager instance to use our test-specific path
        self.manager.kv_cache_mgr = KVCacheManager(cache_dir=str(self.test_cache_dir))

        # Ensure no runner is loaded initially for each test
        self.manager.unload()

    def tearDown(self):
        # Clean up the test-specific cache directory after all tests in the class
        if self.test_cache_super_dir.exists():
            shutil.rmtree(self.test_cache_super_dir)

    def test_lora_operations_on_compatible_runner(self):
        # Load a LoRA-compatible placeholder runner (LlamaCppRunner)
        print("\n[TestLoraIntegration] Loading LlamaCppRunner for LoRA compatibility tests...")
        self.manager.load(
            model_type='gguf',
            model_path_or_name='test_lora_compatible_model.gguf',
            n_gpu_layers=0 # Example kwarg for LlamaCppRunner
        )
        self.assertIsNotNone(self.manager.current_runner, "Failed to load LlamaCppRunner.")
        self.assertIsInstance(self.manager.current_runner, LlamaCppRunner, "Loaded runner is not LlamaCppRunner.")

        lora_id1 = "test_lora_adapter_1"
        lora_path1 = "/path/to/test_lora_adapter_1"

        # Test load_lora
        print(f"\n[TestLoraIntegration] Calling load_lora_on_current_runner for {lora_id1}...")
        load_status = self.manager.load_lora_on_current_runner(lora_id1, lora_path1, alpha=0.5)
        self.assertTrue(load_status, "load_lora_on_current_runner should return True for compatible runner.")

        # Test get_active_loras
        print("\n[TestLoraIntegration] Calling get_active_loras_on_current_runner...")
        active_loras = self.manager.get_active_loras_on_current_runner()
        self.assertEqual(active_loras, [lora_id1], "Active LoRAs list mismatch after load.")

        # Test merge_loras
        print(f"\n[TestLoraIntegration] Calling merge_loras_on_current_runner for {[lora_id1]}...")
        merge_status = self.manager.merge_loras_on_current_runner([lora_id1])
        self.assertTrue(merge_status, "merge_loras_on_current_runner should return True.")

        # Placeholder LlamaCppRunner removes LoRA from active list after merge
        active_loras_post_merge = self.manager.get_active_loras_on_current_runner()
        self.assertEqual(active_loras_post_merge, [], "Active LoRAs should be empty after merge (for this placeholder).")
        self.assertTrue(self.manager.current_runner.is_merged, "Runner's is_merged flag should be True after merge.")

        # Test unmerge_loras
        print("\n[TestLoraIntegration] Calling unmerge_loras_on_current_runner...")
        unmerge_status = self.manager.unmerge_loras_on_current_runner()
        self.assertTrue(unmerge_status, "unmerge_loras_on_current_runner should return True.")
        self.assertFalse(self.manager.current_runner.is_merged, "Runner's is_merged flag should be False after unmerge.")

        # Test unload_lora (LoRA was "consumed" by merge in placeholder, so it might not be found now)
        # Let's load another one to test unload more cleanly
        lora_id2 = "test_lora_adapter_2"
        self.manager.load_lora_on_current_runner(lora_id2, "/path/to/lora2")
        self.assertIn(lora_id2, self.manager.get_active_loras_on_current_runner())

        print(f"\n[TestLoraIntegration] Calling unload_lora_on_current_runner for {lora_id2}...")
        unload_status = self.manager.unload_lora_on_current_runner(lora_id2)
        self.assertTrue(unload_status, "unload_lora_on_current_runner should return True.")
        self.assertNotIn(lora_id2, self.manager.get_active_loras_on_current_runner())

    def test_lora_operations_on_incompatible_runner(self):
        # Load an "incompatible" runner (APIRunner placeholder returns False for LoRA ops)
        print("\n[TestLoraIntegration] Loading APIRunner for LoRA incompatibility tests...")
        self.manager.load(
            model_type='api',
            model_path_or_name='test_api_for_lora',
            api_url='http://dummyapi.example.com/v1'
        )
        self.assertIsNotNone(self.manager.current_runner, "Failed to load APIRunner.")
        self.assertIsInstance(self.manager.current_runner, APIRunner, "Loaded runner is not APIRunner.")

        print("\n[TestLoraIntegration] Calling load_lora_on_current_runner (on APIRunner)...")
        load_status = self.manager.load_lora_on_current_runner('api_lora', 'path/api_lora')
        self.assertFalse(load_status, "load_lora should return False for APIRunner placeholder.")

        print("\n[TestLoraIntegration] Calling get_active_loras_on_current_runner (on APIRunner)...")
        active_loras = self.manager.get_active_loras_on_current_runner()
        self.assertEqual(active_loras, [], "Active LoRAs should be empty for APIRunner placeholder.")

        print("\n[TestLoraIntegration] Calling merge_loras_on_current_runner (on APIRunner)...")
        merge_status = self.manager.merge_loras_on_current_runner(['api_lora'])
        self.assertFalse(merge_status, "merge_loras should return False for APIRunner placeholder.")

    def test_lora_operations_no_runner_loaded(self):
        print("\n[TestLoraIntegration] Testing LoRA operations with no runner loaded...")
        # setUp ensures no runner is loaded initially
        self.assertIsNone(self.manager.current_runner, "Pre-condition: No runner should be loaded.")

        load_status = self.manager.load_lora_on_current_runner('any_lora', 'any/path')
        self.assertFalse(load_status, "load_lora should return False if no runner is loaded.")

        active_loras = self.manager.get_active_loras_on_current_runner()
        self.assertEqual(active_loras, [], "get_active_loras should return empty list if no runner.")

        merge_status = self.manager.merge_loras_on_current_runner(['any_lora'])
        self.assertFalse(merge_status, "merge_loras should return False if no runner.")

        unmerge_status = self.manager.unmerge_loras_on_current_runner()
        self.assertFalse(unmerge_status, "unmerge_loras should return False if no runner.")

        unload_status = self.manager.unload_lora_on_current_runner('any_lora')
        self.assertFalse(unload_status, "unload_lora should return False if no runner.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
