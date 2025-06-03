# llm_context_os/tests/test_llava_cpp_runner.py
import unittest
from llm_context_os.runners.llava_cpp_runner import LlavaCppRunner
import typing as t
from unittest.mock import patch # For suppressing prints in setUp
from io import StringIO # For suppressing prints in setUp

class TestLlavaCppRunner(unittest.TestCase):
    def setUp(self):
        self.model_path = "path/to/llava_model.gguf"
        self.mmproj_path = "path/to/mmproj.gguf"
        # Suppress init prints to keep test output clean
        with patch('sys.stdout', new_callable=StringIO):
            self.runner = LlavaCppRunner(
                model_path=self.model_path,
                mmproj_path=self.mmproj_path,
                n_gpu_layers=0,
                verbose=False # Explicitly set for less output if runner used it
            )

    def test_instantiation(self):
        self.assertIsNotNone(self.runner)
        self.assertEqual(self.runner.model_path, self.model_path)
        self.assertEqual(self.runner.mmproj_path, self.mmproj_path)

    def test_generate_placeholder_no_image(self):
        prompt = "Describe the world."
        # Expected from LlavaCppRunner placeholder: f"[LLaVA-CPP Response to: {prompt[:30]}...{img_processed_text}More text...]"
        # img_processed_text is "" if no image_paths
        expected_response = f"[LLaVA-CPP Response to: {prompt[:30]}...More text...]"
        with patch('sys.stdout', new_callable=StringIO): # Suppress generate prints
            response = self.runner.generate(prompt)
        self.assertEqual(response, expected_response)
        # Check that the "Processed image data" part is NOT in the response
        self.assertNotIn("Processed image data", response)


    def test_generate_placeholder_with_image(self):
        prompt = "What is this?"
        image_paths = ["dummy/image.jpg"]
        # Expected: f"[LLaVA-CPP Response to: {prompt[:30]}... (Processed 1 image(s) like '{image_paths[0]}'). More text...]"
        expected_response = f"[LLaVA-CPP Response to: {prompt[:30]}... (Processed 1 image(s) like '{image_paths[0]}'). More text...]"
        with patch('sys.stdout', new_callable=StringIO):
            response = self.runner.generate(prompt, image_paths=image_paths)
        self.assertEqual(response, expected_response)
        self.assertIn("Processed 1 image(s) like 'dummy/image.jpg'", response)

    def test_stream_placeholder_with_image(self):
        prompt = "Explain this image."
        image_paths = ["dummy/image.png"]
        # Expected chunks from LlavaCppRunner placeholder:
        # yield "[LLaVA-CPP Stream Chunk 1] "
        # if image_paths: yield f"(Streaming with {len(image_paths)} image(s) like '{image_paths[0]}') "
        # yield f"Data for '{prompt[:20]}...' "
        # yield "[LLaVA-CPP Stream End]"
        expected_chunks = [
            "[LLaVA-CPP Stream Chunk 1] ",
            f"(Streaming with {len(image_paths)} image(s) like '{image_paths[0]}') ",
            f"Data for '{prompt[:20]}...' ",
            "[LLaVA-CPP Stream End]"
        ]
        with patch('sys.stdout', new_callable=StringIO):
            chunks = list(self.runner.stream(prompt, image_paths=image_paths))
        self.assertEqual(chunks, expected_chunks)

    def test_kv_cache_and_lora_placeholders(self):
        # Simple calls to ensure methods exist and run without error (placeholders)
        # and that mock state is updated as expected by placeholder logic.
        with patch('sys.stdout', new_callable=StringIO): # Suppress prints for cleaner test
            self.runner.preload_kv("prefix")
            self.assertIsNotNone(self.runner.mock_kv_cache_data)
            self.assertEqual(self.runner.mock_kv_cache_data['type'], 'text_prefix_llava')

            cache = self.runner.export_kv_cache()
            self.assertIsNotNone(cache)

            new_cache_data = {"imported": "data"}
            self.runner.import_kv_cache(new_cache_data)
            self.assertEqual(self.runner.mock_kv_cache_data, new_cache_data)

            self.assertTrue(self.runner.load_lora_adapter("test_lora", "path/lora"))
            self.assertIn("test_lora", self.runner.get_active_lora_adapters())
            self.assertTrue(self.runner.unload_lora_adapter("test_lora"))
            self.assertNotIn("test_lora", self.runner.get_active_lora_adapters())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
