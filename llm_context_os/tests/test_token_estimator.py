# llm_context_os/tests/test_token_estimator.py
import unittest
import typing as t

from llm_context_os.context.token_estimator import (
    BaseTokenEstimator,
    CharTokenEstimator,
    HFTokenEstimator,
    LlamaCppTokenEstimator,
    EXL2TokenEstimator,
    count_tokens
)

# --- Conditional imports for HFTokenizer ---
HAVE_TRANSFORMERS = False
HF_GPT2_TOKENIZER_INSTANCE: t.Optional[HFTokenEstimator] = None
_HF_MODEL_NAME_FOR_TEST = "gpt2" # Using a common model

try:
    from transformers import AutoTokenizer # To check if lib is there AND for direct use in tests
    _hf_auto_tokenizer_gpt2_for_test_comparison = AutoTokenizer.from_pretrained(_HF_MODEL_NAME_FOR_TEST)
    HAVE_TRANSFORMERS = True
    try:
        HF_GPT2_TOKENIZER_INSTANCE = HFTokenEstimator(model_name=_HF_MODEL_NAME_FOR_TEST)
        print(f"Successfully loaded HFTokenEstimator with model '{_HF_MODEL_NAME_FOR_TEST}' for tests.")
    except Exception as e:
        print(f"Note: Could not load HFTokenEstimator model '{_HF_MODEL_NAME_FOR_TEST}' for tests: {e}")
        HF_GPT2_TOKENIZER_INSTANCE = None
except ImportError:
    print("Note: 'transformers' library not found. Skipping some HFTokenEstimator tests.")
    AutoTokenizer = None # Ensure it's defined for type hinting or conditional checks if needed later
    _hf_auto_tokenizer_gpt2_for_test_comparison = None
except Exception as e: # Catch potential errors from AutoTokenizer.from_pretrained directly
    print(f"Note: Could not load AutoTokenizer for '{_HF_MODEL_NAME_FOR_TEST}' for tests: {e}")
    AutoTokenizer = None
    _hf_auto_tokenizer_gpt2_for_test_comparison = None
    HF_GPT2_TOKENIZER_INSTANCE = None # Ensure this is also None
# --- End conditional imports ---


class TestCharTokenEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = CharTokenEstimator()

    def test_count_tokens_empty_string(self):
        self.assertEqual(self.estimator.count_tokens(""), 0)

    def test_count_tokens_simple_string(self):
        self.assertEqual(self.estimator.count_tokens("hello"), 5)

    def test_count_tokens_with_spaces(self):
        self.assertEqual(self.estimator.count_tokens(" hello world "), 13)

    def test_count_tokens_none(self):
        self.assertEqual(self.estimator.count_tokens(None), 0)


class TestHFTokenEstimator(unittest.TestCase):
    def setUp(self):
        if not HF_GPT2_TOKENIZER_INSTANCE:
            self.skipTest(f"Hugging Face tokenizer '{_HF_MODEL_NAME_FOR_TEST}' not available for tests.")
        self.hf_tokenizer_instance = HF_GPT2_TOKENIZER_INSTANCE
        # For direct comparison if needed and AutoTokenizer loaded successfully
        self.gpt2_auto_tokenizer = _hf_auto_tokenizer_gpt2_for_test_comparison


    def test_count_tokens_empty_string(self):
        self.assertEqual(self.hf_tokenizer_instance.count_tokens(""), 0)

    def test_count_tokens_simple_sentence(self):
        text = "Hello, world!"
        # Expected for gpt2: "Hello" (1), "," (1), " world" (1), "!" (1) -> 4 tokens
        # Or using AutoTokenizer directly to get expected count:
        if self.gpt2_auto_tokenizer:
            expected_count = len(self.gpt2_auto_tokenizer.encode(text))
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), expected_count)
        else: # Fallback if direct AutoTokenizer failed but instance somehow loaded
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), 4)


    def test_count_tokens_common_words(self):
        text = "This is a simple sentence."
        if self.gpt2_auto_tokenizer:
            expected_count = len(self.gpt2_auto_tokenizer.encode(text))
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), expected_count)
        else:
            # "This" (1) " is" (1) " a" (1) " simple" (1) " sentence" (1) "." (1) -> 6 tokens for gpt2
            self.assertEqual(self.hf_tokenizer_instance.count_tokens(text), 6)

    def test_count_tokens_none(self):
        self.assertEqual(self.hf_tokenizer_instance.count_tokens(None), 0)

    @unittest.skipIf(not HAVE_TRANSFORMERS, "Transformers library not available for this test.")
    def test_invalid_model_name(self):
        with self.assertRaisesRegex(ValueError, "Failed to load tokenizer for model 'invalid-model-name-does-not-exist'"):
            HFTokenEstimator("invalid-model-name-does-not-exist")

    # Testing ImportError if transformers is not installed is hard when it IS installed.
    # This would typically be done by manipulating sys.modules or in a separate test environment.
    # The HFTokenEstimator's __init__ checks `if AutoTokenizer is None`, which is covered
    # by the global setup if transformers import fails. An explicit test for that path:
    @unittest.skipIf(HAVE_TRANSFORMERS, "Skipping test for ImportError as transformers IS installed.")
    def test_hf_estimator_import_error_if_no_transformers(self):
        # This test will only run if the initial import of AutoTokenizer failed.
        with self.assertRaisesRegex(ImportError, "transformers library is not installed"):
            HFTokenEstimator("gpt2")


class TestPlaceholderTokenEstimators(unittest.TestCase):
    def test_llama_cpp_not_implemented(self):
        estimator = LlamaCppTokenEstimator("dummy/path/model.gguf")
        with self.assertRaisesRegex(NotImplementedError, "LlamaCppTokenEstimator is not yet implemented"):
            estimator.count_tokens("test text")

    def test_exl2_not_implemented(self):
        estimator = EXL2TokenEstimator("dummy/path/exl2_model")
        with self.assertRaisesRegex(NotImplementedError, "EXL2TokenEstimator is not yet implemented"):
            estimator.count_tokens("test text")


class TestCountTokensFunction(unittest.TestCase):
    def test_char_method(self):
        self.assertEqual(count_tokens("hello", method='char'), 5)

    @unittest.skipIf(not HF_GPT2_TOKENIZER_INSTANCE, f"HF tokenizer '{_HF_MODEL_NAME_FOR_TEST}' not available.")
    def test_hf_method_valid(self):
        text = "Hello, world!"
        if _hf_auto_tokenizer_gpt2_for_test_comparison:
            expected_count = len(_hf_auto_tokenizer_gpt2_for_test_comparison.encode(text))
            self.assertEqual(count_tokens(text, method='hf', model_name=_HF_MODEL_NAME_FOR_TEST), expected_count)
        else:
             self.assertEqual(count_tokens(text, method='hf', model_name=_HF_MODEL_NAME_FOR_TEST), 4) # Fallback


    @unittest.skipIf(not HAVE_TRANSFORMERS, "Transformers library not available for this test.")
    def test_hf_method_invalid_model(self):
        with self.assertRaisesRegex(ValueError, "Error using 'hf' method: Failed to load tokenizer"):
            count_tokens("test", method='hf', model_name="invalid-model-name-for-sure")

    def test_hf_method_missing_model_name(self):
        with self.assertRaisesRegex(ValueError, "model_name must be provided for 'hf' method"):
            count_tokens("test", method='hf')

    def test_llama_cpp_method_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "LlamaCppTokenEstimator is not yet implemented"):
            count_tokens("test", method='llama_cpp', model_path="dummy/path.gguf")

    def test_llama_cpp_method_missing_model_path(self):
        with self.assertRaisesRegex(ValueError, "model_path must be provided for 'llama_cpp' method"):
            count_tokens("test", method='llama_cpp')

    def test_exl2_method_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "EXL2TokenEstimator is not yet implemented"):
            count_tokens("test", method='exl2', model_dir="dummy/exl2_dir")

    def test_exl2_method_missing_model_dir(self):
        with self.assertRaisesRegex(ValueError, "model_dir must be provided for 'exl2' method"):
            count_tokens("test", method='exl2')

    def test_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown token estimation method: unknown_method"):
            count_tokens("test", method='unknown_method')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
