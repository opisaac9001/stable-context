# llm_context_os/context/token_estimator.py
import typing as t
from abc import ABC, abstractmethod

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None # Placeholder if transformers is not installed

class BaseTokenEstimator(ABC):
    @abstractmethod
    def count_tokens(self, text_content: str) -> int:
        """
        Estimates the number of tokens in a given text.
        """
        pass

class CharTokenEstimator(BaseTokenEstimator):
    """
    A simple token estimator that counts characters.
    """
    def count_tokens(self, text_content: str) -> int:
        if text_content is None:
            return 0
        return len(text_content)

class HFTokenEstimator(BaseTokenEstimator):
    """
    Token estimator using Hugging Face's transformers library.
    """
    def __init__(self, model_name: str):
        if AutoTokenizer is None:
            raise ImportError(
                "transformers library is not installed. "
                "Please install it with `pip install transformers` to use HFTokenEstimator."
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer for model '{model_name}': {e}")

    def count_tokens(self, text_content: str) -> int:
        if text_content is None:
            return 0
        # Some tokenizers might return a list of token IDs, others might have a more complex structure.
        # For most common use cases, len(tokenizer.encode(text)) is standard.
        return len(self.tokenizer.encode(text_content))

class LlamaCppTokenEstimator(BaseTokenEstimator):
    """
    Placeholder for Llama.cpp token estimator.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        # In a real implementation, this would load the Llama.cpp model/tokenizer.
        # For now, it does nothing.

    def count_tokens(self, text_content: str) -> int:
        raise NotImplementedError(
            f"LlamaCppTokenEstimator is not yet implemented. Model path: {self.model_path}"
        )

class EXL2TokenEstimator(BaseTokenEstimator):
    """
    Placeholder for EXL2 (e.g., ExLlamaV2) token estimator.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        # In a real implementation, this would set up the EXL2 model/tokenizer
        # from the specified directory. For now, it does nothing.

    def count_tokens(self, text_content: str) -> int:
        raise NotImplementedError(
            f"EXL2TokenEstimator is not yet implemented. Model dir: {self.model_dir}"
        )

def count_tokens(
    text_content: str,
    method: str = 'char',
    model_name: t.Optional[str] = None,
    model_path: t.Optional[str] = None,
    model_dir: t.Optional[str] = None
) -> int:
    """
    Provides a generic way to count tokens, supporting multiple methods.
    - 'char': Counts characters.
    - 'hf': Uses a Hugging Face tokenizer. Requires 'model_name'.
    - 'llama_cpp': Uses a Llama.cpp tokenizer. Requires 'model_path'. (Not Implemented)
    - 'exl2': Uses an EXL2 tokenizer. Requires 'model_dir'. (Not Implemented)
    """
    if method == 'char':
        estimator = CharTokenEstimator()
        return estimator.count_tokens(text_content)
    elif method == 'hf':
        if model_name is None:
            raise ValueError("model_name must be provided for 'hf' method.")
        try:
            estimator = HFTokenEstimator(model_name=model_name)
            return estimator.count_tokens(text_content)
        except ImportError as e: # Catch ImportError from HFTokenEstimator constructor
            # Re-raise or handle more gracefully depending on desired behavior
            # For now, let's make it clear that it's an installation issue for this method.
            raise ImportError(
                f"{e} The 'hf' method requires the 'transformers' library and a valid model name."
            )
        except ValueError as e: # Catch ValueError from HFTokenEstimator (e.g. model not found)
            raise ValueError(f"Error using 'hf' method: {e}")
    elif method == 'llama_cpp':
        if model_path is None:
            raise ValueError("model_path must be provided for 'llama_cpp' method.")
        estimator = LlamaCppTokenEstimator(model_path=model_path)
        return estimator.count_tokens(text_content) # This will raise NotImplementedError
    elif method == 'exl2':
        if model_dir is None:
            raise ValueError("model_dir must be provided for 'exl2' method.")
        estimator = EXL2TokenEstimator(model_dir=model_dir)
        return estimator.count_tokens(text_content) # This will raise NotImplementedError
    else:
        raise ValueError(f"Unknown token estimation method: {method}")

if __name__ == '__main__':
    # Basic test
    estimator = CharTokenEstimator()
    text = "Hello, world!"
    print(f"'{text}' has {estimator.count_tokens(text)} tokens (chars).")

    text_2 = "Another example sentence."
    print(f"'{text_2}' has {count_tokens(text_2)} tokens (using standalone char method).")

    # Test HFTokenEstimator - this will only work if transformers is installed
    # and the model is accessible.
    if AutoTokenizer is not None:
        try:
            # Using a small, common model for testing
            hf_model = "gpt2"
            text_3 = "This is a test for Hugging Face tokenizer."
            num_tokens_hf = count_tokens(text_3, method='hf', model_name=hf_model)
            print(f"'{text_3}' has {num_tokens_hf} tokens (using HF model '{hf_model}').")

            # Test invalid model name
            try:
                count_tokens("test", method='hf', model_name="invalid-model-name-that-does-not-exist")
            except ValueError as e:
                print(f"Caught expected error for invalid HF model: {e}")

        except (ImportError, ValueError) as e:
            print(f"Skipping HFTokenizer test due to: {e}")
    else:
        print("Skipping HFTokenizer tests as 'transformers' library is not installed.")
        # Test that using 'hf' method without transformers installed raises ImportError
        try:
            count_tokens("test", method='hf', model_name="gpt2")
        except ImportError as e:
            print(f"Caught expected ImportError for 'hf' method when transformers is missing: {e}")


    try:
        count_tokens("test", method="unknown")
    except ValueError as e:
        print(f"Caught expected error for unknown method: {e}")

    try:
        count_tokens("test", method="hf") # Missing model_name
    except ValueError as e:
        print(f"Caught expected error for hf method without model_name: {e}")

    # Test LlamaCppTokenEstimator placeholder
    try:
        print("\nTesting LlamaCppTokenEstimator (placeholder)...")
        count_tokens("test text for llama.cpp", method='llama_cpp', model_path="/path/to/dummy.gguf")
    except NotImplementedError as e:
        print(f"Caught expected NotImplementedError for llama_cpp: {e}")
    except ValueError as e: # Should not happen if model_path is provided
        print(f"Caught unexpected ValueError for llama_cpp: {e}")

    try:
        count_tokens("test", method='llama_cpp') # Missing model_path
    except ValueError as e:
        print(f"Caught expected error for llama_cpp method without model_path: {e}")


    # Test EXL2TokenEstimator placeholder
    try:
        print("\nTesting EXL2TokenEstimator (placeholder)...")
        count_tokens("test text for exl2", method='exl2', model_dir="/path/to/dummy_exl2_model/")
    except NotImplementedError as e:
        print(f"Caught expected NotImplementedError for exl2: {e}")
    except ValueError as e: # Should not happen if model_dir is provided
        print(f"Caught unexpected ValueError for exl2: {e}")

    try:
        count_tokens("test", method='exl2') # Missing model_dir
    except ValueError as e:
        print(f"Caught expected error for exl2 method without model_dir: {e}")
