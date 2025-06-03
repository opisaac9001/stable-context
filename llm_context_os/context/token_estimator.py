# llm_context_os/context/token_estimator.py
from abc import ABC, abstractmethod

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

def count_tokens(text_content: str, method: str = 'char') -> int:
    """
    Provides a generic way to count tokens, potentially supporting multiple methods.
    For now, only 'char' (character count) is implemented.
    """
    if method == 'char':
        estimator = CharTokenEstimator()
        return estimator.count_tokens(text_content)
    # In the future, other methods like 'hf', 'llama_cpp' could be added here
    # by instantiating different estimator classes.
    else:
        raise ValueError(f"Unknown token estimation method: {method}")

if __name__ == '__main__':
    # Basic test
    estimator = CharTokenEstimator()
    text = "Hello, world!"
    print(f"'{text}' has {estimator.count_tokens(text)} tokens (chars).")

    text_2 = "Another example sentence."
    print(f"'{text_2}' has {count_tokens(text_2)} tokens (using standalone char method).")

    try:
        count_tokens("test", method="unknown")
    except ValueError as e:
        print(f"Caught expected error: {e}")
