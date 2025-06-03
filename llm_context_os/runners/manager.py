# llm_context_os/runners/manager.py
import typing as t
from .base import BaseRunner
from .api_runner import APIRunner
from .llama_cpp_runner import LlamaCppRunner
from .awq_runner import AWQRunner
from .exl2_runner import EXL2Runner

class ModelManager:
    """
    Manages the loading and retrieval of different model runners.
    Handles idle auto-unload in a future phase.
    """
    def __init__(self):
        self.current_runner: t.Optional[BaseRunner] = None
        self.current_model_type: t.Optional[str] = None
        self.current_model_identifier: t.Optional[str] = None # Path or name
        print("ModelManager initialized. No model loaded initially.")

    def load(self, model_type: str, model_path_or_name: str, **kwargs: t.Any) -> None:
        """
        Loads a model runner based on the specified type and path/name.

        Args:
            model_type (str): The type of model to load
                              (e.g., 'api', 'gguf', 'awq', 'exl2').
            model_path_or_name (str): The path to the model file/directory or
                                      a model identifier (e.g., repo_id for HF, name for API).
            **kwargs: Additional keyword arguments to pass to the runner's constructor.
                      These can include things like api_url, api_key, n_gpu_layers, device, etc.
        """
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name: {model_path_or_name}")
        print(f"  Additional args: {kwargs}")

        # Unload previous model if any (basic for now, future versions might have more complex unloading)
        if self.current_runner:
            print(f"Unloading previous model: {self.current_model_identifier} ({self.current_model_type})")
            # In a real scenario, this might involve deleting the runner instance
            # or calling a specific unload method on the runner if it holds resources.
            self.current_runner = None
            self.current_model_type = None
            self.current_model_identifier = None

        try:
            if model_type.lower() == 'api':
                # For API, model_path_or_name is often the model identifier (e.g., "gpt-3.5-turbo")
                # api_url and api_key should be in kwargs
                api_url = kwargs.pop('api_url', None)
                api_key = kwargs.pop('api_key', None)
                if not api_url:
                    print("Error: 'api_url' is required for API runner.")
                    return
                self.current_runner = APIRunner(model_name=model_path_or_name, api_url=api_url, api_key=api_key, **kwargs)

            elif model_type.lower() == 'gguf':
                self.current_runner = LlamaCppRunner(model_path=model_path_or_name, **kwargs)

            elif model_type.lower() == 'awq':
                self.current_runner = AWQRunner(model_path_or_repo_id=model_path_or_name, **kwargs)

            elif model_type.lower() == 'exl2':
                self.current_runner = EXL2Runner(model_path=model_path_or_name, **kwargs)

            else:
                print(f"Error: Unknown model type '{model_type}'. No model loaded.")
                return # current_runner remains as it was (None or previous runner if error handling is different)

            self.current_model_type = model_type.lower()
            self.current_model_identifier = model_path_or_name
            print(f"Successfully loaded {self.current_model_type} model: {self.current_model_identifier}")

        except Exception as e:
            print(f"Error during model loading or runner initialization: {e}")
            self.current_runner = None # Ensure no partially loaded runner state
            self.current_model_type = None
            self.current_model_identifier = None


    def get(self) -> t.Optional[BaseRunner]:
        """
        Retrieves the currently loaded model runner.

        Returns:
            t.Optional[BaseRunner]: The active BaseRunner instance, or None if no model is loaded.
                                    (Changed from raising ValueError to returning None for easier use in some contexts)
        """
        if self.current_runner:
            # In a future phase, this would refresh a keep-alive timer for idle auto-unload
            print(f"\nRetrieved current runner: {self.current_model_identifier} ({self.current_model_type})")
            return self.current_runner
        else:
            print("Error: No model is currently loaded. Call load() first.")
            return None
            # Alternatively, raise ValueError("No model loaded. Call load() first.")

if __name__ == '__main__':
    manager = ModelManager()

    # Attempt to get runner when none is loaded
    runner = manager.get()
    print(f"Runner initially: {runner}")

    # Load an API runner
    manager.load(
        model_type='api',
        model_path_or_name='gpt-4-dummy',
        api_url='https://api.openai.com/v1',
        api_key='sk-xxxxxxxxxxxxxx'
    )
    api_runner = manager.get()
    if api_runner:
        api_runner.generate("Test prompt for API runner")

    # Load a GGUF (LlamaCpp) runner
    manager.load(
        model_type='gguf',
        model_path_or_name='models/dummy-llama-7b.Q4_K_M.gguf',
        n_gpu_layers=20,
        n_ctx=2048
    )
    gguf_runner = manager.get()
    if gguf_runner:
        gguf_runner.stream("Test stream for GGUF runner", temperature=0.5)

    # Load an AWQ runner
    manager.load(
        model_type='awq',
        model_path_or_repo_id='quantized/dummy-awq-model',
        device='cuda',
        use_flash_attention_2=False
    )
    awq_runner = manager.get()
    if awq_runner:
        awq_runner.generate("Test prompt for AWQ runner")

    # Load an EXL2 runner
    manager.load(
        model_type='exl2',
        model_path_or_name='models/dummy-exl2-model-dir',
        gpu_split='auto'
    )
    exl2_runner = manager.get()
    if exl2_runner:
        exl2_runner.stream("Test stream for EXL2 runner")

    # Attempt to load an unknown model type
    manager.load(model_type='unknown_type', model_path_or_name='some/path')
    unknown_runner = manager.get() # Should still be the EXL2 runner or None if load clears on unknown
                                  # Current implementation clears previous before attempting new load.
                                  # If new load fails due to unknown type, current_runner becomes None.
    if unknown_runner:
        print(f"Runner after unknown attempt: {type(unknown_runner)}")
    else:
        print("Runner is None after attempting to load unknown type, as expected.")

    # Demonstrate getting a runner after a successful load again
    manager.load(model_type='api', model_path_or_name='another-api', api_url='http://localhost:8080/v1')
    another_api_runner = manager.get()
    if another_api_runner:
        print("Successfully got another API runner.")
        another_api_runner.generate("Final test.")

    print("\nModelManager demonstration complete.")
