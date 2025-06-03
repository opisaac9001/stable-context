# llm_context_os/runners/manager.py
import typing as t
import time # Added for idle tracking

from .base import BaseRunner
from .api_runner import APIRunner
from .llama_cpp_runner import LlamaCppRunner
from .awq_runner import AWQRunner
from .exl2_runner import EXL2Runner

class ModelManager:
    """
    Manages the loading and retrieval of different model runners.
    Includes basic idle auto-unload functionality.
    """
    def __init__(self):
        self.current_runner: t.Optional[BaseRunner] = None
        self.current_model_type: t.Optional[str] = None
        self.current_model_identifier: t.Optional[str] = None # Path or name

        self.idle_unload_sec: t.Optional[int] = None
        self.last_accessed_time: t.Optional[float] = None

        print("ModelManager initialized. No model loaded initially.")

    def load(self,
             model_type: str,
             model_path_or_name: str,
             idle_unload_sec: t.Optional[int] = None, # New parameter
             **kwargs: t.Any) -> None:
        """
        Loads a model runner based on the specified type and path/name.

        Args:
            model_type (str): The type of model to load
                              (e.g., 'api', 'gguf', 'awq', 'exl2').
            model_path_or_name (str): The path to the model file/directory or
                                      a model identifier (e.g., repo_id for HF, name for API).
            idle_unload_sec (t.Optional[int]): If provided, the model will be automatically
                                               unloaded after this many seconds of inactivity.
            **kwargs: Additional keyword arguments to pass to the runner's constructor.
        """
        print(f"\nAttempting to load model...")
        print(f"  Type: {model_type}")
        print(f"  Path/Name: {model_path_or_name}")
        if idle_unload_sec is not None:
            print(f"  Idle Unload Sec: {idle_unload_sec}")
        print(f"  Additional args: {kwargs}")

        if self.current_runner:
            self.unload() # Use the new unload method

        try:
            if model_type.lower() == 'api':
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
                self.current_runner = None # Ensure it's None if type is unknown
                return

            self.current_model_type = model_type.lower()
            self.current_model_identifier = model_path_or_name
            self.idle_unload_sec = idle_unload_sec
            self.last_accessed_time = time.time() # Set last accessed time on successful load
            print(f"Successfully loaded {self.current_model_type} model: {self.current_model_identifier}")

        except Exception as e:
            print(f"Error during model loading or runner initialization: {e}")
            self.current_runner = None
            self.current_model_type = None
            self.current_model_identifier = None
            self.idle_unload_sec = None
            self.last_accessed_time = None

    def get(self) -> t.Optional[BaseRunner]:
        """
        Retrieves the currently loaded model runner.
        Updates the last_accessed_time if a runner is present.
        """
        if self.current_runner:
            print(f"\nRetrieved current runner: {self.current_model_identifier} ({self.current_model_type})")
            self.last_accessed_time = time.time() # Update access time
            return self.current_runner
        else:
            print("Error: No model is currently loaded. Call load() first.")
            return None

    def unload(self) -> None:
        """
        Unloads the currently active model and resets associated state.
        """
        if self.current_runner:
            print(f"Unloading model: {self.current_model_identifier} ({self.current_model_type})")
            # TODO: Add actual model resource cleanup here (e.g., del self.current_runner.model for HF, exl2; specific unload for llama-cpp)
            # For placeholder runners, direct deletion is fine.
            # If runners have specific cleanup methods (e.g., llama_instance.__del__ or similar), call them.
            if hasattr(self.current_runner, '__del__'): # Basic check, might not be sufficient for all types
                try:
                    self.current_runner.__del__()
                except Exception as e:
                    print(f"Error during explicit runner __del__: {e}")
        else:
            print("No model to unload.")

        self.current_runner = None
        self.current_model_type = None
        self.current_model_identifier = None
        self.idle_unload_sec = None
        self.last_accessed_time = None
        print("Model unloaded and manager reset.")

    def check_idle(self) -> bool:
        """
        Checks if the current model has been idle for longer than its configured
        idle_unload_sec. If so, unloads the model.

        Returns:
            bool: True if the model was unloaded due to idleness, False otherwise.
        """
        if self.current_runner and \
           self.idle_unload_sec is not None and \
           self.last_accessed_time is not None:

            idle_time = time.time() - self.last_accessed_time
            if idle_time > self.idle_unload_sec:
                print(f"Model '{self.current_model_identifier}' idle for {idle_time:.2f}s (limit: {self.idle_unload_sec}s), unloading.")
                self.unload()
                return True
        return False

if __name__ == '__main__':
    manager = ModelManager()

    # Attempt to get runner when none is loaded
    runner = manager.get()
    print(f"Runner initially: {runner}")

    # Load an API runner with idle unload
    print("\n--- Testing Idle Unload ---")
    idle_seconds = 2
    manager.load(
        model_type='api',
        model_path_or_name='idle-test-api',
        api_url='https://dummy.api/v1',
        idle_unload_sec=idle_seconds
    )

    api_runner = manager.get() # Access to update last_accessed_time
    self.assertIsNotNone(api_runner, "API runner should be loaded.")
    print(f"API Runner '{api_runner.model_name if api_runner else None}' loaded. Waiting for {idle_seconds + 1} seconds...")

    time.sleep(idle_seconds + 1)

    unloaded = manager.check_idle()
    self.assertTrue(unloaded, "Model should have been unloaded due to idle timeout.")

    runner_after_idle = manager.get()
    self.assertIsNone(runner_after_idle, "Runner should be None after idle unload.")
    print("Idle unload test successful.")

    # Load a GGUF (LlamaCpp) runner
    print("\n--- Testing GGUF Load (no idle) ---")
    manager.load(
        model_type='gguf',
        model_path_or_name='models/dummy-llama-7b.Q4_K_M.gguf',
        n_gpu_layers=20,
        n_ctx=2048
    )
    gguf_runner = manager.get()
    if gguf_runner:
        gguf_runner.stream("Test stream for GGUF runner", temperature=0.5)
    self.assertIsNotNone(gguf_runner, "GGUF runner should be loaded.")
    self.assertIsInstance(gguf_runner, LlamaCppRunner, "Runner should be LlamaCppRunner.")


    # Test manual unload
    print("\n--- Testing Manual Unload ---")
    self.assertIsNotNone(manager.get(), "Model should be loaded before manual unload.")
    manager.unload()
    self.assertIsNone(manager.get(), "Model should be None after manual unload.")
    print("Manual unload test successful.")


    # Attempt to load an unknown model type
    print("\n--- Testing Unknown Model Type ---")
    manager.load(model_type='unknown_type', model_path_or_name='some/path')
    unknown_runner = manager.get()
    self.assertIsNone(unknown_runner, "Runner should be None after attempting to load unknown type.")

    print("\nModelManager demonstration complete.")

# Need to wrap assertions in a unittest structure or remove for standalone script run
# For now, replacing self.assertX with print and manual check for __main__
if __name__ == '__main__':
    manager = ModelManager()

    runner = manager.get()
    print(f"Runner initially: {runner is None}")

    print("\n--- Testing Idle Unload ---")
    idle_seconds = 2
    manager.load(
        model_type='api',
        model_path_or_name='idle-test-api',
        api_url='https://dummy.api/v1',
        idle_unload_sec=idle_seconds
    )

    api_runner_obj = manager.get()
    print(f"API Runner loaded: {api_runner_obj is not None}")
    print(f"Waiting for {idle_seconds + 1} seconds for idle check...")

    time.sleep(idle_seconds + 1)

    unloaded = manager.check_idle()
    print(f"Model unloaded due to idle: {unloaded}")

    runner_after_idle_obj = manager.get()
    print(f"Runner is None after idle unload: {runner_after_idle_obj is None}")
    print("Idle unload test presumed successful if above is True.")

    print("\n--- Testing GGUF Load (no idle) ---")
    manager.load(
        model_type='gguf',
        model_path_or_name='models/dummy-llama-7b.Q4_K_M.gguf',
        n_gpu_layers=20,
        n_ctx=2048
    )
    gguf_runner_obj = manager.get()
    if gguf_runner_obj:
        # gguf_runner_obj.stream("Test stream for GGUF runner", temperature=0.5) # Placeholder would print
        print(f"GGUF runner loaded: {isinstance(gguf_runner_obj, LlamaCppRunner)}")
    else:
        print("GGUF runner failed to load.")


    print("\n--- Testing Manual Unload ---")
    print(f"Model loaded before manual unload: {manager.get() is not None}")
    manager.unload()
    print(f"Model is None after manual unload: {manager.get() is None}")
    print("Manual unload test presumed successful if above is True.")

    print("\n--- Testing Unknown Model Type ---")
    manager.load(model_type='unknown_type', model_path_or_name='some/path')
    unknown_runner_obj = manager.get()
    print(f"Runner is None after attempting unknown type: {unknown_runner_obj is None}")

    print("\nModelManager demonstration complete.")
