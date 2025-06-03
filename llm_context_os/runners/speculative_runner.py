# llm_context_os/runners/speculative_runner.py
import typing as t
from .base import BaseRunner

class SpeculativeRunner(BaseRunner):
    def __init__(self, draft_runner: BaseRunner, target_runner: BaseRunner, speculative_k: int = 5, **kwargs):
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.speculative_k = speculative_k
        print(f"[SpeculativeRunner] Initialized with draft_runner: {type(draft_runner).__name__}, "
              f"target_runner: {type(target_runner).__name__}, k={speculative_k}")

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> str:
        print(f"[SpeculativeRunner] generate called for prompt: '{prompt[:50]}...'")
        if image_paths:
            print(f"  Image Paths: {image_paths}")

        draft_kwargs = kwargs.copy()
        print(f"[SpeculativeRunner] Simulating draft generation (approx {self.speculative_k} tokens)...")
        # Pass image_paths to draft runner's generate method
        _ = self.draft_runner.generate(prompt, image_paths=image_paths, max_new_tokens=self.speculative_k * 5, **draft_kwargs)
        draft_output_simulation = " ".join([f"draft{i+1}" for i in range(self.speculative_k)]) + " "
        print(f"[SpeculativeRunner] Draft simulation produced: '{draft_output_simulation}'")

        print(f"[SpeculativeRunner] Target runner validating and completing from original prompt + draft...")
        # Pass image_paths to target runner's generate method
        # In a real scenario, if draft processes images, target might not need them again, or vice-versa.
        # For placeholder, pass to both if provided.
        final_output = self.target_runner.generate(prompt + draft_output_simulation, image_paths=image_paths, **kwargs)
        print(f"[SpeculativeRunner] Final output from target: '{final_output}'")
        return final_output

    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> t.Generator[str, None, None]:
        print(f"[SpeculativeRunner] stream called for prompt: '{prompt[:50]}...'")
        if image_paths:
            print(f"  Image Paths: {image_paths}")

        print("[SpeculativeRunner] Streaming from draft_runner (simulated)...")
        draft_stream_count = 0
        # Pass image_paths to draft runner's stream method
        for chunk in self.draft_runner.stream(prompt, image_paths=image_paths, max_new_tokens=self.speculative_k * 5, **kwargs):
            yield chunk
            draft_stream_count +=1
            if draft_stream_count >= self.speculative_k :
                break

        print("\n[SpeculativeRunner] Switching to target_runner for validation and continuation...")
        # Pass image_paths to target runner's stream method
        for chunk in self.target_runner.stream(prompt, image_paths=image_paths, **kwargs):
            yield chunk

        yield " [EndOfSpecStream]"
        print("[SpeculativeRunner] Streaming complete.")

    # --- KV Cache and LoRA methods remain the same (delegation logic) ---
    def export_kv_cache(self) -> t.Any:
        print("[SpeculativeRunner] export_kv_cache called. Delegating to target_runner.")
        return self.target_runner.export_kv_cache()

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print("[SpeculativeRunner] import_kv_cache called. Delegating to both runners.")
        print("  Importing to target_runner...")
        self.target_runner.import_kv_cache(cache_data)
        print("  Importing to draft_runner (using same data for placeholder)...")
        self.draft_runner.import_kv_cache(cache_data)

    def preload_kv(self, prompt: str, **kwargs) -> None:
        print(f"[SpeculativeRunner] preload_kv called for prompt: '{prompt[:50]}...'.")
        print("  Preloading KV for draft_runner...")
        self.draft_runner.preload_kv(prompt, **kwargs) # image_paths not typically sent to preload_kv
        print("  Preloading KV for target_runner...")
        self.target_runner.preload_kv(prompt, **kwargs)

    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"[SpeculativeRunner] load_lora_adapter '{adapter_id}' called.")
        print("  Attempting to load LoRA on target_runner...")
        target_success = self.target_runner.load_lora_adapter(adapter_id, adapter_path, **kwargs)
        print("  Attempting to load LoRA on draft_runner (if different or supports it)...")
        draft_success = self.draft_runner.load_lora_adapter(f"draft_{adapter_id}", adapter_path, **kwargs)
        return target_success

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"[SpeculativeRunner] unload_lora_adapter '{adapter_id}' called.")
        print("  Attempting to unload LoRA from target_runner...")
        target_success = self.target_runner.unload_lora_adapter(adapter_id, **kwargs)
        print("  Attempting to unload LoRA from draft_runner...")
        draft_success = self.draft_runner.unload_lora_adapter(f"draft_{adapter_id}", **kwargs)
        return target_success

    def get_active_lora_adapters(self) -> t.List[str]:
        print("[SpeculativeRunner] get_active_lora_adapters called.")
        print("  Delegating to target_runner for active LoRA list.")
        return self.target_runner.get_active_lora_adapters()

    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"[SpeculativeRunner] merge_lora_adapters for IDs '{adapter_ids}' called.")
        print("  Merging typically applies to the target_runner.")
        return self.target_runner.merge_lora_adapters(adapter_ids, **kwargs)

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print("[SpeculativeRunner] unmerge_lora_adapters called.")
        print("  Unmerging typically applies to the target_runner.")
        return self.target_runner.unmerge_lora_adapters(**kwargs)

if __name__ == '__main__':
    import unittest

    class EchoRunner(BaseRunner):
        # ... (EchoRunner implementation as before, but needs image_paths in generate/stream) ...
        def __init__(self, name: str, delay_factor: float = 0.01):
            self.name = name
            self.delay_factor = delay_factor
            self.kv_cache_data: t.Optional[t.Dict[str, str]] = None
            self.active_loras: t.Dict[str, t.Any] = {}
            self.is_merged_state: bool = False
            print(f"[EchoRunner-{self.name}] Initialized.")

        def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> str:
            img_info = f" (Images: {image_paths})" if image_paths else ""
            print(f"[EchoRunner-{self.name}] generate. Prompt: '{prompt[:30]}...'{img_info}")
            response = f"Echo-{self.name}: {prompt}{img_info}"
            self.kv_cache_data = {f'{self.name}_last_prompt': prompt[:10]}
            return response

        def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> t.Generator[str, None, None]:
            img_info = f" (Images: {image_paths})" if image_paths else ""
            print(f"[EchoRunner-{self.name}] stream. Prompt: '{prompt[:30]}...'{img_info}")
            self.kv_cache_data = {f'{self.name}_last_stream': prompt[:10]}
            yield f"EchoStream-{self.name}: "
            if image_paths: yield f"ImagesSeen-{len(image_paths)} "
            for char_idx, char_val in enumerate(prompt):
                if char_idx < 15 : yield char_val
                else: break
            yield " ... (end of echo stream)"

        def export_kv_cache(self) -> t.Any: return self.kv_cache_data
        def import_kv_cache(self, cache_data: t.Any) -> None: self.kv_cache_data = cache_data
        def preload_kv(self, prompt: str, **kwargs) -> None: self.kv_cache_data = {f'{self.name}_preloaded': prompt[:20]}
        def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool: self.active_loras[adapter_id] = True; return True
        def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool: return self.active_loras.pop(adapter_id, None) is not None
        def get_active_lora_adapters(self) -> t.List[str]: return list(self.active_loras.keys())
        def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool: self.is_merged_state = True; return True
        def unmerge_lora_adapters(self, **kwargs) -> bool: self.is_merged_state = False; return True


    class MainTests(unittest.TestCase):
        def run_main_demo(self):
            print("--- Testing SpeculativeRunner (within MainTests) --- ")
            draft_model = EchoRunner(name="DraftModel")
            target_model = EchoRunner(name="TargetModel")
            spec_runner = SpeculativeRunner(draft_runner=draft_model, target_runner=target_model, speculative_k=3)

            example_images = ["/path/img1.jpg"]

            print("\n--- Generate Demo with Image ---")
            output = spec_runner.generate("Describe the image.", image_paths=example_images)
            print(f"Final Generate Output from SpeculativeRunner: '{output}'")
            self.assertIn("Echo-TargetModel", output)
            self.assertIn("(Images: ['/path/img1.jpg'])", output) # Check if image info propagated

            print("\n--- Stream Demo with Image ---")
            full_streamed_output = []
            for chunk in spec_runner.stream("Describe this other image.", image_paths=example_images):
                full_streamed_output.append(chunk)
            full_stream_str = "".join(full_streamed_output)
            print(f"\nFull streamed output from SpeculativeRunner: {full_stream_str}")
            self.assertIn("ImagesSeen-1", full_stream_str) # Check if EchoRunner saw the image in stream

            # ... (rest of KV and LoRA demos can be kept or simplified) ...
            print("\nSpeculativeRunner multimodal demonstration complete.")

    if __name__ == '__main__':
        demo = MainTests() # This requires unittest to be available
        demo.run_main_demo()
