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

    def generate(self, prompt: str, **kwargs) -> str:
        print(f"[SpeculativeRunner] generate called for prompt: '{prompt[:50]}...'")
        draft_kwargs = kwargs.copy()
        print(f"[SpeculativeRunner] Simulating draft generation (approx {self.speculative_k} tokens)...")
        # Placeholder: draft_runner.generate might be called here in a real impl.
        # For this version, we keep the simplified draft_output_simulation
        _ = self.draft_runner.generate(prompt, max_new_tokens=self.speculative_k * 5, **draft_kwargs)
        draft_output_simulation = " ".join([f"draft{i+1}" for i in range(self.speculative_k)]) + " "
        print(f"[SpeculativeRunner] Draft simulation produced: '{draft_output_simulation}'")

        print(f"[SpeculativeRunner] Target runner validating and completing from original prompt + draft...")
        final_output = self.target_runner.generate(prompt + draft_output_simulation, **kwargs)
        print(f"[SpeculativeRunner] Final output from target: '{final_output}'")
        return final_output

    def stream(self, prompt: str, **kwargs) -> t.Generator[str, None, None]:
        print(f"[SpeculativeRunner] stream called for prompt: '{prompt[:50]}...'")
        print("[SpeculativeRunner] Streaming from draft_runner (simulated)...")
        draft_stream_count = 0
        for chunk in self.draft_runner.stream(prompt, max_new_tokens=self.speculative_k * 5, **kwargs):
            yield chunk
            draft_stream_count +=1
            if draft_stream_count >= self.speculative_k :
                break

        print("\n[SpeculativeRunner] Switching to target_runner for validation and continuation...")
        for chunk in self.target_runner.stream(prompt, **kwargs):
            yield chunk

        yield " [EndOfSpecStream]"
        print("[SpeculativeRunner] Streaming complete.")

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
        self.draft_runner.preload_kv(prompt, **kwargs)
        print("  Preloading KV for target_runner...")
        self.target_runner.preload_kv(prompt, **kwargs)

    # --- LoRA Adapter Methods (Delegation Strategy) ---
    # For SpeculativeRunner, LoRA operations often apply to the target_runner primarily,
    # but draft_runner might also be LoRA-aware. Delegation can be complex.
    # Simplest approach: Delegate to both if applicable, prioritize target for status.

    def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
        print(f"[SpeculativeRunner] load_lora_adapter '{adapter_id}' called.")
        print("  Attempting to load LoRA on target_runner...")
        target_success = self.target_runner.load_lora_adapter(adapter_id, adapter_path, **kwargs)
        print("  Attempting to load LoRA on draft_runner (if different or supports it)...")
        # Pass a modified ID or flag if draft LoRA should be distinct or handled differently
        draft_success = self.draft_runner.load_lora_adapter(f"draft_{adapter_id}", adapter_path, **kwargs)
        # For placeholder, success if target succeeded, or if both did if required.
        # Here, let's say success if target succeeded.
        return target_success

    def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
        print(f"[SpeculativeRunner] unload_lora_adapter '{adapter_id}' called.")
        print("  Attempting to unload LoRA from target_runner...")
        target_success = self.target_runner.unload_lora_adapter(adapter_id, **kwargs)
        print("  Attempting to unload LoRA from draft_runner...")
        draft_success = self.draft_runner.unload_lora_adapter(f"draft_{adapter_id}", **kwargs)
        return target_success # Prioritize target's status

    def get_active_lora_adapters(self) -> t.List[str]:
        print("[SpeculativeRunner] get_active_lora_adapters called.")
        # Could try to get from both and combine/deduplicate, or prioritize target.
        print("  Delegating to target_runner for active LoRA list.")
        return self.target_runner.get_active_lora_adapters()

    def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
        print(f"[SpeculativeRunner] merge_lora_adapters for IDs '{adapter_ids}' called.")
        print("  Merging typically applies to the target_runner. Draft runner is usually not merged or handled separately.")
        return self.target_runner.merge_lora_adapters(adapter_ids, **kwargs)

    def unmerge_lora_adapters(self, **kwargs) -> bool:
        print("[SpeculativeRunner] unmerge_lora_adapters called.")
        print("  Unmerging typically applies to the target_runner.")
        return self.target_runner.unmerge_lora_adapters(**kwargs)

if __name__ == '__main__':
    import unittest # For MainTests class

    class EchoRunner(BaseRunner):
        def __init__(self, name: str, delay_factor: float = 0.01):
            self.name = name
            self.delay_factor = delay_factor
            self.kv_cache_data: t.Optional[t.Dict[str, str]] = None
            self.active_loras: t.Dict[str, t.Any] = {}
            self.is_merged_state: bool = False
            print(f"[EchoRunner-{self.name}] Initialized.")

        def generate(self, prompt: str, **kwargs) -> str:
            active_lora_info = f" (ActiveLoRAs: {list(self.active_loras.keys())})" if self.active_loras else ""
            merged_info = " (Merged)" if self.is_merged_state else ""
            kv_info = f" (UsingKV: {self.kv_cache_data is not None})" if self.kv_cache_data else ""
            print(f"[EchoRunner-{self.name}] generate. Prompt: '{prompt[:30]}...'{active_lora_info}{merged_info}{kv_info}")
            response = f"Echo-{self.name}: {prompt}"
            self.kv_cache_data = {f'{self.name}_last_prompt': prompt[:10]}
            return response

        def stream(self, prompt: str, **kwargs) -> t.Generator[str, None, None]:
            active_lora_info = f" (ActiveLoRAs: {list(self.active_loras.keys())})" if self.active_loras else ""
            merged_info = " (Merged)" if self.is_merged_state else ""
            kv_info = f" (UsingKV: {self.kv_cache_data is not None})" if self.kv_cache_data else ""
            print(f"[EchoRunner-{self.name}] stream. Prompt: '{prompt[:30]}...'{active_lora_info}{merged_info}{kv_info}")
            self.kv_cache_data = {f'{self.name}_last_stream': prompt[:10]}
            yield f"EchoStream-{self.name}: "
            for char_idx, char_val in enumerate(prompt):
                if char_idx < 15 : yield char_val
                else: break
            yield " ... (end of echo stream)"

        def export_kv_cache(self) -> t.Any: return self.kv_cache_data
        def import_kv_cache(self, cache_data: t.Any) -> None: self.kv_cache_data = cache_data
        def preload_kv(self, prompt: str, **kwargs) -> None: self.kv_cache_data = {f'{self.name}_preloaded': prompt[:20]}

        def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool:
            print(f"[EchoRunner-{self.name}] load_lora_adapter: ID='{adapter_id}', Path='{adapter_path}', Kwargs={kwargs}")
            self.active_loras[adapter_id] = {"path": adapter_path, **kwargs}
            return True
        def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool:
            print(f"[EchoRunner-{self.name}] unload_lora_adapter: ID='{adapter_id}'")
            return self.active_loras.pop(adapter_id, None) is not None
        def get_active_lora_adapters(self) -> t.List[str]: return list(self.active_loras.keys())
        def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool:
            print(f"[EchoRunner-{self.name}] merge_lora_adapters: IDs={adapter_ids}")
            for aid in adapter_ids: self.active_loras.pop(aid, None) # Simulate consumption
            self.is_merged_state = True
            return True
        def unmerge_lora_adapters(self, **kwargs) -> bool:
            print(f"[EchoRunner-{self.name}] unmerge_lora_adapters")
            self.is_merged_state = False
            return True

    class MainTests(unittest.TestCase):
        def run_main_demo(self):
            print("--- Testing SpeculativeRunner (within MainTests) --- ")
            draft_model = EchoRunner(name="DraftModel")
            target_model = EchoRunner(name="TargetModel")
            spec_runner = SpeculativeRunner(draft_runner=draft_model, target_runner=target_model, speculative_k=3)

            print("\n--- LoRA Demo for SpeculativeRunner ---")
            spec_runner.load_lora_adapter("spec_lora_1", "/path/spec_lora1", alpha=0.5)
            self.assertIn("spec_lora_1", target_model.get_active_lora_adapters())
            self.assertIn("draft_spec_lora_1", draft_model.get_active_lora_adapters()) # Placeholder adds "draft_"

            spec_runner.generate("Test prompt with LoRA loaded.")

            active_adapters = spec_runner.get_active_lora_adapters() # Should be target's
            self.assertEqual(active_adapters, ["spec_lora_1"])

            spec_runner.merge_lora_adapters(["spec_lora_1"])
            self.assertTrue(target_model.is_merged_state)

            spec_runner.unmerge_lora_adapters()
            self.assertFalse(target_model.is_merged_state)

            spec_runner.unload_lora_adapter("spec_lora_1")
            self.assertNotIn("spec_lora_1", target_model.get_active_lora_adapters())
            spec_runner.unload_lora_adapter("draft_spec_lora_1") # Unload from draft too
            self.assertNotIn("draft_spec_lora_1", draft_model.get_active_lora_adapters())

            print("\nSpeculativeRunner LoRA demo complete.")

    if __name__ == '__main__':
        demo = MainTests()
        demo.run_main_demo()
