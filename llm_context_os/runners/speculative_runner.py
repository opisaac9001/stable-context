# llm_context_os/runners/speculative_runner.py
import typing as t
from .base import BaseRunner

class SpeculativeRunner(BaseRunner):
    def __init__(self, draft_runner: BaseRunner, target_runner: BaseRunner, speculative_k: int = 5, **kwargs):
        # In Python 3, super() without arguments works if the class is defined.
        # If BaseRunner had its own __init__ that took args, we'd use super().__init__(**kwargs)
        # For now, assuming BaseRunner.__init__ is just object.__init__ or simple.
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.speculative_k = speculative_k
        print(f"[SpeculativeRunner] Initialized with draft_runner: {type(draft_runner).__name__}, "
              f"target_runner: {type(target_runner).__name__}, k={speculative_k}")

    def generate(self, prompt: str, **kwargs) -> str:
        print(f"[SpeculativeRunner] generate called for prompt: '{prompt[:50]}...'")

        # Placeholder logic for speculative decoding:
        # 1. Draft runner generates k tokens (or a short sequence).
        # 2. Target runner takes original prompt + draft tokens.
        #    In a real scenario, it would efficiently validate the draft tokens
        #    and only generate new tokens from the point of first mismatch, or extend.

        draft_kwargs = kwargs.copy() # Pass relevant kwargs to draft if needed
        # Modify draft_kwargs if draft model needs different settings (e.g., fewer tokens)
        # draft_kwargs['max_new_tokens'] = self.speculative_k * some_avg_token_length_estimate

        print(f"[SpeculativeRunner] Simulating draft generation (approx {self.speculative_k} tokens)...")
        # For this placeholder, let's have draft_runner generate a small fixed number of tokens or a short string.
        # The actual number of tokens for 'k' depends on the tokenizer.
        # We'll just simulate it by asking for a short generation.
        # A more realistic placeholder might involve generating k actual tokens.
        simulated_draft_extension = self.draft_runner.generate(prompt, max_new_tokens=self.speculative_k * 5, **draft_kwargs) # Crude estimate
        # Let's assume the draft runner returns only the *newly* generated part
        # For EchoRunner, it returns "Echo-DraftModel: <full_prompt_plus_draft>"
        # We need to refine this to get just the "drafted part".
        # For simplicity, let's assume draft_output is just a few placeholder words.
        draft_output_simulation = " ".join([f"draft{i+1}" for i in range(self.speculative_k)]) + " "
        print(f"[SpeculativeRunner] Draft simulation produced: '{draft_output_simulation}'")

        # Target runner validates and generates the final sequence
        print(f"[SpeculativeRunner] Target runner validating and completing from original prompt + draft...")
        # In a real system, the target runner would take the original prompt's state
        # and the draft tokens, then efficiently verify and continue.
        # For this placeholder, we just append the draft to the original prompt.
        final_output = self.target_runner.generate(prompt + draft_output_simulation, **kwargs)
        print(f"[SpeculativeRunner] Final output from target: '{final_output}'")
        return final_output

    def stream(self, prompt: str, **kwargs) -> t.Generator[str, None, None]:
        print(f"[SpeculativeRunner] stream called for prompt: '{prompt[:50]}...'")

        # Placeholder streaming logic:
        # 1. Yield a few (k) tokens from the draft_runner.
        # 2. Then, switch to target_runner to validate and continue streaming.
        # This is highly simplified. Real speculative streaming is complex.

        print("[SpeculativeRunner] Streaming from draft_runner (simulated)...")
        draft_stream_count = 0
        # Simulate getting k tokens from draft runner's stream
        for chunk in self.draft_runner.stream(prompt, max_new_tokens=self.speculative_k * 5, **kwargs): # Crude estimate for k tokens
            yield chunk
            draft_stream_count +=1
            if draft_stream_count >= self.speculative_k : # Simulate getting k "chunks" as k "tokens"
                break

        print("\n[SpeculativeRunner] Switching to target_runner for validation and continuation...")
        # Pass the original prompt to target. A real implementation would also pass draft tokens.
        for chunk in self.target_runner.stream(prompt, **kwargs): # Simplified: target starts from original prompt
            yield chunk

        yield " [EndOfSpecStream]"
        print("[SpeculativeRunner] Streaming complete.")


    def export_kv_cache(self) -> t.Any:
        print("[SpeculativeRunner] export_kv_cache called.")
        # The authoritative KV cache is typically the target runner's.
        # However, one might want to save both if the draft runner is stateful and its state is useful.
        # For this placeholder, just target.
        print("  Delegating export to target_runner.")
        return self.target_runner.export_kv_cache()

    def import_kv_cache(self, cache_data: t.Any) -> None:
        print("[SpeculativeRunner] import_kv_cache called.")
        # If cache_data is a single object, it's likely for the target_runner.
        # If it's structured (e.g., a dict), it could contain caches for both.
        # Placeholder: Assume it's for the target, and maybe draft if it's simple.
        # A more robust solution would involve structured cache data.
        print("  Importing to target_runner...")
        self.target_runner.import_kv_cache(cache_data) # Or cache_data.get('target_cache')

        # Draft runner might also benefit from a KV cache, especially if it's based on the same architecture.
        # For simplicity, we can try to pass the same cache or a relevant part.
        # If cache_data is structured like {'target_cache': ..., 'draft_cache': ...}
        # draft_cache_data = cache_data.get('draft_cache') if isinstance(cache_data, dict) else cache_data
        print("  Importing to draft_runner (using same data for placeholder)...")
        self.draft_runner.import_kv_cache(cache_data)

    def preload_kv(self, prompt: str, **kwargs) -> None:
        print(f"[SpeculativeRunner] preload_kv called for prompt: '{prompt[:50]}...'.")
        print("  Preloading KV for draft_runner...")
        self.draft_runner.preload_kv(prompt, **kwargs)
        print("  Preloading KV for target_runner...")
        self.target_runner.preload_kv(prompt, **kwargs)

if __name__ == '__main__':
    # A simple EchoRunner for demonstration
    class EchoRunner(BaseRunner):
        def __init__(self, name: str, delay_factor: float = 0.01):
            self.name = name
            self.delay_factor = delay_factor # Not used in this placeholder
            self.kv_cache_data: t.Optional[t.Dict[str, str]] = None # Initialize KV cache
            print(f"[EchoRunner-{self.name}] Initialized.")

        def generate(self, prompt: str, **kwargs) -> str:
            print(f"[EchoRunner-{self.name}] generate called. Prompt: '{prompt[:30]}...'. Using KV: {self.kv_cache_data is not None}")
            # Simulate generating based on prompt
            response = f"Echo-{self.name}: {prompt}"
            self.kv_cache_data = {f'{self.name}_last_prompt_summary': prompt[:10]} # Update mock KV
            return response

        def stream(self, prompt: str, **kwargs) -> t.Generator[str, None, None]:
            print(f"[EchoRunner-{self.name}] stream called. Prompt: '{prompt[:30]}...'. Using KV: {self.kv_cache_data is not None}")
            self.kv_cache_data = {f'{self.name}_last_stream_summary': prompt[:10]}
            yield f"EchoStream-{self.name}: "
            for char_idx, char_val in enumerate(prompt):
                if char_idx < 15 : # Stream first 15 chars
                    yield char_val
                else:
                    break
            yield " ... (end of echo stream)"

        def export_kv_cache(self) -> t.Any:
            print(f"[EchoRunner-{self.name}] export_kv_cache called. Returning: {self.kv_cache_data}")
            return self.kv_cache_data

        def import_kv_cache(self, cache_data: t.Any) -> None:
            print(f"[EchoRunner-{self.name}] import_kv_cache called with: {str(cache_data)[:100]}...")
            if isinstance(cache_data, dict):
                self.kv_cache_data = cache_data
            else:
                self.kv_cache_data = {"imported_generic_data": str(cache_data)[:50]}


        def preload_kv(self, prompt: str, **kwargs) -> None:
            print(f"[EchoRunner-{self.name}] preload_kv called with prompt: '{prompt[:30]}...'")
            self.kv_cache_data = {f'{self.name}_preloaded_prefix': prompt[:20]}
            print(f"[EchoRunner-{self.name}] KV cache is now: {self.kv_cache_data}")

    print("--- Testing SpeculativeRunner --- ")
    draft_model = EchoRunner(name="DraftModel")
    target_model = EchoRunner(name="TargetModel")

    spec_runner = SpeculativeRunner(draft_runner=draft_model, target_runner=target_model, speculative_k=3)

    print("\n--- Generate Demo ---")
    test_prompt_gen = "This is a test prompt for speculative generation."
    output = spec_runner.generate(test_prompt_gen)
    print(f"Final Generate Output from SpeculativeRunner: '{output}'")
    # Expected: Echo-TargetModel: This is a test prompt for speculative generation.draft1 draft2 draft3
    # (since EchoRunner echoes the full prompt it receives)

    print("\n--- Stream Demo ---")
    test_prompt_stream = "This is a test prompt for speculative streaming."
    full_streamed_output = []
    for chunk in spec_runner.stream(test_prompt_stream):
        print(chunk, end='')
        full_streamed_output.append(chunk)
    print("\nFull streamed output from SpeculativeRunner:", "".join(full_streamed_output))
    print("\n")


    print("\n--- KV Cache Demo ---")
    preload_text = "Common system prefix for KV cache."
    spec_runner.preload_kv(preload_text)
    # Check if individual runners' caches were populated by preload
    print(f"Draft model KV after spec_runner.preload_kv: {draft_model.kv_cache_data}")
    print(f"Target model KV after spec_runner.preload_kv: {target_model.kv_cache_data}")
    self.assertIsNotNone(draft_model.kv_cache_data) # Using self for unittest style, but this is __main__
    self.assertIsNotNone(target_model.kv_cache_data)


    exported_cache = spec_runner.export_kv_cache() # Should be target_model's cache
    print(f"Exported cache from SpeculativeRunner (target's cache): {exported_cache}")
    if exported_cache:
         assert exported_cache.get(f'{target_model.name}_preloaded_prefix') == preload_text[:20]


    # Simulate clearing and importing
    print("\nImporting new cache back into SpeculativeRunner...")
    # For current simple import, it passes the whole thing to both
    new_cache_data_target = {'TargetModel_imported_key': 'TargetValue', 'common_key': 'common_target_val'}
    new_cache_data_draft = {'DraftModel_imported_key': 'DraftValue', 'common_key': 'common_draft_val'}

    # In a real scenario, export_kv_cache might return a dict like {'target_cache': ..., 'draft_cache': ...}
    # and import_kv_cache would expect a similar structure.
    # For this placeholder, we'll test importing a cache intended for the target,
    # and see how both runners handle it with the current simple passthrough.
    spec_runner.import_kv_cache(new_cache_data_target)
    print(f"Draft model KV after import: {draft_model.kv_cache_data}") # Will get target's data
    print(f"Target model KV after import: {target_model.kv_cache_data}") # Will get target's data
    assert target_model.kv_cache_data == new_cache_data_target
    assert draft_model.kv_cache_data == new_cache_data_target # Due to simple passthrough

    # Test generate again to see if EchoRunners report using KV (they should)
    spec_runner.generate("Another prompt after cache import.")

    print("\nSpeculativeRunner demonstration complete.")

# Add assertions to __main__ for clarity if running as script
if __name__ == '__main__':
    class MainTests(unittest.TestCase): # Wrap __main__ in a test class to use asserts
        def run_main_demo(self):
            print("--- Testing SpeculativeRunner (within MainTests) --- ")
            draft_model = EchoRunner(name="DraftModel")
            target_model = EchoRunner(name="TargetModel")

            spec_runner = SpeculativeRunner(draft_runner=draft_model, target_runner=target_model, speculative_k=3)

            print("\n--- Generate Demo ---")
            test_prompt_gen = "This is a test prompt for speculative generation."
            output = spec_runner.generate(test_prompt_gen)
            print(f"Final Generate Output from SpeculativeRunner: '{output}'")
            self.assertIn("Echo-TargetModel", output)
            self.assertIn(test_prompt_gen, output)
            self.assertIn("draft1 draft2 draft3", output)


            print("\n--- Stream Demo ---")
            test_prompt_stream = "This is a test prompt for speculative streaming."
            full_streamed_output = []
            for chunk in spec_runner.stream(test_prompt_stream):
                # print(chunk, end='') # Already printed by spec_runner/echorunner
                full_streamed_output.append(chunk)
            # print("\nFull streamed output from SpeculativeRunner:", "".join(full_streamed_output))
            self.assertTrue(any("EchoStream-DraftModel" in s for s in full_streamed_output))
            self.assertTrue(any("EchoStream-TargetModel" in s for s in full_streamed_output))
            self.assertTrue(any("[EndOfSpecStream]" in s for s in full_streamed_output))


            print("\n--- KV Cache Demo ---")
            preload_text = "Common system prefix for KV cache."
            spec_runner.preload_kv(preload_text)

            self.assertIsNotNone(draft_model.kv_cache_data)
            self.assertEqual(draft_model.kv_cache_data.get(f'{draft_model.name}_preloaded_prefix'), preload_text[:20])
            self.assertIsNotNone(target_model.kv_cache_data)
            self.assertEqual(target_model.kv_cache_data.get(f'{target_model.name}_preloaded_prefix'), preload_text[:20])

            exported_cache = spec_runner.export_kv_cache()
            self.assertEqual(exported_cache, target_model.kv_cache_data)

            print("\nImporting new cache back into SpeculativeRunner...")
            new_cache_data_for_import = {'imported_key': 'imported_value_for_both'}
            spec_runner.import_kv_cache(new_cache_data_for_import)
            self.assertEqual(draft_model.kv_cache_data, new_cache_data_for_import)
            self.assertEqual(target_model.kv_cache_data, new_cache_data_for_import)

            spec_runner.generate("Another prompt after cache import.")
            print("\nSpeculativeRunner demonstration complete.")

    # Due to the tool environment, directly running unittest.main() inside __main__
    # after defining classes might be tricky or not show output well.
    # The print statements serve as basic visual checks for now.
    # For actual CI/testing, these would be proper unittest classes in test files.
    # Running the demo part:
    if __name__ == '__main__': # Re-check for clarity, original was fine
        demo = MainTests()
        demo.run_main_demo()
