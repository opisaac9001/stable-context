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
        print(f"[SpeculativeRunner] generate called. Prompt: '{prompt[:50]}...', k: {self.speculative_k}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: Speculative runner passes images to both draft and target)")

        # Determine max_new_tokens, providing a default if not in kwargs
        # Check if target_runner and its model exist and have n_ctx method for a smarter default
        default_max_tokens = 50
        if hasattr(self.target_runner, 'model') and self.target_runner.model and hasattr(self.target_runner.model, 'n_ctx') and callable(self.target_runner.model.n_ctx):
            try:
                default_max_tokens = self.target_runner.model.n_ctx() // 4
            except Exception: # pragma: no cover
                pass # Keep default_max_tokens = 50 if n_ctx() fails

        max_new_tokens = kwargs.get('max_new_tokens', default_max_tokens)

        draft_gen_params = kwargs.copy()
        draft_gen_params.pop('stop', None)

        target_gen_params = kwargs.copy()

        generated_tokens_list = [] # Stores strings of accepted tokens from each iteration
        current_prompt_for_runners = prompt # This is the prompt passed to draft/target for their internal generation state

        # This will hold the full generated sequence including the initial prompt
        full_generated_sequence_so_far = prompt
        total_words_generated = 0 # Using word count as a proxy for token count

        eos_token_str = "</s>" # Default EOS, should be from tokenizer
        if hasattr(self.target_runner, 'tokenizer') and self.target_runner.tokenizer and hasattr(self.target_runner.tokenizer, 'eos_token') and self.target_runner.tokenizer.eos_token:
            eos_token_str = self.target_runner.tokenizer.eos_token

        # Max iterations safeguard: max_new_tokens iterations (roughly one token per iter) + k for final draft
        for iteration_count in range(max_new_tokens + self.speculative_k):
            if total_words_generated >= max_new_tokens:
                print(f"[SpeculativeRunner] Reached max_new_tokens ({max_new_tokens} words). Iteration: {iteration_count}")
                break

            # 1. Draft Phase
            try:
                # Use full_generated_sequence_so_far for draft model to build upon previous step's accepted tokens
                draft_tokens_str = self.draft_runner.generate(
                    full_generated_sequence_so_far, # Draft from the latest confirmed state
                    image_paths=image_paths,
                    max_new_tokens=self.speculative_k,
                    **draft_gen_params
                )
                if not draft_tokens_str.strip():
                    print(f"[SpeculativeRunner] Draft model produced no output or only whitespace. Iteration: {iteration_count}")
                    # If target also produces nothing from current_prompt_for_runners, then finish.
                    # For now, we'll let target try to generate from current_prompt_for_runners.
                    # If draft is empty, target will effectively do normal generation for 1 token.
                    draft_tokens_for_comp = [] # Ensure it's an empty list
                else:
                    if eos_token_str and eos_token_str in draft_tokens_str:
                        print(f"[SpeculativeRunner] Draft model produced EOS. Content: '{draft_tokens_str}'. Iteration: {iteration_count}")
                        # Process draft tokens up to EOS for this iteration's comparison
                        draft_tokens_str = draft_tokens_str.split(eos_token_str, 1)[0]
                        # No + eos_token_str, because target will confirm/generate its own EOS
                    draft_tokens_for_comp = draft_tokens_str.strip().split()

            except Exception as e: # pragma: no cover
                print(f"[SpeculativeRunner] Error during draft generation: {e}. Finishing. Iteration: {iteration_count}")
                break

            # 2. Validation Phase
            target_candidate_words = []
            try:
                # Target generates from the same point as draft did (full_generated_sequence_so_far)
                target_candidate_tokens_str = self.target_runner.generate(
                    full_generated_sequence_so_far,
                    image_paths=image_paths,
                    max_new_tokens=self.speculative_k + 1, # Target needs to verify k and potentially provide one more
                    **target_gen_params
                )
                if not target_candidate_tokens_str.strip():
                    print(f"[SpeculativeRunner] Target model produced no primary output. Iteration: {iteration_count}")
                    # If draft also had nothing, we break. Otherwise, draft might be hallucinating.
                    if not draft_tokens_for_comp:
                        break # Both are empty, nothing more to generate
                else:
                    if eos_token_str and eos_token_str in target_candidate_tokens_str:
                         print(f"[SpeculativeRunner] Target model produced EOS. Content: '{target_candidate_tokens_str}'. Iteration: {iteration_count}")
                         target_candidate_tokens_str = target_candidate_tokens_str.split(eos_token_str,1)[0] + eos_token_str # Keep EOS
                    target_candidate_words = target_candidate_tokens_str.strip().split()

            except Exception as e: # pragma: no cover
                print(f"[SpeculativeRunner] Error during target validation generation: {e}. Finishing. Iteration: {iteration_count}")
                break

            num_matched = 0
            if draft_tokens_for_comp and target_candidate_words:
                for i in range(min(len(draft_tokens_for_comp), len(target_candidate_words))):
                    if draft_tokens_for_comp[i] == target_candidate_words[i]:
                        num_matched += 1
                    else:
                        break

            final_accepted_words_this_iter = []
            made_progress_this_iteration = False

            if num_matched > 0:
                # Accept the matched part from target (it's verified)
                final_accepted_words_this_iter.extend(target_candidate_words[:num_matched])
                made_progress_this_iteration = True

            # Add the (num_matched + 1)th token from target model, if it exists
            if num_matched < len(target_candidate_words):
                final_accepted_words_this_iter.append(target_candidate_words[num_matched])
                made_progress_this_iteration = True
            elif not target_candidate_words and num_matched == 0: # Target is empty, draft might also be empty or hallucinatory
                # If draft_tokens_for_comp is also empty, loop will break due to no progress.
                # If draft_tokens_for_comp is not empty, it means draft was hallucinating, num_matched is 0.
                # No tokens from target to add.
                pass


            if not made_progress_this_iteration:
                print(f"[SpeculativeRunner] No tokens accepted in this iteration ({iteration_count}). Finishing.")
                break

            accepted_str_this_iter = " ".join(final_accepted_words_this_iter)

            # Append to the list of generated segments
            generated_tokens_list.append(accepted_str_this_iter)

            # Update full_generated_sequence_so_far for the next iteration's input
            # Add space only if full_generated_sequence_so_far is not empty and does not end with space,
            # and accepted_str_this_iter is not empty.
            if full_generated_sequence_so_far and \
               not full_generated_sequence_so_far.endswith(tuple(' \n\t')) and \
               accepted_str_this_iter:
                full_generated_sequence_so_far += " "
            full_generated_sequence_so_far += accepted_str_this_iter

            total_words_generated += len(final_accepted_words_this_iter)

            if eos_token_str and accepted_str_this_iter.endswith(eos_token_str):
                print(f"[SpeculativeRunner] EOS token accepted. Iteration: {iteration_count}. Finishing.")
                break

        # Construct the final output string from the initially passed prompt
        newly_generated_text_segments = []
        temp_full_text = prompt

        for segment in generated_tokens_list:
            if temp_full_text and not temp_full_text.endswith(tuple(' \n\t')) and segment:
                newly_generated_text_segments.append(" ")
                temp_full_text += " "
            newly_generated_text_segments.append(segment)
            temp_full_text += segment

        final_output_str = "".join(newly_generated_text_segments)

        # Remove potential leading space if original prompt was empty or ended with space
        # and the first segment started with a space (unlikely with current logic but good to be safe)
        # More importantly, strip any final EOS token if it's part of the string,
        # as typical generation calls expect text without the EOS marker.
        if eos_token_str and final_output_str.endswith(eos_token_str):
            final_output_str = final_output_str[:-len(eos_token_str)]

        return final_output_str.strip()


    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> t.Generator[str, None, None]:
        print(f"[SpeculativeRunner] stream called. Prompt: '{prompt[:50]}...', k: {self.speculative_k}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: Speculative runner passes images to both draft and target)")

        # Determine max_new_tokens, providing a default if not in kwargs
        default_max_tokens = 50
        if hasattr(self.target_runner, 'model') and self.target_runner.model and hasattr(self.target_runner.model, 'n_ctx') and callable(self.target_runner.model.n_ctx):
            try:
                default_max_tokens = self.target_runner.model.n_ctx() // 4
            except Exception: # pragma: no cover
                pass
        max_new_tokens = kwargs.get('max_new_tokens', default_max_tokens)

        draft_gen_params = kwargs.copy()
        draft_gen_params.pop('stop', None)

        target_gen_params = kwargs.copy()

        current_prompt_for_iteration = prompt
        total_words_yielded = 0

        eos_token_str = "</s>" # Default EOS
        if hasattr(self.target_runner, 'tokenizer') and self.target_runner.tokenizer and hasattr(self.target_runner.tokenizer, 'eos_token') and self.target_runner.tokenizer.eos_token:
            eos_token_str = self.target_runner.tokenizer.eos_token
        found_eos = False

        for _iter_num in range(max_new_tokens + self.speculative_k):
            if total_words_yielded >= max_new_tokens or found_eos:
                break

            # 1. Draft Phase
            draft_full_chunk_str = ""
            try:
                draft_full_chunk_str = self.draft_runner.generate(
                    current_prompt_for_iteration,
                    image_paths=image_paths,
                    max_new_tokens=self.speculative_k,
                    **draft_gen_params
                )
                # Check for EOS or empty content after stripping
                stripped_draft_chunk = draft_full_chunk_str.strip()
                if eos_token_str and eos_token_str in draft_full_chunk_str: # Check raw string for EOS
                    # If EOS is present, take content up to it for comparison
                    draft_full_chunk_str = draft_full_chunk_str.split(eos_token_str, 1)[0]
                    stripped_draft_chunk = draft_full_chunk_str.strip() # Re-strip after split
                    # Don't set found_eos here, let target confirm.

                if not stripped_draft_chunk:
                    print("[SpeculativeRunner Stream] Draft model produced no actual content.")
                    # Fall through to target generation. draft_words_for_comp will be empty.
            except Exception as e: # pragma: no cover
                print(f"[SpeculativeRunner Stream] Error during draft generation: {e}. Attempting target generation.")

            draft_words_for_comp = draft_full_chunk_str.strip().split()

            # 2. Validation Phase
            target_candidate_str = ""
            try:
                target_candidate_str = self.target_runner.generate(
                    current_prompt_for_iteration,
                    image_paths=image_paths,
                    max_new_tokens=self.speculative_k + 1,
                    **target_gen_params
                )
                stripped_target_candidate = target_candidate_str.strip()
                if eos_token_str and eos_token_str in target_candidate_str: # Check raw string for EOS
                    # If EOS is present, take content up to it for comparison, but keep EOS for yielding
                    target_candidate_str = target_candidate_str.split(eos_token_str, 1)[0] + eos_token_str
                    stripped_target_candidate = target_candidate_str.strip()
                    # Don't set found_eos here yet, will be checked after yielding accepted block.

                if not stripped_target_candidate:
                     print("[SpeculativeRunner Stream] Target model produced no output. Finishing.")
                     found_eos = True; break
            except Exception as e: # pragma: no cover
                print(f"[SpeculativeRunner Stream] Error during target generation: {e}. Finishing.")
                found_eos = True; break

            target_candidate_words = target_candidate_str.strip().split() # Split after potential EOS handling

            # 3. Comparison & Acceptance
            num_matched = 0
            if draft_words_for_comp and target_candidate_words:
                for i in range(min(len(draft_words_for_comp), len(target_candidate_words))):
                    # Be careful if target_candidate_words[i] is EOS and draft is not.
                    if draft_words_for_comp[i] == target_candidate_words[i]:
                        if eos_token_str and target_candidate_words[i] == eos_token_str and i < len(draft_words_for_comp)-1:
                            # Draft continued after target EOS'd at this matched token.
                            # This means only tokens up to EOS are truly matched.
                            num_matched +=1 # count the EOS as matched
                            break # stop matching here
                        num_matched += 1
                    else:
                        break

            accepted_words_this_block = []
            if num_matched > 0:
                accepted_words_this_block.extend(target_candidate_words[:num_matched])

            if num_matched < len(target_candidate_words):
                accepted_words_this_block.append(target_candidate_words[num_matched])
            elif not target_candidate_words: # Target had nothing
                if draft_words_for_comp: # Draft was hallucinating
                    print("[SpeculativeRunner Stream] Target empty, draft hallucinated. No tokens accepted.")
                else: # Both empty
                    print("[SpeculativeRunner Stream] Both draft and target produced no new words. Finishing.")
                    found_eos = True # Effectively EOS or stuck
                # Break from outer loop if found_eos is true after this block
                if found_eos: break


            if not accepted_words_this_block:
                print("[SpeculativeRunner Stream] No words accepted in this block. Finishing.")
                break

            # 4. Yield accepted tokens for this block and update state
            block_str_to_yield = " ".join(accepted_words_this_block)

            # Determine if a leading space is needed before yielding the block
            prefix_space = ""
            if total_words_yielded > 0: # Not the very first block of the entire generation
                if current_prompt_for_iteration and not current_prompt_for_iteration.endswith(tuple(' \n\t')):
                    prefix_space = " "

            yield prefix_space + block_str_to_yield

            # Update current_prompt_for_iteration
            current_prompt_for_iteration += prefix_space + block_str_to_yield
            total_words_yielded += len(accepted_words_this_block)

            # Check for EOS in the yielded block
            if eos_token_str:
                # Check if any word in accepted_words_this_block *is* the EOS token,
                # or if the joined block_str_to_yield *ends with* the EOS token.
                # The latter is more robust if EOS can be part of a larger "word" from split().
                if block_str_to_yield.endswith(eos_token_str):
                    print(f"[SpeculativeRunner Stream] EOS token detected at end of yielded block: '{block_str_to_yield}'.")
                    found_eos = True
                    break # Break from iteration loop

        print("[SpeculativeRunner Stream] Streaming finished.")

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
    from unittest.mock import MagicMock # Required for the mock tokenizer in EchoRunner
    import time # For potential delays in EchoRunner stream, though not used in final provided example

    class EchoRunner(BaseRunner):
        def __init__(self, name: str, delay_factor: float = 0.001, token_prefix: str = ""):
            self.name = name
            self.delay_factor = delay_factor
            self.token_prefix = token_prefix
            self.kv_cache_data: t.Optional[t.Dict[str, str]] = None
            self.active_loras: t.Dict[str, t.Any] = {}
            self.is_merged_state: bool = False
            self._eos_token = "</s>"
            print(f"[EchoRunner-{self.name}] Initialized with token_prefix='{self.token_prefix}'.")

        @property
        def tokenizer(self):
            mock_tok = MagicMock()
            mock_tok.eos_token = self._eos_token
            return mock_tok

        def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> str:
            max_new = kwargs.get('max_new_tokens', 3)
            img_info = f" (Images: {image_paths})" if image_paths else ""
            # Minimal print to avoid clutter during spec demo
            # print(f"[EchoRunner-{self.name}] generate. Prompt: '{prompt[-50:]}...'{img_info}, MaxNew: {max_new}")

            if prompt.strip().endswith(self.token_prefix + self._eos_token): # Check for its own EOS
                 return self.token_prefix + self._eos_token
            if prompt.strip().endswith(self._eos_token): # Check for generic EOS
                 return self.token_prefix + self._eos_token

            words = []
            for i in range(max_new):
                current_word = f"{self.token_prefix}w{i+1}"
                # Simulate EOS if prompt is long or specific content
                if "eos_please" in prompt.lower() and i >= 1: # EOS as second word if "eos_please"
                    current_word = self.token_prefix + self._eos_token
                    words.append(current_word)
                    break
                words.append(current_word)

            response = " ".join(words)
            return response

        def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs) -> t.Generator[str, None, None]:
            max_new = kwargs.get('max_new_tokens', 3)
            img_info = f" (Images: {image_paths})" if image_paths else ""
            # print(f"[EchoRunner-{self.name}] stream. Prompt: '{prompt[-50:]}...'{img_info}, MaxNew: {max_new}")

            if prompt.strip().endswith(self.token_prefix + self._eos_token):
                yield (" " if prompt else "") + self.token_prefix + self._eos_token # Add leading space if not first
                return
            if prompt.strip().endswith(self._eos_token):
                 yield (" " if prompt else "") + self.token_prefix + self._eos_token
                 return

            for i in range(max_new):
                word = f"{self.token_prefix}s_w{i+1}"
                is_first_word_in_stream = (i == 0)

                if "eos_please" in prompt.lower() and i >= 1: # EOS as second word
                    yield (" " if not is_first_word_in_stream else "") + self.token_prefix + self._eos_token
                    return

                yield (" " if not is_first_word_in_stream else "") + word

        # --- Minimal implementations for other abstract methods ---
        def count_tokens(self, text:str, **kwargs) -> t.Optional[int]: return len(text.split())
        def export_kv_cache(self) -> t.Any: print(f"[{self.name}] export_kv_cache"); return None
        def import_kv_cache(self, cache_data: t.Any) -> None: print(f"[{self.name}] import_kv_cache")
        def preload_kv(self, prompt: str, **kwargs) -> None: print(f"[{self.name}] preload_kv for '{prompt[:20]}...'")
        def load_lora_adapter(self, adapter_id: str, adapter_path: str, **kwargs) -> bool: return True
        def unload_lora_adapter(self, adapter_id: str, **kwargs) -> bool: return True
        def get_active_lora_adapters(self) -> t.List[str]: return []
        def merge_lora_adapters(self, adapter_ids: t.List[str], **kwargs) -> bool: return True
        def unmerge_lora_adapters(self, **kwargs) -> bool: return True

    print("--- Testing SpeculativeRunner ---")
    # Use token prefixes to distinguish draft and target model outputs
    draft_model = EchoRunner(name="Draft", token_prefix="D_")
    target_model = EchoRunner(name="Target", token_prefix="T_")

    # Speculative_k=3 means draft model suggests 3 tokens (words in this EchoRunner)
    spec_runner = SpeculativeRunner(draft_runner=draft_model, target_runner=target_model, speculative_k=3)

    # 1. Demonstrate generate
    initial_prompt = "The quick brown fox"
    print(f"\n--- Generate Demo ---")
    print(f"Initial Prompt: '{initial_prompt}'")
    # max_new_tokens is for the SpeculativeRunner itself (total new tokens to generate)
    generated_text = spec_runner.generate(initial_prompt, max_new_tokens=7)
    print(f"Speculative Output: '{initial_prompt}{' ' if generated_text else ''}{generated_text}'")
    # Expected: A mix of T_wX tokens, possibly D_wX if they matched.
    # Example: Initial: "The quick brown fox"
    # Output: "The quick brown fox T_w1 T_w2 T_w3 T_w4 T_w5 T_w6 T_w7" (if draft always wrong or k=0)
    # With k=3, draft generates 3 words. Target generates up to 4.
    # If draft "D_w1 D_w2 D_w3" matches target's first 3 "T_w1 T_w2 T_w3",
    # then accepted is "T_w1 T_w2 T_w3" and then target's 4th "T_w4". Total 4 words this step.
    # This repeats. The exact output depends on the simplified matching logic in SpeculativeRunner.

    # 2. Demonstrate generate with EOS
    prompt_for_eos = "Tell me a story and eos_please" # EchoRunner will EOS after "eos_please" + 1 word
    print(f"\n--- Generate Demo with EOS ---")
    print(f"Prompt for EOS: '{prompt_for_eos}'")
    generated_text_eos = spec_runner.generate(prompt_for_eos, max_new_tokens=7)
    print(f"Speculative Output (EOS): '{prompt_for_eos}{' ' if generated_text_eos else ''}{generated_text_eos}'")
    # Expected: Output should stop around where EchoRunner produced EOS, e.g., "T_w1 T_</s>"

    # 3. Demonstrate stream
    stream_prompt = "The lazy dog"
    print(f"\n--- Stream Demo ---")
    print(f"Stream Prompt: '{stream_prompt}'")
    print("Streaming output: ", end="")
    full_streamed_text_list = []
    # Keep stream max_new_tokens also relatively small for demo
    for chunk in spec_runner.stream(stream_prompt, max_new_tokens=7):
        print(chunk, end="", flush=True)
        full_streamed_text_list.append(chunk)
    print("\nEnd of stream.")
    full_streamed_text = "".join(full_streamed_text_list)
    # print(f"Full streamed text via SpeculativeRunner: '{stream_prompt}{full_streamed_text}'") # full_streamed_text is already the new part

    # 4. Demonstrate generate with image paths (conceptual, EchoRunner just prints them)
    image_prompt = "Describe this image:"
    example_images = ["/path/to/image1.jpg"]
    print(f"\n--- Generate with Image Paths Demo ---")
    print(f"Image Prompt: '{image_prompt}', Images: {example_images}")
    generated_text_img = spec_runner.generate(image_prompt, image_paths=example_images, max_new_tokens=4)
    print(f"Speculative Output (Image): '{image_prompt}{' ' if generated_text_img else ''}{generated_text_img}'")
    # EchoRunner's generate will append image_paths info to its output,
    # so we expect to see that in the final result from SpeculativeRunner.
    # The SpeculativeRunner itself prints that it passes images to both runners.

    print("\n\nSpeculativeRunner __main__ demo complete.")
