# yawl/runners/speculative_runner.py
import typing as t
from .base import BaseRunner

class SpeculativeRunner(BaseRunner):
    def __init__(self, draft_runner: BaseRunner, target_runner: BaseRunner, speculative_k: int = 5, **kwargs):
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.speculative_k = speculative_k
        print(f"[SpeculativeRunner] Initialized with draft_runner: {type(draft_runner).__name__}, "
              f"target_runner: {type(target_runner).__name__}, k={speculative_k}")

    def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str, int]]:
        print(f"[SpeculativeRunner] generate called. Prompt: '{prompt[:50]}...', k: {self.speculative_k}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: Speculative runner passes images to both draft and target)")

        # Determine max_new_tokens, providing a default if not in kwargs
        # Check if target_runner and its model exist and have n_ctx method for a smarter default
        default_max_tokens = 50

        # Placeholder for actual token counting for prompt for now
        prompt_tokens_count = len(prompt.split()) # Simple word count as a proxy

        # if hasattr(self.target_runner, 'model') and self.target_runner.model and hasattr(self.target_runner.model, 'n_ctx') and callable(self.target_runner.model.n_ctx):
        #     try:
        #         default_max_tokens = self.target_runner.model.n_ctx() // 4
        #     except Exception: # pragma: no cover
        #         pass # Keep default_max_tokens = 50 if n_ctx() fails
        # elif hasattr(self.target_runner, 'count_tokens') and callable(self.target_runner.count_tokens):
        #     # This is a rough way to estimate remaining context; real context length is needed
        #     prompt_tokens_for_target = self.target_runner.count_tokens(prompt)
        #     if prompt_tokens_for_target is not None and prompt_tokens_for_target > 0:
        #          # Assuming a generic max context of 2048 if model specific not found
        #          default_max_tokens = (2048 - prompt_tokens_for_target) // 2
        #          if default_max_tokens < 10: default_max_tokens = 10 # Ensure some generation
        #     prompt_tokens_count = prompt_tokens_for_target or prompt_tokens_count


        max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_tokens', default_max_tokens))
        if max_new_tokens <=0: max_new_tokens = default_max_tokens # Ensure positive

        draft_gen_params = kwargs.copy()
        draft_gen_params.pop('stop', None)

        target_gen_params = kwargs.copy()

        generated_tokens_list = []
        full_generated_sequence_so_far = prompt
        total_completion_tokens_count = 0

        eos_token_str = "</s>"
        if hasattr(self.target_runner, 'tokenizer') and self.target_runner.tokenizer and hasattr(self.target_runner.tokenizer, 'eos_token') and self.target_runner.tokenizer.eos_token:
            eos_token_str = self.target_runner.tokenizer.eos_token

        final_eos_detected = False

        # Max iterations safeguard
        # Each iteration aims to accept at least one token from target, up to k+1.
        # So, roughly max_new_tokens iterations needed in the best case.
        # Add some buffer for safety, e.g. max_new_tokens * 2 or max_new_tokens + some_fixed_amount
        max_iterations = max_new_tokens + self.speculative_k * 2

        for iteration_count in range(max_iterations):
            if total_completion_tokens_count >= max_new_tokens:
                print(f"[SpeculativeRunner] Reached target max_new_tokens ({max_new_tokens}). Iteration: {iteration_count}")
                break

            # 1. Draft Phase
            draft_text_segment = ""
            try:
                # Draft from the latest confirmed state
                draft_response = self.draft_runner.generate(
                    full_generated_sequence_so_far,
                    image_paths=image_paths,
                    max_new_tokens=self.speculative_k, # Draft k tokens
                    **draft_gen_params
                )
                draft_text_segment = draft_response[0] if isinstance(draft_response, tuple) else draft_response

                if not draft_text_segment.strip():
                    print(f"[SpeculativeRunner] Draft model produced no output. Iteration: {iteration_count}")
                if eos_token_str and eos_token_str in draft_text_segment:
                    draft_text_segment = draft_text_segment.split(eos_token_str, 1)[0]
            except Exception as e:
                print(f"[SpeculativeRunner] Error during draft generation: {e}. Iteration: {iteration_count}")
                # Don't break yet, let target try to generate from current full_generated_sequence_so_far

            draft_tokens_for_comp = draft_text_segment.strip().split() # Simple word tokenization for comparison

            # 2. Validation Phase
            target_candidate_text = ""
            try:
                target_response = self.target_runner.generate(
                    full_generated_sequence_so_far,
                    image_paths=image_paths,
                    max_new_tokens=len(draft_tokens_for_comp) + 1, # Target verifies k and provides one more
                    **target_gen_params
                )
                target_candidate_text = target_response[0] if isinstance(target_response, tuple) else target_response

                if not target_candidate_text.strip():
                    print(f"[SpeculativeRunner] Target model produced no output. Iteration: {iteration_count}")
                    if not draft_tokens_for_comp: final_eos_detected = True; break # Both empty
            except Exception as e:
                print(f"[SpeculativeRunner] Error during target validation generation: {e}. Iteration: {iteration_count}")
                final_eos_detected = True; break

            target_candidate_words = target_candidate_text.strip().split()

            num_matched = 0
            if draft_tokens_for_comp and target_candidate_words:
                for i in range(min(len(draft_tokens_for_comp), len(target_candidate_words))):
                    if draft_tokens_for_comp[i] == target_candidate_words[i]:
                        if eos_token_str and target_candidate_words[i] == eos_token_str and i < len(draft_tokens_for_comp)-1:
                            num_matched +=1; final_eos_detected = True; break
                        num_matched += 1
                    else:
                        break

            accepted_words_this_iteration = []
            if num_matched > 0:
                accepted_words_this_iteration.extend(target_candidate_words[:num_matched])

            if not final_eos_detected and num_matched < len(target_candidate_words):
                # Accept the (n+1)th token from target if no EOS yet in matched part
                accepted_words_this_iteration.append(target_candidate_words[num_matched])
                if eos_token_str and target_candidate_words[num_matched] == eos_token_str:
                    final_eos_detected = True
            elif not target_candidate_words and not draft_tokens_for_comp : # Both empty
                 final_eos_detected = True


            if not accepted_words_this_iteration:
                print(f"[SpeculativeRunner] No tokens accepted in this iteration ({iteration_count}). Finishing.")
                break

            accepted_segment_str = " ".join(accepted_words_this_iteration)
            generated_tokens_list.append(accepted_segment_str)

            # Update full_generated_sequence_so_far for the next iteration's input
            if full_generated_sequence_so_far and not full_generated_sequence_so_far.endswith(tuple(' \n\t')) and accepted_segment_str:
                full_generated_sequence_so_far += " "
            full_generated_sequence_so_far += accepted_segment_str

            # Use target_runner's tokenizer if available for more accurate token counting
            if hasattr(self.target_runner, 'count_tokens') and callable(self.target_runner.count_tokens):
                segment_token_count = self.target_runner.count_tokens(accepted_segment_str)
                total_completion_tokens_count += segment_token_count if segment_token_count is not None else len(accepted_words_this_iteration)
            else:
                total_completion_tokens_count += len(accepted_words_this_iteration)


            if final_eos_detected:
                print(f"[SpeculativeRunner] EOS token confirmed by target or end of target output. Iteration: {iteration_count}. Finishing.")
                break

        final_output_str = " ".join(generated_tokens_list)
        if eos_token_str and final_output_str.endswith(eos_token_str): # Remove trailing EOS
            final_output_str = final_output_str[:-len(eos_token_str)].rstrip()

        return final_output_str, {"prompt_tokens": prompt_tokens_count, "completion_tokens": total_completion_tokens_count}


    def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]:
        print(f"[SpeculativeRunner] stream called. Prompt: '{prompt[:50]}...', k: {self.speculative_k}")
        if image_paths:
            print(f"  Image Paths: {image_paths} (Note: Speculative runner passes images to both draft and target)")

        default_max_tokens = 50
        prompt_tokens_count = len(prompt.split()) # Simple word count for now

        if hasattr(self.target_runner, 'count_tokens') and callable(self.target_runner.count_tokens):
            pt_count = self.target_runner.count_tokens(prompt)
            if pt_count is not None: prompt_tokens_count = pt_count

        yield {"prompt_tokens": prompt_tokens_count}


        max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_tokens', default_max_tokens))
        if max_new_tokens <=0: max_new_tokens = default_max_tokens

        draft_gen_params = kwargs.copy(); draft_gen_params.pop('stop', None)
        target_gen_params = kwargs.copy()

        current_prompt_for_iteration = prompt
        total_yielded_tokens_count = 0

        eos_token_str = "</s>"
        if hasattr(self.target_runner, 'tokenizer') and self.target_runner.tokenizer and hasattr(self.target_runner.tokenizer, 'eos_token') and self.target_runner.tokenizer.eos_token:
            eos_token_str = self.target_runner.tokenizer.eos_token
        final_eos_detected = False

        max_iterations = max_new_tokens + self.speculative_k * 2

        for iteration_count in range(max_iterations):
            if total_yielded_tokens_count >= max_new_tokens or final_eos_detected:
                break

            draft_text_segment = ""
            try:
                draft_response = self.draft_runner.generate(
                    current_prompt_for_iteration, image_paths=image_paths,
                    max_new_tokens=self.speculative_k, **draft_gen_params
                )
                draft_text_segment = draft_response[0] if isinstance(draft_response, tuple) else draft_response
                if eos_token_str and eos_token_str in draft_text_segment:
                    draft_text_segment = draft_text_segment.split(eos_token_str, 1)[0]
            except Exception as e:
                print(f"[SpeculativeRunner Stream] Error in draft: {e}")

            draft_words_for_comp = draft_text_segment.strip().split()

            target_candidate_text = ""
            try:
                target_response = self.target_runner.generate(
                    current_prompt_for_iteration, image_paths=image_paths,
                    max_new_tokens=len(draft_words_for_comp) + 1, **target_gen_params
                )
                target_candidate_text = target_response[0] if isinstance(target_response, tuple) else target_response
                if not target_candidate_text.strip():
                    if not draft_words_for_comp: final_eos_detected = True; break
            except Exception as e:
                print(f"[SpeculativeRunner Stream] Error in target: {e}")
                final_eos_detected = True; break

            target_candidate_words = target_candidate_text.strip().split()

            num_matched = 0
            if draft_words_for_comp and target_candidate_words:
                for i in range(min(len(draft_words_for_comp), len(target_candidate_words))):
                    if draft_words_for_comp[i] == target_candidate_words[i]:
                        if eos_token_str and target_candidate_words[i] == eos_token_str and i < len(draft_words_for_comp)-1:
                            num_matched +=1; final_eos_detected = True; break
                        num_matched += 1
                    else:
                        break

            accepted_words_this_iteration = []
            if num_matched > 0:
                accepted_words_this_iteration.extend(target_candidate_words[:num_matched])

            if not final_eos_detected and num_matched < len(target_candidate_words):
                accepted_words_this_iteration.append(target_candidate_words[num_matched])
                if eos_token_str and target_candidate_words[num_matched] == eos_token_str:
                    final_eos_detected = True
            elif not target_candidate_words and not draft_words_for_comp:
                 final_eos_detected = True

            if not accepted_words_this_iteration:
                break

            block_str_to_yield = " ".join(accepted_words_this_iteration)
            prefix_space = " " if current_prompt_for_iteration and not current_prompt_for_iteration.endswith(tuple(' \n\t')) and block_str_to_yield else ""

            # Yield chunk by chunk (word by word for this simple setup)
            for i, word in enumerate(accepted_words_this_iteration):
                current_chunk_to_yield = word
                is_first_word_of_block = (i == 0)

                actual_prefix = ""
                if is_first_word_of_block and prefix_space: # Add space only before the first word of a new block
                    actual_prefix = prefix_space
                elif not is_first_word_of_block: # Add space between words within the block
                    actual_prefix = " "

                chunk_with_prefix = actual_prefix + current_chunk_to_yield

                # Estimate tokens for this specific chunk
                chunk_token_count = 0
                if hasattr(self.target_runner, 'count_tokens') and callable(self.target_runner.count_tokens):
                    ct = self.target_runner.count_tokens(chunk_with_prefix) # Count with its prefix for accuracy
                    chunk_token_count = ct if ct is not None else len(chunk_with_prefix.split())
                else:
                    chunk_token_count = len(chunk_with_prefix.split())

                yield (chunk_with_prefix, chunk_token_count)
                total_yielded_tokens_count += chunk_token_count # Or use word count: +=1
                current_prompt_for_iteration += chunk_with_prefix # Update for next draft

                if final_eos_detected and word == eos_token_str: # If EOS was part of this word
                    break

            if final_eos_detected: break

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

    def count_tokens(self, text: str) -> t.Optional[int]:
        # Delegate to target_runner for token counting, as it's the more "authoritative" model
        if hasattr(self.target_runner, 'count_tokens') and callable(self.target_runner.count_tokens):
            return self.target_runner.count_tokens(text)
        return len(text.split()) # Fallback

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

        def generate(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Tuple[str, t.Dict[str,int]]: # Updated return type
            max_new = kwargs.get('max_new_tokens', 3)
            # Minimal print to avoid clutter during spec demo
            # print(f"[EchoRunner-{self.name}] generate. Prompt: '{prompt[-50:]}...'{img_info}, MaxNew: {max_new}")

            if prompt.strip().endswith(self.token_prefix + self._eos_token): # Check for its own EOS
                 return self.token_prefix + self._eos_token, {"prompt_tokens":0, "completion_tokens":1}
            if prompt.strip().endswith(self._eos_token): # Check for generic EOS
                 return self.token_prefix + self._eos_token, {"prompt_tokens":0, "completion_tokens":1}

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
            return response, {"prompt_tokens": len(prompt.split()), "completion_tokens": len(words)}


        def stream(self, prompt: str, image_paths: t.Optional[t.List[str]] = None, **kwargs: t.Any) -> t.Generator[t.Union[t.Dict[str, int], t.Tuple[str, int]], None, None]: # Updated yield type
            max_new = kwargs.get('max_new_tokens', 3)

            prompt_token_count = len(prompt.split())
            yield {"prompt_tokens": prompt_token_count}


            if prompt.strip().endswith(self.token_prefix + self._eos_token):
                yield ((" " if prompt else "") + self.token_prefix + self._eos_token, 1)
                return
            if prompt.strip().endswith(self._eos_token):
                 yield ((" " if prompt else "") + self.token_prefix + self._eos_token, 1)
                 return

            for i in range(max_new):
                word = f"{self.token_prefix}s_w{i+1}"
                is_first_word_in_stream = (i == 0)
                chunk_to_yield = (" " if not is_first_word_in_stream and prompt else "") + word


                if "eos_please" in prompt.lower() and i >= 1: # EOS as second word
                    yield ((" " if not is_first_word_in_stream else "") + self.token_prefix + self._eos_token, 1)
                    return

                yield (chunk_to_yield, 1) # Assume 1 token per word for EchoRunner stream

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
    generated_text, gen_token_counts = spec_runner.generate(initial_prompt, max_new_tokens=7)
    print(f"Speculative Output: '{initial_prompt}{' ' if generated_text else ''}{generated_text}'")
    print(f"Token counts: {gen_token_counts}")
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
    generated_text_eos, eos_token_counts = spec_runner.generate(prompt_for_eos, max_new_tokens=7)
    print(f"Speculative Output (EOS): '{prompt_for_eos}{' ' if generated_text_eos else ''}{generated_text_eos}'")
    print(f"Token counts (EOS): {eos_token_counts}")
    # Expected: Output should stop around where EchoRunner produced EOS, e.g., "T_w1 T_</s>"

    # 3. Demonstrate stream
    stream_prompt = "The lazy dog"
    print(f"\n--- Stream Demo ---")
    print(f"Stream Prompt: '{stream_prompt}'")
    print("Streaming output: ", end="")
    full_streamed_text_list = []
    stream_prompt_tokens = 0
    stream_completion_tokens = 0
    # Keep stream max_new_tokens also relatively small for demo
    stream_gen = spec_runner.stream(stream_prompt, max_new_tokens=7)

    first_item = next(stream_gen)
    if isinstance(first_item, dict) and "prompt_tokens" in first_item:
        stream_prompt_tokens = first_item["prompt_tokens"]
        print(f"(Prompt Tokens: {stream_prompt_tokens}) ", end="")

    for item in stream_gen:
        if isinstance(item, tuple):
            chunk, chunk_tokens = item
            print(chunk, end="", flush=True)
            full_streamed_text_list.append(chunk)
            stream_completion_tokens += chunk_tokens
        else: # Should not happen with current stream spec
            print(f"\nUnexpected item in stream: {item}", end="")


    print("\nEnd of stream.")
    # full_streamed_text = "".join(full_streamed_text_list) # This was adding spaces between words
    # Correct reconstruction of streamed output:
    # The first chunk from stream often doesn't need a leading space if it follows the prompt directly.
    # Subsequent chunks usually do if they are separate words/tokens.
    # The EchoRunner stream mock adds leading spaces for non-first words in its own generation.
    # SpeculativeRunner stream also adds leading space if current_prompt_for_iteration is not empty.
    # So, direct join should be fine.
    full_streamed_text = "".join(full_streamed_text_list)
    print(f"Full streamed text via SpeculativeRunner: '{full_streamed_text}'")
    print(f"Stream token counts: prompt={stream_prompt_tokens}, completion={stream_completion_tokens}")


    # 4. Demonstrate generate with image paths (conceptual, EchoRunner just prints them)
    image_prompt = "Describe this image:"
    example_images = ["/path/to/image1.jpg"]
    print(f"\n--- Generate with Image Paths Demo ---")
    print(f"Image Prompt: '{image_prompt}', Images: {example_images}")
    generated_text_img, img_token_counts = spec_runner.generate(image_prompt, image_paths=example_images, max_new_tokens=4)
    print(f"Speculative Output (Image): '{image_prompt}{' ' if generated_text_img else ''}{generated_text_img}'")
    print(f"Token counts (Image): {img_token_counts}")
    # EchoRunner's generate will append image_paths info to its output,
    # so we expect to see that in the final result from SpeculativeRunner.
    # The SpeculativeRunner itself prints that it passes images to both runners.

    print("\n\nSpeculativeRunner __main__ demo complete.")
