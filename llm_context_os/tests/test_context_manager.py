# llm_context_os/tests/test_context_manager.py
import unittest
from llm_context_os.context.context_manager import ContextManager, MockTokenizer

# --- Conditional imports for HFTokenizer ---
HAVE_TRANSFORMERS = False
HF_TOKENIZER_INSTANCE = None
HF_MODEL_NAME = "gpt2" # Using a common model, fairly small

try:
    # Try to import AutoTokenizer to confirm transformers library is available
    from transformers import AutoTokenizer
    from llm_context_os.context.token_estimator import HFTokenEstimator
    HAVE_TRANSFORMERS = True
    try:
        HF_TOKENIZER_INSTANCE = HFTokenEstimator(model_name=HF_MODEL_NAME)
        print(f"Successfully loaded HFTokenizer with model '{HF_MODEL_NAME}'.")
    except Exception as e:
        # This can happen if the model is not found, network issues, etc.
        print(f"Note: Could not load HFTokenizer model '{HF_MODEL_NAME}': {e}")
        HF_TOKENIZER_INSTANCE = None # Ensure it's None if model loading fails
except ImportError:
    print("Note: 'transformers' library not found or HFTokenEstimator not found. Skipping HFTokenizer tests.")
    # HAVE_TRANSFORMERS remains False, HF_TOKENIZER_INSTANCE remains None
    pass
# --- End conditional imports ---

class TestContextManager(unittest.TestCase):

    def setUp(self):
        self.mock_tokenizer = MockTokenizer() # Using the char-counting mock
        self.system_prompt_str = "SYSTEM:" # Renamed from self.system_prompt to avoid conflict if HF has own
        # For MockTokenizer (char-based):
        # System prompt "SYSTEM:" is 7 chars. Newline is 1 char. So 8 chars for system part.

    def test_initialization(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        self.assertEqual(cm.system_prompt, self.system_prompt_str)
        self.assertEqual(cm._messages, [])
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.max_tokens, 100)
        # Initial prompt should just be the system prompt + newline
        # Corrected based on ContextManager.build_prompt() logic which adds newline if messages follow
        # If no messages, it returns system_prompt as is.
        self.assertEqual(cm.build_prompt(), self.system_prompt_str)


    def test_add_message_simple(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "Hello") # image_path defaults to None
        self.assertEqual(len(cm._messages), 1)
        self.assertEqual(cm._messages[0], {"r": "user", "c": "Hello", "image_path": None})
        # _auto_scroll sets _start to len - 1, so 0.
        # _fit is called. Prompt: "SYSTEM:\nuser: Hello\n" (7+1+12 = 20 chars). Fits. _start remains 0.
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt_str + "\nuser: Hello\n")

    def test_add_multiple_messages_and_auto_scroll(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "Msg1") # "user: Msg1\n" (11)
        cm.add("ai", "Msg2")   # "ai: Msg2\n" (9)
        self.assertEqual(len(cm._messages), 2)
        # _auto_scroll sets _start=1. Window is msg2 for _fit.
        # Prompt for _fit: "SYSTEM:\nai: Msg2\n" (7+1+9 = 17 chars). Fits. _start remains 1.
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt_str + "\nai: Msg2\n")

    def _run_fit_logic_truncation_test(self, tokenizer, system_prompt_str, max_tokens_val, msg_contents, expected_final_start_idx, test_label=""):
        """
        Helper function to test the fit logic of ContextManager.
        Adds 3 messages, then jumps to the start and checks if _start is adjusted correctly
        and the final prompt is within token limits.
        """
        cm = ContextManager(system_prompt_str, tokenizer, max_tokens=max_tokens_val)

        # Add messages
        roles = ["u", "a", "u"]
        for i in range(3):
            cm.add(roles[i], msg_contents[i])
            # Basic check after each add (auto-scroll behavior)
            self.assertEqual(cm._start, i, f"[{test_label}] After {i+1} add(s), _start should be {i}")
            current_prompt = cm.build_prompt()
            self.assertTrue(tokenizer.count_tokens(current_prompt) <= max_tokens_val,
                            f"[{test_label}] Prompt after {i+1} add(s) should fit within max_tokens. "
                            f"Got {tokenizer.count_tokens(current_prompt)}, max {max_tokens_val}. Prompt: {current_prompt}")

        # Test _fit by jumping to cause potential overflow
        cm.jump_to(0)

        final_prompt = cm.build_prompt()
        final_tokens = tokenizer.count_tokens(final_prompt)

        self.assertEqual(cm._start, expected_final_start_idx,
                         f"[{test_label}] After jump_to(0) and _fit, _start should be {expected_final_start_idx}, but got {cm._start}. "
                         f"Final prompt: '{final_prompt}', tokens: {final_tokens}, max_tokens: {max_tokens_val}")

        self.assertTrue(final_tokens <= max_tokens_val,
                        f"[{test_label}] Final prompt tokens ({final_tokens}) should not exceed max_tokens ({max_tokens_val}). "
                        f"Prompt: '{final_prompt}'")

        # Verify that if truncation occurred (expected_final_start_idx > 0), the first message is not in the prompt
        if expected_final_start_idx > 0 and len(cm._messages) > 0:
            first_message_text = cm._messages[0]['c']
            self.assertNotIn(first_message_text, final_prompt,
                             f"[{test_label}] First message content '{first_message_text}' should not be in the final prompt if _start is {cm._start} (>0).")


    def test_fit_logic_truncation(self):
        # Using MockTokenizer (char based)
        system_prompt = self.system_prompt_str # "SYSTEM:" (7 chars) + NL (1 char) = 8 system part
        # Message "u: content01\n": 1(role) + 2(: ) + 9(content) + 1(\n) = 13 chars
        msg_contents = ["content01", "content02", "content03"]

        # Case 1: Original test case where truncation leads to _start = 2
        # Sys (8) + M1(13) + M2(13) + M3(13) = 47. max_tokens = 30.
        # After jump_to(0):
        #   _start=0, prompt=47 > 30 -> _start=1
        #   _start=1, prompt=Sys+M2+M3 = 8+13+13=34 > 30 -> _start=2
        #   _start=2, prompt=Sys+M3 = 8+13=21 <= 30. Final _start=2.
        self._run_fit_logic_truncation_test(self.mock_tokenizer, system_prompt,
                                            max_tokens_val=30, msg_contents=msg_contents,
                                            expected_final_start_idx=2, test_label="Mock-TruncateToLast")

        # Case 2: All messages fit
        # Sys (8) + M1(13) + M2(13) + M3(13) = 47. max_tokens = 50.
        # After jump_to(0): prompt=47 <= 50. Final _start=0.
        self._run_fit_logic_truncation_test(self.mock_tokenizer, system_prompt,
                                            max_tokens_val=50, msg_contents=msg_contents,
                                            expected_final_start_idx=0, test_label="Mock-AllFit")

        # Case 3: Truncation leads to _start = 1
        # Sys (8) + M1(13) + M2(13) + M3(13) = 47. max_tokens = 35.
        # After jump_to(0):
        #   _start=0, prompt=47 > 35 -> _start=1
        #   _start=1, prompt=Sys+M2+M3 = 8+13+13=34 <= 35. Final _start=1.
        self._run_fit_logic_truncation_test(self.mock_tokenizer, system_prompt,
                                            max_tokens_val=35, msg_contents=msg_contents,
                                            expected_final_start_idx=1, test_label="Mock-TruncateToOne")


    @unittest.skipIf(not HF_TOKENIZER_INSTANCE, "HFTokenizer not available or model could not be loaded")
    def test_fit_logic_truncation_hf(self):
        system_prompt = self.system_prompt_str # "SYSTEM:" (gpt2: 2 tokens) + NL (1 token) = 3 system part
        # Message "u: content01\n" (gpt2: "u"(1) + ":"(1) + " content"(1) + "01"(1) + "\n"(1) = 5 tokens)
        msg_contents = ["content01", "content02", "content03"] # Each gives 5 tokens for "role: contentXX\n"

        # Total tokens if all included: Sys(3) + M1(5) + M2(5) + M3(5) = 18 tokens

        # Case 1 HF: All messages fit
        # max_tokens = 20. Full prompt is 18 tokens. Expected _start = 0.
        self._run_fit_logic_truncation_test(HF_TOKENIZER_INSTANCE, system_prompt,
                                            max_tokens_val=20, msg_contents=msg_contents,
                                            expected_final_start_idx=0, test_label="HF-AllFit")

        # Case 2 HF: One message truncated
        # max_tokens = 15. Full prompt is 18 tokens.
        # After jump_to(0):
        #   _start=0, prompt=18 > 15 -> _start=1
        #   _start=1, prompt=Sys+M2+M3 = 3+5+5=13 <= 15. Final _start=1.
        self._run_fit_logic_truncation_test(HF_TOKENIZER_INSTANCE, system_prompt,
                                            max_tokens_val=15, msg_contents=msg_contents,
                                            expected_final_start_idx=1, test_label="HF-TruncateToOne")

        # Case 3 HF: Two messages truncated
        # max_tokens = 9. Full prompt is 18 tokens.
        # After jump_to(0):
        #   _start=0, prompt=18 > 9 -> _start=1
        #   _start=1, prompt=Sys+M2+M3 = 3+5+5=13 > 9 -> _start=2
        #   _start=2, prompt=Sys+M3 = 3+5=8 <= 9. Final _start=2.
        self._run_fit_logic_truncation_test(HF_TOKENIZER_INSTANCE, system_prompt,
                                            max_tokens_val=9, msg_contents=msg_contents,
                                            expected_final_start_idx=2, test_label="HF-TruncateToLast")

    def test_system_prompt_never_dropped(self):
        # System prompt "SYSTEM:" = 7. Max tokens 10 (meaning 7+1 system part, 2 for messages).
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=10)
        # "u: 12345\n" is 9 chars. Prompt: "SYSTEM:\nu: 12345\n" (7+1+9 = 17 tokens). Too long.
        cm.add("u", "12345")
        # _auto_scroll to _start=0. _fit runs.
        # Prompt tokens = 17 > 10.
        # _fit loop condition: `self._start < len(cm._messages) - 1` (0 < 0) is false. Loop doesn't run.
        # So the single message remains. _start is 0.
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt_str + "\nu: 12345\n")
        self.assertTrue(self.mock_tokenizer.count_tokens(cm.build_prompt()) > cm.max_tokens)

        # Add another message, short.
        # Messages: [{"r":"u", "c":"12345"}, {"r":"a", "c":"1"}]
        # "a: 1\n" is 5 chars.
        # _auto_scroll sets _start=1.
        # _fit runs with window=[msg2]: "SYSTEM:\na: 1\n" (7+1+5=13 tokens). Still > 10.
        # _fit loop condition: `self._start < len(cm._messages) - 1` (1 < 1) is false. Loop doesn't run. _start is 1.
        cm.add("a", "1")
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt_str + "\na: 1\n")


    def test_shift_navigation(self):
        cm = ContextManager(self.system_prompt_str, self.mock_tokenizer, max_tokens=55) # Increased max_tokens
        # Each message "role: content\n" will be:
        # msg1: "u: msg1\n" (9)
        # msg2: "a: msg2\n" (9)
        # msg3: "u: msg3\n" (9)
        # msg4: "a: msg4\n" (9)
        # System part: 7+1=8
        cm.add("u", "msg1")
        cm.add("a", "msg2")
        cm.add("u", "msg3")
        cm.add("a", "msg4")
        # _messages = [m1, m2, m3, m4]. _start is 3 (points to m4).
        # Prompt for _fit: SYSTEM:\na: msg4\n (8+9=17). Fits.
        self.assertEqual(cm._start, 3)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: msg4\n")

        # Shift older 1 step (negative). _start becomes 2 (m3). Window: [m3, m4].
        # Prompt: SYSTEM:\nu: msg3\na: msg4\n (8+9+9 = 26). Fits.
        cm.shift(-1)
        self.assertEqual(cm._start, 2)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: msg3\na: msg4\n")

        # Shift older 2 steps. _start was 2, becomes 0 (m1). Window: [m1, m2, m3, m4].
        # Prompt: SYSTEM:\nu: msg1\na: msg2\nu: msg3\na: msg4\n (8+9+9+9+9 = 44). Fits.
        cm.shift(-2)
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: msg1\na: msg2\nu: msg3\na: msg4\n")

        # Shift newer 1 step (positive). _start was 0, becomes 1 (m2). Window: [m2, m3, m4]
        # Prompt: SYSTEM:\na: msg2\nu: msg3\na: msg4\n (8+9+9+9 = 35). Fits.
        cm.shift(1)
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: msg2\nu: msg3\na: msg4\n")

        # Shift beyond end (newer)
        cm.shift(10) # Was 1, 1+10=11. Clamps to _start = 3 (len is 4)
        self.assertEqual(cm._start, 3)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: msg4\n")

        # Shift beyond beginning (older)
        cm.shift(-20) # Was 3, 3-20=-17. Clamps to _start = 0
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: msg1\na: msg2\nu: msg3\na: msg4\n")


    def test_jump_to_navigation(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=55)
        cm.add("u", "msg1")
        cm.add("a", "msg2")
        cm.add("u", "msg3")
        cm.add("a", "msg4")
        # _start is 3.

        cm.jump_to(0) # Window: [m1,m2,m3,m4]. Prompt: 44 tokens. Fits.
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: msg1\na: msg2\nu: msg3\na: msg4\n")

        cm.jump_to(2) # Window: [m3,m4]. Prompt: 26 tokens. Fits.
        self.assertEqual(cm._start, 2)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: msg3\na: msg4\n")

        cm.jump_to(3) # Window: [m4]. Prompt: 17 tokens. Fits.
        self.assertEqual(cm._start, 3)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: msg4\n")

        # Jump to out of bounds
        cm.jump_to(10) # Clamps to 3
        self.assertEqual(cm._start, 3)
        cm.jump_to(-5) # Clamps to 0
        self.assertEqual(cm._start, 0)

    def test_empty_messages_navigation(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=20)
        self.assertEqual(cm.build_prompt(), self.system_prompt) # Only system prompt
        cm.shift(1) # No messages, _start remains 0
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt)
        cm.shift(-1) # No messages, _start remains 0
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt)
        cm.jump_to(0) # No messages, _start remains 0
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt)
        cm.jump_to(5) # No messages, _start remains 0
        self.assertEqual(cm._start, 0)


    def test_build_prompt_with_extra_messages(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=50)
        cm.add("u", "msg1") # "u: msg1\n" (9). Stored message.
        # _start is 0. Window: [msg1]. Prompt for _fit: Sys+NL+msg1 = 8+9 = 17. Fits.

        extra_msgs = [{"r": "system", "c": "hint"}] # "system: hint\n" (13)
        # Expected: SYSTEM:\nsystem: hint\nu: msg1\n
        # Tokens: 7(sys)+1(NL)+13(extra)+9(msg1) = 30. Fits.
        prompt = cm.build_prompt(extra_messages=extra_msgs) # build_prompt takes `extra`, not `extra_messages`
        self.assertEqual(prompt, self.system_prompt + "\nsystem: hint\nu: msg1\n")

        # Check that extra messages don't affect stored messages or _start
        self.assertEqual(len(cm._messages), 1)
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: msg1\n") # Original prompt unchanged

    def test_build_prompt_with_extra_messages_causing_overflow_no_fit_on_build(self):
        # build_prompt itself does not call _fit. _fit is called by add/shift/jump_to.
        # System: 7. NL: 1. Max_tokens: 20. Budget for messages: 12
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=20)
        cm.add("u", "msg1") # "u: msg1\n" (9). Prompt: 7+1+9 = 17. Fits. _start=0.

        extra_msgs = [{"r": "system", "c": "verylonghint"}] # "system: verylonghint\n" (23)
        # Expected prompt: SYSTEM:\nsystem: verylonghint\nu: msg1\n
        # Tokens: 7+1+23+9 = 40. Exceeds max_tokens (20).
        # build_prompt does not truncate, it just builds.
        prompt = cm.build_prompt(extra_messages=extra_msgs) # build_prompt takes `extra`, not `extra_messages`
        self.assertEqual(prompt, self.system_prompt + "\nsystem: verylonghint\nu: msg1\n")
        self.assertTrue(self.mock_tokenizer.count_tokens(prompt) > cm.max_tokens)

    def test_fit_with_single_very_long_message(self):
        # System: 7. NL: 1. Max_tokens: 10. Budget for messages: 2
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=10)
        # "u: 12345\n" (9 chars). Prompt "SYSTEM:\nu: 12345\n" (7+1+9=17 tokens).
        cm.add("u", "12345")
        # _auto_scroll sets _start=0. _fit is called.
        # Inside _fit: prompt tokens = 17. max_tokens is 10.
        # Condition `self._start < len(self._messages) - 1` (0 < 0) is false.
        # So, _start is not incremented. The message remains.
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: 12345\n")
        self.assertTrue(self.mock_tokenizer.count_tokens(cm.build_prompt()) > cm.max_tokens)

    def test_add_until_first_message_is_pushed_out(self):
        # Sys:7, NL:1. Max: 25. Budget for msgs: 17 after sys+NL
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=25)
        # msg1: "u: item_A\n" (11). Prompt: 7+1+11=19. Fits. _start=0.
        cm.add("u", "item_A")
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: item_A\n")

        # msg2: "a: item_B\n" (11).
        # _auto_scroll -> _start=1.
        # _fit checks prompt with window=[m2]: "SYSTEM:\na: item_B\n" (7+1+11 = 19). Fits. _start remains 1.
        cm.add("a", "item_B")
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: item_B\n")

        self.assertEqual(cm._messages[0]['c'], "item_A") # Check item_A is still in _messages

        # Now, if we jump to the beginning, the window should be [item_A, item_B]
        # Prompt: SYSTEM:\nu: item_A\na: item_B\n (7+1+11+11 = 30) > 25.
        cm.jump_to(0)
        # _fit will run:
        # 1. _start=0. Prompt tokens=30. >25. _start < len-1 (0<1) is true. _start becomes 1.
        # 2. _start=1. Prompt (window=[m2]) tokens=19. <25. Loop ends. _start is 1.
        self.assertEqual(cm._start, 1) # item_A got pushed out of context window
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: item_B\n")

    def test_add_and_format_with_image_path(self):
        cm = ContextManager("SysPrompt:", self.mock_tokenizer, max_tokens=200)
        cm.add(role="user", content="Look at this picture of a cat.", image_path="path/to/cat.jpg")
        cm.add(role="user", content="And this one of a dog.", image_path="path/to/dog.png")
        cm.add(role="assistant", content="Interesting pictures!")
        cm.add(role="user", content="What was the first picture about?")

        # Current _auto_scroll behavior makes window only contain the last message if it fits.
        # To test full formatting, we jump_to(0) to include all history.
        cm.jump_to(0)

        prompt = cm.build_prompt()
        # print(f"\nPrompt for image test:\n{prompt}") # For debugging

        # Check for presence of image placeholders and text in order
        self.assertIn("SysPrompt:\n", prompt)
        self.assertIn("user: [IMAGE: path/to/cat.jpg] Look at this picture of a cat.\n", prompt)
        self.assertIn("user: [IMAGE: path/to/dog.png] And this one of a dog.\n", prompt)
        self.assertIn("assistant: Interesting pictures!\n", prompt)
        self.assertIn("user: What was the first picture about?\n", prompt)

        # Test _fit logic with image path lengths
        # SysPrompt: (10) + NL (1) = 11
        # msg0: "user: [IMAGE: path/to/cat.jpg] Look at this picture of a cat.\n"
        #       (6 + 28 + 1 + 30 + 1 = 66)
        # msg1: "user: [IMAGE: path/to/dog.png] And this one of a dog.\n"
        #       (6 + 28 + 1 + 23 + 1 = 59)
        # msg2: "assistant: Interesting pictures!\n" (10 + 20 + 1 = 31)
        # msg3: "user: What was the first picture about?\n" (6 + 33 + 1 = 40)
        # Total with all messages from _start=0: 11 + 66 + 59 + 31 + 40 = 207

        cm.max_tokens = 150 # Set a limit that will force truncation
        cm.jump_to(0) # This will trigger _fit

        # Expected: msg0 (cat.jpg) should be dropped.
        # Window: msg1, msg2, msg3. Tokens: 11 + 59 + 31 + 40 = 141. This fits.
        # So, _start should become 1.
        self.assertEqual(cm._start, 1, "ContextManager _fit did not correctly adjust _start with image paths.")

        prompt_after_fit = cm.build_prompt()
        self.assertNotIn("[IMAGE: path/to/cat.jpg]", prompt_after_fit)
        self.assertIn("[IMAGE: path/to/dog.png]", prompt_after_fit)
        self.assertIn("Interesting pictures!", prompt_after_fit)
        self.assertIn("What was the first picture about?", prompt_after_fit)
        self.assertTrue(self.mock_tokenizer.count_tokens(prompt_after_fit) <= 150)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
