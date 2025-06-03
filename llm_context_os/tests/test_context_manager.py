# llm_context_os/tests/test_context_manager.py
import unittest
from llm_context_os.context.context_manager import ContextManager, MockTokenizer

class TestContextManager(unittest.TestCase):

    def setUp(self):
        self.mock_tokenizer = MockTokenizer() # Using the char-counting mock
        self.system_prompt = "SYSTEM:"
        # max_tokens includes system_prompt length + 1 for newline + messages length
        # System prompt "SYSTEM:" is 7 chars. Newline is 1 char. So 8 chars for system part.

    def test_initialization(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=100)
        self.assertEqual(cm.system_prompt, self.system_prompt)
        self.assertEqual(cm._messages, [])
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.max_tokens, 100)
        # Initial prompt should just be the system prompt + newline
        # Corrected based on ContextManager.build_prompt() logic which adds newline if messages follow
        # If no messages, it returns system_prompt as is.
        self.assertEqual(cm.build_prompt(), self.system_prompt)


    def test_add_message_simple(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "Hello") # "user: Hello\n" is 12 chars (role: content\n)
        self.assertEqual(len(cm._messages), 1)
        self.assertEqual(cm._messages[0], {"r": "user", "c": "Hello"})
        # _auto_scroll sets _start to len - 1, so 0.
        # _fit is called. Prompt: "SYSTEM:\nuser: Hello\n" (7+1+12 = 20 chars). Fits. _start remains 0.
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nuser: Hello\n")

    def test_add_multiple_messages_and_auto_scroll(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=100)
        cm.add("user", "Msg1") # "user: Msg1\n" (11)
        cm.add("ai", "Msg2")   # "ai: Msg2\n" (9)
        self.assertEqual(len(cm._messages), 2)
        # _auto_scroll sets _start=1. Window is msg2 for _fit.
        # Prompt for _fit: "SYSTEM:\nai: Msg2\n" (7+1+9 = 17 chars). Fits. _start remains 1.
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nai: Msg2\n")

    def test_fit_logic_truncation(self):
        # System prompt "SYSTEM:" = 7 chars. Newline = 1 char. Total 8 for system part in prompt.
        # max_tokens = 30.
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=30)

        # msg1 = "u: content01\n" (14 chars)
        # Prompt with msg1: "SYSTEM:\nu: content01\n" (7+1+14 = 22 chars). Fits. _start=0.
        cm.add("u", "content01")
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: content01\n")

        # msg2 = "a: content02\n" (14 chars)
        # _auto_scroll sets _start=1.
        # _fit checks prompt with window=[msg2]: "SYSTEM:\na: content02\n" (7+1+14 = 22 chars). Fits. _start remains 1.
        cm.add("a", "content02")
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: content02\n")

        # msg3 = "u: content03\n" (14 chars)
        # _auto_scroll sets _start=2.
        # _fit checks prompt with window=[msg3]: "SYSTEM:\nu: content03\n" (7+1+14 = 22 chars). Fits. _start remains 2.
        cm.add("u", "content03")
        self.assertEqual(cm._start, 2)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: content03\n")

        # Now, let's test _fit explicitly by navigating.
        # Messages: [msg1, msg2, msg3] (all 14 chars each)
        # Jump to msg1 (idx 0).
        cm.jump_to(0)
        # _start is 0. _fit is called.
        # Window for build_prompt: [msg1, msg2, msg3]
        # Prompt: "SYSTEM:\nu: content01\na: content02\nu: content03\n"
        # Tokens: 7(sys) + 1(nl) + 14(m1) + 14(m2) + 14(m3) = 50. (max_tokens = 30)
        # _fit loop:
        # 1. _start = 0. Prompt tokens = 50. > 30. _start < (3-1) is true. _start becomes 1.
        # 2. _start = 1. Window: [msg2, msg3]. Prompt: "SYSTEM:\na: content02\nu: content03\n"
        #    Tokens: 7(sys) + 1(nl) + 14(m2) + 14(m3) = 36. > 30. _start < (3-1) is true. _start becomes 2.
        # 3. _start = 2. Window: [msg3]. Prompt: "SYSTEM:\nu: content03\n"
        #    Tokens: 7(sys) + 1(nl) + 14(m3) = 22. < 30. Loop ends. _start stays 2.
        self.assertEqual(cm._start, 2)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: content03\n")

    def test_system_prompt_never_dropped(self):
        # System prompt "SYSTEM:" = 7. Max tokens 10 (meaning 7+1 system part, 2 for messages).
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=10)
        # "u: 12345\n" is 9 chars. Prompt: "SYSTEM:\nu: 12345\n" (7+1+9 = 17 tokens). Too long.
        cm.add("u", "12345")
        # _auto_scroll to _start=0. _fit runs.
        # Prompt tokens = 17 > 10.
        # _fit loop condition: `self._start < len(cm._messages) - 1` (0 < 0) is false. Loop doesn't run.
        # So the single message remains. _start is 0.
        self.assertEqual(cm._start, 0)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\nu: 12345\n")
        self.assertTrue(self.mock_tokenizer.count_tokens(cm.build_prompt()) > cm.max_tokens)

        # Add another message, short.
        # Messages: [{"r":"u", "c":"12345"}, {"r":"a", "c":"1"}]
        # "a: 1\n" is 5 chars.
        # _auto_scroll sets _start=1.
        # _fit runs with window=[msg2]: "SYSTEM:\na: 1\n" (7+1+5=13 tokens). Still > 10.
        # _fit loop condition: `self._start < len(cm._messages) - 1` (1 < 1) is false. Loop doesn't run. _start is 1.
        cm.add("a", "1")
        self.assertEqual(cm._start, 1)
        self.assertEqual(cm.build_prompt(), self.system_prompt + "\na: 1\n")


    def test_shift_navigation(self):
        cm = ContextManager(self.system_prompt, self.mock_tokenizer, max_tokens=55) # Increased max_tokens
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


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
