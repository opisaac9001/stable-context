import typing as t

class MockTokenizer:
    """
    A mock tokenizer for testing purposes.
    Counts tokens by character length.
    """
    def count_tokens(self, text: str) -> int:
        return len(text)

class ContextManager:
    def __init__(self, system_prompt: str, tokenizer: t.Any, max_tokens: int = 4096):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer # Expected to have a count_tokens method
        self.max_tokens = max_tokens

        self._messages: t.List[t.Dict[str, str]] = []
        self._start: int = 0 # Pointer to the start of the context window in _messages

    def _fmt(self, messages: t.List[t.Dict[str, str]]) -> str:
        """Formats a list of messages into a single string."""
        if not messages:
            return ""
        return "".join([f"{msg['r']}: {msg['c']}\n" for msg in messages])

    def add(self, role: str, content: str):
        """Adds a message to the conversation history."""
        if not role or not content: # Basic validation
            # print(f"Warning: Empty role or content provided. Message not added.")
            return
        self._messages.append({"r": role, "c": content})
        self._auto_scroll()

    def build_prompt(self, extra_messages: t.Optional[t.List[t.Dict[str, str]]] = None) -> str:
        """Builds the prompt string from the system prompt and current message window."""
        window_messages = []
        # Ensure _start is valid before slicing, especially if _messages can be cleared externally
        if self._messages:
            current_start = max(0, min(self._start, len(self._messages) -1 if self._messages else 0))
            if not self._messages: # if messages became empty after clamp
                 current_start = 0
            # if _start was valid for non-empty, but list became empty, current_start could be -1 from len-1
            # so after clamping, if list IS empty, start must be 0.
            # if list is not empty, current_start is clamped to [0, len-1]
            if self._messages: # only slice if list is not empty
                 window_messages = self._messages[current_start:]
            elif self._start != 0 and not self._messages : # if _start was non-zero and list became empty
                 pass # window_messages remains []

        # A simpler way for window_messages if _start is always maintained correctly:
        if self._messages and self._start < len(self._messages):
            window_messages = self._messages[self._start:]
        else:
            window_messages = []


        current_messages_formatted = self._fmt((extra_messages or []) + window_messages)

        # Avoid double newline if system_prompt is empty or current_messages_formatted is empty
        if not self.system_prompt:
            return current_messages_formatted.lstrip('\n') # remove leading newline if any
        if not current_messages_formatted:
            return self.system_prompt

        return self.system_prompt + "\n" + current_messages_formatted


    def _fit(self):
        """
        Adjusts the context window (_start pointer) to ensure the prompt
        fits within max_tokens. It slides the window from the left (increases _start).
        """
        if not self._messages:
            self._start = 0
            # Check if system_prompt itself is too long
            # if self.tokenizer.count_tokens(self.system_prompt + "\n") > self.max_tokens:
            #     print(f"Warning: System prompt alone ('{self.system_prompt[:30]}...') exceeds max_tokens ({self.max_tokens}).")
            return

        # Ensure _start is valid before fitting, could be OOB if messages were cleared.
        # Max valid _start is len(self._messages) - 1. If list is empty, it should be 0.
        self._start = max(0, min(self._start, len(self._messages) - 1 if self._messages else 0))

        current_prompt_str = self.build_prompt()
        current_tokens = self.tokenizer.count_tokens(current_prompt_str)

        # Loop to shrink window from the left if too many tokens
        # self._start < len(self._messages) -1 : ensures we don't make the window completely empty
        # if the very last message itself (with system prompt) is too long.
        while current_tokens > self.max_tokens and self._start < (len(self._messages) - 1):
            self._start += 1
            current_prompt_str = self.build_prompt()
            current_tokens = self.tokenizer.count_tokens(current_prompt_str)

        # Final check: if prompt is still too long, it might be due to:
        # 1. System prompt alone is too long (if _messages is empty, though handled above).
        # 2. System prompt + the single last message is too long.
        # if current_tokens > self.max_tokens:
        #     if self._start == len(self._messages) - 1 and self._messages: # Pointing to the last message
        #         print(f"Warning: System prompt + last message ('{self._messages[self._start]['c'][:30]}...') exceeds max_tokens. Consider shortening or increasing max_tokens.")
        #     elif not self._messages and self.tokenizer.count_tokens(self.system_prompt + "\n") > self.max_tokens:
        #          # This case is covered by the initial check in _fit if self._messages is empty. Redundant here.
        #          pass


    def _auto_scroll(self):
        """
        Scrolls the window to view the latest messages (sets _start to the last message index)
        and ensures it fits by calling _fit.
        """
        if not self._messages:
            self._start = 0
        else:
            self._start = len(self._messages) - 1 # Point to the last message initially
        self._fit()


    def shift(self, steps: int):
        """
        Shifts the context window by a number of steps.
        Positive steps move the window towards newer messages (increases _start).
        Negative steps move the window towards older messages (decreases _start).
        """
        if not self._messages:
            self._start = 0
            return

        new_start = self._start + steps
        # Clamp new_start to valid range: 0 to len(_messages) - 1
        # If len(_messages) is 0, then len(_messages)-1 is -1. max(0, ...) handles this.
        self._start = max(0, min(new_start, len(self._messages) - 1 if self._messages else 0))
        self._fit()

    def jump_to(self, index: int):
        """
        Jumps the context window to a specific message index.
        The message at 'index' will be the first message in the window if possible.
        """
        if not self._messages:
            self._start = 0
            return

        # Clamp index to valid range: 0 to len(_messages) - 1
        self._start = max(0, min(index, len(self._messages) - 1 if self._messages else 0))
        self._fit()

if __name__ == '__main__':
    # Basic Test Scenario
    tokenizer = MockTokenizer()
    system_prompt = "System: You are a helpful assistant." # len=36

    manager = ContextManager(system_prompt, tokenizer, max_tokens=100)

    print(f"Initial prompt (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")

    manager.add("User", "What is the weather like today?") # User: What is the weather like today?\n (38 tokens)
    # Sys(36) + NL(1) + User_msg(38) = 75 tokens. _start=0.
    print(f"\nAfter 1st message (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    manager.add("Assistant", "It is sunny and warm.") # Assistant: It is sunny and warm.\n (28 tokens)
    # _auto_scroll: _start = 1.
    # _fit: prompt = Sys(36) + NL(1) + Asst_msg(28) = 65. Fits. _start remains 1.
    print(f"\nAfter 2nd message (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Test _fit by reducing max_tokens temporarily for the next add
    manager.max_tokens = 70
    manager.add("User", "Thanks!") # User: Thanks!\n (13 tokens)
    # _auto_scroll: _start = 2 (idx of "Thanks!")
    # _fit: prompt = Sys(36) + NL(1) + User_Thanks(13) = 50. Fits. _start remains 2.
    print(f"\nAfter 3rd message (max_tokens=70, tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    manager.max_tokens = 100 # Reset max_tokens
    manager.add("Assistant", "You are welcome.") # Assistant: You are welcome.\n (21 tokens)
    # _auto_scroll: _start = 3
    # _fit: prompt = Sys(36) + NL(1) + Asst_welcome(21) = 58. Fits. _start remains 3.
    print(f"\nAfter 4th message (max_tokens=100, tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")
    # Messages:
    # 0: User: What is the weather like today? (38)
    # 1: Assistant: It is sunny and warm. (28)
    # 2: User: Thanks! (13)
    # 3: Assistant: You are welcome. (21)
    # System prompt (36) + NL (1) = 37

    # Test shift (positive means newer, negative means older)
    # Current _start = 3. Window: [msg3]. Prompt = Sys + msg3 = 37 + 21 = 58.
    manager.shift(-2) # Shift towards older messages by 2 (i.e. _start becomes 3 - 2 = 1)
    # _start = 1.
    # _fit: Prompt = Sys + msg1 + msg2 + msg3 = 37 + 28 + 13 + 21 = 99. Fits. _start remains 1.
    print(f"\nAfter shifting window start by -2 (towards older msgs) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}") # Should be 1

    manager.shift(1) # Shift towards newer by 1 (i.e. _start becomes 1 + 1 = 2)
    # _start = 2.
    # _fit: Prompt = Sys + msg2 + msg3 = 37 + 13 + 21 = 71. Fits. _start remains 2.
    print(f"\nAfter shifting window start by +1 (towards newer msgs) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}") # Should be 2

    # Test jump_to
    manager.jump_to(0)
    # _start = 0.
    # _fit: Prompt = Sys + msg0 + msg1 + msg2 + msg3 = 37 + 38 + 28 + 13 + 21 = 137. Too large for max_tokens=100.
    #   Loop in _fit: current_tokens=137, _start=0. `0 < 3` is true. _start becomes 1.
    #     Prompt = Sys + msg1 + msg2 + msg3 = 37 + 28 + 13 + 21 = 99. Fits.
    #   Loop terminates. _start is 1.
    print(f"\nAfter jumping to index 0 (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}") # Should be 1

    extra = [{"r": "system", "c": "This is an extra system message."}] # system: ...\n (34 tokens)
    # Current window: Sys + msg1 + msg2 + msg3 (99 tokens). _start=1.
    # build_prompt will generate: Sys + NL + extra_fmt + msg1_fmt + msg2_fmt + msg3_fmt
    # 36 + 1 + 34 + 28 + 13 + 21 = 133 tokens.
    prompt_with_extra = manager.build_prompt(extra_messages=extra)
    print(f"\nPrompt with extra message (tokens: {tokenizer.count_tokens(prompt_with_extra)}):\n{prompt_with_extra}")

    empty_manager = ContextManager("System: Empty here.", tokenizer, 100) # Sys(18)+NL(1)=19
    print(f"\nEmpty manager prompt (tokens: {tokenizer.count_tokens(empty_manager.build_prompt())}):\n{empty_manager.build_prompt()}")
    empty_manager.add("User", "Hello") # User: Hello\n (12 tokens). Sys+NL+UsrMsg = 19+12=31. _start=0.
    print(f"Empty manager after 1 add (tokens: {tokenizer.count_tokens(empty_manager.build_prompt())}):\n{empty_manager.build_prompt()}")
    print(f"Empty manager window starts at: {empty_manager._start}")

    long_sys_manager = ContextManager("System: This system prompt is very very long and will exceed max tokens by itself.", tokenizer, max_tokens=50)
    # System prompt len = 75. max_tokens = 50. Sys+NL = 76.
    print(f"\nManager with long system prompt (tokens: {tokenizer.count_tokens(long_sys_manager.build_prompt())}):\n{long_sys_manager.build_prompt()}")
    long_sys_manager.add("User","Hi") # User: Hi\n (9 tokens).
    # _auto_scroll: _start = 0.
    # _fit: Prompt = Sys + NL + UserMsg = 76 + 9 = 85. max_tokens=50.
    #   Loop: current_tokens=85, _start=0. `0 < (1-1)` is `0 < 0` which is false. Loop doesn't run.
    #   _start remains 0. (Warning should appear if uncommented in _fit)
    print(f"\nManager with long system prompt after add (tokens: {tokenizer.count_tokens(long_sys_manager.build_prompt())}):\n{long_sys_manager.build_prompt()}")
    print(f"Manager window starts at: {long_sys_manager._start}")

    manager.add("User", "") # Try adding empty content
    print(f"\nMessages count after adding empty content: {len(manager._messages)}") # Should not change from 4
    manager.add("", "Test") # Try adding empty role
    print(f"Messages count after adding empty role: {len(manager._messages)}")   # Should not change from 4

    # Test _fmt with empty list
    assert manager._fmt([]) == ""
    # Test build_prompt with empty system prompt
    no_sys_manager = ContextManager("", tokenizer, 100)
    no_sys_manager.add("User", "Hi") # User: Hi\n
    assert no_sys_manager.build_prompt() == "User: Hi\n"
    no_sys_manager.add("Assistant", "Hello") # Assistant: Hello\n
    # _start = 1 (_auto_scroll)
    # build_prompt = Assistant: Hello\n
    assert no_sys_manager.build_prompt() == "Assistant: Hello\n"
    no_sys_manager.jump_to(0)
    # _start = 0
    # build_prompt = User: Hi\nAssistant: Hello\n
    assert no_sys_manager.build_prompt() == "User: Hi\nAssistant: Hello\n"
    print("\nTests with empty system prompt passed.")

    print("\nAll basic tests seem to pass with current logic.")
