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
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self._messages: t.List[t.Dict[str, str]] = []
        self._start: int = 0

    def _fmt(self, messages: t.List[t.Dict[str, str]]) -> str:
        """
        Formats a list of messages into a single string.
        Each message is a dict {"r": role, "c": content}.
        Example: "user: Hello\nassistant: Hi there!\ntool_result: {\"status\": \"success\"...}\n"

        Future enhancement: Could apply specific formatting for different roles
        (e.g., 'system', 'user', 'assistant', 'tool_call', 'tool_result')
        if the underlying LLM requires a more structured format like ChatML.
        """
        if not messages:
            return ""
        # Current basic formatting: "role: content\n"
        return "".join([f"{msg['r']}: {msg['c']}\n" for msg in messages])

    def add(self, role: str, content: str):
        """Adds a message to the conversation history."""
        if not role:
            print(f"Warning: Empty role provided. Message '{content[:50]}...' not added.")
            return
        # Allow empty content for some roles (e.g. assistant response that is just a function call)
        # if not content:
        #     print(f"Warning: Empty content for role '{role}'. Message not added.")
        #     return
        self._messages.append({"r": role, "c": content})
        self._auto_scroll()

    def build_prompt(self, extra_messages: t.Optional[t.List[t.Dict[str, str]]] = None) -> str:
        """Builds the prompt string from the system prompt and current message window."""
        window_messages = []

        # Determine current_start safely
        current_start_idx = 0
        if self._messages: # Only if there are messages
            current_start_idx = max(0, min(self._start, len(self._messages) - 1))

        if self._messages: # Only slice if list is not empty and _start is valid
            window_messages = self._messages[current_start_idx:]

        # Combine extra messages with the current window messages
        # Extra messages usually come first if they are, for example, RAG snippets or tool descriptions.
        # If they are part of the conversational flow, their order might be different.
        # For now, prepending them as was the previous behavior.
        all_messages_to_format = (extra_messages or []) + window_messages
        current_messages_formatted = self._fmt(all_messages_to_format)

        if not self.system_prompt:
            return current_messages_formatted.lstrip('\n')
        if not current_messages_formatted:
            return self.system_prompt # Return only system prompt if no other messages

        return self.system_prompt + "\n" + current_messages_formatted


    def _fit(self):
        """
        Adjusts the context window (_start pointer) to ensure the prompt
        fits within max_tokens. It slides the window from the left (increases _start).
        """
        if not self._messages:
            self._start = 0
            # Check if system_prompt itself is too long (optional logging)
            # if self.tokenizer.count_tokens(self.system_prompt + "\n" if self.system_prompt else "") > self.max_tokens:
            #     print(f"Warning: System prompt alone exceeds max_tokens ({self.max_tokens}).")
            return

        # Ensure _start is valid before fitting.
        self._start = max(0, min(self._start, len(self._messages) - 1))

        current_prompt_str = self.build_prompt() # Uses current _start
        current_tokens = self.tokenizer.count_tokens(current_prompt_str)

        while current_tokens > self.max_tokens and self._start < (len(self._messages) - 1):
            self._start += 1
            current_prompt_str = self.build_prompt()
            current_tokens = self.tokenizer.count_tokens(current_prompt_str)

        # Optional: Log if still too long after fitting
        # if current_tokens > self.max_tokens:
        #     print(f"Warning: Context (len {current_tokens}) still exceeds max_tokens ({self.max_tokens}) after fitting. Last message might be too long.")


    def _auto_scroll(self):
        """
        Scrolls the window to make the latest message the start of the context window if it fits.
        Then ensures the window fits by calling _fit.
        This typically means the window will primarily show the end of the conversation.
        """
        if not self._messages:
            self._start = 0
        else:
            # Point to the last message. _fit will then try to include earlier messages if space allows
            # by potentially NOT incrementing _start if this single message + system prompt fits.
            # The behavior of _fit is to shrink from left (increment _start) only if current window is too big.
            self._start = len(self._messages) - 1
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

        self._start = max(0, min(index, len(self._messages) - 1 if self._messages else 0))
        self._fit()

if __name__ == '__main__':
    tokenizer = MockTokenizer()
    system_prompt = "System: You are a helpful assistant."
    manager = ContextManager(system_prompt, tokenizer, max_tokens=100)

    print(f"Initial prompt (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")

    manager.add("user", "What is the weather like today?")
    print(f"\nAfter 1st message (user) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    manager.add("assistant", "[FUNCALL]{'name': 'get_weather', 'args': {'location': 'London'}}")
    print(f"\nAfter 2nd message (assistant/funcall) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Demonstrate adding a 'tool_result'
    tool_result_content = "{'tool_name': 'get_weather', 'result': 'The weather in London is 15 degrees celsius.', 'status': 'success'}"
    manager.add("tool_result", tool_result_content)
    print(f"\nAfter 3rd message (tool_result) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")
    # At this point, _auto_scroll would have set _start = 2 (index of tool_result).
    # _fit would then check if System + tool_result fits.
    # System (36) + NL (1) + "tool_result: ...\n" (12 + len(tool_result_content) + 1)
    # len(tool_result_content) = 100
    # 37 + 12 + 100 + 1 = 150. max_tokens = 100. So it's too long.
    # _fit's loop: `while current_tokens > self.max_tokens and self._start < (len(self._messages) - 1):`
    # current_tokens = 150, self.max_tokens = 100. self._start = 2. len(self._messages) = 3.
    # (2 < 2) is false. Loop doesn't run. _start remains 2.
    # This means the prompt will be System + tool_result, even if it overflows.

    manager.add("assistant", "The weather in London is 15 degrees Celsius.") # Final AI response
    print(f"\nAfter 4th message (assistant/final_reply) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")
    # _start = 3 (index of final_reply)
    # Prompt: System + final_reply.
    # System (36) + NL (1) + "assistant: The weather in London is 15 degrees Celsius.\n" (10 + 44 + 1 = 55)
    # Total = 37 + 55 = 92. Fits within 100. So _start=3 is fine.

    print("\n--- Testing jump_to to include tool_result and subsequent AI response ---")
    # Messages:
    # 0: user: What is the weather like today?
    # 1: assistant: [FUNCALL]{'name': 'get_weather', 'args': {'location': 'London'}}
    # 2: tool_result: {'tool_name': 'get_weather', ...}
    # 3: assistant: The weather in London is 15 degrees Celsius.

    # Jump to the function call (index 1) to see context for final AI response
    manager.jump_to(1)
    # _start = 1. Window will be messages[1], messages[2], messages[3]
    # Prompt: Sys + NL + funcall_fmt + tool_result_fmt + final_reply_fmt
    # funcall_fmt: "assistant: [FUNCALL]{'name': 'get_weather', 'args': {'location': 'London'}}\n" (10 + 60 + 1 = 71)
    # tool_result_fmt: "tool_result: {'tool_name': 'get_weather', 'result': 'The weather in London is 15 degrees celsius.', 'status': 'success'}\n" (12 + 100 + 1 = 113)
    # final_reply_fmt: "assistant: The weather in London is 15 degrees Celsius.\n" (55)
    # Total prompt tokens: 37 (Sys+NL) + 71 (funcall) + 113 (tool_result) + 55 (final_reply) = 276. max_tokens = 100.
    # _fit loop:
    # _start=1, tokens=276. 1 < 3. _start becomes 2. Window: [msg2, msg3]
    #   Prompt: Sys + NL + tool_result_fmt + final_reply_fmt = 37 + 113 + 55 = 205. > 100.
    # _start=2, tokens=205. 2 < 3. _start becomes 3. Window: [msg3]
    #   Prompt: Sys + NL + final_reply_fmt = 37 + 55 = 92. < 100. Fits.
    # _start becomes 3.
    print(f"\nAfter jump_to(1) (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}") # Expected: 3

    # Add a check for empty content role, e.g. assistant just makes a funcall
    manager.add("assistant", "") # Empty content
    print(f"\nAfter adding message with empty content (tokens: {tokenizer.count_tokens(manager.build_prompt())}):\n{manager.build_prompt()}")
    print(f"Number of messages: {len(manager._messages)}") # Should be 5
    self.assertEqual(manager._messages[-1]['c'], "") # Check it was added

    print("\nContextManager demo with tool_result complete.")
