import typing as t

class MockTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

class ContextManager:
    def __init__(self, system_prompt: str, tokenizer: t.Any, max_tokens: int = 4096):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._messages: t.List[t.Dict[str, t.Any]] = [] # Content can now include image_path
        self._start: int = 0

    def _fmt(self, messages_to_format: t.List[t.Dict[str, t.Any]]) -> str:
        """
        Formats a list of messages into a single string.
        Each message is a dict e.g. {"r": role, "c": content, "image_path": path_or_none}.
        The generic "role: content" formatting is currently used for all roles,
        including special ones like 'tool_result', 'retrieved_context', 'retrieved_pdf_chunk'.

        Future enhancement: Could apply specific formatting for different roles
        or structured content (like function calls/results, images, retrieved snippets)
        if the underlying LLM requires a more structured format (e.g., ChatML, Llama3 format).
        """
        if not messages_to_format:
            return ""

        formatted_strings = []
        for msg in messages_to_format:
            role = msg.get('r', 'unknown') # Default role if not present
            content = msg.get('c', '')     # Default content if not present
            image_path = msg.get('image_path')

            display_content = content
            if image_path:
                # Prepend image placeholder. Ensure space if content exists.
                image_placeholder = f"[IMAGE: {image_path}]"
                if display_content:
                    display_content = f"{image_placeholder} {display_content}"
                else:
                    display_content = image_placeholder

            formatted_strings.append(f"{role}: {display_content}\n")

        return "".join(formatted_strings)

    def add(self, role: str, content: str, image_path: t.Optional[str] = None): # Added image_path
        """Adds a message (text and/or image) to the conversation history."""
        if not role:
            print(f"Warning: Empty role provided. Message not added.")
            return

        # A message can have text, an image, or both.
        # if not content and not image_path:
        #     print(f"Warning: Empty content and no image_path for role '{role}'. Message not added.")
        #     return

        self._messages.append({'r': role, 'c': content, 'image_path': image_path})
        self._auto_scroll()

    def build_prompt(self, extra_messages: t.Optional[t.List[t.Dict[str, t.Any]]] = None) -> str:
        window_messages = []
        current_start_idx = 0
        if self._messages:
            current_start_idx = max(0, min(self._start, len(self._messages) - 1))

        if self._messages:
            window_messages = self._messages[current_start_idx:]

        all_messages_to_format = (extra_messages or []) + window_messages
        # Store for debugging/testing what actually went into prompt formatting
        self._messages_for_prompt_build_debug: t.List[t.Dict[str, t.Any]] = all_messages_to_format
        current_messages_formatted = self._fmt(all_messages_to_format)

        if not self.system_prompt:
            return current_messages_formatted.lstrip('\n')
        if not current_messages_formatted:
            return self.system_prompt

        return self.system_prompt + "\n" + current_messages_formatted

    def _fit(self):
        if not self._messages:
            self._start = 0
            return

        self._start = max(0, min(self._start, len(self._messages) - 1))
        current_prompt_str = self.build_prompt()
        current_tokens = self.tokenizer.count_tokens(current_prompt_str)

        while current_tokens > self.max_tokens and self._start < (len(self._messages) - 1):
            self._start += 1
            current_prompt_str = self.build_prompt()
            current_tokens = self.tokenizer.count_tokens(current_prompt_str)

    def _auto_scroll(self):
        if not self._messages:
            self._start = 0
        else:
            self._start = len(self._messages) - 1
        self._fit()

    def shift(self, steps: int):
        if not self._messages:
            self._start = 0
            return
        new_start = self._start + steps
        self._start = max(0, min(new_start, len(self._messages) - 1 if self._messages else 0))
        self._fit()

    def jump_to(self, index: int):
        if not self._messages:
            self._start = 0
            return
        self._start = max(0, min(index, len(self._messages) - 1 if self._messages else 0))
        self._fit()

if __name__ == '__main__':
    tokenizer = MockTokenizer()
    system_prompt = "System: You are a multimodal helpful assistant."
    # Max tokens set for easier testing of _fit with image paths
    manager = ContextManager(system_prompt, tokenizer, max_tokens=150)

    print(f"Initial prompt:\n{manager.build_prompt()}")

    manager.add("user", "What do you see in this image?", image_path="path/to/image1.jpg")
    print(f"\nAfter 1st message (user with image):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    manager.add("assistant", "I see a cat playing with a ball of yarn.")
    print(f"\nAfter 2nd message (assistant text reply):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Message with only an image (content can be empty string)
    manager.add("user", "", image_path="path/to/image2.png")
    print(f"\nAfter 3rd message (user with only image):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Demonstrate context fitting with image paths (they add to token count via length)
    # Current _start = 2 (points to image2.png message)
    # Prompt: Sys (46) + NL (1) + "user: [IMAGE: path/to/image2.png}]\n" (len of "user: []\n" (8) + len("[IMAGE: path/to/image2.png]") (28) = 36)
    # Total = 46 + 1 + 36 = 83. Fits within 150.

    print("\n--- Testing context fitting with multiple messages including images ---")
    # Add more messages to exceed max_tokens
    manager.add("user", "This is a short text message to add more content to the context window.") # approx 70 chars
    # _start = 3. Prompt: Sys + NL + user_msg4_fmt (46+1 + (7+70+1)=125). Fits.
    print(f"\nAfter 4th message (user text):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    manager.add("assistant", "Okay, I see that text message following the image.") # approx 60 chars
    # _start = 4. Prompt: Sys + NL + assistant_msg5_fmt (46+1 + (10+60+1)=118). Fits.
    print(f"\nAfter 5th message (assistant text):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Now, try to jump to the beginning to include all messages.
    # Sys (46) + NL (1) = 47
    # msg0 (user_img1): "user: [IMAGE: path/to/image1.jpg] What do you see in this image?\n"
    #   (8 + 28 + 1 + 30 + 1 = 68)
    # msg1 (asst_text): "assistant: I see a cat playing with a ball of yarn.\n"
    #   (10 + 46 + 1 = 57)
    # msg2 (user_img2): "user: [IMAGE: path/to/image2.png}]\n" (8 + 28 + 1 = 37)
    # msg3 (user_text): "user: This is a short text message to add more content to the context window.\n" (8 + 70 + 1 = 79)
    # msg4 (asst_text): "assistant: Okay, I see that text message following the image.\n" (10 + 60 + 1 = 71)
    # Total: 47 + 68 + 57 + 37 + 79 + 71 = 359. max_tokens = 150.

    print("\nJumping to start of conversation (index 0):")
    manager.jump_to(0)
    # _fit will be called. It should increment _start until the window fits.
    # Expected: only the last few messages will fit.
    # If _start becomes 3 (msg3, msg4): 47 + 79 + 71 = 197 (too long) -> _start becomes 4
    # If _start becomes 4 (msg4): 47 + 71 = 118 (fits)
    print(f"\nAfter jump_to(0) and _fit:\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}") # Expected: 4

    print("\n--- Testing prompt building with 'extra_messages' (e.g., RAG snippets) ---")
    rag_snippets = [
        {"r": "retrieved_pdf_chunk", "c": "Climate change refers to long-term shifts in temperatures and weather patterns. (Source: report.pdf, Page: 1)"},
        {"r": "retrieved_context", "c": "User previously asked about mitigation strategies."}
    ]
    # Build prompt using current window (which is just the last message due to auto_scroll and fit)
    # plus the extra RAG snippets.
    # Current window (from _start=4): assistant: Okay, I see that text message following the image.
    prompt_with_rag = manager.build_prompt(extra_messages=rag_snippets)
    print(f"\nPrompt with RAG snippets (current window is last msg):\n{prompt_with_rag}")
    # Expected: System prompt, then RAG snippets, then the last message from history.
    assert "retrieved_pdf_chunk: Climate change refers to long-term shifts" in prompt_with_rag
    assert "retrieved_context: User previously asked about mitigation strategies." in prompt_with_rag
    assert "assistant: Okay, I see that text message following the image." in prompt_with_rag

    print("\nContextManager multimodal and RAG snippet demo complete.")
