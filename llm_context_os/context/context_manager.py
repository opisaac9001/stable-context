import typing as t

class MockTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

class ContextManager:
    def __init__(self, system_prompt: str, tokenizer: t.Any, max_tokens: int = 4096):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._messages: t.List[t.Dict[str, t.Any]] = []
        # Message structure:
        # {
        #   'r': role (str: "system", "user", "assistant", "tool", "retrieved_context", etc.)
        #   'c': content (str)
        #   'image_path': image_path (Optional[str]) - for 'user' role primarily
        #   'tcid': tool_call_id (Optional[str]) - for 'tool' role
        #   'nm': name (Optional[str]) - for 'tool' role (name of the tool/function)
        #   'tool_calls': tool_calls (Optional[List[Dict]]) - for 'assistant' role if it issues tool calls
        # }
        self._start: int = 0

    def _format_messages_for_token_counting(self, messages_to_format: t.List[t.Dict[str, t.Any]]) -> str:
        """
        Formats a list of messages into a single string purely for token counting purposes.
        This should try to mimic the structure an LLM would see as closely as possible if it were
        a string-based prompt, including identifiers for tool calls and results.
        """
        if not messages_to_format:
            return ""

        formatted_strings = []
        for msg in messages_to_format:
            role = msg.get('r', 'unknown')
            content = msg.get('c', '')
            image_path = msg.get('image_path')
            tool_call_id = msg.get('tcid')
            name = msg.get('nm')
            # tool_calls from assistant are not directly formatted here, they are part of the assistant's "turn"

            display_content = content
            prefix = f"{role}: "

            if role == "tool":
                prefix = f"tool (tool_call_id={tool_call_id}, name={name}): "

            if image_path:
                image_placeholder = f"[IMAGE: {image_path}]"
                if display_content:
                    display_content = f"{image_placeholder} {display_content}"
                else:
                    display_content = image_placeholder

            # For assistant messages that contain tool_calls, the 'content' might be None or empty.
            # The tool_calls themselves are more structured. For string formatting for token counting,
            # we might want to serialize them or represent their presence.
            # For now, if 'content' is empty for an assistant, and it has tool_calls,
            # we can add a placeholder. The actual 'tool_calls' object is in the message dict.
            if role == "assistant" and not content and msg.get("tool_calls"):
                display_content = "[Requesting tool calls]" # Placeholder for token counting

            formatted_strings.append(f"{prefix}{display_content}\n")

        return "".join(formatted_strings)

    def add(self,
            role: str,
            content: t.Any, # Content can be string, or list of tool_calls for assistant
            image_path: t.Optional[str] = None,
            tool_call_id: t.Optional[str] = None,
            name: t.Optional[str] = None,
            tool_calls: t.Optional[t.List[t.Dict[str, t.Any]]] = None): # For assistant requesting tools
        """Adds a message to the conversation history."""
        if not role:
            print(f"Warning: Empty role provided. Message not added.")
            return

        message: t.Dict[str, t.Any] = {'r': role}

        if content is not None: # Allow content to be explicitly None, e.g. for assistant message with only tool_calls
             message['c'] = content

        if image_path:
            message['image_path'] = image_path

        if role == "tool":
            if not tool_call_id:
                print(f"Warning: 'tool_call_id' is required for role 'tool'. Message not added.")
                return
            if not name: # Tool name is also generally expected for OpenAI "tool" role messages.
                print(f"Warning: 'name' (tool name) is required for role 'tool'. Message not added.")
                return
            message['tcid'] = tool_call_id
            message['nm'] = name

        if role == "assistant" and tool_calls:
            message['tool_calls'] = tool_calls
            # If assistant provides tool_calls, content might be None or a textual precursor.
            # If content is None and tool_calls are present, it's a pure tool_call request.

        self._messages.append(message)
        self._auto_scroll()

    def build_prompt(self, extra_messages: t.Optional[t.List[t.Dict[str, t.Any]]] = None) -> t.List[t.Dict[str, t.Any]]:
        """
        Builds a list of message dictionaries for the LLM, including system prompt,
        RAG snippets (extra_messages), and the current conversation window.
        This now returns a list of dictionaries, not a string.
        """
        final_message_list: t.List[t.Dict[str, t.Any]] = []

        if self.system_prompt:
            final_message_list.append({'role': 'system', 'content': self.system_prompt})

        window_messages_internal = []
        current_start_idx = 0
        if self._messages:
            current_start_idx = max(0, min(self._start, len(self._messages) - 1))
            window_messages_internal = self._messages[current_start_idx:]

        # Combine extra messages (like RAG snippets) with the main conversation window
        # RAG snippets in extra_messages might need role transformation if they are not already "user" or "assistant"
        # For now, assume they are properly formatted or are simple "user"/"assistant" content.
        # Or, they could be custom roles that the _format_messages_for_token_counting handles.

        all_messages_to_process = (extra_messages or []) + window_messages_internal

        # Store for debugging/testing what actually went into prompt build
        self._messages_for_prompt_build_debug: t.List[t.Dict[str, t.Any]] = all_messages_to_process

        for msg_internal in all_messages_to_process:
            role_internal = msg_internal.get('r', 'unknown')
            content_internal = msg_internal.get('c') # Can be None for assistant with tool_calls

            # Translate internal keys to OpenAI expected keys
            # 'r' -> 'role', 'c' -> 'content', 'nm' -> 'name', 'tcid' -> 'tool_call_id'
            # 'tool_calls' remains 'tool_calls' for assistant message

            output_msg: t.Dict[str, t.Any] = {'role': role_internal}

            if content_internal is not None:
                output_msg['content'] = content_internal

            # Handle image_path for user messages (assuming multimodal model takes it this way or runner adapts)
            # OpenAI format for images is typically a list of content blocks, one text, one image_url.
            # This simplified structure might need adaptation by the runner.
            if msg_internal.get('image_path') and role_internal == 'user':
                # This is a simplification. OpenAI expects content to be a list for multimodal.
                # e.g., "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
                # For now, we'll pass it as a top-level key and assume runner handles it or it's for string formatting only.
                # If build_prompt is to be strictly OpenAI compliant for multimodal, this needs more work.
                # However, the task focuses on tool calls.
                output_msg['image_path_for_runner_handling'] = msg_internal['image_path']


            if role_internal == "tool":
                output_msg['tool_call_id'] = msg_internal.get('tcid')
                output_msg['name'] = msg_internal.get('nm')

            if role_internal == "assistant" and msg_internal.get('tool_calls'):
                output_msg['tool_calls'] = msg_internal.get('tool_calls')
                # If content is None for an assistant message with tool_calls, OpenAI API requires content to be null or not present.
                # If content was explicitly set to None, it will be absent from output_msg if not None.
                # If content was an empty string, it will be included.
                # OpenAI API states: "content is required for all messages except for assistant messages with tool calls."
                if output_msg.get('content') is None and 'tool_calls' in output_msg:
                    pass # content can be null/absent
                elif output_msg.get('content') == "" and 'tool_calls' in output_msg:
                     output_msg['content'] = None # Explicitly set to None if it was empty string but tool_calls are present

            final_message_list.append(output_msg)

        return final_message_list

    def _get_prompt_string_for_fitting(self) -> str:
        """
        Helper function to get the current prompt as a string for token counting (_fit).
        Uses the new build_prompt to get the list of messages, then formats it.
        """
        messages_for_fitting = self.build_prompt() # Gets the structured list
        # The first message in messages_for_fitting is the system prompt if present.
        # _format_messages_for_token_counting expects a list of messages without a separate system prompt arg.

        # Reconstruct a string similar to how it was before for token counting purposes.
        # This is a temporary bridge until tokenizers can handle lists of messages directly.
        string_parts = []
        processed_system_prompt = False
        if messages_for_fitting and messages_for_fitting[0]['role'] == 'system':
            string_parts.append(messages_for_fitting[0]['content']) # System prompt content
            messages_to_format_for_counting = messages_for_fitting[1:]
            processed_system_prompt = True
        else:
            messages_to_format_for_counting = messages_for_fitting

        formatted_user_assistant_etc_string = self._format_messages_for_token_counting(messages_to_format_for_counting)

        if processed_system_prompt:
            if formatted_user_assistant_etc_string: # Avoid double newline if no other messages
                 return string_parts[0] + "\n" + formatted_user_assistant_etc_string
            return string_parts[0] # Only system prompt
        return formatted_user_assistant_etc_string.lstrip('\n')


    def _fit(self):
        if not self._messages:
            self._start = 0
            return

        self._start = max(0, min(self._start, len(self._messages) - 1))

        # Use the helper to get a string representation for token counting
        current_prompt_str_for_counting = self._get_prompt_string_for_fitting()
        current_tokens = self.tokenizer.count_tokens(current_prompt_str_for_counting)

        while current_tokens > self.max_tokens and self._start < (len(self._messages) - 1):
            self._start += 1
            current_prompt_str_for_counting = self._get_prompt_string_for_fitting()
            current_tokens = self.tokenizer.count_tokens(current_prompt_str_for_counting)

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
    built_prompt_list = manager.build_prompt()
    print(f"\nAfter 1st message (user with image):\n{built_prompt_list}")
    print(f"Manager window starts at: {manager._start}")

    manager.add("assistant", "I see a cat playing with a ball of yarn.")
    print(f"\nAfter 2nd message (assistant text reply):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Message with only an image (content can be empty string or None)
    manager.add("user", None, image_path="path/to/image2.png")
    print(f"\nAfter 3rd message (user with only image):\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # Test tool call scenario
    print("\n--- Testing Tool Call Scenario ---")
    # Assistant requests a tool call
    manager.add(
        role="assistant",
        content=None, # No textual reply, just tool calls
        tool_calls=[{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}}]
    )
    print(f"\nAfter assistant requests tool call:\n{manager.build_prompt()}")
    # User (or system) adds the tool's response
    manager.add(role="tool", content='{"temperature": "20", "unit": "celsius"}', tool_call_id="call_123", name="get_weather")
    print(f"\nAfter tool response:\n{manager.build_prompt()}")

    # Check the string version for token counting
    print(f"String for token counting after tool response:\n{manager._get_prompt_string_for_fitting()}")


    print("\n--- Testing context fitting with multiple messages including images and tools ---")
    manager.add("user", "This is a short text message to add more content to the context window.")
    print(f"\nAfter user text message:\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    manager.add("assistant", "Okay, I see that text message following the image and tool interaction.")
    print(f"\nAfter assistant text message:\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")

    # max_tokens is 150. Let's check the string representation for fitting for the current prompt.
    # System: You are a multimodal helpful assistant. (46)
    # user: [IMAGE: path/to/image1.jpg] What do you see in this image?\n (user:  (6) + [IMAGE: path/to/image1.jpg] (28) + What do you see in this image? (30) + NL (1) = 65)
    # assistant: I see a cat playing with a ball of yarn.\n (assistant:  (11) + I see a cat playing with a ball of yarn. (46) + NL (1) = 58)
    # user: [IMAGE: path/to/image2.png}]\n (user:  (6) + [IMAGE: path/to/image2.png}] (29) + NL (1) = 36)
    # assistant: [Requesting tool calls]\n (assistant:  (11) + [Requesting tool calls] (24) + NL(1) = 36)
    # tool (tool_call_id=call_123, name=get_weather): {"temperature": "20", "unit": "celsius"}\n
    # (tool (tool_call_id=call_123, name=get_weather):  (50) + {"temperature": "20", "unit": "celsius"} (42) + NL(1) = 93)
    # user: This is a short text message to add more content to the context window.\n (user:  (6) + This is a short text message to add more content to the context window. (70) + NL(1) = 77)
    # assistant: Okay, I see that text message following the image and tool interaction.\n (assistant:  (11) + Okay, I see that text message following the image and tool interaction. (76) + NL(1) = 88)
    # Total if all included: 46 + 65 + 58 + 36 + 36 + 93 + 77 + 88 = 499. Max is 150.
    # _fit should have moved _start.

    print("\nJumping to start of conversation (index 0):")
    manager.jump_to(0)
    print(f"\nAfter jump_to(0) and _fit:\n{manager.build_prompt()}")
    print(f"Manager window starts at: {manager._start}")
    # Expected: _start to be high, e.g. 6 or 7 if only last message fits with system prompt.
    # Sys (46) + last assistant msg (88) = 134. Fits. So _start should be 7 (index of last message)
    # Let's verify:
    # If _start = 6 (user: This is a short text message...): Sys (46) + user (77) + assistant (88) = 211 (too much)
    # So _start should indeed be 7 if my manual token counts for the string format are roughly correct.
    # The MockTokenizer counts characters, so actual counts would differ with a real tokenizer.
    # For MockTokenizer:
    # Sys: 46, NL:1
    # msg6 (user): "user: This is a short text message to add more content to the context window.\n" -> 6 + 70 + 1 = 77
    # msg7 (asst): "assistant: Okay, I see that text message following the image and tool interaction.\n" -> 11 + 76 + 1 = 88
    # If _start = 6: 46 + 1 + 77 + 88 = 212. > 150. So _start moves to 7.
    # If _start = 7: 46 + 1 + 88 = 135. Fits.
    assert manager._start == 7, f"Expected _start to be 7, got {manager._start}"


    print("\n--- Testing prompt building with 'extra_messages' (e.g., RAG snippets) ---")
    rag_snippets_internal_format = [ # Assuming these are already in internal format
        {"r": "retrieved_pdf_chunk", "c": "Climate change refers to long-term shifts in temperatures and weather patterns. (Source: report.pdf, Page: 1)"},
        {"r": "retrieved_context", "c": "User previously asked about mitigation strategies."}
    ]
    prompt_list_with_rag = manager.build_prompt(extra_messages=rag_snippets_internal_format)
    print(f"\nPrompt list with RAG snippets (current window is last msg):\n{prompt_list_with_rag}")

    # Check formatting for token counting for these RAG snippets
    rag_string_for_counting = manager._format_messages_for_token_counting(rag_snippets_internal_format)
    print(f"RAG snippets formatted for token counting:\n{rag_string_for_counting}")
    assert "retrieved_pdf_chunk: Climate change refers to long-term shifts" in rag_string_for_counting
    assert "retrieved_context: User previously asked about mitigation strategies." in rag_string_for_counting

    # The final list should contain system, then RAG, then the window message(s)
    assert prompt_list_with_rag[0]['role'] == 'system'
    assert prompt_list_with_rag[1]['role'] == 'retrieved_pdf_chunk'
    assert prompt_list_with_rag[2]['role'] == 'retrieved_context'
    assert prompt_list_with_rag[3]['role'] == 'assistant' # Because _start is 7 (last message)

    print("\nContextManager tool call, multimodal, and RAG snippet demo complete.")
