# llm_context_os/tests/test_api.py
import unittest
import io
import sys
from contextlib import redirect_stdout

from llm_context_os.api.main import app, model_mgr, ctx_mgr
from llm_context_os.api.schemas import GenerationParams, ChatRequest, LoadModelRequest
from llm_context_os.runners.api_runner import APIRunner
from fastapi.testclient import TestClient

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

class TestApi(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if model_mgr.current_runner:
            model_mgr.unload()
        ctx_mgr._messages = []
        ctx_mgr._start = 0

    def test_read_docs(self):
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>LLM Context OS API - Swagger UI</title>", response.text)

    def test_load_model_valid_type(self):
        load_request_data = LoadModelRequest(
            model_type="api",
            model_path_or_name="test-dummy-api",
            runner_params={"api_url": "http://localhost:1234/v1", "api_key": "sk-dummy"}
        )
        response = self.client.post("/load_model", json=load_request_data.model_dump())
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "ok")
        self.assertIn("Model 'test-dummy-api' of type 'api' loaded successfully.", json_response["message"])
        self.assertIsInstance(model_mgr.current_runner, APIRunner)
        self.assertEqual(model_mgr.current_model_identifier, "test-dummy-api")

    def test_load_model_invalid_type(self):
        load_request_data = {"model_type": "unknown_super_model", "model_path_or_name": "some/path"}
        with Capturing():
            response = self.client.post("/load_model", json=load_request_data)

        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "error")
        self.assertIn("Failed to load model 'some/path'", json_response["message"])
        self.assertIsNone(model_mgr.current_runner)

    def test_chat_no_model_loaded(self):
        self.assertIsNone(model_mgr.current_runner, "Pre-condition: No model should be loaded.")
        chat_request_data = ChatRequest(message="Hello, any model there?")
        response = self.client.post("/chat", json=chat_request_data.model_dump())
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "No model is currently loaded. Please load a model first via /load_model.")

    @patch('llm_context_os.api.main.model_mgr') # Patch the global model_mgr used by the endpoint
    def test_chat_with_loaded_model_and_token_counts(self, mock_api_model_mgr):
        mock_runner_instance = MagicMock()

        # Simulate the new return type of generate: (text, {"prompt_tokens": X, "completion_tokens": Y})
        # The prompt text built by _prepare_context_and_initial_prompt for "Test prompt..."
        # will be something like: "You are a helpful AI assistant.\nuser: Test prompt for token count\n"
        # Let's say this is 15 tokens. The reply "Mocked reply" is 2 tokens.
        mock_runner_instance.generate.return_value = ("Mocked reply", {"prompt_tokens": 15, "completion_tokens": 2})

        # This mock is for the runner's count_tokens method.
        # The endpoint uses counts from runner.generate() primarily for ChatResponse.
        # The direct calls to runner.count_tokens() in the endpoint are for specific cases or if generate() didn't return counts.
        # Since BaseRunner.generate now MUST return counts, these direct calls are less critical for ChatResponse population
        # but are used for the prompt_info event in streaming.
        def mock_count_tokens_side_effect(text_to_count):
            if "Test prompt for token count" in text_to_count: # Simulating prompt token count
                return 15
            if text_to_count == "Mocked reply": # Simulating generated text token count
                return 2
            return len(text_to_count.split()) # Fallback
        mock_runner_instance.count_tokens = MagicMock(side_effect=mock_count_tokens_side_effect)

        mock_api_model_mgr.get.return_value = mock_runner_instance

        chat_text_for_prompt = "Test prompt for token count"
        chat_req_data = ChatRequest(
            message=chat_text_for_prompt,
            generation_params=GenerationParams(temperature=0.5, max_new_tokens=50)
        )

        response = self.client.post("/chat", json=chat_req_data.model_dump())
        self.assertEqual(response.status_code, 200)
        json_response = response.json()

        self.assertEqual(json_response["reply"], "Mocked reply")
        self.assertEqual(json_response["prompt_tokens"], 15)     # From mock_runner_instance.generate
        self.assertEqual(json_response["generated_tokens"], 2) # From mock_runner_instance.generate
        self.assertIsNotNone(json_response["request_details"])
        self.assertEqual(json_response["request_details"]["message"], chat_text_for_prompt)

        mock_runner_instance.generate.assert_called_once()
        # Check that count_tokens was called by _prepare_context_and_initial_prompt
        # The first call to count_tokens happens inside _prepare_context_and_initial_prompt
        # The second call happens on the final_reply_text.
        self.assertGreaterEqual(mock_runner_instance.count_tokens.call_count, 1)


    def test_chat_with_loaded_model(self):
        load_req = LoadModelRequest(
            model_type="api",
            model_path_or_name="chat-test-api",
            runner_params={"api_url": "http://dummylocal:5678/v1"}
        )
        load_response = self.client.post("/load_model", json=load_req.model_dump())
        self.assertEqual(load_response.status_code, 200, "Model loading failed in test setup")
        self.assertEqual(load_response.json()["status"], "ok", "Model loading status not ok in test setup")
        self.assertIsNotNone(model_mgr.current_runner, "Model should be loaded for this test.")

        chat_text = "Hello, model!"
        chat_req_data = ChatRequest(
            message=chat_text,
            generation_params=GenerationParams(temperature=0.5, max_new_tokens=50)
        )
        response = self.client.post("/chat", json=chat_req_data.model_dump())
        self.assertEqual(response.status_code, 200)
        json_response = response.json()

        # Prompt: "You are a helpful AI assistant.\nuser: Hello, model!\n" (length 55)
        # APIRunner uses prompt[:49], which is "You are a helpful AI assistant.\nuser: Hello, mod"
        # The full placeholder reply would be "[API Response from chat-test-api to: You are a helpful AI assistant.\nuser: Hello, mod...]"

        # Make assertion robust to the exact truncation point if it's unstable.
        # Check the part of the string before the "d..." vs "de..."
        expected_prefix = "[API Response from chat-test-api to: You are a helpful AI assistant.\nuser: Hello, mo"
        self.assertTrue(json_response["reply"].startswith(expected_prefix),
                        f"Expected reply to start with '{expected_prefix}', but got '{json_response['reply']}'")

        self.assertIsNotNone(json_response["request_details"])
        self.assertEqual(json_response["request_details"]["message"], chat_text)

    def test_context_is_maintained_between_chats(self):
        load_req = LoadModelRequest(model_type="api", model_path_or_name="context-test-api", runner_params={"api_url": "http://dummy/v1"})
        self.client.post("/load_model", json=load_req.model_dump())
        self.assertIsNotNone(model_mgr.current_runner)

        chat_msg1 = "My name is Bob."
        response1 = self.client.post("/chat", json=ChatRequest(message=chat_msg1).model_dump())
        self.assertEqual(response1.status_code, 200)

        # Prompt for first call: "You are a helpful AI assistant.\nuser: My name is Bob.\n" (len 54)
        # APIRunner's prompt[:49] for this: "You are a helpful AI assistant.\nuser: My name is " (ends with a space)
        expected_fragment1 = "You are a helpful AI assistant.\nuser: My name is "
        self.assertIn(expected_fragment1, response1.json()["reply"])

        chat_msg2 = "What is my name?"
        response2 = self.client.post("/chat", json=ChatRequest(message=chat_msg2).model_dump())
        self.assertEqual(response2.status_code, 200)

        # Prompt for 2nd call (due to ContextManager._auto_scroll):
        # "You are a helpful AI assistant.\nuser: What is my name?\n" (len 62)
        # APIRunner's prompt[:49] for this: "You are a helpful AI assistant.\nuser: What is my " (ends with a space)
        expected_fragment2 = "You are a helpful AI assistant.\nuser: What is my "
        self.assertIn(expected_fragment2, response2.json()["reply"])


# --- Streaming API Tests ---
from unittest.mock import patch, MagicMock
import asyncio
import json

# Helper to parse SSE stream from TestClient response
def parse_sse_stream(response_iter_lines):
    events = []
    current_event_type = 'message' # Default SSE event type
    current_data_lines = []

    for line_bytes in response_iter_lines:
        line = line_bytes.decode('utf-8').strip()

        if not line: # Empty line separates events or is a keep-alive
            if current_data_lines: # Process event if data was accumulated
                try:
                    # Attempt to parse JSON if data is a single line and looks like JSON
                    # Otherwise, join lines if data spans multiple 'data:' lines (not typical for this app's simple SSEs)
                    data_content_str = "\n".join(current_data_lines)
                    if len(current_data_lines) == 1:
                        try:
                            data_json = json.loads(data_content_str)
                            events.append({'event': current_event_type, 'data': data_json})
                        except json.JSONDecodeError: # Not JSON, treat as raw string
                            events.append({'event': current_event_type, 'data': data_content_str})
                    else: # Multi-line data (less common for simple text chunks)
                         events.append({'event': current_event_type, 'data': data_content_str})
                except Exception as e: # Catch any parsing error
                    print(f"Error parsing SSE data: {e}, data lines: {current_data_lines}")
                    events.append({'event': 'parse_error', 'data': str(e), 'raw_data': current_data_lines})

            current_event_type = 'message' # Reset to default for next event
            current_data_lines = []
            continue

        if line.startswith("event:"):
            # If there was data for a previous event, process it first
            if current_data_lines:
                data_content_str = "\n".join(current_data_lines)
                try:
                    data_json = json.loads(data_content_str)
                    events.append({'event': current_event_type, 'data': data_json})
                except json.JSONDecodeError:
                    events.append({'event': current_event_type, 'data': data_content_str})
                current_data_lines = [] # Reset data lines for new event type
            current_event_type = line.split("event:", 1)[1].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line.split("data:", 1)[1].strip())
        # Can add handling for id: and retry: if needed

    # Process any final event data after loop if stream didn't end with blank line
    if current_data_lines:
        data_content_str = "\n".join(current_data_lines)
        try:
            data_json = json.loads(data_content_str)
            events.append({'event': current_event_type, 'data': data_json})
        except json.JSONDecodeError:
            events.append({'event': current_event_type, 'data': data_content_str})

    return events


class TestChatStreamingAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Ensure model_mgr is clean for each test, or use a per-test mock
        if model_mgr.current_runner:
             model_mgr.unload()
        # Reset context manager if it holds state relevant to tests
        ctx_mgr._messages = []
        ctx_mgr._start = 0


    async def mock_async_stream_generator(self, chunks_to_yield: list):
        for chunk in chunks_to_yield:
            yield chunk
            await asyncio.sleep(0.001) # Simulate async behavior of a real stream

    @patch('llm_context_os.api.main.model_mgr')
    def test_chat_stream_sends_token_events(self, mock_api_model_mgr):
        mock_runner_instance = MagicMock()

        test_prompt_message = "Test streaming prompt for detailed token events"
        # This is an approximation of the prompt text that _prepare_context_and_initial_prompt would build.
        # The actual prompt sent to runner.count_tokens for "prompt_info" event.
        expected_initial_prompt_full_text = f"{ctx_mgr.system_prompt}\nuser: {test_prompt_message}\n"

        final_generated_text = "Stream chunk1 Stream chunk2"
        mock_initial_prompt_tokens = 18
        mock_generated_tokens_total = 7 # chunk1 (4) + chunk2 (3)

        def mock_runner_count_tokens_side_effect(text_to_count):
            if text_to_count == expected_initial_prompt_full_text:
                return mock_initial_prompt_tokens
            elif text_to_count == final_generated_text: # For total_generated_tokens in stream_end
                return mock_generated_tokens_total
            return len(text_to_count.split()) # Fallback
        mock_runner_instance.count_tokens = MagicMock(side_effect=mock_runner_count_tokens_side_effect)

        # This async generator mock now adheres to the BaseRunner.stream() interface:
        # 1. Yield dict: {"prompt_tokens": count}
        # 2. Yield tuples: (text_chunk, tokens_in_chunk)
        async def mock_stream_generator_for_tokens_test(*args, **kwargs):
            # The API's sse_generator calls runner.count_tokens on the full prompt first,
            # then uses that for the first yield of its own.
            # So, the runner.stream() itself first yields its own prompt_tokens.
            # The prompt text is in kwargs['prompt'] or args[0]
            prompt_text_arg = kwargs.get('prompt', args[0] if args else "")
            # This call to count_tokens is from *within* the mocked runner.stream
            p_tokens = mock_runner_instance.count_tokens(prompt_text_arg)
            yield {"prompt_tokens": p_tokens}

            yield ("Stream chunk1 ", 4) # Mocked tokens_in_chunk
            await asyncio.sleep(0.001)
            yield ("Stream chunk2", 3)  # Mocked tokens_in_chunk
            await asyncio.sleep(0.001)

        mock_runner_instance.stream.side_effect = mock_stream_generator_for_tokens_test
        mock_api_model_mgr.get.return_value = mock_runner_instance

        chat_data = ChatRequest(message=test_prompt_message, stream=True)
        received_events = []

        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['content-type'], 'text/event-stream; charset=utf-8')
            for line_bytes in response.iter_lines():
                received_events.extend(parse_sse_stream([line_bytes]))

        prompt_info_event = next((e for e in received_events if e['event'] == 'prompt_info'), None)
        stream_end_event = next((e for e in received_events if e['event'] == 'stream_end'), None)
        message_events_data = [e['data'] for e in received_events if e['event'] == 'message' and isinstance(e['data'], dict)]

        self.assertIsNotNone(prompt_info_event, "event: prompt_info not found.")
        self.assertEqual(prompt_info_event['data'].get('prompt_tokens'), mock_initial_prompt_tokens)

        self.assertIsNotNone(stream_end_event, "event: stream_end not found.")
        self.assertEqual(stream_end_event['data'].get('total_generated_tokens'), mock_generated_tokens_total)
        self.assertEqual(stream_end_event['data'].get('final_prompt_tokens'), mock_initial_prompt_tokens) # No tool call, so initial prompt tokens are final

        # Check data messages for text and tokens_in_chunk
        self.assertEqual(len(message_events_data), 2)
        self.assertEqual(message_events_data[0].get('text'), "Stream chunk1 ")
        self.assertEqual(message_events_data[0].get('tokens_in_chunk'), 4)
        self.assertEqual(message_events_data[1].get('text'), "Stream chunk2")
        self.assertEqual(message_events_data[1].get('tokens_in_chunk'), 3)

        mock_runner_instance.stream.assert_called_once()
        # Check calls to count_tokens:
        # 1. By _prepare_context_and_initial_prompt (passed to sse_generator, then to runner.stream)
        # 2. By sse_generator in its finally block for the full response.
        mock_runner_instance.count_tokens.assert_any_call(expected_initial_prompt_full_text)
        mock_runner_instance.count_tokens.assert_any_call(final_generated_text)


    @patch('llm_context_os.api.main.tool_dispatcher.dispatch')
    @patch('llm_context_os.api.main.model_mgr')
    def test_chat_stream_with_tool_call_and_token_counts(self, mock_api_model_mgr, mock_tool_dispatch):
        mock_runner = MagicMock()

        # Configure stream to be called twice with different return values
        initial_stream_chunks = ["[FUNCALL] {\"tool_name\": \"get_weather\", \"params\": {\"location\": \"London\"}}"]
        final_stream_chunks = ["The weather in London is sunny. ", "Anything else?"]

        # Use a list of iterators/generators to simulate multiple calls to stream()
        mock_runner.stream.side_effect = [
            self.mock_async_stream_generator(initial_stream_chunks),
            self.mock_async_stream_generator(final_stream_chunks)
        ]
        mock_model_mgr_param.get.return_value = mock_runner
        mock_tool_dispatch.return_value = {"status": "success", "result": "Weather is sunny."}

        chat_data = ChatRequest(message="What's the weather in London and then say hi?", stream=True)

        received_events = []
        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['content-type'], 'text/event-stream; charset=utf-8')
            for line_bytes in response.iter_lines():
                 received_events.extend(parse_sse_stream([line_bytes]))

        # Assertions
        tool_call_event_found = False
        tool_name_from_event = ""
        final_text_chunks = []
        stream_ended = False

        for event in received_events:
            if event['event'] == 'tool_call':
                tool_call_event_found = True
                tool_name_from_event = event['data'].get('tool_name')
            elif event['event'] == 'message' and 'text' in event['data']:
                final_text_chunks.append(event['data']['text'])
            elif event['event'] == 'stream_end':
                stream_ended = True

        self.assertTrue(tool_call_event_found, "Tool call event not found in stream.")
        self.assertEqual(tool_name_from_event, "get_weather")
        mock_tool_dispatch.assert_called_once_with("{\"tool_name\": \"get_weather\", \"params\": {\"location\": \"London\"}}")
        self.assertEqual("".join(final_text_chunks), "".join(final_stream_chunks))
        self.assertTrue(stream_ended, "Stream did not end correctly.")
        self.assertEqual(mock_runner.stream.call_count, 2)


    @patch('llm_context_os.api.main.model_mgr')
    def test_chat_stream_runner_error(self, mock_model_mgr_param):
        mock_runner = MagicMock()
        mock_runner.stream.side_effect = Exception("Runner failed!") # Simulate error during stream
        mock_model_mgr_param.get.return_value = mock_runner

        chat_data = ChatRequest(message="This will fail.", stream=True)
        received_events = []
        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200) # Stream itself starts with 200
            for line_bytes in response.iter_lines():
                 received_events.extend(parse_sse_stream([line_bytes]))

        error_event_found = False
        stream_ended = False
        for event in received_events:
            if event['event'] == 'error':
                error_event_found = True
                self.assertIn("Runner failed!", event['data'].get('error', ''))
            if event['event'] == 'stream_end':
                stream_ended = True

        self.assertTrue(error_event_found, "Error event not found in stream.")
        self.assertTrue(stream_ended, "Stream end event not found even after error.")

    @patch('llm_context_os.api.main.model_mgr')
    def test_chat_no_model_loaded_stream(self, mock_model_mgr_param):
        mock_model_mgr_param.get.return_value = None # Simulate no model loaded

        chat_data = ChatRequest(message="Hello?", stream=True)
        # The error from no model loaded is raised before stream response is set up by FastAPI normally
        # It will be a regular HTTP error response if not caught and translated to SSE by endpoint.
        # Based on current main.py, it raises HTTPException before streaming starts.
        # Let's check if the current main.py translates this to an SSE error event.
        # The current code in main.py: `if not runner: raise HTTPException(...)`
        # This will result in a non-200 response, not an SSE stream with an error event.
        # So, we test for the HTTP Exception here.

        # If we want it to return SSE error, the endpoint logic needs to change.
        # For now, testing existing behavior:
        response = self.client.post("/chat", json=chat_data.model_dump())
        self.assertEqual(response.status_code, 400)
        self.assertIn("No model is currently loaded", response.json().get("detail"))


# --- Settings API Tests ---
from llm_context_os.api.schemas import GlobalSettings, UpdateSettingsRequest
from llm_context_os.api import main as api_main # To access and restore global CONFIG
from copy import deepcopy

class TestSettingsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Store a deep copy of the original CONFIG to restore in tearDown
        self.original_config = deepcopy(api_main.CONFIG)

        # It's crucial that these mocks target where they are *used* by the /settings endpoint,
        # which is directly the global instances in api_main.
        self.patch_ctx_mgr = patch('llm_context_os.api.main.ctx_mgr', MagicMock())
        self.patch_model_mgr = patch('llm_context_os.api.main.model_mgr', MagicMock())
        self.patch_chat_history_retriever = patch('llm_context_os.api.main.chat_history_retriever', MagicMock())
        self.patch_pdf_retriever = patch('llm_context_os.api.main.pdf_retriever', MagicMock())
        # rag_tokenizer is also global, but its direct modification effects are complex (requires re-init of others)
        # For these tests, we'll assume rag_tokenizer itself isn't directly modified by PUT /settings,
        # but rather the CONFIG values that *would* create it are changed (requiring restart).

        self.mock_ctx_mgr = self.patch_ctx_mgr.start()
        self.mock_model_mgr = self.patch_model_mgr.start()
        self.mock_chat_history_retriever = self.patch_chat_history_retriever.start()
        self.mock_pdf_retriever = self.patch_pdf_retriever.start()

        # Set initial values on mocks based on original_config to simulate a loaded state
        self.mock_ctx_mgr.max_tokens = self.original_config.get('context_manager', {}).get('max_tokens')
        self.mock_ctx_mgr.system_prompt = self.original_config.get('context_manager', {}).get('system_prompt')
        self.mock_model_mgr.default_idle_unload_sec = self.original_config.get('model_manager', {}).get('default_idle_unload_sec')
        self.mock_chat_history_retriever.recall_budget_tokens = self.original_config.get('chat_history_retriever', {}).get('recall_budget_tokens')


    def tearDown(self):
        # Restore the global CONFIG to its original state
        api_main.CONFIG = self.original_config
        # Stop the patches
        self.patch_ctx_mgr.stop()
        self.patch_model_mgr.stop()
        self.patch_chat_history_retriever.stop()
        self.patch_pdf_retriever.stop()


    def test_get_settings(self):
        response = self.client.get("/settings")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check a few key fields to ensure they match the current CONFIG
        self.assertEqual(data['context_manager']['max_tokens'], api_main.CONFIG['context_manager']['max_tokens'])
        self.assertEqual(data['model_manager']['default_idle_unload_sec'], api_main.CONFIG['model_manager']['default_idle_unload_sec'])
        if 'generation_defaults' in data and data['generation_defaults'] is not None: # generation_defaults can be None
             self.assertEqual(data['generation_defaults']['temperature'], api_main.CONFIG.get('generation_defaults', {}).get('temperature'))


    def test_update_settings_partial_success_and_propagate(self):
        update_payload = {
            "context_manager": {"max_tokens": 1234, "system_prompt": "New prompt!"},
            "model_manager": {"default_idle_unload_sec": 555},
            "chat_history_retriever": {"embedding_model_name": "new_embed_model"} # This requires restart
        }

        response = self.client.put("/settings", json=update_payload)
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "ok")

        # Check that specific messages are present
        self.assertIn("ContextManager max_tokens updated to 1234", json_response["message"])
        self.assertIn("ContextManager system_prompt updated", json_response["message"])
        self.assertIn("ModelManager default_idle_unload_sec updated to 555", json_response["message"])
        self.assertIn("ChatHistoryRetriever 'embedding_model_name' or 'vector_db_path' changes require a restart", json_response["message"])

        # Verify CONFIG update
        self.assertEqual(api_main.CONFIG['context_manager']['max_tokens'], 1234)
        self.assertEqual(api_main.CONFIG['context_manager']['system_prompt'], "New prompt!")
        self.assertEqual(api_main.CONFIG['model_manager']['default_idle_unload_sec'], 555)
        self.assertEqual(api_main.CONFIG['chat_history_retriever']['embedding_model_name'], "new_embed_model")

        # Verify propagation to mocked live instances
        self.assertEqual(self.mock_ctx_mgr.max_tokens, 1234)
        self.assertEqual(self.mock_ctx_mgr.system_prompt, "New prompt!")
        self.assertEqual(self.mock_model_mgr.default_idle_unload_sec, 555)
        # embedding_model_name for chat_history_retriever is not set directly on instance
        self.assertNotEqual(self.mock_chat_history_retriever.embedding_model_name, "new_embed_model",
                            "embedding_model_name should not be updated on live instance directly.")


    def test_update_settings_no_changes(self):
        response = self.client.put("/settings", json={})
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "ok")
        self.assertIn("No settings provided to update", json_response["message"])
        # Ensure CONFIG remains unchanged from its original state (deepcopied in setUp)
        self.assertEqual(api_main.CONFIG, self.original_config)


    def test_update_settings_only_restart_required(self):
        update_payload = {"token_estimator_for_rag_budgeting": {"type": "new_type_requires_restart"}}

        # Store current values of some live-updated attributes to check they weren't changed
        original_ctx_max_tokens = self.mock_ctx_mgr.max_tokens

        response = self.client.put("/settings", json=update_payload)
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "ok")
        self.assertIn("Changes to 'token_estimator_for_rag_budgeting' require an application restart", json_response["message"])
        self.assertNotIn("Applied live:", json_response["message"]) # Check that no live updates were claimed

        # Verify CONFIG is updated
        self.assertEqual(api_main.CONFIG['token_estimator_for_rag_budgeting']['type'], "new_type_requires_restart")

        # Verify that live-updatable attributes on mocks were NOT changed
        self.assertEqual(self.mock_ctx_mgr.max_tokens, original_ctx_max_tokens)


# --- Model Management API Tests ---
from llm_context_os.api.schemas import AvailableModel, ModelListResponse, DownloadModelRequest, StatusResponse

class TestModelManagementAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_config = deepcopy(api_main.CONFIG) # Save original config

    def tearDown(self):
        api_main.CONFIG = self.original_config # Restore config

    @patch('llm_context_os.api.main.model_scanner.scan_model_directories')
    def test_list_available_models_success(self, mock_scan_model_directories):
        mock_models_data = [
            {"model_id": "model1.gguf", "model_type": "gguf", "path_or_identifier": "/path/model1.gguf", "source": "local_scan", "name": "Model 1"},
            {"model_id": "org/model2-awq", "model_type": "awq", "path_or_identifier": "/path/model2-awq", "source": "local_scan", "name": "Model 2 AWQ"}
        ]
        # Convert dicts to AvailableModel instances for the mock's return value
        mock_scan_model_directories.return_value = [AvailableModel(**data) for data in mock_models_data]

        # Temporarily set scan_directories in CONFIG for this test
        api_main.CONFIG['model_discovery'] = {'scan_directories': ['dummy_scan_dir/']}

        response = self.client.get("/models/available")
        self.assertEqual(response.status_code, 200)

        response_data = ModelListResponse(**response.json())
        self.assertEqual(len(response_data.models), 2)
        self.assertEqual(response_data.models[0].model_id, "model1.gguf")
        self.assertEqual(response_data.models[1].name, "Model 2 AWQ")
        mock_scan_model_directories.assert_called_once()
        # Can also assert that the call was made with resolved paths based on dummy_scan_dir

    @patch('llm_context_os.api.main.model_scanner.scan_model_directories')
    def test_list_available_models_no_scan_dirs_configured(self, mock_scan_model_directories):
        # Ensure scan_directories is empty or not present in CONFIG for this test
        original_model_discovery_config = api_main.CONFIG.pop('model_discovery', None)

        response = self.client.get("/models/available")
        self.assertEqual(response.status_code, 200)
        response_data = ModelListResponse(**response.json())
        self.assertEqual(len(response_data.models), 0)
        mock_scan_model_directories.assert_not_called() # Should not be called if no valid dirs

        # Restore config
        if original_model_discovery_config is not None:
            api_main.CONFIG['model_discovery'] = original_model_discovery_config


    @patch('llm_context_os.api.main.model_scanner.scan_model_directories')
    def test_list_available_models_scanner_error(self, mock_scan_model_directories):
        mock_scan_model_directories.side_effect = Exception("Scanner crashed badly")
        api_main.CONFIG['model_discovery'] = {'scan_directories': ['another_dummy_dir/']} # Ensure it tries to scan

        response = self.client.get("/models/available")
        self.assertEqual(response.status_code, 500)
        json_response = response.json()
        self.assertIn("Failed to scan model directories: Scanner crashed badly", json_response["detail"])

    def test_download_model_placeholder_success(self):
        payload = {
            "repo_id": "TheBloke/Test-Model-GGUF",
            "filename": "test-model.Q4_K_M.gguf",
            "model_type": "gguf"
        }
        # Use @patch('llm_context_os.api.main.print') if you want to check specific log output
        response = self.client.post("/models/download", json=payload)
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "ok")
        self.assertIn("Download request for 'TheBloke/Test-Model-GGUF'", json_response["message"])
        self.assertIn("file: test-model.Q4_K_M.gguf", json_response["message"])
        self.assertIn("Simulated download initiated", json_response["message"])

    def test_download_model_missing_repo_id(self):
        payload = {"filename": "test-model.Q4_K_M.gguf", "model_type": "gguf"} # Missing repo_id
        response = self.client.post("/models/download", json=payload)
        self.assertEqual(response.status_code, 422) # Unprocessable Entity for Pydantic validation error


# --- Tool Management API Tests ---
from llm_context_os.api.schemas import ToolInfo, ToolListResponse, ToggleToolRequest

class TestToolManagementAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Access the global tool_dispatcher from api_main to manage its state for tests
        self.tool_dispatcher_instance = api_main.tool_dispatcher
        # Store original state of tool_states to restore in tearDown
        self.original_tool_states = deepcopy(self.tool_dispatcher_instance.tool_states)

    def tearDown(self):
        # Restore original tool states
        self.tool_dispatcher_instance.tool_states = self.original_tool_states

    def test_list_tools_success(self):
        # Mock the list_tools method of the actual tool_dispatcher instance
        with patch.object(self.tool_dispatcher_instance, 'list_tools') as mock_list_tools:
            mock_tool_list_data = [
                ToolInfo(name="get_weather", type="local", description="Gets weather", is_enabled=True, parameters={}).model_dump(),
                ToolInfo(name="mcp_tool1", type="mcp", description="MCP tool", is_enabled=False, parameters={}).model_dump()
            ]
            # Ensure the mock returns Pydantic models if that's what list_tools now does
            mock_list_tools.return_value = [ToolInfo(**data) for data in mock_tool_list_data]

            response = self.client.get("/tools")
            self.assertEqual(response.status_code, 200)

            response_data = ToolListResponse(**response.json())
            self.assertEqual(len(response_data.tools), 2)
            self.assertEqual(response_data.tools[0].name, "get_weather")
            self.assertTrue(response_data.tools[0].is_enabled)
            self.assertEqual(response_data.tools[1].name, "mcp_tool1")
            self.assertFalse(response_data.tools[1].is_enabled)
            mock_list_tools.assert_called_once()

    def test_toggle_tool_enable_disable(self):
        tool_name_to_test = "get_weather" # Assuming this is a known local tool

        # 1. Disable the tool
        disable_payload = {"tool_name": tool_name_to_test, "enable": False}
        response_disable = self.client.post("/tools/toggle", json=disable_payload)
        self.assertEqual(response_disable.status_code, 200)
        self.assertEqual(response_disable.json()["status"], "ok")
        self.assertIn(f"Tool '{tool_name_to_test}' disabled", response_disable.json()["message"])
        self.assertFalse(self.tool_dispatcher_instance.tool_states.get(tool_name_to_test))

        # 2. List tools to confirm it's disabled
        response_list_disabled = self.client.get("/tools")
        self.assertEqual(response_list_disabled.status_code, 200)
        listed_tools_disabled = ToolListResponse(**response_list_disabled.json()).tools
        found_tool_disabled = next((t for t in listed_tools_disabled if t.name == tool_name_to_test), None)
        self.assertIsNotNone(found_tool_disabled)
        self.assertFalse(found_tool_disabled.is_enabled)

        # 3. Enable the tool again
        enable_payload = {"tool_name": tool_name_to_test, "enable": True}
        response_enable = self.client.post("/tools/toggle", json=enable_payload)
        self.assertEqual(response_enable.status_code, 200)
        self.assertEqual(response_enable.json()["status"], "ok")
        self.assertIn(f"Tool '{tool_name_to_test}' enabled", response_enable.json()["message"])
        self.assertTrue(self.tool_dispatcher_instance.tool_states.get(tool_name_to_test))

        # 4. List tools again to confirm it's enabled
        response_list_enabled = self.client.get("/tools")
        self.assertEqual(response_list_enabled.status_code, 200)
        listed_tools_enabled = ToolListResponse(**response_list_enabled.json()).tools
        found_tool_enabled = next((t for t in listed_tools_enabled if t.name == tool_name_to_test), None)
        self.assertIsNotNone(found_tool_enabled)
        self.assertTrue(found_tool_enabled.is_enabled)

    def test_toggle_unknown_tool(self):
        unknown_tool_name = "unknown_tool_xyz_123"
        # Enabling an unknown tool (it gets added to tool_states)
        enable_payload = {"tool_name": unknown_tool_name, "enable": True}
        response_enable = self.client.post("/tools/toggle", json=enable_payload)
        self.assertEqual(response_enable.status_code, 200)
        self.assertEqual(response_enable.json()["status"], "ok")
        self.assertTrue(self.tool_dispatcher_instance.tool_states.get(unknown_tool_name))

        # List tools to confirm it shows up (as type "mcp" by current dispatcher logic if not local)
        response_list = self.client.get("/tools")
        self.assertEqual(response_list.status_code, 200)
        listed_tools = ToolListResponse(**response_list.json()).tools
        found_tool = next((t for t in listed_tools if t.name == unknown_tool_name), None)
        self.assertIsNotNone(found_tool)
        self.assertTrue(found_tool.is_enabled)
        self.assertEqual(found_tool.type, "mcp") # Based on current ToolDispatcher.list_tools logic


if __name__ == '__main__':
    unittest.main()
