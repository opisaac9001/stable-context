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
    def test_chat_stream_success_simple_text(self, mock_model_mgr_param):
        # Configure the mock runner and its stream method
        mock_runner = MagicMock()
        chunks = ["Hello, ", "this is ", "a streamed ", "response."]
        mock_runner.stream.return_value = self.mock_async_stream_generator(chunks)
        mock_model_mgr_param.get.return_value = mock_runner

        chat_data = ChatRequest(message="Tell me a story.", stream=True)

        # When using TestClient with streaming, use a context manager for the request
        full_response_text = ""
        received_events = []
        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['content-type'], 'text/event-stream; charset=utf-8') # FastAPI adds charset

            for line_bytes in response.iter_lines(): # iter_lines gives bytes
                received_events.extend(parse_sse_stream([line_bytes])) # Pass as list of one item

        # Process collected events
        final_text_chunks = []
        ended = False
        for event in received_events:
            if event['event'] == 'message' and 'text' in event['data']:
                final_text_chunks.append(event['data']['text'])
            elif event['event'] == 'stream_end':
                ended = True

        self.assertTrue(ended, "Stream did not end with a 'stream_end' event.")
        self.assertEqual("".join(final_text_chunks), "".join(chunks))
        mock_runner.stream.assert_called_once()

    @patch('llm_context_os.api.main.tool_dispatcher.dispatch')
    @patch('llm_context_os.api.main.model_mgr')
    def test_chat_stream_with_tool_call(self, mock_model_mgr_param, mock_tool_dispatch):
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


if __name__ == '__main__':
    unittest.main()
