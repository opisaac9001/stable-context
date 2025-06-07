# yawl/tests/test_api.py
import unittest
import io
import sys
from contextlib import redirect_stdout

from yawl.api.main import app, model_mgr, ctx_mgr # Updated
from yawl.api.schemas import GenerationParams, ChatRequest, LoadModelRequest # Updated
from yawl.runners.api_runner import APIRunner # Updated
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
        self.assertIn("<title>YAWL API - Swagger UI</title>", response.text) # Updated

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

    @patch('yawl.api.main.model_mgr') # Updated
    def test_chat_with_loaded_model_and_token_counts(self, mock_api_model_mgr):
        mock_runner_instance = MagicMock()
        mock_runner_instance.generate.return_value = ("Mocked reply", {"prompt_tokens": 15, "completion_tokens": 2})
        def mock_count_tokens_side_effect(text_to_count):
            if "Test prompt for token count" in text_to_count: return 15
            if text_to_count == "Mocked reply": return 2
            return len(text_to_count.split())
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
        self.assertEqual(json_response["prompt_tokens"], 15)
        self.assertEqual(json_response["generated_tokens"], 2)
        self.assertIsNotNone(json_response["request_details"])
        self.assertEqual(json_response["request_details"]["message"], chat_text_for_prompt)
        mock_runner_instance.generate.assert_called_once()
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
        expected_fragment1 = "You are a helpful AI assistant.\nuser: My name is "
        self.assertIn(expected_fragment1, response1.json()["reply"])
        chat_msg2 = "What is my name?"
        response2 = self.client.post("/chat", json=ChatRequest(message=chat_msg2).model_dump())
        self.assertEqual(response2.status_code, 200)
        expected_fragment2 = "You are a helpful AI assistant.\nuser: What is my "
        self.assertIn(expected_fragment2, response2.json()["reply"])

# --- Streaming API Tests ---
from unittest.mock import patch, MagicMock # Already imported, but good for clarity
import asyncio
import json

# Helper to parse SSE stream from TestClient response (content unchanged)
def parse_sse_stream(response_iter_lines):
    events = []
    current_event_type = 'message'
    current_data_lines = []
    for line_bytes in response_iter_lines:
        line = line_bytes.decode('utf-8').strip()
        if not line:
            if current_data_lines:
                try:
                    data_content_str = "\n".join(current_data_lines)
                    if len(current_data_lines) == 1:
                        try: data_json = json.loads(data_content_str); events.append({'event': current_event_type, 'data': data_json})
                        except json.JSONDecodeError: events.append({'event': current_event_type, 'data': data_content_str})
                    else: events.append({'event': current_event_type, 'data': data_content_str})
                except Exception as e: events.append({'event': 'parse_error', 'data': str(e), 'raw_data': current_data_lines})
            current_event_type = 'message'; current_data_lines = []
            continue
        if line.startswith("event:"):
            if current_data_lines:
                data_content_str = "\n".join(current_data_lines)
                try: data_json = json.loads(data_content_str); events.append({'event': current_event_type, 'data': data_json})
                except json.JSONDecodeError: events.append({'event': current_event_type, 'data': data_content_str})
                current_data_lines = []
            current_event_type = line.split("event:", 1)[1].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line.split("data:", 1)[1].strip())
    if current_data_lines:
        data_content_str = "\n".join(current_data_lines)
        try: data_json = json.loads(data_content_str); events.append({'event': current_event_type, 'data': data_json})
        except json.JSONDecodeError: events.append({'event': current_event_type, 'data': data_content_str})
    return events

class TestChatStreamingAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        if model_mgr.current_runner: model_mgr.unload()
        ctx_mgr._messages = []; ctx_mgr._start = 0
    async def mock_async_stream_generator(self, chunks_to_yield: list):
        for chunk in chunks_to_yield: yield chunk; await asyncio.sleep(0.001)

    @patch('yawl.api.main.model_mgr') # Updated
    def test_chat_stream_sends_token_events(self, mock_api_model_mgr):
        mock_runner_instance = MagicMock()
        test_prompt_message = "Test streaming prompt for detailed token events"
        expected_initial_prompt_full_text = f"{ctx_mgr.system_prompt}\nuser: {test_prompt_message}\n"
        final_generated_text = "Stream chunk1 Stream chunk2"; mock_initial_prompt_tokens = 18; mock_generated_tokens_total = 7
        def mock_runner_count_tokens_side_effect(text_to_count):
            if text_to_count == expected_initial_prompt_full_text: return mock_initial_prompt_tokens
            elif text_to_count == final_generated_text: return mock_generated_tokens_total
            return len(text_to_count.split())
        mock_runner_instance.count_tokens = MagicMock(side_effect=mock_runner_count_tokens_side_effect)
        async def mock_stream_generator_for_tokens_test(*args, **kwargs):
            prompt_text_arg = kwargs.get('prompt', args[0] if args else ""); p_tokens = mock_runner_instance.count_tokens(prompt_text_arg)
            yield {"prompt_tokens": p_tokens}; yield ("Stream chunk1 ", 4); await asyncio.sleep(0.001); yield ("Stream chunk2", 3); await asyncio.sleep(0.001)
        mock_runner_instance.stream.side_effect = mock_stream_generator_for_tokens_test
        mock_api_model_mgr.get.return_value = mock_runner_instance
        chat_data = ChatRequest(message=test_prompt_message, stream=True); received_events = []
        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200); self.assertEqual(response.headers['content-type'], 'text/event-stream; charset=utf-8')
            for line_bytes in response.iter_lines(): received_events.extend(parse_sse_stream([line_bytes]))
        prompt_info_event = next((e for e in received_events if e['event'] == 'prompt_info'), None)
        stream_end_event = next((e for e in received_events if e['event'] == 'stream_end'), None)
        message_events_data = [e['data'] for e in received_events if e['event'] == 'message' and isinstance(e['data'], dict)]
        self.assertIsNotNone(prompt_info_event); self.assertEqual(prompt_info_event['data'].get('prompt_tokens'), mock_initial_prompt_tokens)
        self.assertIsNotNone(stream_end_event); self.assertEqual(stream_end_event['data'].get('total_generated_tokens'), mock_generated_tokens_total)
        self.assertEqual(stream_end_event['data'].get('final_prompt_tokens'), mock_initial_prompt_tokens)
        self.assertEqual(len(message_events_data), 2); self.assertEqual(message_events_data[0].get('text'), "Stream chunk1 "); self.assertEqual(message_events_data[0].get('tokens_in_chunk'), 4)
        self.assertEqual(message_events_data[1].get('text'), "Stream chunk2"); self.assertEqual(message_events_data[1].get('tokens_in_chunk'), 3)
        mock_runner_instance.stream.assert_called_once()
        mock_runner_instance.count_tokens.assert_any_call(expected_initial_prompt_full_text); mock_runner_instance.count_tokens.assert_any_call(final_generated_text)

    @patch('yawl.api.main.tool_dispatcher.dispatch') # Updated
    @patch('yawl.api.main.model_mgr') # Updated
    def test_chat_stream_with_tool_call_and_token_counts(self, mock_api_model_mgr, mock_tool_dispatch):
        mock_runner_instance = MagicMock(); prompt_msg1 = "What's the weather in London and then say hi?"
        expected_prompt1_text = f"{ctx_mgr.system_prompt}\nuser: {prompt_msg1}\n"; mock_prompt1_tokens = 25
        func_call_text_content = "[FUNCALL] {\"tool_name\": \"get_weather\", \"params\": {\"location\": \"London\"}}"; mock_funcall_generated_tokens = 15
        expected_prompt2_text = "System: ... User: ... ToolResult: ... Assistant: [FUNCALL] ... User: ... "; mock_prompt2_tokens = 35
        final_reply_text = "The weather in London is sunny. Anything else?"; mock_final_reply_tokens = 10
        def mock_runner_count_tokens_side_effect(text_to_count):
            if text_to_count == expected_prompt1_text: return mock_prompt1_tokens
            if text_to_count == func_call_text_content: return mock_funcall_generated_tokens
            if expected_prompt2_text in text_to_count : return mock_prompt2_tokens
            if text_to_count == final_reply_text: return mock_final_reply_tokens
            return len(text_to_count.split())
        mock_runner_instance.count_tokens = MagicMock(side_effect=mock_runner_count_tokens_side_effect)
        async def mock_stream_initial_call(*args, **kwargs):
            prompt_text_arg = kwargs.get('prompt'); yield {"prompt_tokens": mock_runner_instance.count_tokens(prompt_text_arg)}
            yield (func_call_text_content, mock_funcall_generated_tokens); await asyncio.sleep(0.001)
        async def mock_stream_final_call(*args, **kwargs):
            prompt_text_arg = kwargs.get('prompt'); yield {"prompt_tokens": mock_runner_instance.count_tokens(prompt_text_arg)}
            yield ("The weather in London is sunny. ", 7); await asyncio.sleep(0.001); yield ("Anything else?", 3)
        mock_runner_instance.stream.side_effect = [mock_stream_initial_call, mock_stream_final_call]
        mock_api_model_mgr.get.return_value = mock_runner_instance
        mock_tool_dispatch.return_value = {"status": "success", "result": "Weather is sunny."}
        chat_data = ChatRequest(message=prompt_msg1, stream=True); received_events = []
        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200)
            for line_bytes in response.iter_lines(): received_events.extend(parse_sse_stream([line_bytes]))
        prompt_info_events = [e for e in received_events if e['event'] == 'prompt_info']
        tool_call_event = next((e for e in received_events if e['event'] == 'tool_call'), None)
        message_events_data = [e['data'] for e in received_events if e['event'] == 'message' and isinstance(e['data'], dict)]
        stream_end_event = next((e for e in received_events if e['event'] == 'stream_end'), None)
        self.assertTrue(len(prompt_info_events) >= 2); self.assertEqual(prompt_info_events[0]['data'].get('prompt_tokens'), mock_prompt1_tokens)
        self.assertEqual(prompt_info_events[0]['data'].get('context'), "initial_prompt"); self.assertIsNotNone(tool_call_event)
        self.assertEqual(tool_call_event['data'].get('tool_name'), "get_weather"); self.assertEqual(prompt_info_events[1]['data'].get('prompt_tokens'), mock_prompt2_tokens)
        self.assertEqual(prompt_info_events[1]['data'].get('context'), "after_tool_call"); self.assertEqual("".join(d['text'] for d in message_events_data if 'text' in d), final_reply_text)
        self.assertEqual(message_events_data[0].get('tokens_in_chunk'), 7); self.assertEqual(message_events_data[1].get('tokens_in_chunk'), 3)
        self.assertIsNotNone(stream_end_event); self.assertEqual(stream_end_event['data'].get('total_generated_tokens'), mock_final_reply_tokens)
        self.assertEqual(stream_end_event['data'].get('final_prompt_tokens'), mock_prompt2_tokens); self.assertEqual(mock_runner_instance.stream.call_count, 2)
        mock_tool_dispatch.assert_called_once()

    @patch('yawl.api.main.model_mgr') # Updated
    def test_chat_stream_runner_error(self, mock_api_model_mgr):
        mock_runner = MagicMock(); mock_runner.stream.side_effect = Exception("Runner failed!")
        mock_api_model_mgr.get.return_value = mock_runner # mock_api_model_mgr was mock_model_mgr_param
        chat_data = ChatRequest(message="This will fail.", stream=True); received_events = []
        with self.client.stream("POST", "/chat", json=chat_data.model_dump()) as response:
            self.assertEqual(response.status_code, 200)
            for line_bytes in response.iter_lines(): received_events.extend(parse_sse_stream([line_bytes]))
        error_event_found = False; stream_ended = False
        for event in received_events:
            if event['event'] == 'error': error_event_found = True; self.assertIn("Runner failed!", event['data'].get('error', ''))
            if event['event'] == 'stream_end': stream_ended = True
        self.assertTrue(error_event_found); self.assertTrue(stream_ended)

    @patch('yawl.api.main.model_mgr') # Updated
    def test_chat_no_model_loaded_stream(self, mock_model_mgr_param):
        mock_model_mgr_param.get.return_value = None
        chat_data = ChatRequest(message="Hello?", stream=True)
        response = self.client.post("/chat", json=chat_data.model_dump())
        self.assertEqual(response.status_code, 400)
        self.assertIn("No model is currently loaded", response.json().get("detail"))

# --- Settings API Tests ---
from yawl.api.schemas import GlobalSettings, UpdateSettingsRequest # Updated
from yawl.api import main as api_main # Updated
from copy import deepcopy

class TestSettingsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_config = deepcopy(api_main.CONFIG)
        self.patch_ctx_mgr = patch('yawl.api.main.ctx_mgr', MagicMock()) # Updated
        self.patch_model_mgr = patch('yawl.api.main.model_mgr', MagicMock()) # Updated
        self.patch_chat_history_retriever = patch('yawl.api.main.chat_history_retriever', MagicMock()) # Updated
        self.patch_pdf_retriever = patch('yawl.api.main.pdf_retriever', MagicMock()) # Updated
        self.mock_ctx_mgr = self.patch_ctx_mgr.start(); self.mock_model_mgr = self.patch_model_mgr.start()
        self.mock_chat_history_retriever = self.patch_chat_history_retriever.start(); self.mock_pdf_retriever = self.patch_pdf_retriever.start()
        self.mock_ctx_mgr.max_tokens = self.original_config.get('context_manager', {}).get('max_tokens')
        self.mock_ctx_mgr.system_prompt = self.original_config.get('context_manager', {}).get('system_prompt')
        self.mock_model_mgr.default_idle_unload_sec = self.original_config.get('model_manager', {}).get('default_idle_unload_sec')
        self.mock_chat_history_retriever.recall_budget_tokens = self.original_config.get('chat_history_retriever', {}).get('recall_budget_tokens')
    def tearDown(self):
        api_main.CONFIG = self.original_config; self.patch_ctx_mgr.stop(); self.patch_model_mgr.stop()
        self.patch_chat_history_retriever.stop(); self.patch_pdf_retriever.stop()
    def test_get_settings(self): # Content unchanged
        response = self.client.get("/settings"); self.assertEqual(response.status_code, 200); data = response.json()
        self.assertEqual(data['context_manager']['max_tokens'], api_main.CONFIG['context_manager']['max_tokens'])
        self.assertEqual(data['model_manager']['default_idle_unload_sec'], api_main.CONFIG['model_manager']['default_idle_unload_sec'])
        if 'generation_defaults' in data and data['generation_defaults'] is not None:
             self.assertEqual(data['generation_defaults']['temperature'], api_main.CONFIG.get('generation_defaults', {}).get('temperature'))
    def test_update_settings_partial_success_and_propagate(self): # Content unchanged
        update_payload = {"context_manager": {"max_tokens": 1234, "system_prompt": "New prompt!"}, "model_manager": {"default_idle_unload_sec": 555}, "chat_history_retriever": {"embedding_model_name": "new_embed_model"}}
        response = self.client.put("/settings", json=update_payload); self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response["status"], "ok")
        self.assertIn("ContextManager max_tokens updated to 1234", json_response["message"]); self.assertIn("ContextManager system_prompt updated", json_response["message"])
        self.assertIn("ModelManager default_idle_unload_sec updated to 555", json_response["message"]); self.assertIn("ChatHistoryRetriever 'embedding_model_name' or 'vector_db_path' changes require a restart", json_response["message"])
        self.assertEqual(api_main.CONFIG['context_manager']['max_tokens'], 1234); self.assertEqual(api_main.CONFIG['context_manager']['system_prompt'], "New prompt!")
        self.assertEqual(api_main.CONFIG['model_manager']['default_idle_unload_sec'], 555); self.assertEqual(api_main.CONFIG['chat_history_retriever']['embedding_model_name'], "new_embed_model")
        self.assertEqual(self.mock_ctx_mgr.max_tokens, 1234); self.assertEqual(self.mock_ctx_mgr.system_prompt, "New prompt!")
        self.assertEqual(self.mock_model_mgr.default_idle_unload_sec, 555)
        self.assertNotEqual(self.mock_chat_history_retriever.embedding_model_name, "new_embed_model", "embedding_model_name should not be updated on live instance directly.")
    def test_update_settings_no_changes(self): # Content unchanged
        response = self.client.put("/settings", json={}); self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response["status"], "ok")
        self.assertIn("No settings provided to update", json_response["message"]); self.assertEqual(api_main.CONFIG, self.original_config)
    def test_update_settings_only_restart_required(self): # Content unchanged
        update_payload = {"token_estimator_for_rag_budgeting": {"type": "new_type_requires_restart"}}; original_ctx_max_tokens = self.mock_ctx_mgr.max_tokens
        response = self.client.put("/settings", json=update_payload); self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response["status"], "ok")
        self.assertIn("Changes to 'token_estimator_for_rag_budgeting' require an application restart", json_response["message"]); self.assertNotIn("Applied live:", json_response["message"])
        self.assertEqual(api_main.CONFIG['token_estimator_for_rag_budgeting']['type'], "new_type_requires_restart"); self.assertEqual(self.mock_ctx_mgr.max_tokens, original_ctx_max_tokens)

# --- Model Management API Tests ---
from yawl.api.schemas import AvailableModel, ModelListResponse, DownloadModelRequest, StatusResponse # Updated
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

class TestModelManagementAPI(unittest.TestCase):
    def setUp(self): self.client = TestClient(app); self.original_config = deepcopy(api_main.CONFIG)
    def tearDown(self): api_main.CONFIG = self.original_config
    @patch('yawl.api.main.model_scanner.scan_model_directories') # Updated
    def test_list_available_models_success(self, mock_scan_model_directories): # Content mostly unchanged
        mock_models_data = [{"model_id": "model1.gguf", "model_type": "gguf", "path_or_identifier": "/path/model1.gguf", "source": "local_scan", "name": "Model 1"}, {"model_id": "org/model2-awq", "model_type": "awq", "path_or_identifier": "/path/model2-awq", "source": "local_scan", "name": "Model 2 AWQ"}]
        mock_scan_model_directories.return_value = [AvailableModel(**data) for data in mock_models_data]
        api_main.CONFIG['model_discovery'] = {'scan_directories': ['dummy_scan_dir/']}
        response = self.client.get("/models/available"); self.assertEqual(response.status_code, 200)
        response_data = ModelListResponse(**response.json()); self.assertEqual(len(response_data.models), 2)
        self.assertEqual(response_data.models[0].model_id, "model1.gguf"); self.assertEqual(response_data.models[1].name, "Model 2 AWQ"); mock_scan_model_directories.assert_called_once()
    @patch('yawl.api.main.model_scanner.scan_model_directories') # Updated
    def test_list_available_models_no_scan_dirs_configured(self, mock_scan_model_directories): # Content mostly unchanged
        original_model_discovery_config = api_main.CONFIG.pop('model_discovery', None)
        response = self.client.get("/models/available"); self.assertEqual(response.status_code, 200)
        response_data = ModelListResponse(**response.json()); self.assertEqual(len(response_data.models), 0); mock_scan_model_directories.assert_not_called()
        if original_model_discovery_config is not None: api_main.CONFIG['model_discovery'] = original_model_discovery_config
    @patch('yawl.api.main.model_scanner.scan_model_directories') # Updated
    def test_list_available_models_scanner_error(self, mock_scan_model_directories): # Content mostly unchanged
        mock_scan_model_directories.side_effect = Exception("Scanner crashed badly")
        api_main.CONFIG['model_discovery'] = {'scan_directories': ['another_dummy_dir/']}
        response = self.client.get("/models/available"); self.assertEqual(response.status_code, 500)
        json_response = response.json(); self.assertIn("Failed to scan model directories: Scanner crashed badly", json_response["detail"])
    def test_download_model_placeholder_success(self): # Content mostly unchanged
        payload = {"repo_id": "TheBloke/Test-Model-GGUF", "filename": "test-model.Q4_K_M.gguf", "model_type": "gguf"}
        response = self.client.post("/models/download", json=payload); self.assertEqual(response.status_code, 200); json_response = response.json()
        self.assertEqual(json_response["status"], "ok"); self.assertIn("Download request for 'TheBloke/Test-Model-GGUF'", json_response["message"])
        self.assertIn("file: test-model.Q4_K_M.gguf", json_response["message"]); self.assertIn("Simulated download initiated", json_response["message"])
    def test_download_model_missing_repo_id(self): # Content unchanged
        payload = {"filename": "test-model.Q4_K_M.gguf", "model_type": "gguf"}
        response = self.client.post("/models/download", json=payload); self.assertEqual(response.status_code, 422)
    @patch('yawl.api.main.download_model_from_hf') # Updated
    def test_download_model_success_single_file(self, mock_download_func): # Content mostly unchanged
        mock_download_func.return_value = (True, Path("/fake/downloaded/model_repo_name/file.gguf"))
        payload = DownloadModelRequest(repo_id="test-org/model-repo-name", filename="file.gguf", revision="test-rev").model_dump()
        response = self.client.post("/models/download", json=payload); self.assertEqual(response.status_code, 200); json_response = response.json()
        self.assertEqual(json_response["status"], "ok"); self.assertIn("Download for 'test-org/model-repo-name'", json_response["message"])
        self.assertIn("File: file.gguf", json_response["message"]); self.assertIn("Revision: test-rev", json_response["message"])
        self.assertIn("Saved to: /fake/downloaded/model_repo_name/file.gguf", json_response["message"])
        mock_download_func.assert_called_once_with(repo_id="test-org/model-repo-name", target_dir=ANY, filename="file.gguf", hf_token=api_main.CONFIG.get('hf_token'), ignore_patterns=api_main.CONFIG.get('hf_snapshot_ignore_patterns'), allow_patterns=api_main.CONFIG.get('hf_snapshot_allow_patterns'), repo_type=api_main.CONFIG.get('hf_repo_type'), revision="test-rev")
        args, _ = mock_download_func.call_args; self.assertIsInstance(args[1], Path)
    @patch('yawl.api.main.download_model_from_hf') # Updated
    def test_download_model_success_snapshot(self, mock_download_func): # Content mostly unchanged
        mock_download_func.return_value = (True, Path("/fake/downloaded/snapshot_repo_name/"))
        payload = DownloadModelRequest(repo_id="test-org/snapshot-repo-name", revision="main").model_dump()
        response = self.client.post("/models/download", json=payload); self.assertEqual(response.status_code, 200); json_response = response.json()
        self.assertEqual(json_response["status"], "ok"); self.assertIn("Download for 'test-org/snapshot-repo-name'", json_response["message"])
        self.assertIn("File: snapshot", json_response["message"]); self.assertIn("Revision: main", json_response["message"])
        self.assertIn("Saved to: /fake/downloaded/snapshot_repo_name", json_response["message"])
        mock_download_func.assert_called_once_with(repo_id="test-org/snapshot-repo-name", target_dir=ANY, filename=None, hf_token=api_main.CONFIG.get('hf_token'), ignore_patterns=api_main.CONFIG.get('hf_snapshot_ignore_patterns'), allow_patterns=api_main.CONFIG.get('hf_snapshot_allow_patterns'), repo_type=api_main.CONFIG.get('hf_repo_type'), revision="main")
    @patch('yawl.api.main.download_model_from_hf') # Updated
    def test_download_model_failure_downloader_returns_error(self, mock_download_func): # Content mostly unchanged
        mock_download_func.return_value = (False, "Simulated downloader network error")
        payload = DownloadModelRequest(repo_id="test-org/failing-repo", filename="some.file").model_dump()
        response = self.client.post("/models/download", json=payload); self.assertEqual(response.status_code, 200)
        json_response = response.json(); self.assertEqual(json_response["status"], "error"); self.assertEqual(json_response["message"], "Simulated downloader network error")
        mock_download_func.assert_called_once()

# --- Tool Management API Tests ---
from yawl.api.schemas import ToolInfo, ToolListResponse, ToggleToolRequest # Updated

class TestToolManagementAPI(unittest.TestCase): # Content mostly unchanged
    def setUp(self): self.client = TestClient(app); self.tool_dispatcher_instance = api_main.tool_dispatcher; self.original_tool_states = deepcopy(self.tool_dispatcher_instance.tool_states)
    def tearDown(self): self.tool_dispatcher_instance.tool_states = self.original_tool_states
    def test_list_tools_success(self):
        with patch.object(self.tool_dispatcher_instance, 'list_tools') as mock_list_tools:
            mock_tool_list_data = [ToolInfo(name="get_weather", type="local", description="Gets weather", is_enabled=True, parameters={}).model_dump(), ToolInfo(name="mcp_tool1", type="mcp", description="MCP tool", is_enabled=False, parameters={}).model_dump()]
            mock_list_tools.return_value = [ToolInfo(**data) for data in mock_tool_list_data]
            response = self.client.get("/tools"); self.assertEqual(response.status_code, 200)
            response_data = ToolListResponse(**response.json()); self.assertEqual(len(response_data.tools), 2)
            self.assertEqual(response_data.tools[0].name, "get_weather"); self.assertTrue(response_data.tools[0].is_enabled)
            self.assertEqual(response_data.tools[1].name, "mcp_tool1"); self.assertFalse(response_data.tools[1].is_enabled); mock_list_tools.assert_called_once()
    def test_toggle_tool_enable_disable(self): # Content mostly unchanged
        tool_name_to_test = "get_weather"
        disable_payload = {"tool_name": tool_name_to_test, "enable": False}; response_disable = self.client.post("/tools/toggle", json=disable_payload)
        self.assertEqual(response_disable.status_code, 200); self.assertEqual(response_disable.json()["status"], "ok"); self.assertIn(f"Tool '{tool_name_to_test}' disabled", response_disable.json()["message"]); self.assertFalse(self.tool_dispatcher_instance.tool_states.get(tool_name_to_test))
        response_list_disabled = self.client.get("/tools"); self.assertEqual(response_list_disabled.status_code, 200); listed_tools_disabled = ToolListResponse(**response_list_disabled.json()).tools
        found_tool_disabled = next((t for t in listed_tools_disabled if t.name == tool_name_to_test), None); self.assertIsNotNone(found_tool_disabled); self.assertFalse(found_tool_disabled.is_enabled)
        enable_payload = {"tool_name": tool_name_to_test, "enable": True}; response_enable = self.client.post("/tools/toggle", json=enable_payload)
        self.assertEqual(response_enable.status_code, 200); self.assertEqual(response_enable.json()["status"], "ok"); self.assertIn(f"Tool '{tool_name_to_test}' enabled", response_enable.json()["message"]); self.assertTrue(self.tool_dispatcher_instance.tool_states.get(tool_name_to_test))
        response_list_enabled = self.client.get("/tools"); self.assertEqual(response_list_enabled.status_code, 200); listed_tools_enabled = ToolListResponse(**response_list_enabled.json()).tools
        found_tool_enabled = next((t for t in listed_tools_enabled if t.name == tool_name_to_test), None); self.assertIsNotNone(found_tool_enabled); self.assertTrue(found_tool_enabled.is_enabled)
    def test_toggle_unknown_tool(self): # Content mostly unchanged
        unknown_tool_name = "unknown_tool_xyz_123"; enable_payload = {"tool_name": unknown_tool_name, "enable": True}
        response_enable = self.client.post("/tools/toggle", json=enable_payload); self.assertEqual(response_enable.status_code, 200); self.assertEqual(response_enable.json()["status"], "ok"); self.assertTrue(self.tool_dispatcher_instance.tool_states.get(unknown_tool_name))
        response_list = self.client.get("/tools"); self.assertEqual(response_list.status_code, 200); listed_tools = ToolListResponse(**response_list.json()).tools
        found_tool = next((t for t in listed_tools if t.name == unknown_tool_name), None); self.assertIsNotNone(found_tool); self.assertTrue(found_tool.is_enabled); self.assertEqual(found_tool.type, "mcp")
    def test_get_openai_tool_schemas_endpoint(self): # Content mostly unchanged
        self.tool_dispatcher_instance.enable_tool("get_weather"); response = self.client.get("/tools/openai_schemas"); self.assertEqual(response.status_code, 200)
        schemas = response.json(); self.assertIsInstance(schemas, list)
        get_weather_schema = next((s for s in schemas if s.get("function", {}).get("name") == "get_weather"), None); self.assertIsNotNone(get_weather_schema)
        self.assertEqual(get_weather_schema["type"], "function"); function_def = get_weather_schema["function"]; self.assertEqual(function_def["name"], "get_weather"); self.assertIn("current weather", function_def["description"].lower())
        parameters = function_def["parameters"]; self.assertEqual(parameters["type"], "object"); self.assertIn("properties", parameters); self.assertIn("location", parameters["properties"])
        self.assertEqual(parameters["properties"]["location"]["type"], "str"); self.assertIn("unit", parameters["properties"]); self.assertEqual(parameters["properties"]["unit"]["type"], "str")
        self.assertEqual(parameters["properties"]["unit"]["default"], "celsius"); self.assertIn("location", parameters["required"])
        self.tool_dispatcher_instance.disable_tool("get_weather"); response_disabled = self.client.get("/tools/openai_schemas"); self.assertEqual(response_disabled.status_code, 200)
        schemas_disabled = response_disabled.json(); get_weather_schema_disabled = next((s for s in schemas_disabled if s.get("function", {}).get("name") == "get_weather"), None)
        self.assertIsNone(get_weather_schema_disabled); self.tool_dispatcher_instance.enable_tool("get_weather")

# --- Test class for Chat Endpoint with OpenAI Tool Calls ---
class TestChatWithOpenAITools(unittest.TestCase):
    def setUp(self): # Content mostly unchanged
        self.client = TestClient(app);
        if model_mgr.current_runner: model_mgr.unload()
        ctx_mgr._messages = []; ctx_mgr._start = 0
        api_main.tool_dispatcher.enable_tool("get_weather")
    def tearDown(self): # Content mostly unchanged
        if model_mgr.current_runner: model_mgr.unload()
        ctx_mgr._messages = []; ctx_mgr._start = 0
    @patch('yawl.api.main.model_mgr.get') # Updated
    def test_chat_sync_openai_tool_call_success(self, mock_get_runner): # Content mostly unchanged
        mock_runner = MagicMock(spec=APIRunner); mock_get_runner.return_value = mock_runner
        tool_call_request = {"tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": "Paris", "unit": "celsius"})}}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}}
        final_text_response = {"text": "The weather in Paris is 20 degrees Celsius.", "usage": {"prompt_tokens": 50, "completion_tokens": 15}}
        mock_runner.generate.side_effect = [tool_call_request, final_text_response]; mock_runner.count_tokens.return_value = 15
        chat_payload = ChatRequest(message="What's the weather in Paris like?", stream=False); response = self.client.post("/chat", json=chat_payload.model_dump())
        self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response["reply"], "The weather in Paris is 20 degrees Celsius.")
        self.assertEqual(json_response["prompt_tokens"], 50); self.assertEqual(json_response["generated_tokens"], 10 + 15); self.assertEqual(mock_runner.generate.call_count, 2)
        user_message_in_ctx = next((m for m in ctx_mgr._messages if m['r'] == 'user'), None); self.assertIsNotNone(user_message_in_ctx); self.assertEqual(user_message_in_ctx['c'], "What's the weather in Paris like?")
        assistant_tool_request_msg = next((m for m in ctx_mgr._messages if m['r'] == 'assistant' and m.get('tool_calls')), None); self.assertIsNotNone(assistant_tool_request_msg); self.assertEqual(assistant_tool_request_msg['tool_calls'][0]['id'], "call_abc123"); self.assertEqual(assistant_tool_request_msg['tool_calls'][0]['function']['name'], "get_weather")
        tool_result_msg = next((m for m in ctx_mgr._messages if m['r'] == 'tool'), None); self.assertIsNotNone(tool_result_msg); self.assertEqual(tool_result_msg['tcid'], "call_abc123"); self.assertEqual(tool_result_msg['nm'], "get_weather"); self.assertIn("Paris is 20 degrees celsius", tool_result_msg['c'])
    @patch('yawl.api.main.model_mgr.get') # Updated
    def test_chat_sync_openai_tool_call_tool_disabled(self, mock_get_runner): # Content mostly unchanged
        mock_runner = MagicMock(spec=APIRunner); mock_get_runner.return_value = mock_runner
        tool_call_request_for_disabled_tool = {"tool_calls": [{"id": "call_disabled_tool", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": "Moon"})}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        final_text_response_after_disabled_tool = {"text": "Sorry, I couldn't use the get_weather tool because it's disabled.", "usage": {"prompt_tokens": 30, "completion_tokens": 8}}
        mock_runner.generate.side_effect = [tool_call_request_for_disabled_tool, final_text_response_after_disabled_tool]; mock_runner.count_tokens.return_value = 10
        api_main.tool_dispatcher.disable_tool("get_weather"); chat_payload = ChatRequest(message="Weather on Moon?", stream=False); response = self.client.post("/chat", json=chat_payload.model_dump())
        self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response["reply"], "Sorry, I couldn't use the get_weather tool because it's disabled.")
        tool_msg = next((m for m in ctx_mgr._messages if m['r'] == 'tool' and m['tcid'] == 'call_disabled_tool'), None); self.assertIsNotNone(tool_msg); self.assertIn("Tool 'get_weather' is currently disabled", tool_msg['c'])
        api_main.tool_dispatcher.enable_tool("get_weather")
    @patch('yawl.api.main.model_mgr.get') # Updated
    def test_chat_stream_openai_tool_call_success(self, mock_get_runner): # Content mostly unchanged
        mock_runner = MagicMock(spec=APIRunner); mock_get_runner.return_value = mock_runner
        tool_call_request_stream_item = {"tool_calls": [{"id": "stream_call_xyz789", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": "Mars", "unit": "celsius"})}}]}
        initial_stream_yield = [{"prompt_tokens": 25}, tool_call_request_stream_item]
        final_text_chunks_yield = [{"prompt_tokens": 60}, ("Weather on Mars is very cold. ", 7), ("Bring a warm jacket!", 5)]
        mock_runner.stream.side_effect = [self.mock_async_stream_generator(initial_stream_yield), self.mock_async_stream_generator(final_text_chunks_yield)]
        mock_runner.count_tokens.return_value = 25
        chat_payload = ChatRequest(message="What is the weather on Mars and what should I wear?", stream=True); received_events = []
        with self.client.stream("POST", "/chat", json=chat_payload.model_dump()) as response:
            self.assertEqual(response.status_code, 200); self.assertEqual(response.headers['content-type'], 'text/event-stream; charset=utf-8')
            for line_bytes in response.iter_lines(): received_events.extend(parse_sse_stream([line_bytes]))
        prompt_info_initial = next((e for e in received_events if e['event'] == 'prompt_info' and e['data'].get('context') == 'initial_prompt'), None); self.assertIsNotNone(prompt_info_initial); self.assertEqual(prompt_info_initial['data']['prompt_tokens'], 25)
        tool_calls_processing_event = next((e for e in received_events if e['event'] == 'tool_calls_processing'), None); self.assertIsNotNone(tool_calls_processing_event); self.assertEqual(tool_calls_processing_event['data'][0]['id'], "stream_call_xyz789"); self.assertEqual(tool_calls_processing_event['data'][0]['function']['name'], "get_weather")
        tool_result_event = next((e for e in received_events if e['event'] == 'tool_result'), None); self.assertIsNotNone(tool_result_event); self.assertEqual(tool_result_event['data']['tool_call_id'], "stream_call_xyz789"); self.assertEqual(tool_result_event['data']['name'], "get_weather"); self.assertIn("Mars is -65 degrees celsius", tool_result_event['data']['result'])
        prompt_info_after_tools = next((e for e in received_events if e['event'] == 'prompt_info' and e['data'].get('context') == 'runner_provided_after_tools'), None); self.assertIsNotNone(prompt_info_after_tools); self.assertEqual(prompt_info_after_tools['data']['prompt_tokens'], 60)
        text_chunk_events = [e['data'] for e in received_events if e['event'] == 'message' and 'text' in e['data']]; self.assertTrue(len(text_chunk_events) >= 2)
        reconstructed_reply = "".join(chunk['text'] for chunk in text_chunk_events); self.assertEqual(reconstructed_reply, "Weather on Mars is very cold. Bring a warm jacket!")
        self.assertEqual(text_chunk_events[0]['tokens_in_chunk'], 7); self.assertEqual(text_chunk_events[1]['tokens_in_chunk'], 5)
        stream_end_event = next((e for e in received_events if e['event'] == 'stream_end'), None); self.assertIsNotNone(stream_end_event)
        self.assertEqual(stream_end_event['data']['total_generated_tokens'], 7 + 5) ; self.assertEqual(stream_end_event['data']['final_prompt_tokens'], 60)
        self.assertEqual(mock_runner.stream.call_count, 2)
        user_msg = next((m for m in ctx_mgr._messages if m['r'] == 'user'), None); self.assertEqual(user_msg['c'], "What is the weather on Mars and what should I wear?")
        assistant_tool_request = next((m for m in ctx_mgr._messages if m['r'] == 'assistant' and m.get('tool_calls')), None); self.assertIsNotNone(assistant_tool_request); self.assertEqual(assistant_tool_request['tool_calls'][0]['id'], 'stream_call_xyz789')
        tool_response_msg = next((m for m in ctx_mgr._messages if m['r'] == 'tool' and m.get('tcid') == 'stream_call_xyz789'), None); self.assertIsNotNone(tool_response_msg); self.assertIn("Mars is -65 degrees celsius", tool_response_msg['c'])

# --- HyDE Integration Tests ---
class TestChatWithHyDE(unittest.TestCase):
    def setUp(self): # Content mostly unchanged
        self.client = TestClient(app); self.original_config = deepcopy(api_main.CONFIG)
        self.original_pdf_retriever_enable_hyde = api_main.pdf_retriever.enable_hyde; self.original_chr_enable_hyde = api_main.chat_history_retriever.enable_hyde
        self.patch_model_mgr = patch('yawl.api.main.model_mgr') # Updated
        self.patch_pdf_retriever_instance = patch('yawl.api.main.pdf_retriever') # Updated
        self.patch_chat_history_retriever_instance = patch('yawl.api.main.chat_history_retriever') # Updated
        self.mock_model_mgr = self.patch_model_mgr.start(); self.mock_pdf_retriever = self.patch_pdf_retriever_instance.start(); self.mock_chat_history_retriever = self.patch_chat_history_retriever_instance.start()
        self.mock_runner = MagicMock(spec=APIRunner); self.mock_model_mgr.get.return_value = self.mock_runner
        self.mock_pdf_retriever.embedding_model = MagicMock(); self.mock_pdf_retriever.embedding_model.encode.return_value = MagicMock(); self.mock_pdf_retriever.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        self.mock_pdf_retriever.retrieve_from_pdf.return_value = [{'r': 'retrieved_pdf_chunk', 'c': 'Mocked PDF snippet.'}]
        self.mock_chat_history_retriever.embedding_model = MagicMock(); self.mock_chat_history_retriever.embedding_model.encode.return_value = MagicMock(); self.mock_chat_history_retriever.embedding_model.encode.return_value.tolist.return_value = [0.4, 0.5, 0.6]
        self.mock_chat_history_retriever.retrieve.return_value = [{'r': 'retrieved_context', 'c': 'Mocked chat history snippet.'}]
        ctx_mgr._messages = []; ctx_mgr._start = 0
    def tearDown(self): # Content mostly unchanged
        api_main.CONFIG = self.original_config; api_main.pdf_retriever.enable_hyde = self.original_pdf_retriever_enable_hyde; api_main.chat_history_retriever.enable_hyde = self.original_chr_enable_hyde
        self.patch_model_mgr.stop(); self.patch_pdf_retriever_instance.stop(); self.patch_chat_history_retriever_instance.stop()
        if model_mgr.current_runner: model_mgr.unload()
        ctx_mgr._messages = []; ctx_mgr._start = 0
    def test_chat_with_hyde_enabled_pdf_retriever(self): # Content mostly unchanged
        api_main.CONFIG['pdf_retriever']['enable_hyde'] = True; self.mock_api_pdf_retriever.enable_hyde = True # Use self.mock_api_pdf_retriever
        api_main.CONFIG['hyde']['prompt_template'] = "PDF HyDE: {query}"; hypothetical_doc_pdf = "This is a hypothetical PDF document about Paris."; final_response_text = "Final response after PDF RAG."
        self.mock_runner.generate.side_effect = [(hypothetical_doc_pdf, {"prompt_tokens": 5, "completion_tokens": 10}), (final_response_text, {"prompt_tokens": 50, "completion_tokens": 20})]
        chat_payload = ChatRequest(message="What about Paris?", use_rag=True, pdf_doc_ids_for_rag=["doc1"]).model_dump()
        response = self.client.post("/chat", json=chat_payload); self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response['reply'], final_response_text)
        self.mock_runner.generate.assert_any_call(prompt="PDF HyDE: What about Paris?", max_new_tokens=api_main.CONFIG['hyde']['max_tokens_hyde_doc'])
        self.mock_pdf_retriever.embedding_model.encode.assert_called_with(hypothetical_doc_pdf) # Use self.mock_pdf_retriever
        self.mock_pdf_retriever.retrieve_from_pdf.assert_called_once_with(query_text="What about Paris?", query_embedding_override=[0.1, 0.2, 0.3], doc_ids=["doc1"]) # Use self.mock_pdf_retriever
    def test_chat_with_hyde_enabled_chat_history_retriever(self): # Content mostly unchanged
        api_main.CONFIG['chat_history_retriever']['enable_hyde'] = True; self.mock_api_chat_history_retriever.enable_hyde = True # Use self.mock_api_chat_history_retriever
        api_main.CONFIG['hyde']['prompt_template'] = "Chat HyDE: {query}"; hypothetical_doc_chat = "This is a hypothetical chat history snippet about meetings."; final_response_text = "Final response after Chat RAG."
        self.mock_runner.generate.side_effect = [(hypothetical_doc_chat, {"prompt_tokens": 6, "completion_tokens": 12}), (final_response_text, {"prompt_tokens": 55, "completion_tokens": 22})]
        chat_payload = ChatRequest(message="Any meetings today?", use_rag=True).model_dump()
        response = self.client.post("/chat", json=chat_payload); self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response['reply'], final_response_text)
        self.mock_runner.generate.assert_any_call(prompt="Chat HyDE: Any meetings today?", max_new_tokens=api_main.CONFIG['hyde']['max_tokens_hyde_doc'])
        self.mock_chat_history_retriever.embedding_model.encode.assert_called_with(hypothetical_doc_chat) # Use self.mock_chat_history_retriever
        self.mock_chat_history_retriever.retrieve.assert_called_once_with(query_text="Any meetings today?", query_embedding_override=[0.4, 0.5, 0.6], current_chat_history=ANY) # Use self.mock_chat_history_retriever
    def test_chat_with_hyde_disabled(self): # Content mostly unchanged
        api_main.CONFIG['pdf_retriever']['enable_hyde'] = False; self.mock_api_pdf_retriever.enable_hyde = False
        api_main.CONFIG['chat_history_retriever']['enable_hyde'] = False; self.mock_api_chat_history_retriever.enable_hyde = False
        final_response_text = "Response without HyDE."; self.mock_runner.generate.return_value = (final_response_text, {"prompt_tokens": 10, "completion_tokens": 5})
        chat_payload = ChatRequest(message="A query that could use RAG.", use_rag=True, pdf_doc_ids_for_rag=["doc1"]).model_dump()
        response = self.client.post("/chat", json=chat_payload); self.assertEqual(response.status_code, 200)
        self.mock_runner.generate.assert_called_once()
        self.mock_pdf_retriever.retrieve_from_pdf.assert_called_once_with(query_text="A query that could use RAG.", query_embedding_override=None, doc_ids=["doc1"])
        self.mock_pdf_retriever.embedding_model.encode.assert_not_called()
    def test_chat_with_hyde_llm_failure(self): # Content mostly unchanged
        api_main.CONFIG['pdf_retriever']['enable_hyde'] = True; self.mock_api_pdf_retriever.enable_hyde = True
        api_main.CONFIG['hyde']['prompt_template'] = "PDF HyDE: {query}"
        final_response_text = "Final response after HyDE failure."
        self.mock_runner.generate.side_effect = [Exception("LLM error during HyDE!"), (final_response_text, {"prompt_tokens": 30, "completion_tokens": 10})]
        chat_payload = ChatRequest(message="Query for PDF RAG with failing HyDE", use_rag=True, pdf_doc_ids_for_rag=["doc_fail"]).model_dump()
        response = self.client.post("/chat", json=chat_payload); self.assertEqual(response.status_code, 200); json_response = response.json(); self.assertEqual(json_response['reply'], final_response_text)
        self.mock_runner.generate.assert_any_call(prompt="PDF HyDE: Query for PDF RAG with failing HyDE", max_new_tokens=ANY)
        self.mock_pdf_retriever.embedding_model.encode.assert_not_called()
        self.mock_pdf_retriever.retrieve_from_pdf.assert_called_once_with(query_text="Query for PDF RAG with failing HyDE", query_embedding_override=None, doc_ids=["doc_fail"])

if __name__ == '__main__':
    unittest.main()
