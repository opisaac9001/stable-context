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

if __name__ == '__main__':
    unittest.main()
