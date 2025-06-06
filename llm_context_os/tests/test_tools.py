# llm_context_os/tests/test_tools.py
# llm_context_os/tests/test_tools.py
import unittest
import json
from unittest.mock import patch, MagicMock # Added for MCP testing

from llm_context_os.tools.tool_dispatcher import ToolDispatcher, MCP_ADAPTERS_AVAILABLE # Import flag
from llm_context_os.tools import builtin_weather

# If McpClient is imported in tool_dispatcher, we might need its path for patching
# from langchain_mcp_adapters import McpClient # Only if needed for spec in MagicMock

class TestToolDispatcherAndTools(unittest.TestCase):

    def setUp(self):
        # Create a new dispatcher for each test to ensure a clean state,
        # especially important if MCP_ADAPTERS_AVAILABLE changes or if McpClient is stateful.
        # Patching MCP_ADAPTERS_AVAILABLE for specific tests will require re-instantiation
        # of ToolDispatcher within those tests if its __init__ depends on the flag.
        self.dispatcher = ToolDispatcher()


    def test_builtin_weather_get_weather_known_location(self):
        result_ny_c = builtin_weather.get_weather("New York")
        self.assertIn("New York is 22 degrees celsius", result_ny_c)

        result_london_f = builtin_weather.get_weather("London", unit="fahrenheit")
        self.assertIn("London is 15 degrees fahrenheit", result_london_f)

    def test_builtin_weather_get_weather_unknown_location(self):
        result = builtin_weather.get_weather("Paris")
        self.assertIn("Sorry, I don't have weather information for Paris", result)

    def test_tool_dispatcher_instantiation(self):
        self.assertIsNotNone(self.dispatcher)
        self.assertIn("get_weather", self.dispatcher.local_tools)

    def test_dispatch_local_tool_get_weather_valid(self):
        weather_call_json = json.dumps({
            "name": "get_weather",
            "arguments": {"location": "London", "unit": "celsius"}
        })
        result_dict = self.dispatcher.dispatch(weather_call_json)
        self.assertEqual(result_dict['tool_name'], 'get_weather')
        self.assertEqual(result_dict['status'], 'success')
        self.assertIn("London is 15 degrees celsius", result_dict['result'])

    def test_dispatch_local_tool_missing_required_arg(self):
        # 'location' is required by get_weather function.
        weather_call_json = json.dumps({"name": "get_weather", "arguments": {"unit": "celsius"}})
        result_dict = self.dispatcher.dispatch(weather_call_json)
        self.assertEqual(result_dict['tool_name'], 'get_weather')
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("missing 1 required positional argument: 'location'", result_dict['error'].lower())

    def test_dispatch_local_tool_extra_arg(self):
        # Extra arguments should ideally be ignored by the tool or cause a TypeError
        # if the tool function doesn't accept **kwargs.
        # The current get_weather doesn't accept **kwargs, so it should be a TypeError.
        weather_call_json = json.dumps({
            "name": "get_weather",
            "arguments": {"location": "New York", "unit": "celsius", "extra_param": "test"}
        })
        result_dict = self.dispatcher.dispatch(weather_call_json)
        self.assertEqual(result_dict['tool_name'], 'get_weather')
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("got an unexpected keyword argument 'extra_param'", result_dict['error'].lower())

    # --- MCP Tool Dispatching Tests ---

    @unittest.skipUnless(MCP_ADAPTERS_AVAILABLE, "langchain-mcp-adapters not installed, skipping MCP tests.")
    @patch('llm_context_os.tools.tool_dispatcher.McpClient') # Patch the McpClient class where it's imported by ToolDispatcher
    def test_dispatch_mcp_tool_success(self, MockMcpClient):
        # This test runs if MCP_ADAPTERS_AVAILABLE is True.
        # ToolDispatcher will attempt to initialize McpClient in its __init__.
        # So, the MockMcpClient from the patch is the class, and dispatcher.mcp_client is an instance of this mock.

        # Re-initialize dispatcher to ensure it uses the patched McpClient from this test's context
        dispatcher = ToolDispatcher()
        self.assertIsNotNone(dispatcher.mcp_client, "MCP Client should be initialized (mocked).")

        # Configure the mock mcp_client instance
        mock_mcp_instance = dispatcher.mcp_client
        mock_mcp_instance.invoke.return_value = {"mcp_data": "successful result from mcp tool"}

        mcp_tool_name = "mcp_actual_tool"
        mcp_tool_args = {"arg1": "val1", "arg2": 123}
        mcp_call_json = json.dumps({"name": mcp_tool_name, "arguments": mcp_tool_args})

        result_dict = dispatcher.dispatch(mcp_call_json)

        self.assertEqual(result_dict['tool_name'], mcp_tool_name)
        self.assertEqual(result_dict['status'], 'success')
        self.assertEqual(result_dict['result'], {"mcp_data": "successful result from mcp tool"})
        mock_mcp_instance.invoke.assert_called_once_with(mcp_tool_name, mcp_tool_args)


    @unittest.skipUnless(MCP_ADAPTERS_AVAILABLE, "langchain-mcp-adapters not installed, skipping MCP tests.")
    @patch('llm_context_os.tools.tool_dispatcher.McpClient')
    def test_dispatch_mcp_tool_error_from_client(self, MockMcpClient):
        dispatcher = ToolDispatcher()
        self.assertIsNotNone(dispatcher.mcp_client)

        mock_mcp_instance = dispatcher.mcp_client
        mock_mcp_instance.invoke.side_effect = Exception("MCP client experienced a critical failure")

        mcp_tool_name = "mcp_failing_tool"
        mcp_tool_args = {"param": "boom"}
        mcp_call_json = json.dumps({"name": mcp_tool_name, "arguments": mcp_tool_args})

        result_dict = dispatcher.dispatch(mcp_call_json)

        self.assertEqual(result_dict['tool_name'], mcp_tool_name)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("MCP Error for tool mcp_failing_tool: MCP client experienced a critical failure", result_dict['error'])
        mock_mcp_instance.invoke.assert_called_once_with(mcp_tool_name, mcp_tool_args)

    # This test specifically checks behavior when MCP_ADAPTERS_AVAILABLE is False *during* ToolDispatcher init
    @patch('llm_context_os.tools.tool_dispatcher.MCP_ADAPTERS_AVAILABLE', False)
    def test_dispatch_mcp_tool_unavailable_client_falls_to_unknown(self):
        # With MCP_ADAPTERS_AVAILABLE patched to False, ToolDispatcher should set self.mcp_client to None
        dispatcher_no_mcp = ToolDispatcher()
        self.assertIsNone(dispatcher_no_mcp.mcp_client, "MCP client should be None when MCP_ADAPTERS_AVAILABLE is False.")

        mcp_tool_name = "mcp_tool_when_client_is_none"
        mcp_call_json = json.dumps({"name": mcp_tool_name, "arguments": {}})
        result_dict = dispatcher_no_mcp.dispatch(mcp_call_json)

        self.assertEqual(result_dict['tool_name'], mcp_tool_name)
        self.assertEqual(result_dict['status'], 'error')
        # The error message includes "client unavailable/tool not found"
        self.assertIn(f"Unknown tool: {mcp_tool_name}. Not found in local tools or MCP (client unavailable/tool not found).", result_dict['error'])

    def test_dispatch_unknown_tool(self):
        # This test now implicitly also covers the case where MCP_ADAPTERS_AVAILABLE is True,
        # McpClient initializes, but the tool is not local and not found by MCP (if MCP client itself
        # raises an error that the dispatcher catches and then moves to 'unknown tool').
        # The current dispatcher logic: local -> mcp (if client) -> unknown.
        # If mcp_client.invoke raises an error, it's reported as an MCP error, not "Unknown tool".
        # So this test primarily covers "not local AND (MCP client is None OR MCP_ADAPTERS_AVAILABLE is False)"
        # or if MCP was available but the tool name was not even attempted via MCP (e.g. due to naming convention not met, though current code doesn't have such a convention check before trying MCP)

        # To specifically test "unknown to local AND unknown to a functioning MCP client":
        # This would require the mocked mcp_client.invoke to raise a specific "ToolNotFoundOnMCPError"
        # which the dispatcher then specifically catches to return the "Unknown tool" message.
        # The current implementation returns "MCP Error: ToolNotFoundOnMCPError" instead.
        # So, this test remains for "not local, and no MCP client to ask".

        # If MCP_ADAPTERS_AVAILABLE is true and McpClient is mocked by other tests,
        # self.dispatcher might have a mocked mcp_client.
        # To ensure this test checks "unknown when MCP not available or not relevant":
        if self.dispatcher.mcp_client: # If a mock client got attached from a previous MCP test context
            with patch.object(self.dispatcher, 'mcp_client', None): # Temporarily remove mcp_client
                 unknown_call_json = json.dumps({"name": "fake_tool", "arguments": {}})
                 result_dict = self.dispatcher.dispatch(unknown_call_json)
        else: # mcp_client is already None (e.g. MCP_ADAPTERS_AVAILABLE was false)
            unknown_call_json = json.dumps({"name": "fake_tool", "arguments": {}})
            result_dict = self.dispatcher.dispatch(unknown_call_json)

        self.assertEqual(result_dict['tool_name'], 'fake_tool')
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("Unknown tool: fake_tool", result_dict['error'])


    def test_dispatch_invalid_json(self):
        invalid_json_str = '{"name": "get_weather", "arguments": {"location": "New York"}' # Missing closing brace
        result_dict = self.dispatcher.dispatch(invalid_json_str)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIsNone(result_dict['tool_name']) # No tool name could be parsed
        self.assertIn("Invalid JSON for function call", result_dict['error'])

    def test_dispatch_arguments_not_dict(self):
        # Test case where "arguments" is not a dictionary
        call_json = json.dumps({"name": "get_weather", "arguments": "not_a_dict"})
        result_dict = self.dispatcher.dispatch(call_json)
        self.assertEqual(result_dict['tool_name'], 'get_weather')
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("Tool arguments must be a dictionary", result_dict['error'])

    def test_dispatch_no_tool_name(self):
        call_json = json.dumps({"arguments": {"location": "New York"}})
        result_dict = self.dispatcher.dispatch(call_json)
        self.assertIsNone(result_dict['tool_name'])
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("Missing tool name", result_dict['error'])

    # --- Tests for new ToolDispatcher features: list_tools, enable/disable ---
    def test_list_tools_content_and_schema(self):
        # dispatcher is self.dispatcher from setUp
        tools_info = self.dispatcher.list_tools()
        self.assertIsInstance(tools_info, list)

        get_weather_info = next((t for t in tools_info if t.name == "get_weather"), None)
        self.assertIsNotNone(get_weather_info, "get_weather tool not found in list_tools output.")

        self.assertEqual(get_weather_info.name, "get_weather")
        self.assertEqual(get_weather_info.type, "local")
        self.assertTrue(get_weather_info.is_enabled, "get_weather should be enabled by default.")
        self.assertIn("current weather in a given location", get_weather_info.description.lower()) # Check part of docstring

        # Check parameters schema for get_weather
        self.assertIsNotNone(get_weather_info.parameters)
        self.assertEqual(get_weather_info.parameters.get("type"), "object")
        self.assertIn("location", get_weather_info.parameters.get("properties", {}))
        self.assertIn("unit", get_weather_info.parameters.get("properties", {}))
        self.assertIn("location", get_weather_info.parameters.get("required", []))
        self.assertEqual(get_weather_info.parameters["properties"]["unit"].get("default"), "celsius")

    def test_enable_disable_tool_cycle(self):
        tool_name = "get_weather"

        # 1. Disable
        self.dispatcher.disable_tool(tool_name)
        self.assertFalse(self.dispatcher.tool_states.get(tool_name), f"{tool_name} should be disabled in tool_states.")

        listed_tool_disabled = next((t for t in self.dispatcher.list_tools() if t.name == tool_name), None)
        self.assertIsNotNone(listed_tool_disabled)
        self.assertFalse(listed_tool_disabled.is_enabled, f"{tool_name} should be listed as disabled.")

        # 2. Enable
        self.dispatcher.enable_tool(tool_name)
        self.assertTrue(self.dispatcher.tool_states.get(tool_name), f"{tool_name} should be enabled in tool_states.")

        listed_tool_enabled = next((t for t in self.dispatcher.list_tools() if t.name == tool_name), None)
        self.assertIsNotNone(listed_tool_enabled)
        self.assertTrue(listed_tool_enabled.is_enabled, f"{tool_name} should be listed as enabled.")

    def test_toggle_mcp_tool_state_and_listing(self):
        mcp_tool_name = "mcp_hypothetical_tool"

        # Initially not present in tool_states unless explicitly added by some other means
        self.assertNotIn(mcp_tool_name, self.dispatcher.tool_states)

        # Disable (adds/sets it to False)
        self.dispatcher.disable_tool(mcp_tool_name)
        self.assertFalse(self.dispatcher.tool_states.get(mcp_tool_name))

        mcp_tool_info_disabled = next((t for t in self.dispatcher.list_tools() if t.name == mcp_tool_name), None)
        self.assertIsNotNone(mcp_tool_info_disabled, "Disabled MCP tool should appear in list_tools.")
        self.assertEqual(mcp_tool_info_disabled.type, "mcp")
        self.assertFalse(mcp_tool_info_disabled.is_enabled)
        self.assertIn("Description not available", mcp_tool_info_disabled.description) # Generic desc

        # Enable
        self.dispatcher.enable_tool(mcp_tool_name)
        self.assertTrue(self.dispatcher.tool_states.get(mcp_tool_name))

        mcp_tool_info_enabled = next((t for t in self.dispatcher.list_tools() if t.name == mcp_tool_name), None)
        self.assertIsNotNone(mcp_tool_info_enabled)
        self.assertTrue(mcp_tool_info_enabled.is_enabled)

    def test_dispatch_disabled_tool(self):
        tool_name = "get_weather"
        self.dispatcher.disable_tool(tool_name) # Ensure it's disabled

        call_json = json.dumps({"name": tool_name, "arguments": {"location": "London"}})
        result_dict = self.dispatcher.dispatch(call_json)

        self.assertEqual(result_dict['tool_name'], tool_name)
        self.assertEqual(result_dict['status'], "error_disabled") # Check for specific disabled status
        self.assertIn(f"Tool '{tool_name}' is currently disabled", result_dict['error'])

    # --- Tests for dispatch_openai_tool_call ---

    def test_dispatch_openai_local_tool_get_weather_valid(self):
        args_json_str = json.dumps({"location": "London", "unit": "celsius"})
        result_dict = self.dispatcher.dispatch_openai_tool_call("get_weather", args_json_str)
        self.assertEqual(result_dict['status'], 'success')
        self.assertIn("London is 15 degrees celsius", result_dict['result'])

    def test_dispatch_openai_local_tool_missing_required_arg(self):
        args_json_str = json.dumps({"unit": "celsius"}) # Missing 'location'
        result_dict = self.dispatcher.dispatch_openai_tool_call("get_weather", args_json_str)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("missing 1 required positional argument: 'location'", result_dict['error'].lower())

    def test_dispatch_openai_local_tool_extra_arg(self):
        args_json_str = json.dumps({"location": "New York", "extra_param": "test"})
        result_dict = self.dispatcher.dispatch_openai_tool_call("get_weather", args_json_str)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("got an unexpected keyword argument 'extra_param'", result_dict['error'].lower())

    def test_dispatch_openai_invalid_json_arguments(self):
        args_json_str = '{"location": "Berlin", "unit": "celsius"' # Malformed JSON
        result_dict = self.dispatcher.dispatch_openai_tool_call("get_weather", args_json_str)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("Invalid JSON arguments string", result_dict['error'])
        self.assertIn("get_weather", result_dict['error']) # Ensure tool name is in error

    def test_dispatch_openai_arguments_not_dict_after_parse(self):
        # This tests if the JSON string itself is valid but doesn't parse to a dict (e.g. "[]" or "\"string\"")
        args_json_str = '["not", "a", "dict"]'
        result_dict = self.dispatcher.dispatch_openai_tool_call("get_weather", args_json_str)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("Arguments JSON did not parse to a dictionary", result_dict['error'])

    def test_dispatch_openai_disabled_tool(self):
        tool_name = "get_weather"
        self.dispatcher.disable_tool(tool_name)
        args_json_str = json.dumps({"location": "London"})
        result_dict = self.dispatcher.dispatch_openai_tool_call(tool_name, args_json_str)
        self.assertEqual(result_dict['status'], 'error')
        self.assertIn(f"Tool '{tool_name}' is currently disabled", result_dict['error'])
        self.dispatcher.enable_tool(tool_name) # Clean up for other tests

    def test_dispatch_openai_unknown_tool(self):
        args_json_str = json.dumps({})
        # Similar to test_dispatch_unknown_tool, manage mcp_client presence for specific scenario
        if self.dispatcher.mcp_client:
            with patch.object(self.dispatcher, 'mcp_client', None):
                result_dict = self.dispatcher.dispatch_openai_tool_call("fake_openai_tool", args_json_str)
        else:
            result_dict = self.dispatcher.dispatch_openai_tool_call("fake_openai_tool", args_json_str)

        self.assertEqual(result_dict['status'], 'error')
        self.assertIn("Unknown tool: fake_openai_tool", result_dict['error'])

    @unittest.skipUnless(MCP_ADAPTERS_AVAILABLE, "langchain-mcp-adapters not installed, skipping MCP tests for OpenAI dispatch.")
    @patch('llm_context_os.tools.tool_dispatcher.McpClient')
    def test_dispatch_openai_mcp_tool_success(self, MockMcpClient):
        dispatcher = ToolDispatcher() # Re-init with patch context
        self.assertIsNotNone(dispatcher.mcp_client)
        mock_mcp_instance = dispatcher.mcp_client
        mock_mcp_instance.invoke.return_value = "MCP success via OpenAI call"

        tool_name = "mcp_for_openai"
        args_json_str = json.dumps({"param": "value"})
        result_dict = dispatcher.dispatch_openai_tool_call(tool_name, args_json_str)

        self.assertEqual(result_dict['status'], 'success')
        self.assertEqual(result_dict['result'], "MCP success via OpenAI call")
        mock_mcp_instance.invoke.assert_called_once_with(tool_name, {"param": "value"})

    @unittest.skipUnless(MCP_ADAPTERS_AVAILABLE, "langchain-mcp-adapters not installed, skipping MCP tests for OpenAI dispatch.")
    @patch('llm_context_os.tools.tool_dispatcher.McpClient')
    def test_dispatch_openai_mcp_tool_error(self, MockMcpClient):
        dispatcher = ToolDispatcher() # Re-init
        self.assertIsNotNone(dispatcher.mcp_client)
        mock_mcp_instance = dispatcher.mcp_client
        mock_mcp_instance.invoke.side_effect = Exception("MCP OpenAI call failed")

        tool_name = "mcp_fail_openai"
        args_json_str = json.dumps({})
        result_dict = dispatcher.dispatch_openai_tool_call(tool_name, args_json_str)

        self.assertEqual(result_dict['status'], 'error')
        self.assertIn(f"MCP Error for tool {tool_name} (via OpenAI call): MCP OpenAI call failed", result_dict['error'])

    # --- Tests for get_openai_tool_schemas ---
    def test_get_openai_tool_schemas_basic_structure(self):
        schemas = self.dispatcher.get_openai_tool_schemas()
        self.assertIsInstance(schemas, list)

        # Assuming get_weather is enabled by default
        get_weather_schema = next((s for s in schemas if s.get("function", {}).get("name") == "get_weather"), None)
        self.assertIsNotNone(get_weather_schema, "get_weather schema not found.")

        self.assertEqual(get_weather_schema["type"], "function")
        function_def = get_weather_schema["function"]
        self.assertEqual(function_def["name"], "get_weather")
        self.assertIn("current weather", function_def["description"].lower())

        parameters = function_def["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertIn("properties", parameters)
        self.assertIn("location", parameters["properties"])
        self.assertEqual(parameters["properties"]["location"]["type"], "str") # Based on current _get_tool_parameters
        self.assertIn("unit", parameters["properties"])
        self.assertEqual(parameters["properties"]["unit"]["type"], "str")
        self.assertEqual(parameters["properties"]["unit"]["default"], "celsius")
        self.assertIn("location", parameters["required"])

    def test_get_openai_tool_schemas_disabled_tool_not_included(self):
        tool_name = "get_weather"
        self.dispatcher.disable_tool(tool_name)

        schemas = self.dispatcher.get_openai_tool_schemas()
        get_weather_schema = next((s for s in schemas if s.get("function", {}).get("name") == tool_name), None)
        self.assertIsNone(get_weather_schema, f"Disabled tool '{tool_name}' should not be included in OpenAI schemas.")

        self.dispatcher.enable_tool(tool_name) # Re-enable for other tests

    def test_get_openai_tool_schemas_parameter_schema_edge_cases(self):
        # Temporarily add a tool with problematic parameter schema for testing robustness
        def dummy_tool_no_params(): pass
        def dummy_tool_bad_required_type(): pass

        original_local_tools = self.dispatcher.local_tools.copy()
        original_tool_states = self.dispatcher.tool_states.copy()

        # Mock _get_tool_parameters for these specific dummy tools
        # to simulate problematic schemas returned by it.
        def mock_get_tool_parameters(func):
            if func == dummy_tool_no_params:
                # Simulate a schema that's not a dict or missing keys
                return None # Or some other invalid structure
            if func == dummy_tool_bad_required_type:
                return {"type": "object", "properties": {"p1": {"type": "string"}}, "required": "not_a_list"}
            # Fallback to original method for other tools like get_weather
            return ToolDispatcher._get_tool_parameters(self.dispatcher, func)

        with patch.object(self.dispatcher, '_get_tool_parameters', side_effect=mock_get_tool_parameters):
            self.dispatcher.local_tools["dummy_tool_no_params"] = dummy_tool_no_params
            self.dispatcher.tool_states["dummy_tool_no_params"] = True
            self.dispatcher.local_tools["dummy_tool_bad_required_type"] = dummy_tool_bad_required_type
            self.dispatcher.tool_states["dummy_tool_bad_required_type"] = True

            schemas = self.dispatcher.get_openai_tool_schemas()

            dummy_no_params_schema = next((s for s in schemas if s.get("function", {}).get("name") == "dummy_tool_no_params"), None)
            self.assertIsNotNone(dummy_no_params_schema)
            # Expecting default empty schema because original was None/invalid
            expected_default_params = {"type": "object", "properties": {}, "required": []}
            self.assertEqual(dummy_no_params_schema["function"]["parameters"], expected_default_params)

            dummy_bad_required_schema = next((s for s in schemas if s.get("function", {}).get("name") == "dummy_tool_bad_required_type"), None)
            self.assertIsNotNone(dummy_bad_required_schema)
            # Expecting 'required' to be corrected to an empty list
            self.assertEqual(dummy_bad_required_schema["function"]["parameters"]["required"], [])

        # Restore original tools and states
        self.dispatcher.local_tools = original_local_tools
        self.dispatcher.tool_states = original_tool_states


if __name__ == '__main__':
    unittest.main()
