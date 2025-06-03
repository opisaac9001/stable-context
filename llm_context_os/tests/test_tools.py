# llm_context_os/tests/test_tools.py
import unittest
import json
from llm_context_os.tools.tool_dispatcher import ToolDispatcher
from llm_context_os.tools import builtin_weather

class TestToolDispatcherAndTools(unittest.TestCase):

    def setUp(self):
        # It's good practice to create a new dispatcher for each test
        # if its state could be modified (though current placeholder is stateless after init)
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


    def test_dispatch_mcp_tool_placeholder(self):
        mcp_call_json = json.dumps({"name": "mcp_example_tool", "arguments": {"data": "test"}})
        result_dict = self.dispatcher.dispatch(mcp_call_json)
        self.assertEqual(result_dict['tool_name'], 'mcp_example_tool')
        self.assertEqual(result_dict['status'], 'success_placeholder') # As per ToolDispatcher code
        self.assertIn("Placeholder result from MCP tool", result_dict['result'])
        self.assertIn("'data': 'test'", result_dict['result']) # Check args are passed

    def test_dispatch_unknown_tool(self):
        unknown_call_json = json.dumps({"name": "fake_tool", "arguments": {}}) # Added arguments
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


if __name__ == '__main__':
    unittest.main()
