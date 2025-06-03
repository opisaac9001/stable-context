# llm_context_os/tools/tool_dispatcher.py
import json
from llm_context_os.tools import builtin_weather # Assuming it's in the same package via __init__.py
import typing as t

class ToolDispatcher:
    def __init__(self):
        print("[ToolDispatcher] Initialized. Placeholder: Would load local tools and MCP adapters here.")
        self.local_tools = {
            "get_weather": builtin_weather.get_weather
        }
        # In a real scenario, might load tools from a config file or discover them dynamically.

    def dispatch(self, function_call_json: str) -> t.Dict[str, t.Any]:
        """
        Parses a function call JSON (string) and executes the appropriate tool.
        Returns a dictionary with the result or an error.
        """
        print(f"[ToolDispatcher] Received function call JSON string: {function_call_json}")
        try:
            call_data = json.loads(function_call_json)
            tool_name = call_data.get("name")
            tool_args = call_data.get("arguments", {}) # Default to empty dict if no arguments

            if not tool_name:
                return {"tool_name": None, "error": "Missing tool name in function call.", "status": "error"}

            if not isinstance(tool_args, dict):
                 return {"tool_name": tool_name, "error": "Tool arguments must be a dictionary (JSON object).", "status": "error"}


            print(f"[ToolDispatcher] Parsed Tool Name: {tool_name}, Arguments: {tool_args}")

            if tool_name in self.local_tools:
                tool_function = self.local_tools[tool_name]
                try:
                    # Note: Real implementation needs robust arg handling, validation, and security.
                    # Consider using Pydantic models for tool argument validation.
                    result = tool_function(**tool_args)
                    return {"tool_name": tool_name, "result": result, "status": "success"}
                except TypeError as e: # Catch issues with mismatched arguments
                    return {"tool_name": tool_name, "error": f"Argument mismatch for tool {tool_name}: {str(e)}", "status": "error"}
                except Exception as e: # Catch other execution errors
                    return {"tool_name": tool_name, "error": f"Error executing tool {tool_name}: {str(e)}", "status": "error"}

            elif tool_name.startswith("mcp_"): # Example prefix for MCP tools
                print(f"[ToolDispatcher] Placeholder: Would dispatch to MCP tool: {tool_name}")
                # Real MCP dispatch logic would go here
                return {"tool_name": tool_name, "result": f"Placeholder result from MCP tool {tool_name} with args {tool_args}", "status": "success_placeholder"}

            else:
                return {"tool_name": tool_name, "error": f"Unknown tool: {tool_name}", "status": "error"}

        except json.JSONDecodeError as e:
            return {"tool_name": None, "error": f"Invalid JSON for function call: {str(e)}", "status": "error"}
        except Exception as e: # Catch-all for other unexpected errors during dispatch logic
            return {"tool_name": None, "error": f"Unexpected error in dispatch: {str(e)}", "status": "error"}

if __name__ == '__main__':
    dispatcher = ToolDispatcher()

    print("\n--- Test 1: Valid local tool call (get_weather) ---")
    weather_call_valid = '{"name": "get_weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}'
    result1 = dispatcher.dispatch(weather_call_valid)
    print(f"Dispatch result 1: {result1}")
    # Expected: {'tool_name': 'get_weather', 'result': 'The weather in New York is 22 degrees fahrenheit.', 'status': 'success'}


    print("\n--- Test 2: Local tool with different arguments (unknown location) ---")
    weather_call_unknown_loc = '{"name": "get_weather", "arguments": {"location": "Paris"}}'
    result2 = dispatcher.dispatch(weather_call_unknown_loc)
    print(f"Dispatch result 2: {result2}")
    # Expected: {'tool_name': 'get_weather', 'result': "Sorry, I don't have weather information for Paris.", 'status': 'success'}

    print("\n--- Test 3: Local tool with missing required argument (if any were required beyond defaults) ---")
    # get_weather's 'location' is required. 'unit' has a default.
    weather_call_missing_arg = '{"name": "get_weather", "arguments": {"unit": "celsius"}}'
    result_missing = dispatcher.dispatch(weather_call_missing_arg)
    print(f"Dispatch result (missing arg): {result_missing}")
    # Expected: {'tool_name': 'get_weather', 'error': "Argument mismatch for tool get_weather: get_weather() missing 1 required positional argument: 'location'", 'status': 'error'}


    print("\n--- Test 4: MCP tool placeholder ---")
    mcp_call = '{"name": "mcp_some_enterprise_tool", "arguments": {"param1": "value1"}}'
    result3 = dispatcher.dispatch(mcp_call)
    print(f"Dispatch result 3: {result3}")
    # Expected: {'tool_name': 'mcp_some_enterprise_tool', 'result': "Placeholder result from MCP tool mcp_some_enterprise_tool with args {'param1': 'value1'}", 'status': 'success_placeholder'}

    print("\n--- Test 5: Unknown tool ---")
    unknown_tool_call = '{"name": "non_existent_tool", "arguments": {}}'
    result4 = dispatcher.dispatch(unknown_tool_call)
    print(f"Dispatch result 4: {result4}")
    # Expected: {'tool_name': 'non_existent_tool', 'error': 'Unknown tool: non_existent_tool', 'status': 'error'}

    print("\n--- Test 6: Invalid JSON string ---")
    invalid_json_call = '{"name": "get_weather", "arguments": {"location": "New York"}' # Missing closing brace
    result5 = dispatcher.dispatch(invalid_json_call)
    print(f"Dispatch result 5: {result5}")
    # Expected: {'tool_name': None, 'error': 'Invalid JSON for function call: ...', 'status': 'error'}

    print("\n--- Test 7: Malformed call data (arguments not a dict) ---")
    malformed_args_call = '{"name": "get_weather", "arguments": "New York"}'
    result6 = dispatcher.dispatch(malformed_args_call)
    print(f"Dispatch result 6: {result6}")
    # Expected: {'tool_name': 'get_weather', 'error': 'Tool arguments must be a dictionary (JSON object).', 'status': 'error'}

    print("\n--- Test 8: No tool name ---")
    no_tool_name_call = '{"arguments": {"location": "New York"}}'
    result7 = dispatcher.dispatch(no_tool_name_call)
    print(f"Dispatch result 7: {result7}")
    # Expected: {'tool_name': None, 'error': 'Missing tool name in function call.', 'status': 'error'}

    print("\nToolDispatcher demo complete.")
