# llm_context_os/tools/tool_dispatcher.py
import json
from llm_context_os.tools import builtin_weather
import typing as t
import os # Potentially for MCP config later

# Attempt to import langchain_mcp_adapters
MCP_ADAPTERS_AVAILABLE = False
McpClient = None
try:
    from langchain_mcp_adapters import McpClient # Assuming this is the client class
    MCP_ADAPTERS_AVAILABLE = True
    print("Successfully imported McpClient from langchain_mcp_adapters.")
except ImportError:
    print("Warning: langchain-mcp-adapters not found. MCP tool dispatching will be unavailable.")
except Exception as e: # Catch other potential errors during import
    print(f"Warning: Error importing McpClient from langchain-mcp-adapters: {e}. MCP tool dispatching will be unavailable.")


class ToolDispatcher:
    def __init__(self):
        self.local_tools = {
            "get_weather": builtin_weather.get_weather
        }
        self.mcp_client: t.Optional[McpClient] = None

        if MCP_ADAPTERS_AVAILABLE:
            try:
                # Initialize McpClient.
                # For this task, assuming default constructor works or it uses env vars.
                # Real-world usage might involve passing server_urls from a config file:
                # e.g., mcp_server_url = os.getenv("MCP_SERVER_URL", "http://default_mcp_server/api")
                # self.mcp_client = McpClient(base_url=mcp_server_url)
                self.mcp_client = McpClient()
                print("[ToolDispatcher] McpClient initialized successfully.")
            except Exception as e:
                print(f"[ToolDispatcher] Error initializing McpClient: {e}. MCP tools will be unavailable.")
                self.mcp_client = None
        else:
            print("[ToolDispatcher] langchain-mcp-adapters not available. MCP tools will not be dispatched.")

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
                    result = tool_function(**tool_args)
                    return {"tool_name": tool_name, "result": result, "status": "success"}
                except TypeError as e:
                    return {"tool_name": tool_name, "error": f"Argument mismatch for local tool {tool_name}: {str(e)}", "status": "error"}
                except Exception as e:
                    return {"tool_name": tool_name, "error": f"Error executing local tool {tool_name}: {str(e)}", "status": "error"}

            # If not a local tool, try MCP if available
            elif self.mcp_client:
                print(f"[ToolDispatcher] Attempting to dispatch '{tool_name}' to MCP client with args: {tool_args}")
                try:
                    # Assuming McpClient has an 'invoke' or similar method.
                    # The structure of the response from mcp_client.invoke needs to be known.
                    # For now, assume it returns the direct result of the tool.
                    # If it returns a more complex object, adaptation will be needed.
                    mcp_result = self.mcp_client.invoke(tool_name, tool_args) # Or e.g. tool_args if it expects a dict
                    # Example: mcp_result = self.mcp_client.invoke({"tool_name": tool_name, "tool_input": tool_args})

                    print(f"[ToolDispatcher] MCP tool '{tool_name}' executed. Result: {mcp_result}")
                    return {"tool_name": tool_name, "result": mcp_result, "status": "success"}
                except Exception as e:
                    # This could be various errors: tool not found on MCP, execution error on MCP, network error.
                    # langchain_mcp_adapters might raise specific exceptions to distinguish these.
                    error_message = f"MCP Error for tool {tool_name}: {str(e)}"
                    print(f"[ToolDispatcher] {error_message}")
                    return {"tool_name": tool_name, "error": error_message, "status": "error"}

            # If not local and MCP client not available or tool not found by MCP (if McpClient raises specific error caught above)
            else:
                return {"tool_name": tool_name, "error": f"Unknown tool: {tool_name}. Not found in local tools or MCP (client unavailable/tool not found).", "status": "error"}

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


    print("\n--- Test 4: MCP tool call (mocked success) ---")
    # This test requires McpClient to be available or mocked.
    # We will mock it for this __main__ block for demonstration.
    if MCP_ADAPTERS_AVAILABLE:
        # Mocking the mcp_client instance on the dispatcher for this test
        original_mcp_client = dispatcher.mcp_client

        mock_mcp_client_instance = MagicMock(spec=McpClient) # Use spec for better mocking
        mock_mcp_client_instance.invoke.return_value = "Mocked successful result from MCP tool"
        dispatcher.mcp_client = mock_mcp_client_instance

        mcp_call_success = '{"name": "mcp_some_tool", "arguments": {"param": "value"}}'
        result_mcp_success = dispatcher.dispatch(mcp_call_success)
        print(f"Dispatch result (MCP success): {result_mcp_success}")
        mock_mcp_client_instance.invoke.assert_called_once_with("mcp_some_tool", {"param": "value"})

        print("\n--- Test 4b: MCP tool call (mocked error) ---")
        mock_mcp_client_instance.reset_mock() # Reset call counts etc.
        mock_mcp_client_instance.invoke.side_effect = Exception("MCP tool execution failed on server")
        mcp_call_error = '{"name": "mcp_another_tool", "arguments": {}}'
        result_mcp_error = dispatcher.dispatch(mcp_call_error)
        print(f"Dispatch result (MCP error): {result_mcp_error}")
        mock_mcp_client_instance.invoke.assert_called_once_with("mcp_another_tool", {})

        dispatcher.mcp_client = original_mcp_client # Restore original client
    else:
        print("Skipping MCP tests as langchain-mcp-adapters is not available.")


    print("\n--- Test 5: Unknown tool (MCP client unavailable or tool truly unknown) ---")
    if dispatcher.mcp_client: # If MCP client was initialized (even if later mocked for specific tests)
        original_mcp_client_for_unknown = dispatcher.mcp_client
        # Simulate McpClient raising a "ToolNotFound" style error or generic Exception if tool is not on MCP
        # For simplicity, let's assume it just doesn't find it after local check
        # Or, if the MCP client itself is None because it failed to init or lib not available:
        if not MCP_ADAPTERS_AVAILABLE: # If lib was never there
             dispatcher.mcp_client = None
        else: # If lib is there, but we want to test "MCP does not find it"
             # This depends on McpClient's behavior. If it raises error for unknown tool, test above covers it.
             # If it returns None or specific non-error, that's different.
             # For now, the existing logic assumes if not local, and mcp_client exists, it tries.
             # If mcp_client.invoke itself handles "tool not found" by raising an Exception, Test 4b covers it.
             pass # Covered by existing logic: if not local, try mcp, if mcp fails, error. If no mcp client, error.

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
