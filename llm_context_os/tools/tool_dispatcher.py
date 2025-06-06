# llm_context_os/tools/tool_dispatcher.py
import json
import inspect # For tool parameter and docstring inspection
from llm_context_os.tools import builtin_weather
import typing as t
from llm_context_os.api.schemas import ToolInfo # For listing tools

# Attempt to import langchain_mcp_adapters
MCP_ADAPTERS_AVAILABLE = False
McpClient = None # Define McpClient for type hinting even if import fails
try:
    from langchain_mcp_adapters import McpClient # Assuming this is the client class
    MCP_ADAPTERS_AVAILABLE = True
    print("Successfully imported McpClient from langchain_mcp_adapters.")
except ImportError:
    print("Warning: langchain-mcp-adapters not found. MCP tool dispatching will be unavailable.")
except Exception as e:
    print(f"Warning: Error importing McpClient from langchain-mcp-adapters: {e}. MCP tool dispatching will be unavailable.")


class ToolDispatcher:
    def __init__(self):
        self.local_tools: t.Dict[str, t.Callable] = {
            "get_weather": builtin_weather.get_weather
        }
        self.tool_states: t.Dict[str, bool] = {name: True for name in self.local_tools} # Default all local tools to enabled

        self.mcp_client: t.Optional[McpClient] = None
        if MCP_ADAPTERS_AVAILABLE:
            try:
                self.mcp_client = McpClient()
                print("[ToolDispatcher] McpClient initialized successfully.")
                # Potentially pre-fetch available MCP tools and populate self.tool_states
                # For now, MCP tools are implicitly enabled if client is up and tool is called.
                # Toggling them would require knowing their names beforehand.
            except Exception as e:
                print(f"[ToolDispatcher] Error initializing McpClient: {e}. MCP tools will be unavailable.")
                self.mcp_client = None
        else:
            print("[ToolDispatcher] langchain-mcp-adapters not available. MCP tools will not be dispatched.")

    def _get_tool_parameters(self, func: t.Callable) -> t.Dict[str, t.Any]:
        params_schema = {"type": "object", "properties": {}, "required": []}
        try:
            sig = inspect.signature(func)
            for name, param in sig.parameters.items():
                param_info = {}
                # Type annotation
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, '__name__'):
                        param_info["type"] = param.annotation.__name__.lower() # Simplified type
                    elif hasattr(param.annotation, '__origin__') and param.annotation.__origin__ in [t.Union, t.Optional]:
                         # Handle Optional[type] or Union[type, None]
                        args = t.get_args(param.annotation)
                        type_args = [a for a in args if a is not type(None)]
                        if type_args and hasattr(type_args[0], '__name__'):
                             param_info["type"] = type_args[0].__name__.lower()
                        else: # Fallback for complex Unions or other types
                             param_info["type"] = str(param.annotation)

                # Default value
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                else:
                    params_schema["required"].append(name)

                params_schema["properties"][name] = param_info
        except Exception as e:
            print(f"Warning: Could not generate detailed schema for tool {func.__name__}: {e}")
            # Return a generic schema or indicate schema generation failure
            return {"type": "object", "properties": {}, "description": "Parameter schema unavailable."}
        return params_schema

    def list_tools(self) -> t.List[ToolInfo]:
        tool_info_list = []
        # Local Tools
        for tool_name, func in self.local_tools.items():
            description = inspect.getdoc(func)
            params = self._get_tool_parameters(func)
            is_enabled = self.tool_states.get(tool_name, True) # Default to True
            tool_info_list.append(ToolInfo(
                name=tool_name,
                type="local",
                description=description,
                is_enabled=is_enabled,
                parameters=params
            ))

        # MCP Tools - Conceptual (Requires McpClient to have a method to list tools)
        # if self.mcp_client:
        #     try:
        #         mcp_tool_list = self.mcp_client.list_available_tools() # Hypothetical method
        #         for mcp_tool in mcp_tool_list:
        #             is_enabled = self.tool_states.get(mcp_tool.name, True) # Default to True
        #             tool_info_list.append(ToolInfo(
        #                 name=mcp_tool.name,
        #                 type="mcp",
        #                 description=mcp_tool.description, # Assuming these attributes exist
        #                 is_enabled=is_enabled,
        #                 parameters=mcp_tool.parameters_schema # Assuming this attribute exists
        #             ))
        #     except Exception as e:
        #         print(f"Warning: Could not retrieve tool list from MCP client: {e}")
        # For now, MCP tools are not proactively listed, only discovered upon dispatch attempt.
        # If an MCP tool state was set via enable/disable, it would be reflected if we listed it.

        # Add any MCP tools whose state was explicitly set, even if not discoverable
        if self.mcp_client: # Only relevant if MCP client exists
            for tool_name, is_enabled_state in self.tool_states.items():
                if tool_name not in self.local_tools: # It's an MCP tool we've learned about by toggle
                    # We don't have description or params here, so list with minimal info
                     if not any(t.name == tool_name for t in tool_info_list): # Avoid duplicates if discoverable later
                        tool_info_list.append(ToolInfo(
                            name=tool_name,
                            type="mcp",
                            description="Description not available (MCP tool state managed).",
                            is_enabled=is_enabled_state,
                            parameters={"type": "object", "properties": {}, "description": "Parameters not available."}
                        ))
        return tool_info_list

    def enable_tool(self, tool_name: str) -> bool:
        # This enables the tool for dispatch. If it's not a known local tool,
        # it's assumed to be an MCP tool whose state is being managed.
        self.tool_states[tool_name] = True
        print(f"[ToolDispatcher] Tool '{tool_name}' enabled.")
        return True # Always succeeds in setting the state

    def disable_tool(self, tool_name: str) -> bool:
        self.tool_states[tool_name] = False
        print(f"[ToolDispatcher] Tool '{tool_name}' disabled.")
        return True # Always succeeds in setting the state

    def dispatch(self, function_call_json: str) -> t.Dict[str, t.Any]:
        """
        Parses a function call JSON (string) from [FUNCALL] syntax and executes the appropriate tool.
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

            # Check if tool is enabled before dispatching
            # Defaults to True if tool_name not in self.tool_states (e.g. a new MCP tool not yet toggled)
            if not self.tool_states.get(tool_name, True):
                print(f"[ToolDispatcher] Tool '{tool_name}' is disabled. Not dispatching.")
                return {"tool_name": tool_name, "error": f"Tool '{tool_name}' is currently disabled.", "status": "error_disabled"}

            if tool_name in self.local_tools:
                tool_function = self.local_tools[tool_name]
                try:
                    result = tool_function(**tool_args)
                    return {"tool_name": tool_name, "result": result, "status": "success"}
                except TypeError as e:
                    return {"tool_name": tool_name, "error": f"Argument mismatch for local tool {tool_name}: {str(e)}", "status": "error"}
                except Exception as e:
                    return {"tool_name": tool_name, "error": f"Error executing local tool {tool_name}: {str(e)}", "status": "error"}

            elif self.mcp_client:
                print(f"[ToolDispatcher] Attempting to dispatch '{tool_name}' to MCP client with args: {tool_args}")
                try:
                    mcp_result = self.mcp_client.invoke(tool_name, tool_args) # type: ignore
                    print(f"[ToolDispatcher] MCP tool '{tool_name}' executed. Result: {mcp_result}")
                    # Update tool_states to mark this MCP tool as known and enabled by default if it was successful
                    if tool_name not in self.tool_states: self.tool_states[tool_name] = True
                    return {"tool_name": tool_name, "result": mcp_result, "status": "success"}
                except Exception as e:
                    error_message = f"MCP Error for tool {tool_name}: {str(e)}"
                    print(f"[ToolDispatcher] {error_message}")
                    return {"tool_name": tool_name, "error": error_message, "status": "error"}

            else:
                return {"tool_name": tool_name, "error": f"Unknown tool: {tool_name}. Not found in local tools or MCP (client unavailable/tool not found).", "status": "error"}

        except json.JSONDecodeError as e:
            return {"tool_name": None, "error": f"Invalid JSON for function call: {str(e)}", "status": "error"}
        except Exception as e: # Catch-all for other unexpected errors during dispatch logic
            return {"tool_name": None, "error": f"Unexpected error in dispatch: {str(e)}", "status": "error"}

    def dispatch_openai_tool_call(self, tool_name: str, arguments_json_str: str) -> t.Dict[str, t.Any]:
        """
        Executes the appropriate tool based on OpenAI-style tool name and JSON string arguments.
        Returns a dictionary with the result or an error.
        """
        print(f"[ToolDispatcher] Received OpenAI tool call: Name='{tool_name}', ArgsJSON='{arguments_json_str}'")
        parsed_args: t.Dict[str, t.Any] = {}
        try:
            parsed_args = json.loads(arguments_json_str)
            if not isinstance(parsed_args, dict):
                raise json.JSONDecodeError("Arguments JSON did not parse to a dictionary.", arguments_json_str, 0)
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON arguments string for tool {tool_name}: {str(e)}"}

        # Check if tool is enabled
        if not self.tool_states.get(tool_name, True): # Defaults to True if not in states (e.g. new MCP tool)
            print(f"[ToolDispatcher] OpenAI Tool '{tool_name}' is disabled. Not dispatching.")
            return {"status": "error", "error": f"Tool '{tool_name}' is currently disabled."}

        if tool_name in self.local_tools:
            tool_function = self.local_tools[tool_name]
            try:
                result = tool_function(**parsed_args)
                return {"status": "success", "result": result}
            except TypeError as e:
                return {"status": "error", "error": f"Argument mismatch for local tool {tool_name}: {str(e)}"}
            except Exception as e:
                return {"status": "error", "error": f"Error executing local tool {tool_name}: {str(e)}"}

        elif self.mcp_client:
            print(f"[ToolDispatcher] Attempting to dispatch OpenAI tool '{tool_name}' to MCP client with args: {parsed_args}")
            try:
                mcp_result = self.mcp_client.invoke(tool_name, parsed_args) # type: ignore
                print(f"[ToolDispatcher] MCP tool '{tool_name}' (via OpenAI call) executed. Result: {mcp_result}")
                if tool_name not in self.tool_states: self.tool_states[tool_name] = True
                return {"status": "success", "result": mcp_result}
            except Exception as e:
                error_message = f"MCP Error for tool {tool_name} (via OpenAI call): {str(e)}"
                print(f"[ToolDispatcher] {error_message}")
                return {"status": "error", "error": error_message}
        else:
            return {"status": "error", "error": f"Unknown tool: {tool_name}. Not found in local tools or MCP (client unavailable/tool not found)."}

    def get_openai_tool_schemas(self) -> t.List[t.Dict[str, t.Any]]:
        openai_schemas = []
        all_tools_info = self.list_tools() # This gets all tools, including MCP if states were set

        for tool_info in all_tools_info:
            # For now, only generate OpenAI schemas for local tools where we reliably have parameter info
            # and for tools that are currently enabled.
            if tool_info.type == "local" and tool_info.is_enabled:
                # Ensure description is not None
                description = tool_info.description if tool_info.description is not None else ""

                # Ensure parameters schema is valid, defaulting if necessary
                parameters_schema = tool_info.parameters
                if not isinstance(parameters_schema, dict) or \
                   not isinstance(parameters_schema.get("type"), str) or \
                   not isinstance(parameters_schema.get("properties"), dict):
                    # If schema is malformed or missing, use a default empty schema
                    parameters_schema = {"type": "object", "properties": {}, "required": []}

                # Ensure 'required' field is a list of strings, if present
                if "required" in parameters_schema and not all(isinstance(item, str) for item in parameters_schema["required"]):
                    print(f"Warning: 'required' field for tool '{tool_info.name}' is not a list of strings. Setting to empty list for OpenAI schema.")
                    parameters_schema["required"] = []


                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_info.name,
                        "description": description,
                        "parameters": parameters_schema
                    }
                }
                openai_schemas.append(tool_schema)
            # TODO: Future: Consider how to represent MCP tools if their parameter schemas become available
            # via McpClient.list_available_tools() and are compatible.

        return openai_schemas


if __name__ == '__main__':
    # Mock McpClient for standalone testing if library is not available or for controlled tests
    # This basic mock helps avoid crashing if McpClient cannot be imported/initialized.
    if not MCP_ADAPTERS_AVAILABLE:
        class MockMcpClient:
            def invoke(self, tool_name: str, arguments: t.Dict[str, t.Any]):
                print(f"[MockMcpClient] Invoked tool: {tool_name} with args: {arguments}")
                if tool_name == "mcp_some_tool":
                    return "Mocked successful result from MCP tool"
                elif tool_name == "mcp_another_tool":
                    raise Exception("MCP tool execution failed on server")
                raise Exception(f"Mock MCP tool '{tool_name}' not found.")
        McpClient = MockMcpClient # type: ignore

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

    print("\n--- Testing Tool Listing and Management ---")
    print("Initial tool list:")
    initial_tools = dispatcher.list_tools()
    for tool_info in initial_tools:
        print(f"  - {tool_info.name} (Enabled: {tool_info.is_enabled}, Type: {tool_info.type}, Params: {tool_info.parameters})")

    print("\nDisabling 'get_weather' tool...")
    dispatcher.disable_tool("get_weather")
    get_weather_info_disabled = next((t for t in dispatcher.list_tools() if t.name == "get_weather"), None)
    if get_weather_info_disabled:
        print(f"  - get_weather (Enabled: {get_weather_info_disabled.is_enabled})") # Should be False
        assert not get_weather_info_disabled.is_enabled

    print("\nAttempting to dispatch disabled 'get_weather' using [FUNCALL]:")
    weather_call_disabled = '{"name": "get_weather", "arguments": {"location": "New York"}}'
    result_disabled = dispatcher.dispatch(weather_call_disabled)
    print(f"Dispatch result (disabled): {result_disabled}")
    assert result_disabled['status'] == 'error_disabled'
    assert "disabled" in result_disabled['error']

    print("\nEnabling 'get_weather' tool...")
    dispatcher.enable_tool("get_weather")
    get_weather_info_enabled = next((t for t in dispatcher.list_tools() if t.name == "get_weather"), None)
    if get_weather_info_enabled:
        print(f"  - get_weather (Enabled: {get_weather_info_enabled.is_enabled})") # Should be True
        assert get_weather_info_enabled.is_enabled

    print("\nAttempting to dispatch enabled 'get_weather' using [FUNCALL]:")
    result_enabled_again = dispatcher.dispatch(weather_call_valid) # Use valid call from earlier
    print(f"Dispatch result (enabled again): {result_enabled_again}")
    assert result_enabled_again['status'] == 'success'

    print("\n--- Test OpenAI Dispatch Method ---")
    print("\n--- Test OpenAI 1: Valid local tool call (get_weather) ---")
    openai_result1 = dispatcher.dispatch_openai_tool_call("get_weather", '{"location": "New York", "unit": "fahrenheit"}')
    print(f"OpenAI Dispatch result 1: {openai_result1}")
    assert openai_result1['status'] == 'success'
    assert "New York" in openai_result1.get("result", "")

    print("\n--- Test OpenAI 2: Local tool with argument error ---")
    openai_result2 = dispatcher.dispatch_openai_tool_call("get_weather", '{"unit": "celsius"}') # Missing 'location'
    print(f"OpenAI Dispatch result 2: {openai_result2}")
    assert openai_result2['status'] == 'error'
    assert "Argument mismatch" in openai_result2.get("error", "")

    print("\n--- Test OpenAI 3: Tool disabled ---")
    dispatcher.disable_tool("get_weather")
    openai_result3 = dispatcher.dispatch_openai_tool_call("get_weather", '{"location": "London", "unit": "celsius"}')
    print(f"OpenAI Dispatch result 3 (disabled): {openai_result3}")
    assert openai_result3['status'] == 'error'
    assert "disabled" in openai_result3.get("error", "")
    dispatcher.enable_tool("get_weather") # Re-enable for subsequent tests if any

    print("\n--- Test OpenAI 4: Unknown tool ---")
    openai_result4 = dispatcher.dispatch_openai_tool_call("no_such_tool_openai", '{}')
    print(f"OpenAI Dispatch result 4 (unknown): {openai_result4}")
    assert openai_result4['status'] == 'error'
    assert "Unknown tool" in openai_result4.get("error", "")

    print("\n--- Test OpenAI 5: Invalid JSON arguments ---")
    openai_result5 = dispatcher.dispatch_openai_tool_call("get_weather", '{"location": "Berlin", "unit": "celsius"') # Malformed JSON
    print(f"OpenAI Dispatch result 5 (invalid JSON): {openai_result5}")
    assert openai_result5['status'] == 'error'
    assert "Invalid JSON arguments" in openai_result5.get("error", "")

    # Test MCP via OpenAI dispatch if MCP client is available/mocked
    if dispatcher.mcp_client: # Check if a real or mock client is there
        print("\n--- Test OpenAI 6: MCP tool call (mocked success) ---")
        openai_mcp_success = dispatcher.dispatch_openai_tool_call("mcp_some_tool", '{"param": "value"}')
        print(f"OpenAI Dispatch result (MCP success): {openai_mcp_success}")
        assert openai_mcp_success['status'] == 'success'
        assert openai_mcp_success.get("result") == "Mocked successful result from MCP tool"

        print("\n--- Test OpenAI 7: MCP tool call (mocked error) ---")
        openai_mcp_error = dispatcher.dispatch_openai_tool_call("mcp_another_tool", '{}')
        print(f"OpenAI Dispatch result (MCP error): {openai_mcp_error}")
        assert openai_mcp_error['status'] == 'error'
        assert "MCP tool execution failed on server" in openai_mcp_error.get("error", "")

    print("\nTesting enabling/disabling a non-local (presumed MCP) tool for OpenAI dispatch:")
    dispatcher.disable_tool("mcp_hypothetical_openai_tool")
    mcp_tool_info_disabled_openai = next((t for t in dispatcher.list_tools() if t.name == "mcp_hypothetical_openai_tool"), None)
    # Note: list_tools might create a minimal entry if a tool state is set for an unknown tool
    if mcp_tool_info_disabled_openai:
         assert not mcp_tool_info_disabled_openai.is_enabled
    else: # If it wasn't added to list_tools just by setting state
         assert not dispatcher.tool_states.get("mcp_hypothetical_openai_tool")

    openai_mcp_disabled_call = dispatcher.dispatch_openai_tool_call("mcp_hypothetical_openai_tool", '{}')
    print(f"OpenAI Dispatch result (MCP disabled): {openai_mcp_disabled_call}")
    assert openai_mcp_disabled_call['status'] == 'error'
    assert "disabled" in openai_mcp_disabled_call.get("error", "")

    dispatcher.enable_tool("mcp_hypothetical_openai_tool")
    mcp_tool_info_enabled_openai = next((t for t in dispatcher.list_tools() if t.name == "mcp_hypothetical_openai_tool"), None)
    if mcp_tool_info_enabled_openai:
        assert mcp_tool_info_enabled_openai.is_enabled
    else:
        assert dispatcher.tool_states.get("mcp_hypothetical_openai_tool")

    print("\n--- Test get_openai_tool_schemas ---")
    openai_schemas = dispatcher.get_openai_tool_schemas()
    print("OpenAI Tool Schemas:")
    for schema in openai_schemas:
        print(json.dumps(schema, indent=2))

    # Check if get_weather is in the schemas (assuming it's enabled by default)
    get_weather_schema = next((s for s in openai_schemas if s["function"]["name"] == "get_weather"), None)
    assert get_weather_schema is not None
    assert get_weather_schema["function"]["parameters"]["type"] == "object"
    assert "location" in get_weather_schema["function"]["parameters"]["properties"]

    # Disable get_weather and check again
    print("\nDisabling get_weather and checking OpenAI schemas again...")
    dispatcher.disable_tool("get_weather")
    openai_schemas_after_disable = dispatcher.get_openai_tool_schemas()
    get_weather_schema_after_disable = next((s for s in openai_schemas_after_disable if s["function"]["name"] == "get_weather"), None)
    assert get_weather_schema_after_disable is None, "Disabled tool 'get_weather' should not be in OpenAI schemas."
    dispatcher.enable_tool("get_weather") # Re-enable for other tests


    print("\nTool Management and OpenAI Dispatch Demo complete.")
