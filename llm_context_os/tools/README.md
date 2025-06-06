# Tools in LLM Context OS

The LLM Context OS includes a flexible tool system that allows Large Language Models (LLMs) to interact with external functionalities. This enhances the LLM's capabilities beyond text generation, enabling them to fetch real-time information, interact with other services, or perform specific actions.

## Tool Dispatcher

The core of the tool system is the `ToolDispatcher` class (`llm_context_os.tools.tool_dispatcher.ToolDispatcher`). It is responsible for:

*   Registering available tools (both local Python functions and potentially external tools via Model Capability Packs - MCPs).
*   Listing tools and their configurations.
*   Dispatching calls from the LLM to the appropriate tool with the provided arguments.
*   Managing the enabled/disabled state of tools.

## OpenAI-Compatible Tool Calling

The system has been updated to support an OpenAI-compatible tool calling mechanism. This means:

1.  **Tool Definition as JSON Schema:** Tools, especially local Python functions, can have their parameters and descriptions automatically converted into JSON schemas that align with OpenAI's function calling specifications. This allows LLMs familiar with this format to understand how to use the tools.
    *   These schemas can be retrieved via the API endpoint: `GET /tools/openai_schemas`.

2.  **LLM Interaction Flow:**
    *   When an LLM needs to use a tool, it's expected to generate a `tool_calls` object in its response (this part is handled by the specific model runner and how it's prompted).
    *   The LLM Context OS API (specifically the `/chat` endpoint) detects these `tool_calls`.
    *   For each requested tool call, the API uses the `ToolDispatcher` to execute the tool with the arguments provided by the LLM.
    *   The results from the tool executions are then formatted and sent back to the LLM.
    *   The LLM uses these results to formulate its final textual response to the user.

3.  **Supported Tool Call Format (Internal):**
    *   While the aim is OpenAI compatibility, the internal dispatch mechanism within `ToolDispatcher` (`dispatch_openai_tool_call` method) directly takes the tool name and a JSON string of arguments.
    *   The previous custom `[FUNCALL]` string syntax for tool invocation is being phased out in favor of this more structured OpenAI-aligned approach. While the `ToolDispatcher` might still contain the older `dispatch` method for `[FUNCALL]` strings for backward compatibility or internal use, the primary and recommended flow for new LLM integrations is the OpenAI-style `tool_calls` mechanism.

### Example OpenAI Tool Schema

Here's an example of what an OpenAI-compatible schema for a local tool like `get_weather` might look like (as returned by `/tools/openai_schemas`):

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Gets the current weather in a given location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "str",
          "description": "The city and state, e.g. San Francisco, CA"
        },
        "unit": {
          "type": "str",
          "default": "celsius",
          "description": "The unit of temperature (celsius or fahrenheit)"
        }
      },
      "required": ["location"]
    }
  }
}
```

## Local Tools

Local tools are Python functions. For them to be discoverable and usable by the `ToolDispatcher` and properly represented in the OpenAI schemas:

*   They should have clear function signatures with type hints.
*   Their docstrings are used as the `description` for the tool.
*   Parameter types and default values are inferred from the signature.

Currently, local tools are registered directly within the `ToolDispatcher`'s `__init__` method. Future enhancements might include discovery from designated directories.

## Model Capability Packs (MCPs)

The system is designed with the potential to integrate external tools through Model Capability Packs (MCPs), though the detailed implementation of MCP adapters and schema conversion for them is an ongoing development. The `ToolDispatcher` has placeholder logic for an `McpClient`.
