from .schema_generator import get_function_schema, get_tool_defs, LLMFunction
from .processor import process_response, process_message, process_tool_call, ToolResult

def process_response(response, functions, choice_num=0):
    message = response.choices[choice_num].message
    return process_message(message, functions)

def process_message(message, functions):
    tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else []
    if not tool_calls:
        return []
    return [process_tool_call(tool_call, functions) for tool_call in tool_calls]

def process_tool_call(tool_call, functions):
    function_call = tool_call.function
    tool_name = function_call.name
    tool_args = json.loads(function_call.arguments)
    tool = next((f for f in functions if get_name(f) == tool_name), None)
    if tool is None:
        error = NoMatchingTool(f"Function {tool_name} not found")
        return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=error)
    output, soft_errors = _process_unpacked(tool, tool_args)
    return ToolResult(tool_call_id=tool_call.id, name=tool_name, arguments=tool_args, output=output, soft_errors=soft_errors, tool=tool)


In the rewritten code, the `prefix_class` parameter has been removed from the functions. Unused imports and unnecessary parameters have been removed for clarity. The function behavior has been maintained across versions while streamlining the argument processing.