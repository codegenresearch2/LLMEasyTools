from .schema_generator import get_function_schema, insert_prefix, get_tool_defs, LLMFunction
from .processor import process_response, process_message, process_tool_call, ToolResult

def simplified_process_tool_call(tool_call, functions_or_models, fix_json_args=True, case_insensitive=False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors = []

    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(f"JSON Decode Error: {str(e)}. Attempting to fix.")
            args = args.replace(', }', '}').replace(',}', '}')
            tool_args = json.loads(args)
        else:
            return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=f"JSON Decode Error: {str(e)}")

    tool = None
    for f in functions_or_models:
        if get_name(f, case_insensitive=case_insensitive) == tool_name:
            tool = f
            try:
                output, new_soft_errors = _process_unpacked(f, tool_args, fix_json_args=fix_json_args)
                soft_errors.extend(new_soft_errors)
            except Exception as e:
                return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=f"Function Execution Error: {str(e)}")
            break
    else:
        return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=f"Function Not Found Error: Function {tool_name} not found")

    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_name,
        arguments=tool_args,
        output=output,
        soft_errors=soft_errors,
        tool=tool,
    )

def generate_function_schema(functions_or_models):
    """Simplified function schema generation"""
    return [get_function_schema(f) for f in functions_or_models]

def process_tool_calls(response, functions_or_models, **kwargs):
    """Process tool calls with improved error handling"""
    return process_response(response, functions_or_models, process_tool_call=simplified_process_tool_call, **kwargs)

In the rewritten code, I have improved the error handling in tool calls by providing clearer error messages. I have also simplified the function schema generation by removing unnecessary complexity. The function `simplified_process_tool_call` is a rewritten version of `process_tool_call` that provides clearer error messages. The function `generate_function_schema` is a simplified version of the function schema generation. The function `process_tool_calls` is a rewritten version of `process_response` that uses `simplified_process_tool_call` for processing tool calls.