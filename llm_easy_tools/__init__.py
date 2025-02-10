from .schema_generator import get_function_schema, get_tool_defs, LLMFunction
from .processor import process_response, process_message, process_tool_call, ToolResult

def simplify_functions(functions):
    """
    Simplify a list of functions by extracting the underlying function from LLMFunction instances.

    Args:
        functions (list): A list of functions or LLMFunction instances.

    Returns:
        list: A list of simplified functions.
    """
    simplified_functions = []
    for func in functions:
        if isinstance(func, LLMFunction):
            simplified_functions.append(func.func)
        else:
            simplified_functions.append(func)
    return simplified_functions

def process_llm_response(response, functions):
    """
    Process an LLM response by simplifying the functions and using the process_response function.

    Args:
        response: The LLM response to process.
        functions (list): A list of functions or LLMFunction instances.

    Returns:
        list: A list of ToolResult instances representing the processed tool calls.
    """
    functions = simplify_functions(functions)
    return process_response(response, functions)