from .schema_generator import get_tool_defs, LLMFunction
from .processor import process_response

def process_functions(functions):
    simplified_functions = []
    for func in functions:
        if isinstance(func, LLMFunction):
            simplified_functions.append(func.func)
        else:
            simplified_functions.append(func)
    return simplified_functions

def process_llm_response(response, functions):
    functions = process_functions(functions)
    return process_response(response, functions)