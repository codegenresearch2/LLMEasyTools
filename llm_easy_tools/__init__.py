from .schema_generator import get_function_schema, get_tool_defs, LLMFunction
from .processor import process_response, process_message, process_tool_call, ToolResult
from .types import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall

def process_tool_call(tool_call, functions_or_models, fix_json_args, case_insensitive):
    return process_tool_call(tool_call, functions_or_models, fix_json_args, case_insensitive)

def process_response(response, functions, choice_num=0, **kwargs):
    return process_response(response, functions, choice_num, **kwargs)

def process_message(message, functions, fix_json_args, case_insensitive, executor):
    return process_message(message, functions, fix_json_args, case_insensitive, executor)