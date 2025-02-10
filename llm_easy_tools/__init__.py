from .schema_generator import get_function_schema, get_tool_defs, LLMFunction
from .processor import process_response, process_message, process_tool_call, ToolResult
from typing import Union, Callable, Optional

def process_tool_call_with_rules(tool_call, functions_or_models, fix_json_args=True, case_insensitive=False) -> ToolResult:
    try:
        return process_tool_call(tool_call, functions_or_models, fix_json_args, case_insensitive)
    except Exception as e:
        return ToolResult(tool_call_id=tool_call.id, name=tool_call.function.name, error=str(e), output=None, arguments=None, stack_trace=None, soft_errors=[], tool=None)

def process_response_with_rules(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    try:
        return process_response(response, functions, choice_num, **kwargs)
    except Exception as e:
        return [ToolResult(tool_call_id='', name='', error=str(e), output=None, arguments=None, stack_trace=None, soft_errors=[], tool=None)]

def process_message_with_rules(message: ChatCompletionMessage, functions: list[Union[Callable, LLMFunction]], fix_json_args=True, case_insensitive=False, executor=None) -> list[ToolResult]:
    try:
        return process_message(message, functions, fix_json_args, case_insensitive, executor)
    except Exception as e:
        return [ToolResult(tool_call_id='', name='', error=str(e), output=None, arguments=None, stack_trace=None, soft_errors=[], tool=None)]