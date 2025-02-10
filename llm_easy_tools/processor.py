import json
import inspect
import traceback

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union, Optional, Any, get_origin, get_args
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field

from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionMessage, Function

class NoMatchingTool(Exception):
    def __init__(self, message):
        super().__init__(message)

@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    output: Optional[Any] = None
    arguments: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None
    stack_trace: Optional[str] = None
    soft_errors: list[Exception] = field(default_factory=list)
    prefix: Optional[BaseModel] = None
    tool: Optional[Union[Callable, BaseModel]] = None

    def to_message(self) -> dict[str, str]:
        content = f"{self.error}" if self.error else '' if self.output is None else str(self.output)
        return {"role": "tool", "tool_call_id": self.tool_call_id, "name": self.name, "content": content}

def process_tool_call(tool_call, functions_or_models, fix_json_args=True, case_insensitive=False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors = []
    error = None
    stack_trace = None
    output = None

    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            tool_args = json.loads(args.replace(', }', '}').replace(',}', '}'))
        else:
            return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=e, stack_trace=traceback.format_exc())

    tool = next((f for f in functions_or_models if get_name(f, case_insensitive=case_insensitive) == tool_name), None)

    if tool:
        try:
            output, new_soft_errors = _process_unpacked(tool, tool_args, fix_json_args=fix_json_args)
            soft_errors.extend(new_soft_errors)
        except Exception as e:
            error = e
            stack_trace = traceback.format_exc()
    else:
        error = NoMatchingTool(f"Function {tool_name} not found")

    return ToolResult(tool_call_id=tool_call.id, name=tool_name, arguments=tool_args, output=output, error=error, stack_trace=stack_trace, soft_errors=soft_errors, tool=tool)

def _process_unpacked(function, tool_args={}, fix_json_args=True):
    if isinstance(function, LLMFunction):
        function = function.func
    model = parameters_basemodel_from_function(function)
    soft_errors = []

    if fix_json_args:
        for field, field_info in model.model_fields.items():
            if _is_list_type(field_info.annotation) and field in tool_args and isinstance(tool_args[field], str):
                tool_args[field] = [item.strip() for item in tool_args[field].split(',')]
                soft_errors.append(f"Fixed JSON decode error for field {field}")

    model_instance = model(**tool_args)
    args = {field: getattr(model_instance, field) for field in model.model_fields}
    return function(**args), soft_errors

def _is_list_type(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is list or (origin in (Union, Optional) and any(_is_list_type(arg) for arg in args))

def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(message: ChatCompletionMessage, functions: list[Union[Callable, LLMFunction]], fix_json_args=True, case_insensitive=False, executor: ThreadPoolExecutor=None) -> list[ToolResult]:
    tool_calls = message.tool_calls if message.tool_calls else []
    args_list = [(tool_call, functions, fix_json_args, case_insensitive) for tool_call in tool_calls]
    results = list(executor.map(lambda args: process_tool_call(*args), args_list)) if executor else list(map(lambda args: process_tool_call(*args), args_list))
    return results

def process_one_tool_call(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], index: int = 0, fix_json_args=True, case_insensitive=False) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response)
    return process_tool_call(tool_calls[index], functions, fix_json_args, case_insensitive) if tool_calls and index < len(tool_calls) else None

def _get_tool_calls(response: ChatCompletion) -> list[ChatCompletionMessageToolCall]:
    message = response.choices[0].message
    return message.tool_calls if message.tool_calls else []