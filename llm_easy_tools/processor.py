from typing import Callable, Union, Optional, Any, get_origin, get_args
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
from pydantic import BaseModel, ValidationError
from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionMessage

class NoMatchingTool(Exception):
    def __init__(self, message: str):
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
        content = str(self.error) if self.error else '' if self.output is None else str(self.output) if not isinstance(self.output, BaseModel) else f'{self.name} created'
        return {'role': 'tool', 'tool_call_id': self.tool_call_id, 'name': self.name, 'content': content}

def process_tool_call(tool_call: ChatCompletionMessageToolCall, functions_or_models: list[Union[Callable, BaseModel]], prefix_class=None, fix_json_args=True, case_insensitive=False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors = []
    error = None
    stack_trace = None
    prefix = None
    output = None

    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            tool_args = json.loads(args.replace(', }', '}').replace(',}', '}'))
        else:
            return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=e, stack_trace=traceback.format_exc())

    if prefix_class:
        prefix = _extract_prefix_unpacked(tool_args, prefix_class)
        tool_name = tool_name[len(prefix_class.__name__ + '_and_'):]

    tool = next((f for f in functions_or_models if get_name(f, case_insensitive=case_insensitive) == tool_name), None)

    if tool:
        try:
            output, new_soft_errors = _process_unpacked(tool, tool_args, fix_json_args=fix_json_args)
            soft_errors.extend(new_soft_errors)
        except Exception as e:
            error = e
            stack_trace = traceback.format_exc()
    else:
        error = NoMatchingTool(f'Function {tool_name} not found')

    return ToolResult(tool_call_id=tool_call.id, name=tool_name, arguments=tool_args, output=output, error=error, stack_trace=stack_trace, soft_errors=soft_errors, prefix=prefix, tool=tool)

def _process_unpacked(function, tool_args={}, fix_json_args=True):
    model = parameters_basemodel_from_function(function)
    soft_errors = []

    if fix_json_args:
        for field, field_info in model.model_fields.items():
            if _is_list_type(field_info.annotation):
                if isinstance(tool_args.get(field), str):
                    try:
                        tool_args[field] = json.loads(tool_args[field])
                    except json.JSONDecodeError:
                        tool_args[field] = [item.strip() for item in tool_args[field].split(',')]
                        soft_errors.append(f'Fixed string to list conversion for field {field}')

    model_instance = model(**tool_args)
    args = {field: getattr(model_instance, field) for field in model.model_fields}
    return function(**args), soft_errors

def _is_list_type(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is list or (origin in (Union, Optional) and any(_is_list_type(arg) for arg in args))

def _extract_prefix_unpacked(tool_args, prefix_class):
    prefix_args = {key: tool_args.pop(key) for key in list(tool_args.keys()) if key in prefix_class.__annotations__}
    return prefix_class(**prefix_args)

def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(message: ChatCompletionMessage, functions: list[Union[Callable, LLMFunction]], prefix_class=None, fix_json_args=True, case_insensitive=False, executor: Optional[ThreadPoolExecutor]=None) -> list[ToolResult]:
    tool_calls = message.tool_calls or []
    args_list = [(tool_call, functions, prefix_class, fix_json_args, case_insensitive) for tool_call in tool_calls]
    results = list(map(lambda args: process_tool_call(*args), args_list)) if not executor else list(executor.map(lambda args: process_tool_call(*args), args_list))
    return results

def process_one_tool_call(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], index: int = 0, prefix_class=None, fix_json_args=True, case_insensitive=False) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response)
    return process_tool_call(tool_calls[index], functions, prefix_class, fix_json_args, case_insensitive) if tool_calls and index < len(tool_calls) else None

def _get_tool_calls(response: ChatCompletion) -> list[ChatCompletionMessageToolCall]:
    return response.choices[0].message.tool_calls or []