import json
import inspect
import traceback

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union, Optional, Any, get_origin, get_args
from pprint import pprint

from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field

from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionMessage

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
        content = str(self.error) if self.error else '' if self.output is None else str(self.output)
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": content,
        }

def process_tool_call(tool_call: ChatCompletionMessageToolCall, functions_or_models: list[Union[Callable, BaseModel]], fix_json_args=True, case_insensitive=False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors, error, stack_trace, output = [], None, None, None
    tool_args = {}

    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            args = args.replace(', }', '}').replace(',}', '}')
            tool_args = json.loads(args)
        else:
            error = e
            stack_trace = traceback.format_exc()

    tool = next((f for f in functions_or_models if get_name(f, case_insensitive=case_insensitive) == tool_name.lower()), None)
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

    for field, field_info in model.model_fields.items():
        field_annotation = field_info.annotation
        if _is_list_type(field_annotation):
            if field in tool_args and isinstance(tool_args[field], str):
                tool_args[field] = split_string_to_list(tool_args[field])
                soft_errors.append(f"Fixed JSON decode error for field {field}")

    model_instance = model(**tool_args)
    args = {field: getattr(model_instance, field) for field in model.model_fields}
    return function(**args), soft_errors

def _is_list_type(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        return True
    elif origin is Union or origin is Optional:
        return any(_is_list_type(arg) for arg in args)
    return False

def split_string_to_list(s: str) -> list[str]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(message: ChatCompletionMessage, functions: list[Union[Callable, LLMFunction]], **kwargs) -> list[ToolResult]:
    tool_calls = _get_tool_calls(message)
    args_list = [(tool_call, functions, **kwargs) for tool_call in tool_calls]

    if 'executor' in kwargs and kwargs['executor']:
        results = list(kwargs['executor'].map(lambda args: process_tool_call(*args), args_list))
    else:
        results = list(map(lambda args: process_tool_call(*args), args_list))

    return results

def process_one_tool_call(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], index: int = 0, **kwargs) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response.choices[0].message)
    return process_tool_call(tool_calls[index], functions, **kwargs) if tool_calls and index < len(tool_calls) else None

def _get_tool_calls(message: ChatCompletionMessage) -> list[ChatCompletionMessageToolCall]:
    if hasattr(message, 'function_call') and (function_call := message.function_call):
        return [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(message, 'tool_calls') and message.tool_calls:
        return message.tool_calls
    return []

# Examples
if __name__ == "__main__":
    from llm_easy_tools.types import mk_chat_with_tool_call

    def original_function():
        return 'Result of function_decorated'

    function_decorated = LLMFunction(original_function, name="altered_name")

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            return 'Result of simple_method'

    example_object = ExampleClass()

    class User(BaseModel):
        name: str
        email: str

    pprint(process_response(mk_chat_with_tool_call('altered_name', {}), [function_decorated]))
    call_to_simple_method = mk_chat_with_tool_call('simple_method', {"count": 1, "size": 2.2}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_simple_method, [example_object.simple_method]))
    call_to_model = mk_chat_with_tool_call('User', {"name": 'John', "email": 'john@example.com'}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_model, [User]))