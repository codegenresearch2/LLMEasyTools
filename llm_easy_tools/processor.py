import json
import inspect
import traceback

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Union, Optional, Any
from pprint import pprint

from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field

from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionMessage, ChatCompletionMessageToolCall, Function

class NoMatchingTool(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

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
        if self.error is not None:
            content = f"{self.error}"
        elif self.output is None:
            content = ""
        elif isinstance(self.output, BaseModel):
            content = f"{self.name} created"
        else:
            content = str(self.output)
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": content,
        }

# Define the get_origin function
# This function is necessary for inspecting type annotations
def get_origin(annotation):
    return getattr(annotation, '__origin__', None)

# Define the _is_list_type function
# This function checks if the annotation is a list or a union of lists
def _is_list_type(annotation):
    origin = get_origin(annotation)
    args = getattr(annotation, '__args__', ())
    return origin is list or (origin is Union and any(_is_list_type(arg) for arg in args))

# Define the _process_unpacked function
# This function processes the arguments and returns the function output
def _process_unpacked(function, tool_args={}, fix_json_args=True):
    if isinstance(function, LLMFunction):
        function = function.func
    model = parameters_basemodel_from_function(function)
    soft_errors = []
    if fix_json_args:
        for field, field_info in model.model_fields.items():
            field_annotation = field_info.annotation
            if _is_list_type(field_annotation):
                if field in tool_args and isinstance(tool_args[field], str):
                    tool_args[field] = split_string_to_list(tool_args[field])
                    soft_errors.append(f"Fixed JSON decode error for field {field}")
    model_instance = model(**tool_args)
    args = {}
    for field, _ in model.model_fields.items():
        args[field] = getattr(model_instance, field)
    return function(**args), soft_errors

# Define the process_tool_call function
# This function processes a tool call
def process_tool_call(tool_call, functions_or_models, prefix_class=None, fix_json_args=True, case_insensitive=False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors: list[Exception] = []
    error = None
    stack_trace = None
    prefix = None
    output = None
    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            args = args.replace(', }', '}').replace(',}', '}')
            tool_args = json.loads(args)
        else:
            stack_trace = traceback.format_exc()
            return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=e, stack_trace=stack_trace)

    if prefix_class is not None:
        try:
            prefix = _extract_prefix_unpacked(tool_args, prefix_class)
        except ValidationError as e:
            soft_errors.append(e)
        prefix_name = prefix_class.__name__
        if case_insensitive:
            prefix_name = prefix_name.lower()
        if not tool_name.startswith(prefix_name):
            soft_errors.append(NoMatchingTool(f"Trying to decode function call with a name '{tool_name}' not matching prefix '{prefix_name}'"))
        else:
            tool_name = tool_name[len(prefix_name + '_and_'):]

    tool = None

    for f in functions_or_models:
        if get_name(f, case_insensitive=case_insensitive) == tool_name:
            tool = f
            try:
                output, new_soft_errors = _process_unpacked(f, tool_args, fix_json_args=fix_json_args)
                soft_errors.extend(new_soft_errors)
            except Exception as e:
                error = e
                stack_trace = traceback.format_exc()
            break
    else:
        error = NoMatchingTool(f"Function {tool_name} not found")
    result = ToolResult(
        tool_call_id=tool_call.id,
        name=tool_name,
        arguments=tool_args,
        output=output,
        error=error,
        stack_trace=stack_trace,
        soft_errors=soft_errors,
        prefix=prefix,
        tool=tool,
    )
    return result

# Define the split_string_to_list function
def split_string_to_list(s: str) -> list[str]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

# Define the process_response function
# This function processes a ChatCompletion response
def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

# Define the process_message function
# This function processes a ChatCompletionMessage
def process_message(
    message: ChatCompletionMessage,
    functions: list[Union[Callable, LLMFunction]],
    prefix_class=None,
    fix_json_args=True,
    case_insensitive=False,
    executor: Union[ThreadPoolExecutor, ProcessPoolExecutor, None]=None
    ) -> list[ToolResult]:
    results = []
    if hasattr(message, 'function_call') and (function_call := message.function_call):
        tool_calls = [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = message.tool_calls
    else:
        tool_calls = []
    if not tool_calls:
        return []
    args_list = [(tool_call, functions, prefix_class, fix_json_args, case_insensitive) for tool_call in tool_calls]

    if executor:
        results = list(executor.map(lambda args: process_tool_call(*args), args_list))
    else:
        results = list(map(lambda args: process_tool_call(*args), args_list))
    return results

# Define the process_one_tool_call function
# This function processes a single tool call from a ChatCompletion response at the specified index
def process_one_tool_call(
        response: ChatCompletion,
        functions: list[Union[Callable, LLMFunction]],
        index: int = 0,
        prefix_class=None,
        fix_json_args=True,
        case_insensitive=False
    ) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response)
    if not tool_calls or index >= len(tool_calls):
        return None

    return process_tool_call(tool_calls[index], functions, prefix_class, fix_json_args, case_insensitive)

# Helper function to get tool calls from the response
def _get_tool_calls(response: ChatCompletion) -> list[ChatCompletionMessageToolCall]:
    if hasattr(response.choices[0].message, 'function_call') and (function_call := response.choices[0].message.function_call):
        return [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        return response.choices[0].message.tool_calls
    return []

#######################################
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
    call_to_altered_name = mk_chat_with_tool_call('altered_name', {}).choices[0].message.tool_calls[0]
    pprint(call_to_altered_name)
    pprint(process_tool_call(call_to_altered_name, [function_decorated]))

    call_to_simple_method = mk_chat_with_tool_call('simple_method', {"count": 1, "size": 2.2}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_simple_method, [example_object.simple_method]))

    call_to_model = mk_chat_with_tool_call('User', {"name": 'John', "email": 'john@example.com'}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_model, [User]))
