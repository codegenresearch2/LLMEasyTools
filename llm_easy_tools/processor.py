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
    """Exception raised when no matching tool is found."""
    def __init__(self, message: str):
        super().__init__(message)

@dataclass
class ToolResult:
    """Represents the result of a tool invocation."""
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
        """Converts the ToolResult into a dictionary suitable for returning to a chat interface."""
        content = f"{self.error}" if self.error else '' if self.output is None else str(self.output)
        return {"role": "tool", "tool_call_id": self.tool_call_id, "name": self.name, "content": content}

def process_tool_call(tool_call: ChatCompletionMessageToolCall, functions_or_models: list[Union[Callable, BaseModel]], fix_json_args: bool = True, case_insensitive: bool = False) -> ToolResult:
    """Processes a single tool call and returns the result."""
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors = []
    error = None
    stack_trace = None
    output = None
    tool_args = {}  # Initialize tool_args to an empty dictionary

    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            fixed_args = args.replace(', }', '}').replace(',}', '}')
            try:
                tool_args = json.loads(fixed_args)
            except json.decoder.JSONDecodeError:
                error = e
                stack_trace = traceback.format_exc()

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

def _process_unpacked(function: Union[Callable, BaseModel], tool_args: dict[str, Any] = {}, fix_json_args: bool = True) -> tuple[Any, list[Exception]]:
    """Processes the arguments for a function or model."""
    if isinstance(function, LLMFunction):
        function = function.func
    model = parameters_basemodel_from_function(function)
    soft_errors = []

    if fix_json_args:
        for field, field_info in model.model_fields.items():
            if isinstance(tool_args.get(field), str):
                try:
                    tool_args[field] = json.loads(tool_args[field])
                except json.JSONDecodeError:
                    soft_errors.append(f"Failed to parse JSON for field {field}")
                else:
                    if isinstance(tool_args[field], str):
                        tool_args[field] = split_string_to_list(tool_args[field])

    model_instance = model(**tool_args)
    args = {field: getattr(model_instance, field) for field in model.model_fields}
    return function(**args), soft_errors

def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num: int = 0, **kwargs) -> list[ToolResult]:
    """Processes a ChatCompletion response and returns a list of ToolResult objects."""
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(message: ChatCompletionMessage, functions: list[Union[Callable, LLMFunction]], fix_json_args: bool = True, case_insensitive: bool = False, executor: ThreadPoolExecutor = None) -> list[ToolResult]:
    """Processes a ChatCompletionMessage and returns a list of ToolResult objects."""
    tool_calls = message.tool_calls if message.tool_calls else []
    args_list = [(tool_call, functions, fix_json_args, case_insensitive) for tool_call in tool_calls]
    results = list(executor.map(lambda args: process_tool_call(*args), args_list)) if executor else [process_tool_call(*args) for args in args_list]
    return results

def process_one_tool_call(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], index: int = 0, fix_json_args: bool = True, case_insensitive: bool = False) -> Optional[ToolResult]:
    """Processes a single tool call from a ChatCompletion response at the specified index."""
    tool_calls = _get_tool_calls(response)
    return process_tool_call(tool_calls[index], functions, fix_json_args, case_insensitive) if tool_calls and index < len(tool_calls) else None

def _get_tool_calls(response: ChatCompletion) -> list[ChatCompletionMessageToolCall]:
    """Retrieves the tool calls from a ChatCompletion response."""
    message = response.choices[0].message
    return message.tool_calls if message.tool_calls else []

def _is_list_type(annotation):
    """Checks if a type is a list type."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        return True
    elif origin is Union or origin is Optional:
        return any(_is_list_type(arg) for arg in args)
    return False

def split_string_to_list(s: str) -> list[str]:
    """Splits a string into a list of strings."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

# Examples
if __name__ == "__main__":
    from llm_easy_tools.types import mk_chat_with_tool_call
    from pprint import pprint

    def original_function():
        return 'Result of function_decorated'

    function_decorated = LLMFunction(original_function, name="altered_name")

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            """simple method does something"""
            return 'Result of simple_method'

    example_object = ExampleClass()

    class User(BaseModel):
        name: str
        email: str

    # Process a response with a tool call to a decorated function
    pprint(process_response(mk_chat_with_tool_call('altered_name', {}), [function_decorated]))

    # Process a tool call to a decorated function
    call_to_altered_name = mk_chat_with_tool_call('altered_name', {}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_altered_name, [function_decorated]))

    # Process a tool call to a method of an object
    call_to_simple_method = mk_chat_with_tool_call('simple_method', {"count": 1, "size": 2.2}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_simple_method, [example_object.simple_method]))

    # Process a tool call to a Pydantic model
    call_to_model = mk_chat_with_tool_call('User', {"name": 'John', "email": 'john@example.com'}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_model, [User]))