import json
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Union, Optional, Any
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field
from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import Function, ChatCompletionMessageToolCall, ChatCompletionMessage

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
            content = ''
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

def process_tool_call(tool_call, functions_or_models, fix_json_args=True, case_insensitive=False) -> ToolResult:
    """
    Processes a tool call from a ChatCompletion response.

    Args:
        tool_call (ChatCompletionMessageToolCall): The tool call object.
        functions_or_models (list): A list of functions or Pydantic models to match against the tool call.
        fix_json_args (bool): Whether to attempt to fix JSON decoding errors in arguments.
        case_insensitive (bool): Whether to perform a case-insensitive match for tool names.

    Returns:
        ToolResult: A ToolResult object containing the result of the tool call.
    """
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors: list[Exception] = []
    error = None
    stack_trace = None
    prefix = None
    output = None
    tool = None

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
        if tool is None:
            error = NoMatchingTool(f"Function {tool_name} not found")

    result = ToolResult(
        tool_call_id=tool_call.id, 
        name=tool_name,
        arguments=tool_args,
        output=output, 
        error=error,
        stack_trace=stack_trace,
        soft_errors=soft_errors,
        tool=tool,
    )
    return result

def split_string_to_list(s: str) -> list[str]:
    """
    Converts a string representation of a list into an actual list.

    Args:
        s (str): The string to convert.

    Returns:
        list[str]: The list representation of the string.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

def _process_unpacked(function, tool_args={}, fix_json_args=True):
    """
    Helper function to process unpacked function calls.

    Args:
        function (Callable): The function to call.
        tool_args (dict): The arguments for the function.
        fix_json_args (bool): Whether to attempt to fix JSON decoding errors in arguments.

    Returns:
        tuple: A tuple containing the result of the function call and any soft errors.
    """
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

def _is_list_type(annotation):
    """
    Checks if the given annotation is a list type.

    Args:
        annotation: The annotation to check.

    Returns:
        bool: True if the annotation is a list type, False otherwise.
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        return True
    elif origin is Union or origin is Optional:
        return any(_is_list_type(arg) for arg in args)
    return False

def process_response(response: 'ChatCompletion', functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    """
    Processes a ChatCompletion response, executing contained tool calls.

    Args:
        response (ChatCompletion): The response object containing tool calls.
        functions (list[Callable]): A list of functions or Pydantic models to call.
        choice_num (int, optional): The index of the choice to process from the response. Defaults to 0.

    Returns:
        list[ToolResult]: A list of ToolResult objects, each representing the outcome of a processed tool call.
    """
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(
    message: ChatCompletionMessage,
    functions: list[Union[Callable, LLMFunction]],
    fix_json_args=True,
    case_insensitive=False,
    executor: Union[ThreadPoolExecutor, ProcessPoolExecutor, None]=None
    ) -> list[ToolResult]:
    """
    Processes a ChatCompletionMessage, executing contained tool calls.

    Args:
        message (ChatCompletionMessage): The message object containing tool calls.
        functions (list[Callable]): A list of functions or Pydantic models to call.
        fix_json_args (bool): Whether to attempt to fix JSON decoding errors in arguments.
        case_insensitive (bool): Whether to perform a case-insensitive match for tool names.
        executor (ThreadPoolExecutor, optional): An executor to run the tool calls in parallel.

    Returns:
        list[ToolResult]: A list of ToolResult objects, each representing the outcome of a processed tool call.
    """
    results = []
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = message.tool_calls
    else:
        tool_calls = []

    args_list = [(tool_call, functions, fix_json_args, case_insensitive) for tool_call in tool_calls]

    if executor:
        results = list(executor.map(lambda args: process_tool_call(*args), args_list))
    else:
        results = list(map(lambda args: process_tool_call(*args), args_list)) 
    return results

def process_one_tool_call(
        response: 'ChatCompletion',
        functions: list[Union[Callable, LLMFunction]],
        index: int = 0,
        fix_json_args=True,
        case_insensitive=False
    ) -> Optional[ToolResult]:
    """
    Processes a single tool call from a ChatCompletion response at the specified index.

    Args:
        response (ChatCompletion): The response object containing tool calls.
        functions (list[Callable]): A list of functions or Pydantic models to call.
        index (int, optional): The index of the tool call to process. Defaults to 0.
        fix_json_args (bool): Whether to attempt to fix JSON decoding errors in arguments.
        case_insensitive (bool): Whether to perform a case-insensitive match for tool names.

    Returns:
        Optional[ToolResult]: A ToolResult object representing the outcome of the processed tool call, or None if the index is out of range.
    """
    tool_calls = _get_tool_calls(response)
    if not tool_calls or index >= len(tool_calls):
        return None

    return process_tool_call(tool_calls[index], functions, fix_json_args, case_insensitive)

def _get_tool_calls(response: 'ChatCompletion') -> list[ChatCompletionMessageToolCall]:
    """
    Helper function to get tool calls from a ChatCompletion response.

    Args:
        response (ChatCompletion): The response object containing tool calls.

    Returns:
        list[ChatCompletionMessageToolCall]: A list of tool calls.
    """
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        return response.choices[0].message.tool_calls
    return []