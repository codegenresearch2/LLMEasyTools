from typing import Callable, Union, Optional, Any, get_origin, get_args
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import traceback
from pydantic import BaseModel, ValidationError
from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionMessage

class NoMatchingTool(Exception):
    """Exception raised when no matching tool is found."""
    def __init__(self, message: str):
        super().__init__(f'No matching tool found: {message}')

@dataclass
class ToolResult:
    """Represents the result of a tool invocation.

    Attributes:
        tool_call_id (str): The ID of the tool call.
        name (str): The name of the tool.
        output (Optional[Union[str, BaseModel]]): The output of the tool.
        arguments (Optional[dict[str, Any]]): The arguments passed to the tool.
        error (Optional[Exception]): The error raised during tool execution, if any.
        stack_trace (Optional[str]): The stack trace of the error, if any.
        soft_errors (list[Exception]): A list of soft errors encountered during tool execution.
        prefix (Optional[BaseModel]): The prefix used for the tool, if any.
        tool (Optional[Union[Callable, BaseModel]]): The tool that was executed.
    """
    tool_call_id: str
    name: str
    output: Optional[Union[str, BaseModel]] = None
    arguments: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None
    stack_trace: Optional[str] = None
    soft_errors: list[Exception] = field(default_factory=list)
    prefix: Optional[BaseModel] = None
    tool: Optional[Union[Callable, BaseModel]] = None

    def to_message(self) -> dict[str, str]:
        """Converts the ToolResult into a dictionary suitable for returning to a chat interface.

        Returns:
            dict[str, str]: A dictionary representing the tool result.
        """
        content = str(self.error) if self.error else '' if self.output is None else str(self.output) if not isinstance(self.output, BaseModel) else f'{self.name} created'
        return {'role': 'tool', 'tool_call_id': self.tool_call_id, 'name': self.name, 'content': content}

def process_tool_call(tool_call: ChatCompletionMessageToolCall, functions_or_models: list[Union[Callable, BaseModel]], prefix_class=None, fix_json_args=True, case_insensitive=False) -> ToolResult:
    """Processes a single tool call from a ChatCompletion response.

    Args:
        tool_call (ChatCompletionMessageToolCall): The tool call to process.
        functions_or_models (list[Union[Callable, BaseModel]]): A list of functions or models that can be called.
        prefix_class (Optional[BaseModel]): The prefix class for the tool, if any.
        fix_json_args (bool): Whether to fix JSON arguments if they are invalid.
        case_insensitive (bool): Whether to match tool names case-insensitively.

    Returns:
        ToolResult: The result of the tool call.
    """
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
            tool_args = json.loads(args.replace(', }', '}').replace(',}', '}'))
        else:
            error = e
            stack_trace = traceback.format_exc()

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

def split_string_to_list(s: str) -> list[str]:
    """Converts a string representation of a list into a list.

    Args:
        s (str): The string to convert.

    Returns:
        list[str]: The converted list.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

def _process_unpacked(function: Union[Callable, BaseModel], tool_args: dict[str, Any], fix_json_args: bool) -> tuple[Any, list[Exception]]:
    """Processes the arguments for a function or model.

    Args:
        function (Union[Callable, BaseModel]): The function or model to process.
        tool_args (dict[str, Any]): The arguments to process.
        fix_json_args (bool): Whether to fix JSON arguments if they are invalid.

    Returns:
        tuple[Any, list[Exception]]: The output of the function and a list of soft errors.
    """
    model = parameters_basemodel_from_function(function)
    soft_errors = []

    if fix_json_args:
        for field, field_info in model.model_fields.items():
            field_annotation = field_info.annotation
            if _is_list_type(field_annotation):
                if field in tool_args and isinstance(tool_args[field], str):
                    tool_args[field] = split_string_to_list(tool_args[field])
                    soft_errors.append(f'Fixed JSON decode error for field {field}')

    model_instance = model(**tool_args)
    args = {field: getattr(model_instance, field) for field in model.model_fields}
    return function(**args), soft_errors

def _is_list_type(annotation: Any) -> bool:
    """Checks if a type annotation represents a list.

    Args:
        annotation (Any): The type annotation to check.

    Returns:
        bool: True if the annotation represents a list, False otherwise.
    """
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is list or (origin in (Union, Optional) and any(_is_list_type(arg) for arg in args))

def _extract_prefix_unpacked(tool_args: dict[str, Any], prefix_class: BaseModel) -> BaseModel:
    """Extracts the prefix arguments from the tool arguments.

    Args:
        tool_args (dict[str, Any]): The tool arguments.
        prefix_class (BaseModel): The prefix class.

    Returns:
        BaseModel: The prefix instance.
    """
    prefix_args = {key: tool_args.pop(key) for key in list(tool_args.keys()) if key in prefix_class.__annotations__}
    return prefix_class(**prefix_args)

def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    """Processes a ChatCompletion response, executing contained tool calls.

    Args:
        response (ChatCompletion): The response to process.
        functions (list[Union[Callable, LLMFunction]]): A list of functions or models that can be called.
        choice_num (int): The index of the choice to process.
        **kwargs: Additional keyword arguments to pass to `process_message`.

    Returns:
        list[ToolResult]: A list of tool results.
    """
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(message: ChatCompletionMessage, functions: list[Union[Callable, LLMFunction]], prefix_class=None, fix_json_args=True, case_insensitive=False, executor: Optional[ThreadPoolExecutor]=None) -> list[ToolResult]:
    """Processes a ChatCompletionMessage, executing contained tool calls.

    Args:
        message (ChatCompletionMessage): The message to process.
        functions (list[Union[Callable, LLMFunction]]): A list of functions or models that can be called.
        prefix_class (Optional[BaseModel]): The prefix class for the tool, if any.
        fix_json_args (bool): Whether to fix JSON arguments if they are invalid.
        case_insensitive (bool): Whether to match tool names case-insensitively.
        executor (Optional[ThreadPoolExecutor]): The executor to use for parallel processing, if any.

    Returns:
        list[ToolResult]: A list of tool results.
    """
    tool_calls = message.tool_calls or []
    args_list = [(tool_call, functions, prefix_class, fix_json_args, case_insensitive) for tool_call in tool_calls]
    results = list(map(lambda args: process_tool_call(*args), args_list)) if not executor else list(executor.map(lambda args: process_tool_call(*args), args_list))
    return results

def process_one_tool_call(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], index: int = 0, prefix_class=None, fix_json_args=True, case_insensitive=False) -> Optional[ToolResult]:
    """Processes a single tool call from a ChatCompletion response at the specified index.

    Args:
        response (ChatCompletion): The response to process.
        functions (list[Union[Callable, LLMFunction]]): A list of functions or models that can be called.
        index (int): The index of the tool call to process.
        prefix_class (Optional[BaseModel]): The prefix class for the tool, if any.
        fix_json_args (bool): Whether to fix JSON arguments if they are invalid.
        case_insensitive (bool): Whether to match tool names case-insensitively.

    Returns:
        Optional[ToolResult]: The tool result, or None if the index is out of range.
    """
    tool_calls = _get_tool_calls(response)
    return process_tool_call(tool_calls[index], functions, prefix_class, fix_json_args, case_insensitive) if tool_calls and index < len(tool_calls) else None

def _get_tool_calls(response: ChatCompletion) -> list[ChatCompletionMessageToolCall]:
    """Retrieves the tool calls from a ChatCompletion response.

    Args:
        response (ChatCompletion): The response to retrieve tool calls from.

    Returns:
        list[ChatCompletionMessageToolCall]: A list of tool calls.
    """
    return response.choices[0].message.tool_calls or []

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

    pprint(process_response(mk_chat_with_tool_call('altered_name', {}), [function_decorated]))
    call_to_altered_name = mk_chat_with_tool_call('altered_name', {}).choices[0].message.tool_calls[0]
    pprint(call_to_altered_name)
    pprint(process_tool_call(call_to_altered_name, [function_decorated]))

    call_to_simple_method = mk_chat_with_tool_call('simple_method', {"count": 1, "size": 2.2}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_simple_method, [example_object.simple_method]))

    call_to_model = mk_chat_with_tool_call('User', {"name": 'John', "email": 'john@example.com'}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_model, [User]))