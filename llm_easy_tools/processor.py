import json
import inspect
import traceback

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable
from pprint import pprint
from typing import Optional, List, Union, Any, get_origin, get_args

from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call   import ChatCompletionMessageToolCall, Function

from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction

class NoMatchingTool(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

from typing import Protocol, runtime_checkable

@runtime_checkable
class ToContent(Protocol):
    def to_content(self) -> dict[str, str]:
        ...

@dataclass
class ToolResult:
    """
    Represents the result of a tool invocation within the ToolBox framework.

    Attributes:
        tool_call_id (str): A unique identifier for the tool call.
        name (str): The name of the tool that was called.
        output (Optional[Union[str, BaseModel]]): The output generated by the tool call, if any.
        error (Optional[Exception]): An error message if the tool call failed.
        soft_errors (List[Exception]): A list of non-critical error messages encountered during the tool call.
        prefix (Optional[BaseModel]): The Pydantic model instance used as a prefix in the tool call, if applicable.

    Methods:
        to_message(): Converts the ToolResult into a dictionary suitable for returning to a chat interface.
    """
    tool_call_id: str
    name: str
    output: Optional[Any] = None
    arguments: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None
    stack_trace: Optional[str] = None
    soft_errors: List[Exception] = field(default_factory=list)
    prefix: Optional[BaseModel] = None
    tool: Optional[Callable|BaseModel] = None

    def to_message(self) -> dict[str, str]:
        if self.error is not None:
            content = f"{self.error}"
        elif self.output is None:
            content = ''
        elif isinstance(self.output, ToContent):
            content = self.output.to_content()
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

def split_string_to_list(s: str) -> list[str]:
    return [item.strip() for item in s.split(',')]

def _process_unpacked(function, tool_args={}, fix_json_args=True):
    if isinstance(function, LLMFunction):
        function = function.func
    model = parameters_basemodel_from_function(function)
    soft_errors = []
    if fix_json_args:
        for field, field_info in model.model_fields.items():
            field_annotation = field_info.annotation
            origin = get_origin(field_annotation)
            if origin is list:
                if field in tool_args and isinstance(tool_args[field], str):
                    # this happens in Claude from Anthropic 
                    tool_args[field] = split_string_to_list(tool_args[field])
                    soft_errors.append(f"Fixed JSON decode error for field {field}")

    model_instance = model(**tool_args)
    args = {}
    for field, _ in model.model_fields.items():
        args[field] = getattr(model_instance, field)
    return function(**args), soft_errors

def _extract_prefix_unpacked(tool_args, prefix_class):
    # modifies tool_args
    prefix_args = {}
    for key in list(tool_args.keys()):  # copy keys to list because we modify the dict while iterating over it
        if key in prefix_class.__annotations__:
            prefix_args[key] = tool_args.pop(key)
    prefix = prefix_class(**prefix_args)
    return(prefix)

def process_response( response: ChatCompletion, functions: list[Callable | LLMFunction], choice_num=0, **kwargs) -> list[ToolResult]:
    """
    Processes a ChatCompletion response, executing contained tool calls.
    For each tool call matches a function from the 'functions' list by name.
    The result of the tool call is returned as a ToolResult object.
    If the tool call raises an exception, that exception is saved in the 'error' field in the result.

    Args:
        response (ChatCompletion): The response object containing tool calls.
        functions (List[Callable]): A list of functions or pydantic models to call.
        choice_num (int, optional): The index of the choice to process from the response. Defaults to 0.

    Returns:
        list[ToolResult]: A list of ToolResult objects, each representing the outcome of a processed tool call.
    """
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(
    message: ChatCompletionMessage,
    functions: list[Callable | LLMFunction],
    prefix_class=None,
    fix_json_args=True,
    case_insensitive=False,
    executor: ThreadPoolExecutor|ProcessPoolExecutor|None=None
    ) -> list[ToolResult]:
    results = []
    if hasattr(message, 'function_call') and (function_call:=message.function_call):
        # this is obsolete in openai - but maybe it is used by other llms?
        tool_calls = [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = message.tool_calls
    else:
        tool_calls = []
        # Prepare the arguments for each tool call
    if not tool_calls:
        return []
    args_list = [(tool_call, functions, prefix_class, fix_json_args, case_insensitive) for tool_call in tool_calls]

    if executor:
        results = list(executor.map(lambda args: process_tool_call(*args), args_list))
    else:
        results = list(map(lambda args: process_tool_call(*args), args_list)) 
    return results

def process_one_tool_call(
        response: ChatCompletion,
        functions: list[Callable | LLMFunction],
        index: int = 0,
        prefix_class=None,
        fix_json_args=True,
        case_insensitive=False
    ) -> Optional[ToolResult]:
    """
    Processes a single tool call from a ChatCompletion response at the specified index.
    """
    tool_calls = _get_tool_calls(response)
    if not tool_calls or index >= len(tool_calls):
        return None

    return process_tool_call(tool_calls[index], functions, prefix_class, fix_json_args, case_insensitive)

# Helper function to get tool calls from the response
def _get_tool_calls(response: ChatCompletion) -> List[ChatCompletionMessageToolCall]:
    if hasattr(response.choices[0].message, 'function_call') and (function_call := response.choices[0].message.function_call):
        return [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        return response.choices[0].message.tool_calls
    return []

#######################################
#
# Examples


if __name__ == "__main__":

    def original_function():
        return 'Result of function_decorated'

    function_decorated = LLMFunction(original_function, schema_name="altered_name")

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            """simple method does something"""
            return 'Result of simple_method'

    example_object = ExampleClass()

    class User(BaseModel):
        name: str
        email: str

    def mk_chat_with_tool_call(name, args):
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": 'A',
                    "type": 'function',
                    "function": {
                        "arguments": json.dumps(args),
                        "name": name
                    }
                }
            ]
        )
        chat_completion = ChatCompletion(
            id='A',
            created=0,
            model='A',
            choices=[{'finish_reason': 'stop', 'index': 0, 'message': message}],
            object='chat.completion'
        )
        return chat_completion


    pprint(process_response(mk_chat_with_tool_call('altered_name', {}), [function_decorated]))
    call_to_altered_name = mk_chat_with_tool_call('altered_name', {}).choices[0].message.tool_calls[0]
    pprint(call_to_altered_name)
    pprint(process_tool_call(call_to_altered_name, [function_decorated]))

    call_to_simple_method = mk_chat_with_tool_call('simple_method', {"count": 1, "size": 2.2}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_simple_method, [example_object.simple_method]))

    call_to_model = mk_chat_with_tool_call('User', {"name": 'John', "email": 'john@example.com'}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_model, [User]))

