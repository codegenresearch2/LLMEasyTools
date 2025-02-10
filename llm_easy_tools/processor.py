from typing import List, Optional, Union, Callable, Any
from pydantic import BaseModel
from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import Function, ChatCompletionMessageToolCall, ChatCompletionMessage, ChatCompletion
import json
from dataclasses import dataclass

@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    output: Optional[Union[str, BaseModel]] = None
    error: Optional[Exception] = None
    soft_errors: List[Exception] = None
    prefix: Optional[BaseModel] = None
    tool: Optional[Union[Callable, BaseModel]] = None

    def to_message(self) -> dict:
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

def process_tool_call(tool_call, functions_or_models, fix_json_args: bool = True, case_insensitive: bool = False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors: List[Exception] = []
    error = None
    output = None

    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            args = args.replace(', }', '}').replace(',}', '}')
            tool_args = json.loads(args)
        else:
            return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=e, soft_errors=soft_errors)

    for f in functions_or_models:
        if get_name(f, case_insensitive=case_insensitive) == tool_name:
            try:
                output = f(**tool_args)
            except Exception as e:
                error = e
            break
    else:
        error = NoMatchingTool(f"Function {tool_name} not found")

    return ToolResult(tool_call_id=tool_call.id, name=tool_name, output=output, error=error, soft_errors=soft_errors)

def split_string_to_list(s: str) -> List[str]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

def _process_unpacked(function, tool_args: dict = {}, fix_json_args: bool = True):
    if isinstance(function, LLMFunction):
        function = function.func
    model = parameters_basemodel_from_function(function)
    soft_errors = []
    if fix_json_args:
        for field, field_info in model.model_fields.items():
            field_annotation = field_info.annotation
            if isinstance(field_annotation, list):
                if field in tool_args and isinstance(tool_args[field], str):
                    tool_args[field] = split_string_to_list(tool_args[field])
                    soft_errors.append(f"Fixed JSON decode error for field {field}")

    model_instance = model(**tool_args)
    args = {}
    for field, _ in model.model_fields.items():
        args[field] = getattr(model_instance, field)
    return function(**args), soft_errors

def process_response(response, functions: List[Union[Callable, LLMFunction]], choice_num: int = 0, **kwargs) -> List[ToolResult]:
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(message: ChatCompletionMessage, functions: List[Union[Callable, LLMFunction]], fix_json_args: bool = True, case_insensitive: bool = False, executor=None) -> List[ToolResult]:
    results = []
    if hasattr(message, 'function_call') and (function_call := message.function_call):
        tool_calls = [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = message.tool_calls
    else:
        tool_calls = []

    if not tool_calls:
        return []
    args_list = [(tool_call, functions, fix_json_args, case_insensitive) for tool_call in tool_calls]

    if executor:
        results = list(executor.map(lambda args: process_tool_call(*args), args_list))
    else:
        results = list(map(lambda args: process_tool_call(*args), args_list))
    return results

def process_one_tool_call(response: ChatCompletion, functions: List[Union[Callable, LLMFunction]], index: int = 0, fix_json_args: bool = True, case_insensitive: bool = False) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response)
    if not tool_calls or index >= len(tool_calls):
        return None

    return process_tool_call(tool_calls[index], functions, fix_json_args, case_insensitive)

def _get_tool_calls(response: ChatCompletion) -> List[ChatCompletionMessageToolCall]:
    if hasattr(response.choices[0].message, 'function_call') and (function_call := response.choices[0].message.function_call):
        return [ChatCompletionMessageToolCall(id='A', function=Function(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        return response.choices[0].message.tool_calls
    return []

if __name__ == "__main__":
    from llm_easy_tools.types import mk_chat_with_tool_call
    from pprint import pprint

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