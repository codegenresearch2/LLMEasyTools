import json
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Union, Optional, Any
from pprint import pprint
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

# Ensure ChatCompletion is defined or imported correctly

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

# Helper functions...

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

    pprint(process_response(mk_chat_with_tool_call('altered_name', {})))
    call_to_altered_name = mk_chat_with_tool_call('altered_name', {}).choices[0].message.tool_calls[0]
    pprint(call_to_altered_name)
    pprint(process_tool_call(call_to_altered_name, [function_decorated]))

    call_to_simple_method = mk_chat_with_tool_call('simple_method', {"count": 1, "size": 2.2}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_simple_method, [example_object.simple_method]))

    call_to_model = mk_chat_with_tool_call('User', {"name": 'John', "email": 'john@example.com'}).choices[0].message.tool_calls[0]
    pprint(process_tool_call(call_to_model, [User]))
