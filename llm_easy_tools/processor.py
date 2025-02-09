import json
import inspect
import traceback

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Union, Optional, Any, get_origin, get_args
from pprint import pprint

from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field

from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import Function

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

# Ensure that type annotations can be correctly processed
def _is_list_type(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        return True
    elif origin is Union or origin is Optional:
        return any(_is_list_type(arg) for arg in args)
    return False

# Modify the response structure handling
@get_name
def process_response(response: dict, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    message = response['choices'][choice_num]['message']
    return process_message(message, functions, **kwargs)

# Ensure correct access to the response attributes
def process_message(
    message: dict,
    functions: list[Union[Callable, LLMFunction]],
    prefix_class=None,
    fix_json_args=True,
    case_insensitive=False,
    executor: Union[ThreadPoolExecutor, ProcessPoolExecutor, None]=None
    ) -> list[ToolResult]:
    results = []
    if 'function_call' in message and message['function_call']:
        tool_calls = [{"id": 'A', "function": Function(name=message['function_call']['name'], arguments=message['function_call']['arguments']), "type": 'function'}]
    elif 'tool_calls' in message and message['tool_calls']:
        tool_calls = message['tool_calls']
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
