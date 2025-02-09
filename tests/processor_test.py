import json
from pydantic import BaseModel
from typing import Optional, List, Callable, Union
from llm_easy_tools.types import SimpleMessage, SimpleToolCall, SimpleFunction, SimpleChoice, SimpleCompletion
from llm_easy_tools.processor import process_response, process_tool_call, ToolResult
from llm_easy_tools import LLMFunction
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import sleep, time
from unittest.mock import Mock
from pydantic import ValidationError


def mk_tool_call(name, args):
    arguments = json.dumps(args)
    return SimpleToolCall(id='A', function=SimpleFunction(name=name, arguments=arguments), type='function')


def mk_tool_call_jason(name, args):
    return SimpleToolCall(id='A', function=SimpleFunction(name=name, arguments=args), type='function')


def mk_chat_completion(tool_calls):
    return SimpleCompletion(
        id='A',
        created=0,
        model='gpt-3.5-turbo',
        object='chat.completion',
        choices=[SimpleChoice(finish_reason='stop', index=0, message=SimpleMessage(role='assistant', tool_calls=tool_calls))]
    )


class TestTool:
    def tool_method(self, arg: int) -> str:
        return f'executed tool_method with param: {arg}'

    def no_output(self, arg: int):
        pass

    def failing_method(self, arg: int) -> str:
        raise Exception('Some exception')


def _extract_prefix_unpacked(args, model):
    return model(**{k: v for k, v in args.items() if k in model.__fields__})


# Import Union from typing module
from typing import Union


def process_one_tool_call(
        response: SimpleCompletion,
        functions: List[Union[Callable, LLMFunction]],
        index: int = 0,
        fix_json_args=True,
        case_insensitive=False
    ) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response)
    if not tool_calls or index >= len(tool_calls):
        return None

    return process_tool_call(tool_calls[index], functions, fix_json_args, case_insensitive)


# Helper function to get tool calls from the response
def _get_tool_calls(response: SimpleCompletion) -> List[SimpleToolCall]:
    if hasattr(response.choices[0].message, 'function_call') and (function_call := response.choices[0].message.function_call):
        return [SimpleToolCall(id='A', function=SimpleFunction(name=function_call.name, arguments=function_call.arguments), type='function')]
    elif hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        return response.choices[0].message.tool_calls
    return []
