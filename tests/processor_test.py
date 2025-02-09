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


# Define process_one_tool_call function


# Helper function to get tool calls from the response


# Define the function `process_one_tool_call` as per the test case feedback


# Rest of the code remains unchanged
# ...