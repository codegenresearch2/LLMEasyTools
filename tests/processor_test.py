import pytest
import json
from time import sleep, time
from unittest.mock import Mock
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Optional
from llm_easy_tools.types import SimpleMessage, SimpleToolCall, SimpleFunction, SimpleChoice, SimpleCompletion
from llm_easy_tools.processor import process_response, process_tool_call, ToolResult, process_one_tool_call
from llm_easy_tools import LLMFunction
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def mk_tool_call(name, args):
    arguments = json.dumps(args)
    return SimpleToolCall(id='A', function=SimpleFunction(name=name, arguments=arguments), type='function')

def mk_tool_call_json(name, args):
    return SimpleToolCall(id='A', function=SimpleFunction(name=name, arguments=args), type='function')

def mk_chat_completion(tool_calls):
    return SimpleCompletion(
        id='A',
        created=0,
        model='gpt-3.5-turbo',
        object='chat.completion',
        choices=[
            SimpleChoice(
                finish_reason='stop',
                index=0,
                message=SimpleMessage(role='assistant', tool_calls=tool_calls))
        ]
    )

@pytest.mark.parametrize("tool_name, args, expected_output", [
    ("tool_method", {"arg": 2}, 'executed tool_method with param: 2'),
    ("failing_method", {"arg": 2}, "Some exception"),
])
def test_process_methods(tool_name, args, expected_output):
    class TestTool:
        def tool_method(self, arg: int) -> str:
            return f'executed tool_method with param: {arg}'

        def no_output(self, arg: int):
            pass

        def failing_method(self, arg: int) -> str:
            raise Exception('Some exception')

    tool = TestTool()

    tool_call = mk_tool_call(tool_name, args)
    result = process_tool_call(tool_call, [getattr(tool, tool_name)])
    assert isinstance(result, ToolResult)
    if result.error:
        assert expected_output in str(result.error)
    else:
        assert result.output == expected_output

def test_process_complex():
    class Address(BaseModel):
        street: str
        city: str

    class Company(BaseModel):
        name: str
        speciality: str
        address: Address

    def print_companies(companies: list[Company]):
        return companies

    company_list = [{
        'address': {'city': 'Metropolis', 'street': '150 Futura Plaza'},
        'name': 'Aether Innovations',
        'speciality': 'sustainable energy solutions'
    }]

    tool_call = mk_tool_call("print_companies", {"companies": company_list})
    result = process_tool_call(tool_call, [print_companies])
    assert isinstance(result, ToolResult)
    assert isinstance(result.output, list)
    assert isinstance(result.output[0], Company)

def test_prefixing():
    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Relevance of the last retrieved information")

    args = { 'relevancy': 'good', 'name': 'hammer'}
    prefix = Reflection(**args)
    assert isinstance(prefix, Reflection)
    assert 'reflection' not in args

def test_json_fix():
    class UserDetail(BaseModel):
        name: str
        age: int

    original_user = UserDetail(name="John", age=21)
    json_data = json.dumps(original_user.model_dump())
    json_data = json_data[:-1]
    json_data = json_data + ',}'
    tool_call = mk_tool_call_json("UserDetail", json_data)
    result = process_tool_call(tool_call, [UserDetail])
    assert result.output == original_user
    assert len(result.soft_errors) > 0

    result = process_tool_call(tool_call, [UserDetail], fix_json_args=False)
    assert isinstance(result.error, json.decoder.JSONDecodeError)

    response = mk_chat_completion([tool_call])
    results = process_response(response, [UserDetail])
    assert results[0].output == original_user
    assert len(results[0].soft_errors) > 0

    results = process_response(response, [UserDetail], fix_json_args=False)
    assert isinstance(results[0].error, json.decoder.JSONDecodeError)

def test_list_in_string_fix():
    class User(BaseModel):
        names: Optional[list[str]]

    tool_call = mk_tool_call("User", {"names": "John, Doe"})
    result = process_tool_call(tool_call, [User])
    assert result.output.names == ["John", "Doe"]
    assert len(result.soft_errors) > 0

    tool_call = mk_tool_call("User", {"names": "[\"John\", \"Doe\"]"})
    result = process_tool_call(tool_call, [User])
    assert result.output.names == ["John", "Doe"]
    assert len(result.soft_errors) > 0

    result = process_tool_call(tool_call, [User], fix_json_args=False)
    assert isinstance(result.error, ValidationError)

def test_case_insensitivity():
    class User(BaseModel):
        name: str
        city: str

    response = mk_chat_completion([mk_tool_call("user", {"name": "John", "city": "Metropolis"})])
    results = process_response(response, [User], case_insensitive=True)
    assert results[0].output == User(name="John", city="Metropolis")

def test_parallel_tools():
    class CounterClass:
        def __init__(self):
            self.counter = 0

        def increment_counter(self):
            self.counter += 1
            sleep(1)

    counter = CounterClass()
    tool_call = mk_tool_call("increment_counter", {})
    response = mk_chat_completion([tool_call] * 10)

    executor = ThreadPoolExecutor()
    start_time = time()
    results = process_response(response, [counter.increment_counter], executor=executor)
    end_time = time()

    assert results[9].error is None

    time_taken = end_time - start_time
    assert counter.counter == 10
    assert time_taken <= 3, f"Expected processing time to be less than or equal to 3 seconds, but was {time_taken}"

def test_process_one_tool_call():
    class User(BaseModel):
        name: str
        age: int

    response = mk_chat_completion([
        mk_tool_call("User", {"name": "Alice", "age": 30}),
        mk_tool_call("User", {"name": "Bob", "age": 25})
    ])

    result = process_one_tool_call(response, [User], index=0)
    assert isinstance(result, ToolResult)
    assert result.output == User(name="Alice", age=30)

    result = process_one_tool_call(response, [User], index=1)
    assert isinstance(result, ToolResult)
    assert result.output == User(name="Bob", age=25)

    result = process_one_tool_call(response, [User], index=2)
    assert result is None

    invalid_response = mk_chat_completion([mk_tool_call("InvalidFunction", {})])
    result = process_one_tool_call(invalid_response, [User])
    assert isinstance(result, ToolResult)
    assert result.error is not None