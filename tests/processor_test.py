import pytest
import json
from time import sleep, time
from typing import Any, Optional
from pydantic import BaseModel, Field, ValidationError
from concurrent.futures import ThreadPoolExecutor
from llm_easy_tools.types import SimpleMessage, SimpleToolCall, SimpleFunction, SimpleChoice, SimpleCompletion
from llm_easy_tools.processor import process_response, process_tool_call, ToolResult, _extract_prefix_unpacked, _process_unpacked

def mk_tool_call(name, args):
    arguments = json.dumps(args)
    return SimpleToolCall(id='A', function=SimpleFunction(name=name, arguments=arguments), type='function')

def mk_chat_completion(tool_calls):
    return SimpleCompletion(id='A', created=0, model='gpt-3.5-turbo', object='chat.completion', choices=[SimpleChoice(finish_reason='stop', index=0, message=SimpleMessage(role='assistant', tool_calls=tool_calls))])

def test_process_methods():
    class TestTool:
        def tool_method(self, arg: int) -> str:
            return f'executed tool_method with param: {arg}'

        def no_output(self, arg: int):
            pass

        def failing_method(self, arg: int) -> str:
            raise Exception('Some exception')

    tool = TestTool()
    tool_call = mk_tool_call("tool_method", {"arg": 2})
    result = process_tool_call(tool_call, [tool.tool_method])
    assert result.output == 'executed tool_method with param: 2'

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
    assert isinstance(result.output[0], Company)

def test_json_fix():
    class UserDetail(BaseModel):
        name: str
        age: int

    json_data = json.dumps(UserDetail(name="John", age=21).model_dump())[:-1] + ',}'
    tool_call = mk_tool_call("UserDetail", json_data)
    result = process_tool_call(tool_call, [UserDetail])
    assert result.output == UserDetail(name="John", age=21)

def test_list_in_string_fix():
    class User(BaseModel):
        names: Optional[list[str]]

    tool_call = mk_tool_call("User", {"names": "John, Doe"})
    result = process_tool_call(tool_call, [User])
    assert result.output.names == ["John", "Doe"]

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
    assert counter.counter == 10
    assert end_time - start_time <= 3

def test_process_one_tool_call():
    class User(BaseModel):
        name: str
        age: int

    response = mk_chat_completion([mk_tool_call("User", {"name": "Alice", "age": 30}), mk_tool_call("User", {"name": "Bob", "age": 25})])
    result = process_one_tool_call(response, [User], index=0)
    assert result.output == User(name="Alice", age=30)