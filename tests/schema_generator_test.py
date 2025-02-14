import pytest
from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, field_validator
from llm_easy_tools import get_function_schema, insert_prefix, LLMFunction
from pprint import pprint

def simplify_function(count: int, size: Optional[float] = None):
    """This function simplifies parameters and enhances error handling."""
    if size is not None and size <= 0:
        raise ValueError("Size must be a positive number")
    # Function body here

def test_function_schema():
    function_schema = get_function_schema(simplify_function)
    assert function_schema['name'] == 'simplify_function'
    assert function_schema['description'] == 'This function simplifies parameters and enhances error handling.'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2

def test_no_parameters():
    def function_without_parameters():
        """This function has no parameters."""
        pass
    result = get_function_schema(function_without_parameters)
    assert result['name'] == 'function_without_parameters'
    assert result['description'] == "This function has no parameters."
    assert result['parameters']['properties'] == {}

def test_nested_structure():
    class Structure(BaseModel):
        count: int
        size: Optional[float] = None
    def nested_function(structure_list: List[Structure]):
        """This function handles nested structures."""
        pass
    function_schema = get_function_schema(nested_function)
    assert function_schema['name'] == 'nested_function'

def test_llm_function():
    func = LLMFunction(simplify_function, name='simplified_function')
    function_schema = func.schema
    assert function_schema['name'] == 'simplified_function'

def test_case_insensitivity():
    class User(BaseModel):
        name: str
        city: str
    function_schema = get_function_schema(User, case_insensitive=True)
    assert function_schema['name'] == 'user'

def test_function_no_type_annotation():
    def function_with_missing_type(param):
        return f"Value is {param}"
    with pytest.raises(ValueError) as exc_info:
        get_function_schema(function_with_missing_type)
    assert str(exc_info.value) == "Parameter 'param' has no type annotation"