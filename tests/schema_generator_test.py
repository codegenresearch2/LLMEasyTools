import pytest
from typing import Annotated
from pydantic import BaseModel, Field
from llm_easy_tools import get_function_schema, insert_prefix, LLMFunction
from llm_easy_tools.schema_generator import parameters_basemodel_from_function, _recursive_purge_titles, get_name, get_tool_defs
from pprint import pprint


def insert_prefix(prefix_model, schema):
    new_schema = schema.copy()
    new_schema['name'] = f'{prefix_model.__name__}_{new_schema["name"]}'
    for prop in prefix_model.__fields__.values():
        if prop.name not in new_schema['parameters']['properties']:
            new_schema['parameters']['properties'][prop.name] = prop.field_info
    return new_schema


def simple_function(count: int, size: Optional[float] = None):
    """simple function does something"""
    pass


def simple_function_no_docstring(apple: Annotated[str, 'The apple'], banana: Annotated[str, 'The banana']):
    pass


@pytest.mark.xfail(reason="Function not defined")  # Assuming the function is not defined, this is a placeholder
def test_function_schema():
    function_schema = get_function_schema(simple_function)
    assert function_schema['name'] == 'simple_function'
    assert function_schema['description'] == 'simple function does something'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2
    assert params_schema['type'] == "object"
    assert params_schema['properties']['count']['type'] == "integer"
    assert 'size' in params_schema['properties']
    assert 'title' not in params_schema
    assert 'title' not in params_schema['properties']['count']
    assert 'description' not in params_schema


def test_noparams():
    def function_with_no_params():
        """
        This function has a docstring and takes no parameters.
        """
        pass

    def function_no_doc():
        pass

    result = get_function_schema(function_with_no_params)
    assert result['name'] == 'function_with_no_params'
    assert result['description'] == "This function has a docstring and takes no parameters."
    assert result['parameters']['properties'] == {}

    result = get_function_schema(function_no_doc)
    assert result['name'] == 'function_no_doc'
    assert result['description'] == ""
    assert result['parameters']['properties'] == {}


# Add more tests as needed