# Import necessary modules
from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, field_validator
from llm_easy_tools import get_function_schema, insert_prefix, LLMFunction
from llm_easy_tools.schema_generator import parameters_basemodel_from_function, _recursive_purge_titles, get_name, get_tool_defs
from pprint import pprint
import pytest

# Define the insert_prefix function within the llm_easy_tools module
def insert_prefix(model: BaseModel, schema: dict, case_insensitive: bool = False):
    """
    Inserts a prefix into the name of the schema.
    
    Args:
        model (BaseModel): The Pydantic model to insert the prefix into.
        schema (dict): The function schema to modify.
        case_insensitive (bool, optional): Whether to make the schema name case insensitive. Defaults to False.
    
    Returns:
        dict: The modified function schema with a prefixed name.
    """
    schema_name = schema.get('name', model.__name__)
    if case_insensitive:
        schema_name = schema_name.lower()
    new_schema_name = f"{schema_name}_{model.__name__}"
    schema['name'] = new_schema_name
    return schema

# Define the simple_function with a concise docstring
def simple_function(count: int, size: Optional[float] = None):
    """
    A simple function that takes two parameters: count and size.
    
    Args:
        count (int): The count of items.
        size (Optional[float]): The size of the items.
    """
    pass

# Define the simple_function_no_docstring with a concise description
def simple_function_no_docstring(
        apple: Annotated[str, 'The apple'],
        banana: Annotated[str, 'The banana']
):
    pass

# Test function to check the schema of simple_function
def test_function_schema():
    function_schema = get_function_schema(simple_function)
    assert function_schema['name'] == 'simple_function'
    assert function_schema['description'] == 'A simple function that takes two parameters: count and size.'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2
    assert params_schema['type'] == "object"
    assert params_schema['properties']['count']['type'] == "integer"
    assert 'size' in params_schema['properties']
    assert 'title' not in params_schema
    assert 'title' not in params_schema['properties']['count']
    assert 'description' not in params_schema

# Test function to check the handling of functions with no parameters
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
    assert result['description'] == 'This function has a docstring and takes no parameters.'
    assert result['parameters']['properties'] == {}

    result = get_function_schema(function_no_doc)
    assert result['name'] == 'function_no_doc'
    assert result['description'] == ''
    assert result['parameters']['properties'] == {}

# Test function to check the handling of nested structures
def test_nested():
    class Foo(BaseModel):
        count: int
        size: Optional[float] = None

    class Bar(BaseModel):
        """Some Bar"""
        apple: str = Field(description="The apple")
        banana: str = Field(description="The banana")

    class FooAndBar(BaseModel):
        foo: Foo
        bar: Bar

    def nested_structure_function(foo: Foo, bars: List[Bar]):
        """spams everything"""
        pass

    function_schema = get_function_schema(nested_structure_function)
    assert function_schema['name'] == 'nested_structure_function'
    assert function_schema['description'] == 'spams everything'
    assert len(function_schema['parameters']['properties']) == 2

    function_schema = get_function_schema(FooAndBar)
    assert function_schema['name'] == 'FooAndBar'
    assert len(function_schema['parameters']['properties']) == 2

# Test function to check the handling of methods
def test_methods():
    class ExampleClass:
        def simple_method(self, count: int, size: Optional[float] = None):
            """simple method does something"""
            pass

    example_object = ExampleClass()

    function_schema = get_function_schema(example_object.simple_method)
    assert function_schema['name'] == 'simple_method'
    assert function_schema['description'] == 'simple method does something'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2

# Test function to check the LLMFunction class
def test_LLMFunction():
    def new_simple_function(count: int, size: Optional[float] = None):
        """simple function does something"""
        pass

    func = LLMFunction(new_simple_function, name='changed_name')
    function_schema = func.schema
    assert function_schema['name'] == 'changed_name'
    assert not 'strict' in function_schema or function_schema['strict'] == False

    func = LLMFunction(simple_function, strict=True)
    function_schema = func.schema
    assert function_schema['strict'] == True

# Test function to check the merging of schemas
def test_merge_schemas():
    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    function_schema = get_function_schema(simple_function)
    new_schema = insert_prefix(Reflection, function_schema)
    assert new_schema['name'] == "Reflection_and_simple_function"
    assert len(new_schema['parameters']['properties']) == 4
    assert len(new_schema['parameters']['required']) == 3
    assert len(function_schema['parameters']['properties']) == 2  # the old schema is not changed
    assert len(function_schema['parameters']['required']) == 1  # the old schema is not changed
    param_names = list(new_schema['parameters']['properties'].keys())
    assert param_names == ['relevancy', 'next_actions_plan', 'count', 'size']

    function_schema = get_function_schema(simple_function)
    new_schema = insert_prefix(Reflection, function_schema, case_insensitive=True)
    assert new_schema['name'] == "reflection_and_simple_function"

# Test function to check the merging of schemas with no parameters
def test_noparams_function_merge():
    def function_no_params():
        pass

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")
        next_actions_plan: str = Field(description="What you plan to do next and why")

    function_schema = get_function_schema(function_no_params)
    assert function_schema['name'] == 'function_no_params'
    assert function_schema['parameters']['properties'] == {}
    pprint(function_schema)

    new_schema = insert_prefix(Reflection, function_schema)
    assert len(new_schema['parameters']['properties']) == 2
    assert new_schema['name'] == 'Reflection_and_function_no_params'

# Test function to check the initialization of models
def test_model_init_function():
    class User(BaseModel):
        """A user object"""
        name: str
        city: str

    function_schema = get_function_schema(User)
    assert function_schema['name'] == 'User'
    assert function_schema['description'] == 'A user object'
    assert len(function_schema['parameters']['properties']) == 2
    assert len(function_schema['parameters']['required']) == 2

    new_function = LLMFunction(User, name="extract_user_details")
    assert new_function.schema['name'] == 'extract_user_details'
    assert new_function.schema['description'] == 'A user object'
    assert len(new_function.schema['parameters']['properties']) == 2
    assert len(new_function.schema['parameters']['required']) == 2

# Test function to check case insensitivity in schema names
def test_case_insensitivity():
    class User(BaseModel):
        """A user object"""
        name: str
        city: str

    function_schema = get_function_schema(User, case_insensitive=True)
    assert function_schema['name'] == 'user'
    assert get_name(User, case_insensitive=True) == 'user'

# Test function to check for missing type annotations
def test_function_no_type_annotation():
    def function_with_missing_type(param):
        return f"Value is {param}"

    with pytest.raises(ValueError) as exc_info:
        get_function_schema(function_with_missing_type)
    assert str(exc_info.value) == "Parameter 'param' has no type annotation"

# Test function to check the handling of Pydantic parameters
def test_pydantic_param():
    class Query(BaseModel):
        query: str
        region: str

    def search(query: Query):
        ...

    schema = get_tool_defs([search])
    assert schema[0]['function']['name'] == 'search'
    assert schema[0]['function']['description'] == ''
    assert schema[0]['function']['parameters']['properties']['query']['$ref'] == '#/$defs/Query'

# Test function to check strict mode for schemas
def test_strict():
    class Address(BaseModel):
        street: str
        city: str

    class Company(BaseModel):
        name: str
        speciality: str
        addresses: list[Address]

    def print_companies(companies: list[Company]):
        ...

    schema = get_tool_defs([print_companies], strict=True)
    pprint(schema)

    function_schema = schema[0]['function']
    assert function_schema['name'] == 'print_companies'
    assert function_schema['strict'] == True
    assert function_schema['parameters']['additionalProperties'] == False
    assert function_schema['parameters']['$defs']['Address']['additionalProperties'] == False
    assert function_schema['parameters']['$defs']['Address']['properties']['street']['type'] == 'string'
    assert function_schema['parameters']['$defs']['Company']['additionalProperties'] == False