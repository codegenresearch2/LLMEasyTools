import inspect
from typing import Callable, Dict, Any, get_origin, Type, Union
from typing_extensions import TypeGuard

import pydantic as pd
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from pprint import pprint
import sys

class LLMFunction:
    def __init__(self, func, schema=None, name=None, description=None, strict=False):
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

        if schema:
            self.schema = schema
            if name or description:
                raise ValueError("Cannot specify name or description when providing a complete schema")
        else:
            self.schema = get_function_schema(func, strict=strict)

            if name:
                self.schema['name'] = name

            if description:
                self.schema['description'] = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def tool_def(function_schema: dict) -> dict:
    return {
        "type": "function",
        "function": function_schema,
    }

def get_tool_defs(functions: list[Union[Callable, LLMFunction]], case_insensitive: bool = False, strict: bool = False) -> list[dict]:
    result = []
    for function in functions:
        if isinstance(function, LLMFunction):
            fun_schema = function.schema
        else:
            fun_schema = get_function_schema(function, case_insensitive, strict)
        result.append(tool_def(fun_schema))
    return result

def parameters_basemodel_from_function(function: Callable) -> Type[pd.BaseModel]:
    fields = {}
    parameters = inspect.signature(function).parameters
    function_globals = sys.modules[function.__module__].__dict__ if inspect.ismethod(function) else getattr(function, '__globals__', {})

    for name, parameter in parameters.items():
        type_ = parameter.annotation
        if type_ is inspect._empty:
            raise ValueError(f"Parameter '{name}' has no type annotation")
        if get_origin(type_) is pd.Annotated:
            type_ = type_.__args__[0]
        if isinstance(type_, str):
            type_ = eval(type_, function_globals)
        default = PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (type_, default)
    return pd.create_model(f'{function.__name__}_ParameterModel', **fields)

def _recursive_purge_titles(d: Dict[str, Any]) -> None:
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == 'title' and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_titles(d[key])

def get_name(func: Union[Callable, LLMFunction], case_insensitive: bool = False) -> str:
    schema_name = func.schema['name'] if isinstance(func, LLMFunction) else func.__name__
    return schema_name.lower() if case_insensitive else schema_name

def get_function_schema(function: Union[Callable, LLMFunction], case_insensitive: bool=False, strict: bool=False) -> dict:
    if isinstance(function, LLMFunction):
        if case_insensitive:
            raise ValueError("Cannot case insensitive for LLMFunction")
        return function.schema

    description = function.__doc__.strip() if hasattr(function, '__doc__') and function.__doc__ else ''
    schema_name = function.__name__.lower() if case_insensitive else function.__name__

    function_schema: dict[str, Any] = {
        'name': schema_name,
        'description': description,
    }
    model = parameters_basemodel_from_function(function)
    model_json_schema = model.model_json_schema()
    if strict:
        model_json_schema = to_strict_json_schema(model_json_schema)
        function_schema['strict'] = True
    else:
        _recursive_purge_titles(model_json_schema)
    function_schema['parameters'] = model_json_schema

    return function_schema

def to_strict_json_schema(schema: dict) -> dict[str, Any]:
    return _ensure_strict_json_schema(schema, path=())

def _ensure_strict_json_schema(json_schema: object, path: tuple[str, ...]) -> dict[str, Any]:
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key)) for key, prop_schema in properties.items()}

    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"))

    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        json_schema["anyOf"] = [_ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i))) for i, variant in enumerate(any_of)]

    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        json_schema["allOf"] = [_ensure_strict_json_schema(entry, path=(*path, "anyOf", str(i))) for i, entry in enumerate(all_of)]

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name))

    return json_schema

def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    return isinstance(obj, dict)

#######################################
#
# Examples

if __name__ == "__main__":
    def function_with_doc():
        """This function has a docstring and no parameters."""
        pass

    altered_function = LLMFunction(function_with_doc, name="altered_name")

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            """simple method does something"""
            pass

    example_object = ExampleClass()

    class User(BaseModel):
        name: str
        age: int

    pprint(get_tool_defs([example_object.simple_method, function_with_doc, altered_function, User]))