import inspect
from typing import Annotated, Callable, Dict, Any, get_origin, Type, Union
from typing_extensions import TypeGuard

import copy
import pydantic as pd
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

class LLMFunction:
    def __init__(self, func: Callable, schema: Dict[str, Any] = None, name: str = None, description: str = None, strict: bool = False):
        self.func = func
        self.__name__ = name if name else func.__name__
        self.__doc__ = description if description else func.__doc__
        self.schema = schema if schema else get_function_schema(func, strict=strict)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def get_function_schema(function: Union[Callable, LLMFunction], case_insensitive: bool = False, strict: bool = False) -> dict:
    if isinstance(function, LLMFunction):
        schema = function.schema
    else:
        description = function.__doc__.strip() if function.__doc__ else ''
        schema = {
            'name': function.__name__.lower() if case_insensitive else function.__name__,
            'description': description,
            'parameters': parameters_basemodel_from_function(function).model_json_schema(),
        }
    if strict:
        schema['parameters'] = to_strict_json_schema(schema['parameters'])
        schema['strict'] = True
    else:
        _recursive_purge_titles(schema['parameters'])
    return schema

def parameters_basemodel_from_function(function: Callable) -> Type[pd.BaseModel]:
    fields = {}
    for name, parameter in inspect.signature(function).parameters.items():
        type_ = parameter.annotation
        if type_ is inspect._empty:
            raise ValueError(f"Parameter '{name}' has no type annotation")
        if get_origin(type_) is Annotated:
            type_, description = type_.__args__
        else:
            description = None
        fields[name] = (type_, pd.Field(PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default, description=description))
    return pd.create_model(f'{function.__name__}_ParameterModel', **fields)

def _recursive_purge_titles(d: Dict[str, Any]) -> None:
    if isinstance(d, dict):
        d.pop('title', None)
        for v in d.values():
            _recursive_purge_titles(v)

def to_strict_json_schema(schema: dict) -> dict[str, Any]:
    return _ensure_strict_json_schema(schema, path=())

def _ensure_strict_json_schema(json_schema: object, path: tuple[str, ...]) -> dict[str, Any]:
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")
    if json_schema.get("type") == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    for key in ["properties", "items", "anyOf", "allOf", "$defs"]:
        if key in json_schema:
            if isinstance(json_schema[key], list):
                json_schema[key] = [_ensure_strict_json_schema(item, path=(*path, key, str(i))) for i, item in enumerate(json_schema[key])]
            else:
                json_schema[key] = _ensure_strict_json_schema(json_schema[key], path=(*path, key))
    return json_schema

def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    return isinstance(obj, dict)

def get_tool_defs(functions: list[Union[Callable, LLMFunction]], case_insensitive: bool = False, prefix_class: Union[Type[BaseModel], None] = None, prefix_schema_name: bool = True, strict: bool = False) -> list[dict]:
    result = []
    for function in functions:
        schema = get_function_schema(function, case_insensitive, strict)
        if prefix_class:
            schema = insert_prefix(prefix_class, schema, prefix_schema_name, case_insensitive)
        result.append({"type": "function", "function": schema})
    return result

def insert_prefix(prefix_class: Type[BaseModel], schema: Dict[str, Any], prefix_schema_name: bool = True, case_insensitive: bool = False) -> Dict[str, Any]:
    if not issubclass(prefix_class, BaseModel):
        raise TypeError(f"The given class reference is not a subclass of pydantic BaseModel")
    prefix_schema = prefix_class.model_json_schema()
    _recursive_purge_titles(prefix_schema)
    prefix_schema.pop('description', '')
    required = schema['parameters'].get('required', [])
    prefix_schema['required'].extend(required)
    prefix_schema['properties'].update(schema['parameters']['properties'])
    schema['parameters'] = prefix_schema
    if prefix_schema_name:
        schema['name'] = f"{prefix_class.__name__.lower() if case_insensitive else prefix_class.__name__}_and_{schema['name']}"
    return schema