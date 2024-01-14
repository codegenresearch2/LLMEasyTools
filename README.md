# LLMToolBox

LLMToolBox is a Python package designed to facilitate using 'tools' and 'function calls' in OpenAI APIs.
It is minimal and designed to not get in the way.

It can generate tools definitions (schemas) and then process a function call from an LLM response.
The package leverages Pydantic for data validation and schema generation, ensuring that tools are used with
the correct data types and structures.


## Features

- **Schema Generation**: Automatically generate JSON schemas for tools based on Pydantic models.
- **Function Name Mapping**: Allows for using different names in JSON schemas and python code.
- **Dispatching Function Calls**: Can call the function corresponding to the function call structure received from the LLM

## Installation

You can install ToolDefGenerator by cloning the repository and installing it via pip:

```bash
git clone git@github.com:zby/LLMToolBox.git
cd LLMToolBox
pip install .
```

When working on the package it is useful to install it in editable form:
```bash
pip install -r requirements.txt
pip install -e .
```

Then test:
```bash
pytest -v tests
```

## Usage

### Basic example
```python
from llm_tool_box import ToolBox, ToolResult
from pydantic import BaseModel
from openai import OpenAI
client = OpenAI()

# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    age: int

# Create a ToolBox instance
toolbox = ToolBox()

# Register your tool - if a class is passed an identity function over it is registered
toolbox.register_tool(UserDetail)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract Jason is 25 years old"}],
    tools=toolbox.tool_schemas,
    tool_choice="auto",
)
function_call = response.choices[0].message.function_call
if function_call:
    result = toolbox.process(function_call)
```
More examples in tests.

## License

LLMToolBox is licensed under the Apache License. See LICENSE for more information.
