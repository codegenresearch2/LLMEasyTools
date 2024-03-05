from llm_easy_tools import ToolBox
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()


# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    city: str


def contact_user(user: UserDetail):
    return f"User {user.name} from {user.city} was contactd"

# Create a ToolBox instance
toolbox = ToolBox()

# Register your model -
toolbox.register_model(UserDetail)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    functions=toolbox.function_schemas(),
    function_call= "auto",
)
# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results)
