from llm_easy_tools import get_tool_defs, process_response
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()


# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    city: str


response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    tools=get_tool_defs([UserDetail]),
    tool_choice="auto",
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [UserDetail])

#pprint(results)
pprint(results[0].output)

def contact_user(name: str, city: str) -> str:
    return f"User {name} from {city} was contactd"

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Contact John. John lives in Warsaw"}],
    tools=get_tool_defs([contact_user]),
    tool_choice={"type": "function", "function": {"name": "contact_user"}},
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [contact_user])

pprint(results[0].output)
