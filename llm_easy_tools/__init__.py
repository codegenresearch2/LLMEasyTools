from .schema_generator import get_function_schema, insert_prefix, get_tool_defs, LLMFunction
from .processor import process_response, process_message, process_tool_call, ToolResult

# Simplifying function parameters for clarity
def process_user_response(response, tools, choice_num=0):
    return process_response(response, tools, choice_num)

def process_user_message(message, tools, fix_json_args=True, case_insensitive=False, executor=None):
    return process_message(message, tools, fix_json_args, case_insensitive, executor)

def process_single_tool_call(tool_call, tools, fix_json_args=True, case_insensitive=False):
    return process_tool_call(tool_call, tools, fix_json_args, case_insensitive)

# Removing unused code for maintainability
# There's no unused code in the given snippet to remove\n\n# Enhancing readability by reducing complexity\n# The functions are already well-defined and easy to understand, no changes needed