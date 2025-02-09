import pytest
from pydantic import BaseModel, Field
from typing import Annotated

# Define the Query model with type annotations and descriptions
class Query(BaseModel):
    query: Annotated[str, 'The search query']
    region: Annotated[str, 'The region for the search']

# Implement the search function
def search(query: Query, additional_param: str = None):
    '''
    This function performs a search based on the provided query and an optional additional parameter.
    
    Parameters:
    query (Query): A Pydantic model containing the search query and region.
    additional_param (str, optional): An optional additional parameter for the search.
    
    Returns:
    dict: A dictionary containing the results of the search, including 'query', 'region', and 'additional_param' keys if provided.
    '''
    result = {
        'query': query.query,
        'region': query.region
    }
    if additional_param:
        result['additional_param'] = additional_param
    return result

# Define a fixture for the query data
@pytest.fixture
def query_data():
    return Query(query='example query', region='example region')

# Define a more complex model for nested structures
class Address(BaseModel):
    street: str
    city: str

class Company(BaseModel):
    name: str
    speciality: str
    addresses: list[Address]

# Test function for search using pytest
@pytest.mark.parametrize('query_data, additional_param',
                         [(query_data(), 'extra info')],
                         indirect=['query_data'])
def test_search(query_data, additional_param):
    result = search(query_data, additional_param)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'query' in result, 'The result should contain the query.'
    assert result['query'] == 'example query', 'The query should match the provided query.'
    assert 'region' in result, 'The result should contain the region.'
    assert result['region'] == 'example region', 'The region should match the provided region.'
    if 'additional_param' in result:
        assert result['additional_param'] == additional_param, 'The additional parameter should match the provided value.'
    print('All tests passed!')