import pytest

import just_agents.llm_options
from examples.just_agents.examples.tools import get_current_weather
from typing import List, Dict, Optional, Type, Union, Any, Sequence, Set, Callable, Tuple
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from just_agents.base_agent import BaseAgent
from just_agents.just_profile import JustAgentProfile
from just_agents.just_schema import ModelHelper


# Define the test function
def krex_pex_fex(
    a: List[Dict[str, Union[Sequence[Set[Callable | Optional[Any]]], Tuple[str, ...]]]],
    /,
    b: list[dict[str, int]],
    *,
    c=5,
):
    """
    A test function for testing type handling
    """
    print(a, b, c)


# Define test models with different complexity levels
class SimpleTestModel(BaseModel):
    """A simple model for testing schema serialization"""
    name: str = Field(..., description="Name field")
    value: int = Field(..., description="Value field")


class NestedTestModel(BaseModel):
    """A model with nested types for testing complex schema serialization"""
    id: str = Field(..., description="Unique identifier")
    data: Dict[str, Any] = Field(..., description="Data dictionary")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    config: Optional[SimpleTestModel] = Field(None, description="Optional nested configuration")


class UnionTestModel(BaseModel):
    """A model with union types for testing complex schema serialization"""
    value: Union[str, int, bool] = Field(..., description="Value that can be different types")
    items: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="List of mixed type items")

# Fixture to load environment variables
@pytest.fixture(scope="module")
def load_env():
    load_dotenv(override=True)

def test_just_agent_profile(load_env, tmp_path):
    # Set up paths
    config_path = tmp_path / "yaml_initialization_example_new.yaml"

    # Create a JustAgentProfile instance
    profile = JustAgentProfile(
        tools=[krex_pex_fex],
        config_path=config_path,
    )

    # Save the profile to YAML
    profile.save_to_yaml("TestProfileA")

    # Load the profile from YAML
    loaded_profile = JustAgentProfile.from_yaml("TestProfileA", file_path=config_path)

    # Assert that the loaded profile matches the original
    assert loaded_profile.to_json() == profile.to_json()

    # Optionally, print the JSON representation
    print(loaded_profile.to_json())

def test_just_agent(load_env, tmp_path):
    # Set up paths
    config_path = tmp_path / "yaml_initialization_example_new.yaml"

    # Create a JustAgent instance
    agent = BaseAgent(
        llm_options=just_agents.llm_options.OPENAI_GPT4_1NANO,
        config_path=config_path,
        tools=[get_current_weather],
    )

    # Save the agent to YAML
    agent.save_to_yaml("SimpleWeatherAgent")

    # Load the agent from YAML
    loaded_agent = BaseAgent.from_yaml("SimpleWeatherAgent", file_path=config_path)

    # Assert that the loaded agent matches the original
    assert loaded_agent.to_json() == agent.to_json()

    # Optionally, print the JSON representation
    print(loaded_agent.to_json())


# Test fixture for creating agents with different parsers
@pytest.fixture(params=[
    SimpleTestModel,
    NestedTestModel,
    UnionTestModel,
    None  # Special case for handling None
])
def agent_with_parser(request):
    """Create an agent with the specified parser model"""
    # Use a predefined options dictionary instead of trying to instantiate LLMOptions
    options = {"model": "gpt-4.1-mini", "temperature": 0.7}

    return BaseAgent(
        llm_options=options,
        system_prompt="Test system prompt",
        parser=request.param
    )


def test_parser_serialization(agent_with_parser):
    """Test that parser field is correctly serialized"""
    # Get the original parser
    original_parser = agent_with_parser.parser

    # Convert agent to dict (this will call the field_serializer)
    agent_dict = agent_with_parser.model_dump(mode="json")

    # Check that the serialized parser has the expected format
    serialized_parser = agent_dict.get("parser")

    if original_parser is None:
        assert serialized_parser is None
    else:
        test_model = ModelHelper.create_model_from_flat_yaml(
            "SimpleTestModelOutputParser",
            serialized_parser,
            optional_fields=False  # Make fields required for validation
        )
        assert test_model is not None
        assert test_model.__name__ == "SimpleTestModelOutputParser"
        assert issubclass(original_parser, BaseModel)
        assert issubclass(test_model, BaseModel)
        assert set(test_model.model_fields.keys()) == set(original_parser.model_fields.keys())


def test_validate_assignment():
    agent = BaseAgent(
        llm_options={"model": "gpt-4.1-mini", "temperature": 0.7},
        system_prompt="Test system prompt",
        parser=None
    )
    agent.parser = {'name': 'str', 'value': 'int'}
    assert agent.parser is not None
    assert not isinstance(agent.parser, BaseModel)
    assert not isinstance(agent.parser, dict)
    assert isinstance(agent.parser, type) and issubclass(agent.parser, BaseModel)
    assert 'OutputParser' in agent.parser.__name__
    assert set(agent.parser.model_fields.keys()) == {'name', 'value'}

def test_yaml_to_agent_to_yaml_parser_roundtrip(tmp_path):
    """Test roundtrip of parser field from YAML to agent and back to YAML"""
    # Set up paths
    config_path = tmp_path / "parser_roundtrip_test.yaml"
    
    # Create initial YAML content with parser definition
    yaml_content = {
        "name": "TestAgentWithParser",
        "llm_options": {"model": "gpt-4.1-mini", "temperature": 0.7},
        "system_prompt": "Test system prompt",
        "parser": {
            "name": "str",
            "age": "int",
            "email": "Optional[str]",
            "scores": "Dict[str, float]",
            "tags": "List[str]",
            "is_active": "bool",
            "metadata": "Dict[str, Any]"
        }
    }
    
    # Save initial YAML
    agent = BaseAgent(**yaml_content)
    agent.save_to_yaml("TestAgentWithParser", file_path=config_path)
    
    # Load agent from YAML
    loaded_agent = BaseAgent.from_yaml("TestAgentWithParser", file_path=config_path)
    
    # Verify the parser was correctly loaded
    assert loaded_agent.parser is not None
    assert isinstance(loaded_agent.parser, type)
    assert issubclass(loaded_agent.parser, BaseModel)
    
    # Check all fields were preserved
    expected_fields = set(yaml_content["parser"].keys())
    actual_fields = set(loaded_agent.parser.model_fields.keys())
    assert expected_fields == actual_fields
    
    # Save back to YAML and verify parser field
    loaded_agent.save_to_yaml("TestAgentWithParser2", file_path=config_path)
    final_agent = BaseAgent.from_yaml("TestAgentWithParser2", file_path=config_path)
    
    # Verify the final parser matches the original
    assert final_agent.parser is not None
    assert set(final_agent.parser.model_fields.keys()) == expected_fields
    
    # Test creating an instance with the parser
    test_data = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "scores": {"math": 95.5, "science": 88.0},
        "tags": ["student", "active"],
        "is_active": True,
        "metadata": {"last_login": "2024-03-20"}
    }
    parsed_instance : BaseModel = final_agent.parser(**test_data)
    assert parsed_instance.model_dump() == test_data

def test_agent_to_yaml_to_agent_complex_parser_roundtrip(tmp_path):
    """Test roundtrip of agent with complex parser models to YAML and back"""
    config_path = tmp_path / "complex_parser_roundtrip_test.yaml"
    
    # Create a complex parser model
    class UserProfile(BaseModel):
        id: int
        username: str
        email: Optional[str]
        preferences: Dict[str, Any]
        roles: List[str]
        settings: Optional[Dict[str, Union[str, int, bool]]]
        scores: Dict[str, float]
        
    # Create initial agent with complex parser
    original_agent = BaseAgent(
        llm_options={"model": "gpt-4.1-mini", "temperature": 0.7},
        system_prompt="Test system prompt",
        parser=UserProfile
    )
    
    # Save to YAML
    original_agent.save_to_yaml("ComplexParserAgent", file_path=config_path)
    
    # Load back from YAML
    loaded_agent = BaseAgent.from_yaml("ComplexParserAgent", file_path=config_path)
    
    # Verify parser was preserved
    assert loaded_agent.parser is not None
    assert isinstance(loaded_agent.parser, type)
    assert issubclass(loaded_agent.parser, BaseModel)
    
    # Check all fields were preserved
    original_fields = set(UserProfile.model_fields.keys())
    loaded_fields = set(loaded_agent.parser.model_fields.keys())
    assert original_fields == loaded_fields
    
    # Test the loaded parser with complex data
    test_data = {
        "id": 123,
        "username": "testuser",
        "email": "test@example.com",
        "preferences": {"theme": "dark", "notifications": True},
        "roles": ["admin", "user"],
        "settings": {"timeout": 30, "debug": True, "api_url": "http://api.example.com"},
        "scores": {"project1": 95.5, "project2": 88.0}
    }
    
    # Verify we can create instances with both parsers
    original_instance = original_agent.parser(**test_data)
    loaded_instance = loaded_agent.parser(**test_data)
    
    # Verify both instances have the same data
    assert original_instance.model_dump() == loaded_instance.model_dump()
    
    # Save again and load to verify stability
    loaded_agent.save_to_yaml("ComplexParserAgent2", file_path=config_path)
    final_agent = BaseAgent.from_yaml("ComplexParserAgent2", file_path=config_path)
    
    # Verify the final parser still works
    final_instance = final_agent.parser(**test_data)
    assert final_instance.model_dump() == original_instance.model_dump()
