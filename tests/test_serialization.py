import pytest
from dotenv import load_dotenv
from pathlib import Path
import just_agents.llm_options
from examples.tools.weather import get_current_weather

from just_agents.just_agent import JustAgent
from just_agents.just_profile import JustAgentProfile

from typing import Optional, Dict, Any, Sequence, Union, Set, Callable, List, Tuple

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
    agent = JustAgent(
        llm_options=just_agents.llm_options.OPENAI_GPT4oMINI,
        config_path=config_path,
        tools=[get_current_weather],
    )

    # Save the agent to YAML
    agent.save_to_yaml("SimpleWeatherAgent")

    # Load the agent from YAML
    loaded_agent = JustAgent.from_yaml("SimpleWeatherAgent", file_path=config_path)

    # Assert that the loaded agent matches the original
    assert loaded_agent.to_json() == agent.to_json()

    # Optionally, print the JSON representation
    print(loaded_agent.to_json())
