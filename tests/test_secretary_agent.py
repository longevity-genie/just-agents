import os
import pytest
import dotenv
import json
import just_agents.llm_options
from just_agents.router.secretary_agent import SecretaryAgent

@pytest.fixture(scope='module')
def temp_config_path(tmpdir_factory):
    # Create a temporary directory for YAML files
    dotenv.load_dotenv(override=True)
    tmpdir_factory.mktemp('config')
    return SecretaryAgent.DEFAULT_CONFIG_PATH

@pytest.fixture
def secretary_autoload_false(temp_config_path):
    dotenv.load_dotenv(override=True)
    params = {
        'autoload_from_yaml': False,
        'config_path': temp_config_path,
        'llm_options': just_agents.llm_options.OPENAI_GPT4oMINI,
    }
    secretary = SecretaryAgent(**params)
    info = secretary.get_info(secretary)
    to_populate = secretary.get_to_populate(secretary)
    result = secretary.update_profile(secretary, info, to_populate, verbose=True)
    secretary.save_to_yaml()
    return secretary, result

@pytest.fixture
def secretary_autoload_true(temp_config_path, secretary_autoload_false):
    # Ensure the YAML file exists from the previous fixture
    dotenv.load_dotenv(override=True)
    params = {
        'model_name': None,
        'llm_options': just_agents.llm_options.OPENAI_GPT4oMINI,
        'extra_dict': {
            "personality_traits": "Agent's personality traits go here",
        },
        'config_path': temp_config_path,
    }
    secretary = SecretaryAgent(**params)
    secretary.update_from_yaml(True)
    info = secretary.get_info(secretary)
    to_populate = secretary.get_to_populate(secretary)
    result = secretary.update_profile(secretary, info, to_populate, verbose=True)
    secretary.save_to_yaml()
    return secretary, result

def test_secretary_autoload_false(secretary_autoload_false):
    dotenv.load_dotenv(override=True)
    secretary, result = secretary_autoload_false
    assert result is True, "Failed to update profile when autoload_from_yaml is False."
    assert secretary.description != secretary.DEFAULT_DESCRIPTION, "Description was not updated."
    assert secretary.llm_model_name is not None, "LLM model name is None."

def test_secretary_autoload_true(secretary_autoload_true):
    dotenv.load_dotenv(override=True)
    secretary, result = secretary_autoload_true
    assert result is True, "Failed to update profile when autoload_from_yaml is True."
    assert secretary.description != secretary.DEFAULT_DESCRIPTION, "Description was not updated."
    assert secretary.extras.get("personality_traits") is not None, "Personality traits were not set."
    assert secretary.llm_model_name is not None, "LLM model name is None."

def test_new_secretary(temp_config_path):
    # Load the secretary from the YAML file created in previous tests
    dotenv.load_dotenv(override=True)
    new_secretary = SecretaryAgent.from_yaml(
        'SecretaryAgent'
    )
    assert new_secretary.role is not None, "Role is None in the loaded secretary."
    assert new_secretary.description is not None, "Description is None in the loaded secretary."
    assert new_secretary.extras.get("personality_traits") is not None, "Personality traits missing in loaded secretary."
    assert new_secretary.llm_model_name is not None, "LLM model name is None in the loaded secretary."
    assert not new_secretary.tools, "Tool names should be empty."
    assert not new_secretary.get_to_populate(new_secretary), "There are still fields to populate."
    print("Results:", json.dumps(new_secretary.to_json(), indent=2))
