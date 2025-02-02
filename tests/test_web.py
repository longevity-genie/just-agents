import json
import glob
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
import just_agents.llm_options
from just_agents.web.web_agent import WebAgent
from just_agents.web.chat_ui import ModelConfig

TESTS_DIR = os.path.dirname(__file__)  # Get the directory where this test file is located
MODELS_DIR = os.path.join(TESTS_DIR, "models.d")  # Path to models.d inside tests

# Fixture to load environment variables
@pytest.fixture(scope="module")
def load_env():
    load_dotenv(override=True)


def test_web_agent_profile(load_env, tmp_path):

    config_path = Path(TESTS_DIR)   / "profiles" / "web_agent.yaml"

    agent: WebAgent = WebAgent.from_yaml_auto(file_path=config_path, section_name="example_web_agent", parent_section="agent_profiles")
    agent.write_model_config_to_json(models_dir=MODELS_DIR)
    agent.save_to_yaml()

def test_web_agent_tool(load_env, tmp_path):

    config_path = Path(TESTS_DIR)  / "profiles" / "tool_problem.yaml"
    try:
        bad_agent = WebAgent.from_yaml(file_path=config_path, section_name="sugar_genie_bad",
                                       parent_section="agent_profiles")
    except ValueError as e:
        assert "Tools mismatch" in str(e)
    agent_good: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="sugar_genie_good", parent_section="agent_profiles")
    assert "Zaharia" in agent_good.query("Who is the founder of GlucoseDAO?")
    ill_agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="sugar_genie_good", parent_section="agent_profiles")
    assert "Zaharia" in ill_agent.query("Who is the founder of GlucoseDAO?")




def test_web_agents(load_env, tmp_path):
    config_path = Path(TESTS_DIR)  / "profiles" / "web_agent.yaml"
    agents : dict[str,WebAgent] = WebAgent.from_yaml_dict(yaml_path=config_path, parent_section="agent_profiles")
    for name, agent in agents.items():
        assert name == agent.shortname
        agent.write_model_config_to_json(models_dir=MODELS_DIR)

@pytest.mark.parametrize("model_file", glob.glob(os.path.join(MODELS_DIR, "*.json")))
def test_models_valid(model_file):
    """
    Test that each .json file in models.d:
    1. Loads without error.
    2. Successfully validates against ModelConfig via model_validate.
    3. Round-trips (dict -> model -> dict) without changing data.
    """
    # Step 1: Load the JSON data
    with open(model_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 2: Validate using Pydantic's model_validate (Pydantic v2)
    model_instance = ModelConfig.model_validate(data)

    # Step 3: Serialize back to JSON and then parse into dict for comparison
    reserialized_data = json.loads(model_instance.model_dump_json(exclude_none=True))

    # Step 4: Assert equality with the original dictionary
    assert reserialized_data == data, f"Round-trip mismatch in file: {model_file}"