import json
import glob
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
import just_agents.llm_options
from just_agents.web.web_agent import WebAgent
from just_agents.web.chat_ui import ModelConfig

# Fixture to load environment variables
@pytest.fixture(scope="module")
def load_env():
    load_dotenv(override=True)


@pytest.mark.skip(reason="fix file locations")
def test_web_agent_profile(load_env, tmp_path):
    test_dir = Path(__file__).parent.absolute()
    config_path = test_dir / "agent.yaml"
    models_dir = test_dir / "models.d"
    models_jsons = models_dir / "*.json"
    agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="example_web_agent", parent_section="")
    agent.write_model_config_to_json(models_dir=models_dir)
#    agent.examples=[WebAgent.BLUE_SKY]
    agent.save_to_yaml()


@pytest.mark.skip(reason="fix file locations")
#@pytest.mark.parametrize("model_file", glob.glob(os.path.join("models.d", "*.json")))
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
    reserialized_data = json.loads(model_instance.model_dump_json())

    # Step 4: Assert equality with the original dictionary
    assert reserialized_data == data, f"Round-trip mismatch in file: {model_file}"