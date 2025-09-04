import json
import glob
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
import just_agents.llm_options
from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui_agent import ChatUIAgent
from just_agents.web.web_agent import WebAgent
from just_agents.web.chat_ui import ModelConfig
from just_agents.web.run_agent import validate_agent_config
from just_agents.web.chat_ui_rest_api import ChatUIAgentRestAPI
from just_agents.web.config import ChatUIAgentConfig, BaseModel
from just_agents.just_bus import JustLogBus

TESTS_DIR = os.path.dirname(__file__)  # Get the directory where this test file is located
MODELS_DIR = os.path.join(TESTS_DIR, "models.d")  # Path to models.d inside tests

# Fixture to load environment variables
@pytest.fixture(scope="module")
def load_env():
    load_dotenv(override=True)


def test_web_agent_profile(load_env, tmp_path):

    config_path = Path(TESTS_DIR)   / "profiles" / "web_agent.yaml"
    os.environ["TMP_DIR"] = str(tmp_path)
    agent: WebAgent = WebAgent.from_yaml_auto(file_path=config_path, section_name="example_web_agent", parent_section="agent_profiles")
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
   

def test_google_search_tool(load_env, tmp_path):
    config_path = Path(TESTS_DIR)  / "profiles" / "tool_problem.yaml"
    agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="sugar_genie_generic_google_search", parent_section="agent_profiles")
    result = agent.query("Which model of insulin pumps the users found to be more user friendly?")
    assert "Omnipod" in result or "Tandem" in result
    result = agent.query("Who is a founder of GlucoseDAO?")
    assert "Livia Zaharia" in result



def test_prompt_functions_and_agent_locator(load_env, tmp_path):
    config_path = Path(TESTS_DIR)  / "profiles" / "secret_service.yaml"
    key_agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="keys_agent", parent_section="agent_profiles")
    agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="secret_agent", parent_section="agent_profiles")
    result = agent.query("Decipher me the message please - 'JBYEF0QTGV9IPRIAXEpI' ")
    print(key_agent.shortname) # keep in local scope
    assert key_agent.shortname == "keys_agent"
    assert agent.shortname == "secret_agent"
    assert "Wake up, Neo..." in result

def test_fallback(load_env, tmp_path):
    config_path = Path(TESTS_DIR)  / "profiles" / "secret_service.yaml"
    key_agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="keys_agent", parent_section="agent_profiles")
    secret_agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="secret_agent", parent_section="agent_profiles")
    general_agent: WebAgent = WebAgent.from_yaml(file_path=config_path, section_name="generic_assistant", parent_section="agent_profiles")
    blue_sky = general_agent.query("When the sky is blue?")
    assert "scattering" in blue_sky
    result = general_agent.query("Decipher me the message please - 'JBYEF0QTGV9IPRIAXEpI' ")
    assert key_agent.shortname == "keys_agent"
    assert secret_agent.shortname == "secret_agent"
    assert "Wake up, Neo..." in result


def _test_tool_description_helper(
    tmp_path: Path, 
    use_litellm: bool, 
    section_name: str = "rag_agent", 
    query: str = "Search me some mushrooms in recent clinical trials?"
) -> str:
    """
    Helper function to test tool description functionality
    
    Args:
        tmp_path: pytest fixture for temporary directory
        use_litellm: whether to use litellm tool description
        section_name: name of the agent section in the YAML file (default: "rag_agent")
        query: query string to send to the agent (default: "Search me some mushrooms in recent clinical trials?")
    
    Returns:
        str: query response from the agent
    """
    config_path = Path(TESTS_DIR) / "profiles" / "tool_problem.yaml"
    rag_agent: WebAgent = WebAgent.from_yaml(
        file_path=config_path, 
        section_name=section_name, 
        parent_section="agent_profiles"
    )
    rag_agent.litellm_tool_description = use_litellm
    return rag_agent.query(query)

def test_tool_description_litellm(load_env, tmp_path):
    """Test tool description using litellm implementation"""
    query = _test_tool_description_helper(tmp_path, use_litellm=True)
    assert "ample content" in query
    assert "_trials/document" in query
    query = _test_tool_description_helper(
        tmp_path,
        use_litellm=True,
        section_name="sugar_genie_tool_problem_search",
        query="Please show me some clinical trials about diabetes?"
    )
    assert "beta_cell_pathophysiology" in query

def test_tool_description_builtin(load_env, tmp_path):
    """Test tool description using built-in implementation"""
    query = _test_tool_description_helper(tmp_path, use_litellm=False)
    assert "ample content" in query
    assert "_trials/document" in query
    query = _test_tool_description_helper(
        tmp_path, 
        use_litellm=False, 
        section_name="sugar_genie_tool_problem_search", 
        query="Please show me some clinical trials about diabetes?"
    )
    assert "beta_cell_pathophysiology" in query


def test_web_agents(load_env, tmp_path):
    config_path = Path(TESTS_DIR)  / "profiles" / "web_agent.yaml"
    agents : dict[str,BaseAgent] = WebAgent.from_yaml_dict(yaml_path=config_path, parent_section="agent_profiles")
    for name, agent in agents.items():
        assert name == agent.shortname
        if isinstance(agent, ChatUIAgent):
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

def test_agent_config(load_env):
    os.environ["REMOVE_DD_CONFIGS"] = "test"
    os.environ["AGENT_USE_PROXY"] = "True"
    os.environ["AGENT_CONFIG_PATH"] = "testt.yaml"
    os.environ["MODELS_DIR"] = str(Path(TESTS_DIR) / "models.d")
    os.environ["ENV_MODELS_PATH"] = str(Path(TESTS_DIR) / "env" / ".env.local")
    env_config = ChatUIAgentConfig()
    assert env_config.remove_dd_configs == False
    assert env_config.agent_config_path == "testt.yaml"
    assert env_config.models_dir == str(Path(TESTS_DIR) / "models.d")

def test_validate_agent_config(load_env):
    """Test the validate_agent_config function with regular AgentRestAPI"""
    config_path = Path(TESTS_DIR) / "profiles" / "agent_profiles.yaml"
    env_config = ChatUIAgentConfig()
    # Test successful validation
    api = validate_agent_config(
        config=config_path,
        parent_section="agent_profiles",
        debug=True
    )
    assert api.title == "Just-Agent endpoint"
    # Convert agents to list before checking to make debugging easier
    agents_list = list(api.agents)
    # Add debug print to see what we're actually getting

    assert all(isinstance(agent, BaseAgent) for agent in api.agents.values())
    assert any(isinstance(agent, WebAgent) for agent in api.agents.values())
    assert any(agent.__class__.__name__ == "BaseAgent" for agent in api.agents.values())
    # Test with non-existent config file
    with pytest.raises(FileNotFoundError):
        validate_agent_config(config=Path("nonexistent.yaml"))

    

def test_validate_chat_ui_config(load_env):
    """Test the validate_agent_config function with ChatUIAgentRestAPI"""
    config_path = Path(TESTS_DIR) / "profiles" / "chat_agent_profiles.yaml"
    os.environ["MODELS_DIR"] = str(Path(TESTS_DIR) / "models.d")
    os.environ["ENV_KEYS_PATH"] = str(Path(TESTS_DIR) / "env" / ".env.keys")
    os.environ["REMOVE_DD_CONFIGS"] = "test"

    # Test successful validation
    api = validate_agent_config(
        config=config_path,
        parent_section="agent_profiles",
        api_class=ChatUIAgentRestAPI,
        debug=True
    )
    assert all(isinstance(agent, WebAgent) for agent in api.agents.values())
    assert all(isinstance(agent, ChatUIAgent) for agent in api.agents.values())
    assert isinstance(api, ChatUIAgentRestAPI)