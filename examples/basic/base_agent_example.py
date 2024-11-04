from dotenv import load_dotenv
from pathlib import Path

import just_agents.llm_options

from examples.tools.weather import get_current_weather
from just_agents.just_profile import JustAgentProfile
from just_agents.base_agent import BaseAgent

if __name__ == "__main__":
    load_dotenv(override=True)
    tmp_path =Path('./config/')
    config_path = tmp_path / "yaml_initialization_example_new.yaml"
    base_agent = BaseAgent(
        llm_options=just_agents.llm_options.OPENAI_GPT4oMINI,
        tools=[get_current_weather]
    )
    base_agent.save_to_yaml("BaseAgentSchemaExample", file_path=config_path)
    agent = JustAgentProfile.auto_load("BaseAgentSchemaExample",file_path=config_path)
    print(agent)
    assert isinstance(agent, JustAgentProfile)
    assert isinstance(agent, BaseAgent)
    assert agent.to_json() == base_agent.to_json()
    agent.query("What's the weather like in San Francisco, Tokyo, and Paris?")

    print(agent.to_json())
