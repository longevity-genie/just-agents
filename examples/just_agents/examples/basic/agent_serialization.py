from dotenv import load_dotenv
from pathlib import Path

from just_agents.tools.weather import mock_get_current_weather
from just_agents import llm_options
from just_agents.base_agent import BaseAgent
from just_agents.just_profile import JustAgentProfile

load_dotenv(override=True)

basic_examples_dir = Path(__file__).parent.absolute()

"""
This example shows how to save and load an agent from a yaml file.
In complex use-cases it can be useful to keep agents in yaml files to be able to iterate on them without changing the code.
"""

if __name__ == "__main__":

    config_path = basic_examples_dir / "agent_profiles.yaml"

    created_agent = BaseAgent(
        llm_options=llm_options.LLAMA3_3,
        config_path=config_path,
        tools=[mock_get_current_weather]
    )

    created_agent.save_to_yaml("SimpleWeatherAgent")

    #auto load example
    agent_auto = JustAgentProfile.auto_load("SimpleWeatherAgent", file_path=config_path)
    print(agent_auto)
    assert isinstance(agent_auto, JustAgentProfile) #just testing that types are correct
    assert isinstance(agent_auto, BaseAgent)
    assert agent_auto.to_json() == created_agent.to_json()
    res = agent_auto.query("What's the weather like in San Francisco, Tokyo, and Paris?")
    print("===============")
    print(res)
    #print(agent_auto.to_json())

    #yaml constructor example
    agent = BaseAgent.from_yaml("SimpleWeatherAgent", file_path=config_path)
    print(agent)
    assert isinstance(agent, JustAgentProfile)
    assert isinstance(agent, BaseAgent)
    assert agent.to_json() == created_agent.to_json()
    res = agent.query("What's the weather like in San Francisco, Tokyo, and Paris?")
    print(res)
    #print(agent.to_json())
