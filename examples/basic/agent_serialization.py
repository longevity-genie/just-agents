from dotenv import load_dotenv
from pathlib import Path

from examples.tools.weather import get_current_weather

import just_agents.llm_options
from just_agents.just_agent import JustAgent
from just_agents.just_profile import JustAgentProfile

load_dotenv(override=True)

basic_examples_dir = Path(__file__).parent.absolute()

"""
This example shows how to save and load an agent from a yaml file.
"""



if __name__ == "__main__":

    config_path = basic_examples_dir / "agent_from_yaml.yaml"

    created_agent = JustAgent(
        llm_options=just_agents.llm_options.OPENAI_GPT4oMINI,
        config_path=config_path,
        tools=[get_current_weather]
    )

    created_agent.save_to_yaml("SimpleWeatherAgent")

    loaded_agent = JustAgent.from_yaml("SimpleWeatherAgent", file_path=config_path)
    result = loaded_agent.query(
        "What's the weather like in San Francisco, Tokyo, and Paris?"
    )
    print (result)
