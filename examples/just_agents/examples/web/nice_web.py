from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.agent import IAgent
from just_agents.simple.utils import build_agent
from just_agents.web.rest_api import *
from just_agents.web.run import *

load_dotenv(override=True)

web_examples_dir = Path(__file__).parent.absolute()

"""
This example shows how to wrap an agent into a FastAPI endpoint.
You can use it with a 
"""

if __name__ == "__main__":
    cofig_path = web_examples_dir / "nice_agent.yaml"
    run_server(config=cofig_path, agent_parent_section="agent_profiles", agent_section="SimpleWeatherAgent")
