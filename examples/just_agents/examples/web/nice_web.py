from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.agent import IAgent
from just_agents.web.rest_api import *
from just_agents.web.run_agent import *

from pycomfort.logging import to_nice_stdout

load_dotenv(override=True)

web_examples_dir = Path(__file__).parent.absolute()

"""
This example shows how to wrap an agent into a FastAPI endpoint.
You can use it with a 
"""

if __name__ == "__main__":
    to_nice_stdout()
    cofig_path = web_examples_dir / "agent.yaml"
    run_agent_server(config=cofig_path)
