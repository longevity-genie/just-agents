from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.agent import IAgent
from just_agents.simple.utils import build_agent
from just_agents_web.web import create_app, run_server

load_dotenv(override=True)

web_examples_dir = Path(__file__).parent.absolute()

"""
This example shows how to wrap an agent into a FastAPI endpoint.
You can use it with a 
"""

if __name__ == "__main__":
    cofig_path = web_examples_dir / "nice_agent.yaml"
    run_server(config_path=cofig_path)