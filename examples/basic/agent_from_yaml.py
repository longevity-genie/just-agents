from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import IAgent
from just_agents.utils import build_agent

load_dotenv(override=True)

basic_examples_dir = Path(__file__).parent.absolute()

"""
This example shows how an agent can be built from a yaml file.
In complex use-cases it can be useful to keep agents in yaml files to be able to iterate on them without changing the code.
"""

if __name__ == "__main__":
    assistant: IAgent = build_agent(basic_examples_dir / "agent_from_yaml.yaml")
    print(assistant.query("Count the number of occurrences of the letter ’L’ in the word - ’LOLLAPALOOZA’."))