from pathlib import Path
from dotenv import load_dotenv

from just_agents.interfaces.IAgent import build_agent, IAgent
load_dotenv(override=True)

"""
This example shows how to use a simple code agent to run python code and bash commands, it does not use volumes and is based on basic LLMSession class.
"""


if __name__ == "__main__":

    examples_dir = Path(__file__).parent.absolute()
    assert examples_dir.exists(), f"Examples directory {str(examples_dir)} does not exist, check the current working directory"

    assistant: IAgent = build_agent( examples_dir / "simple_code_agent.yaml")
    assistant.query("Get FGF2 human protein sequence with biopython from uniprot")