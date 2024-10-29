from pathlib import Path
from dotenv import load_dotenv

from examples.coding.mounts import make_mounts, input_dir, output_dir, coding_examples_dir
from just_agents.interfaces.IAgent import build_agent, IAgent
load_dotenv(override=True)

"""
This example shows how to use a simple code agent to run python code and bash commands, it does not use volumes and is based on basic LLMSession class.
"""

if __name__ == "__main__":

    assert coding_examples_dir.exists(), f"Examples directory {str(coding_examples_dir)} does not exist, check the current working directory"

    assistant: IAgent = build_agent( coding_examples_dir / "simple_code_agent.yaml")
    assistant.query("Get FGF2 human protein sequence with biopython from uniprot")