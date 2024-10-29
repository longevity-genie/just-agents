from pathlib import Path
from wsgiref.validate import assert_

from dotenv import load_dotenv
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession

from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from examples.coding.tools import write_thoughts_and_results, amino_match_endswith
from examples.coding.mounts import input_dir, output_dir, coding_examples_dir



load_dotenv(override=True)

"""
This example shows how to use a simple code agent to run python code and bash commands, it does not use volumes and is based on basic LLMSession class.
"""

if __name__ == "__main__":
    ref="FLPMSAKS"
    assistant: IAgent = build_agent( coding_examples_dir / "simple_code_agent.yaml")
    result = assistant.query("Get FGF2 human protein sequence from uniprot using biopython")
    assert amino_match_endswith(result, ref), f"Sequence ending doesn't match reference {ref}: {result}"