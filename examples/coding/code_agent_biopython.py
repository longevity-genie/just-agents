from pathlib import Path
from dotenv import load_dotenv
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession

from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from examples.coding.tools import write_thoughts_and_results
from examples.coding.mounts import input_dir, output_dir, coding_examples_dir

load_dotenv(override=True)


"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands.
It uses volumes (see tools.py) and is based on Chain Of Thought Agent class.
"""
if __name__ == "__main__":
    assistant: LLMSession = build_agent(coding_examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Get FGF2 human protein sequence with biopython from uniprot and save it as FGF2.fasta")
    write_thoughts_and_results("uniprot", thoughts, result)
    
