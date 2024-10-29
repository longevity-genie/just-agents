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
This example shows how to use a Chain Of Thought code agent to run python code and bash commands, it uses volumes and is based on Chain Of Thought Agent class.
"""

if __name__ == "__main__":
    assistant: LLMSession= build_agent(coding_examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Use squidpy for neighborhood enrichment analysis for "
                                       "'https://github.com/antonkulaga/AutoBA/blob/dev-v1.x.x/examples/case4.1/data/slice1.h5ad', "
                                       "'https://github.com/antonkulaga/AutoBA/blob/dev-v1.x.x/examples/case4.1/data/slice1.h5ad', "
                                       "'https://github.com/antonkulaga/AutoBA/blob/dev-v1.x.x/examples/case4.1/data/slice1.h5ad'"
                                       "that are spatial transcriptomics data for slices 1, 2 and 3 in AnnData format'. Save results as reslult.txt")
    write_thoughts_and_results("scRNA_analysis", thoughts, result)
