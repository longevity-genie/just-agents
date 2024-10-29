from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession
from docker.types import Mount
import os

load_dotenv(override=True)

from examples.coding.mounts import make_mounts, input_dir, output_dir, coding_examples_dir

"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands. 
It uses volumes (see tools.py) and is based on Chain Of Thought Agent class.
"""

if __name__ == "__main__":
    assert coding_examples_dir.exists(), f"Examples directory {str(coding_examples_dir)} does not exist, check the current working directory"

    assistant: LLMSession= build_agent(coding_examples_dir / "code_agent.yaml")
    query = "Take two nutritional datasets (GSE176043 and GSE41781) and three partial reprogramming datasets (GSE148911, GSE190986 and GSE144600), download them from GEO and generate PCA plot with them in /output folder"
    result, thoughts = assistant.query(query)
   
