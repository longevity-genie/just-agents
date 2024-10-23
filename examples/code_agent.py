from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession
from docker.types import Mount
import os

load_dotenv(override=True)

"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands, it uses volumes and is based on Chain Of Thought Agent class.
"""

def make_mounts():
    examples_dir = Path(__file__).parent.absolute()
    assert examples_dir.exists(), f"Examples directory {str(examples_dir)} does not exist, check the current working directory"
    input_dir =  examples_dir / "input"
    output_dir =  examples_dir / "output"
    return [
        Mount(target="/input", source=str(input_dir), type="bind"),
        Mount(target="/output", source=str(output_dir), type="bind")
    ]

def run_bash_command(command: str):
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    mounts = make_mounts()

    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", 
                            lang="python", 
                            keep_template=True, 
                            verbose=True,
                            mounts=mounts
                            ) as session:
        result = session.execute_command(command=command)
        return result
        

def run_python_code(code: str):
    """
    code: str # python code to run in micromamba environment
    """
    mounts = make_mounts()

    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", 
                            lang="python", 
                            keep_template=True, 
                            verbose=True,
                            mounts=mounts
                            ) as session:
        result = session.run(code)
        return result

if __name__ == "__main__":
    examples_dir = Path(__file__).parent.absolute()
    assert examples_dir.exists(), f"Examples directory {str(examples_dir)} does not exist, check the current working directory"

    assistant: LLMSession= build_agent(examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Get FGF2 human protein sequence with biopython from uniprot and save it as FGF2.fasta")
    print("Thoughts: ", thoughts)
    print("Result: ", result)