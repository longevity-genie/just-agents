from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from just_agents_coding.micromamba_session import MicromambaSession
from docker.types import Mount
import os

load_dotenv(override=True)

output_dir =  Path.cwd().absolute() / "examples" / "output"


def run_bash_command_with_output(command: str):
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    assert output_dir.exists(), "Output directory does not exist"

    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", 
                            lang="python", 
                            keep_template=True, 
                            verbose=True,
                            mounts=[Mount(target="/tmp", source=str(output_dir))]
                            ) as session:
        result = session.execute_command(command=command)
        return result
        

def run_python_code_with_output(code: str):
    """
    code: str # python code to run in micromamba environment
    """
    assert output_dir.exists(), "Output directory does not exist"

    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", 
                            lang="python", 
                            keep_template=True, 
                            verbose=True,
                            mounts=[Mount(target="/tmp", source=str(output_dir))]
                            ) as session:
        result = session.run(code)
        return result
      

assistant: LLMSession= build_agent("examples/simple_code_agent.yaml")
assistant._prepare_tools( [run_bash_command_with_output, run_python_code_with_output] )
assistant.query("Get FGF2 human protein sequence with biopython from uniprot and save it as output/FGF2.fasta")