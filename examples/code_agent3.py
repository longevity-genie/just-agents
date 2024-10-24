from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession
from docker.types import Mount
import requests
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


def download_file(source_url:str, file_name:str):
    """ Download file from source_url and save it to '/input' folder with file_name that available mount for runtime. """
    examples_dir = Path(__file__).parent.absolute()
    input_path = Path(examples_dir, "input", file_name)
    try:
        print("!!!!!Downloding: ",source_url, " to ", input_path)
        response = requests.get(source_url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        with open(input_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"File downloaded successfully and saved to {input_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    examples_dir = Path(__file__).parent.absolute()
    assert examples_dir.exists(), f"Examples directory {str(examples_dir)} does not exist, check the current working directory"

    assistant: LLMSession= build_agent(examples_dir / "code_agent2.yaml")
    result, thoughts = assistant.query("Use pyvcf3 library to analyze vcf file with human genome. "
                                       "Extract an overall number of SNPs, deletions, and insertions. Show SNPs distribution over chromosomes."
                                       "On this URL you will find vcf file for this task 'https://drive.usercontent.google.com/download?id=13tPNQsVXMtQKFcTeTOZ-bi7QWa9OdPD4'."
                                       " Save results as reslt.txt")
    #("Get FGF2 human protein sequence with biopython from uniprot and save it as FGF2.fasta")
    print("Thoughts: ", thoughts)
    print("Result: ", result)