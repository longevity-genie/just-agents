from pathlib import Path
from dotenv import load_dotenv
from just_agents.interfaces.IAgent import build_agent, IAgent
from just_agents.llm_session import LLMSession
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import SandboxDockerSession
from docker.types import Mount

from examples.coding.tools import write_thoughts_and_results

load_dotenv(override=True)
coding_examples_dir = Path(__file__).parent.absolute()
output_dir = coding_examples_dir / "output"

if __name__ == "__main__":
    assert coding_examples_dir.exists(), f"Examples directory {str(coding_examples_dir)} does not exist, check the current working directory"

    assistant: LLMSession= build_agent(coding_examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Analyze vcf file with human genome using vcflib python binding."
                                       "Extract an overall number of SNPs, deletions, and insertions. Show SNPs distribution over chromosomes."
                                       "On this URL you will find vcf file for this task 'https://drive.usercontent.google.com/download?id=13tPNQsVXMtQKFcTeTOZ-bi7QWa9OdPD4'."
                                       " Save results as genomic_result.txt")
    write_thoughts_and_results("genomics", thoughts, result)