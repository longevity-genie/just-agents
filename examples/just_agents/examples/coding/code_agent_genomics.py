from dotenv import load_dotenv

from just_agents.simple.utils import build_agent
from just_agents.simple.cot_agent import ChainOfThoughtAgent
from just_agents.examples.coding.tools import write_thoughts_and_results
from just_agents.examples.coding.mounts import coding_examples_dir

load_dotenv(override=True)

"""
This example shows how to use code generation for genomic tasks
"""
if __name__ == "__main__":
    assistant: ChainOfThoughtAgent = build_agent(coding_examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Analyze vcf file with human genome using vcflib python binding."
                                       "Extract an overall number of SNPs, deletions, and insertions. Show SNPs distribution over chromosomes."
                                       "On this URL you will find vcf file for this task 'https://drive.usercontent.google.com/download?id=13tPNQsVXMtQKFcTeTOZ-bi7QWa9OdPD4'."
                                       " Save results as genomic_result.txt")
    write_thoughts_and_results("genomics", thoughts, result)