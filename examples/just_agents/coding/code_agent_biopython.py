from dotenv import load_dotenv

from just_agents.simple.utils import build_agent
from examples.coding.tools import write_thoughts_and_results
from examples.coding.mounts import coding_examples_dir
from just_agents.simple.cot_agent import ChainOfThoughtAgent

load_dotenv(override=True)


"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands.
It uses volumes (see tools.py) and is based on Chain Of Thought Agent class.
Progress can be checked in the /output folder.
There will be a code file with the code that AI has written, 
a thoughts file with the thoughts 
and a results file with the results.

"""
if __name__ == "__main__":
    assistant: ChainOfThoughtAgent = build_agent(coding_examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Get a random FGF2_HUMAN protein sequence with biopython from uniprot and save it as FGF2.fasta")
    write_thoughts_and_results("uniprot", thoughts, result)
    