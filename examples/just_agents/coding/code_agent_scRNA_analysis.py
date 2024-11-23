from pathlib import Path
from dotenv import load_dotenv
from just_agents.simple.utils import build_agent
from just_agents.simple.cot_agent import ChainOfThoughtAgent
from examples.coding.tools import write_thoughts_and_results
from examples.coding.mounts import input_dir, output_dir, coding_examples_dir

load_dotenv(override=True)

"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands, it uses volumes and is based on Chain Of Thought Agent class.
The task was taken from then https://github.com/JoshuaChou2018/AutoBA library

WARNING: This example is not working stabily, we have to update the prompt to make it work stably
"""

if __name__ == "__main__":
    assistant: ChainOfThoughtAgent= build_agent(coding_examples_dir / "code_agent.yaml")
    result, thoughts = assistant.query("Use squidpy for neighborhood enrichment analysis for "
                                       "'https://github.com/antonkulaga/AutoBA/blob/dev-v1.x.x/examples/case4.1/data/slice1.h5ad', "
                                       "'https://github.com/antonkulaga/AutoBA/blob/dev-v1.x.x/examples/case4.1/data/slice1.h5ad', "
                                       "'https://github.com/antonkulaga/AutoBA/blob/dev-v1.x.x/examples/case4.1/data/slice1.h5ad'"
                                       "that are spatial transcriptomics data for slices 1, 2 and 3 in AnnData format'. Save results as reslult.txt")
    write_thoughts_and_results("scRNA_analysis", thoughts, result)
