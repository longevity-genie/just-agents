import pprint
from dotenv import load_dotenv

from just_agents.base_agent import BaseAgent
from just_agents.examples.coding.mounts import coding_examples_dir

load_dotenv(override=True)

"""
This example shows how to use a simple code agent to run python code and bash commands, it does not use volumes and is based on basic LLMSession class.
"""

if __name__ == "__main__":
    ref="FLPMSAKS"
    #here we use claude sonnet 3.5 as default mode, if you want to switch to another please edit yaml
    config_path = coding_examples_dir / "coding_agents.yaml"
    agent: BaseAgent = BaseAgent.from_yaml("SimpleCodeAgent", file_path=config_path)
    result = agent.query("Get FGF2 human protein sequence from uniprot using biopython. As a result, return only the sequence")
    print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    pprint.pprint(result)
    #assert amino_match_endswith(result, ref), f"Sequence ending doesn't match reference {ref}: {result}"