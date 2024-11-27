from dotenv import load_dotenv
from just_agents.simple.cot_agent import ChainOfThoughtAgent
from just_agents.simple.utils import build_agent

from just_agents.examples.coding.mounts import coding_examples_dir
from just_agents.simple.utils import build_agent

load_dotenv(override=True)

"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands. 
It uses volumes (see tools.py) and is based on Chain Of Thought Agent class.
Note: current example is a work in progress and the task is too complex to get it solved in one go.


WARNING: This example is not working as expected, some of GSE-s are messed up
"""

if __name__ == "__main__":
    agent = build_agent(coding_examples_dir / "bioinformatic_agent.yaml")
    query_GSE137317 = "Download gene counts from GSE137317, split them by conditions, make PCA plot and differential expression analysis using only python libraries"
    #query_GSE144600 = "Download gene counts from GSE144600"
    #query_two = "add GSE137317 and GSE144600 to the same PCA plot"
    
    #query = "Take two nutritional datasets (GSE176043 and GSE41781) and three partial reprogramming datasets (GSE148911, GSE190986 and GSE144600), download them from GEO and generate PCA plot with them in /output folder"
    result, thoughts = agent.query(query_GSE137317)
   
