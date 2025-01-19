from dotenv import load_dotenv
from pathlib import Path
from just_agents.examples.coding.cot_dev import ChainOfThoughtDevAgent


load_dotenv(override=True)

"""
This example shows how to use a Chain Of Thought code agent to run python code and bash commands. 
It uses volumes (see tools.py) and is based on Chain Of Thought Agent class.
Note: current example is a work in progress and the task is too complex to get it solved in one go.


WARNING: This example is not working as expected, some of GSE-s are messed up
"""

if __name__ == "__main__":
    current_dir = Path(__file__).parent.absolute()

    #test_agent = ChainOfThoughtDevAgent(llm_options=OPENAI_GPT4o)
    #test_agent.save_to_yaml(file_path=current_dir/"agent_profiles.yaml", exclude_unset=False, exclude_defaults=False)

    bio_coder : ChainOfThoughtDevAgent= ChainOfThoughtDevAgent.convert_from_legacy(
        Path(current_dir/"bioinformatic_dev_agent.yaml"),
        Path(current_dir/"cot_dev_agent_profiles.yaml"),
        ChainOfThoughtDevAgent,
        "bioinformatic_cot_agent",
       )

    dev_ops : ChainOfThoughtDevAgent= ChainOfThoughtDevAgent.convert_from_legacy(
        Path(current_dir / "devops_agent.yaml"),
        Path(current_dir / "cot_dev_agent_profiles.yaml"),
        ChainOfThoughtDevAgent,
        "devops_cot_agent",
    )

    query_strong = "Take two nutritional datasets (GSE176043 and GSE41781) and three partial reprogramming datasets (GSE148911, GSE190986 and GSE144600), download them from GEO and generate PCA plot with them in /output folder"
    testrun=5

    #for _ in range(testrun):
    result, cot = bio_coder.think(query_strong)
    bio_coder.thoughts.save_to_yaml("bio_coder_thoughts",file_path=Path(current_dir / "thoughts.yaml"), exclude_unset=False, exclude_defaults=False)

    #agent = build_agent(coding_examples_dir / "bioinformatic_agent.yaml")
    #query_GSE137317 = "Download gene counts from GSE137317, split them by conditions, make PCA plot and differential expression analysis using only python libraries"
    #query_GSE144600 = "Download gene counts from GSE144600"
    #query_two = "add GSE137317 and GSE144600 to the same PCA plot"
    
    #query = "Take two nutritional datasets (GSE176043 and GSE41781) and three partial reprogramming datasets (GSE148911, GSE190986 and GSE144600), download them from GEO and generate PCA plot with them in /output folder"
    #result, thoughts = agent.query(query_GSE137317)
   
