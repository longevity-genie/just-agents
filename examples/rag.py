import pprint

import loguru
from pathlib import Path

import yaml

from just_agents.chat_agent import ChatAgent
from dotenv import load_dotenv
from just_agents.llm_options import LLAMA3
load_dotenv()
from loguru import logger
from just_agents.tools.search import get_semantic_paper, hybrid_search

scientist: ChatAgent = ChatAgent(llm_options = LLAMA3,
                                role = "scientist",
                                goal = "Research the topics in the most comprehensive way, using search in academic literature and providing sources",
                                task="Address the research question in the most comprehensive way",
                                tools = [hybrid_search])

critic: ChatAgent = ChatAgent(llm_options = LLAMA3,
                              role = "critic",
                              goal = "Evaluate the answer according to the criteria provided",
                              task="Evaluate the answer according to the criteria provided and make recommendations to improve")
#TODO: write down cryticism

scientist.memory.add_on_message(lambda m: logger.info(f"MESSAGE: {m}"))

# setting up relative paths to define output and load prompts
current_folder: Path = Path(__file__).parent
example_prompts = current_folder / "example_prompts.yaml"
output = Path(__file__).parent.parent / "output" / "examples"
promtps = yaml.safe_load(example_prompts.open("r"))

rapamycin = promtps["rapamycin_case"]["with_requirements"]

result = scientist.query(rapamycin, output=output / "rapamycin" / "with_requirements.txt")
print("RESULT IS:")
pprint.pprint(result)