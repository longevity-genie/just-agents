import pprint
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv

from just_agents.chat_agent import ChatAgent
from just_agents.llm_options import LLAMA3

from loguru import logger
from just_agents.tools.search import literature_search

import typer

app = typer.Typer(no_args_is_help=True)

@app.command()
def search(query: str, limit: int = 10):
    """
    Command to test if hybrid search works at all
    :param query:
    :return:
    """
    logger.add("logs/rag_search.txt", rotation="1 MB")
    logger.info(f"QUERY:\n{query}")
    result: list[str] = literature_search(query, limit = limit)

    logger.info(f"RESULT:\n{result}")
    return result


@app.command()
def rapamycin():
    logger.add("logs/rag_rapamycin.txt", rotation="1 MB")
    load_dotenv()
    scientist: ChatAgent = ChatAgent(llm_options = LLAMA3,
                                    role = "scientist",
                                    goal = "Research the topics in the most comprehensive way, using search in academic literature and providing sources",
                                    task="Address the research question in the most comprehensive way",
                                    tools = [literature_search])

    critic: ChatAgent = ChatAgent(llm_options = LLAMA3,
                                  role = "critic",
                                  goal = "Evaluate the answer according to the criteria provided",
                                  task="Evaluate the answer according to the criteria provided and make recommendations to improve")
    #TODO: write down cryticism

    scientist.memory.add_on_message(lambda m: logger.info(f"MESSAGE: {m}"))
    scientist.memory.add_on_tool_call(lambda t: logger.info(f"FUNCTION: {t}"))

    # setting up relative paths to define output and load prompts
    current_folder: Path = Path(__file__).parent
    example_prompts = current_folder / "example_prompts.yaml"
    output = Path(__file__).parent.parent / "output" / "examples"
    prompts = yaml.safe_load(example_prompts.open("r"))

    rapamycin = prompts["rapamycin_case"]["with_requirements"]

    result = scientist.query(rapamycin, output=output / "rapamycin" / "with_requirements.txt")
    print("RESULT IS:")
    pprint.pprint(result)



if __name__ == "__main__":
    """
    Using typer to add CLI!
    """
    app()