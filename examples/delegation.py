import pprint
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.chat_agent import ChatAgent
from just_agents import llm_options
import copy
from loguru import logger
from just_agents.tools.search import literature_search

import typer

from just_agents.utils import rotate_env_keys

app = typer.Typer(no_args_is_help=True)

def configure_logger(level: str) -> None:
    """Configure the logger to the specified log level."""
    logger.remove()
    logger.add(sys.stdout, level=level)

@app.command()
def search_with_delegation(query: str = "Dose 5-10 mg rapamycin orally provided once per week.", max_tokens: int = 8000, log_level: str = "DEBUG"):
    """
    Command to test if hybrid search works at all
    :param query:
    :return:
    """
    configure_logger(log_level)
    logger.add("logs/search_with_delegation.txt", rotation="1 MB")

    searcher: ChatAgent = ChatAgent(llm_options = llm_options.LLAMA3,
                                    role = "Searcher",
                                    goal = "Your goal is to run search on the the query. You filter only results that consider relevant together with sources.",
                                    task = "Search on the query, filter only relevant results with sources. In the end you you generate the json array with only relevant results",
                                    backstory= "You are an investigative searcher who checks numerous sources and filters out the most relevant once. You are objective, you do not modify the results in any way, if you have your own opinion about the result you can fill in comment field",
                                    tools = [literature_search])
    searcher.memory.add_on_message(lambda m: logger.debug(f"SEARCHER MESSAGE: {m}"))
    result = searcher.query(query, max_tokens=max_tokens, key_getter=rotate_env_keys)

    print(f"RESULTS WAS: {result}")




if __name__ == "__main__":
    """
    Using typer to add CLI!
    """
    app()