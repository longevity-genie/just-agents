import pprint
import sys
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

def configure_logger(level: str) -> None:
    """Configure the logger to the specified log level."""
    logger.remove()
    logger.add(sys.stdout, level=level)

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
def rapamycin(prompt_name: str = "rapamycin_case", sub_prompt: str = "with_requirements", log_level: str = "INFO"):
    configure_logger(log_level)
    logger.add("logs/rag_rapamycin.txt", rotation="1 MB")
    load_dotenv()


    # setting up relative paths to define output and load prompts
    current_folder: Path = Path(__file__).parent
    example_prompts = current_folder / "example_prompts.yaml"
    output = Path(__file__).parent.parent / "output" / "examples"
    prompts = yaml.safe_load(example_prompts.open("r"))
    question = prompts[prompt_name][sub_prompt]


    scientist: ChatAgent = ChatAgent(llm_options = LLAMA3,
                                    role = "scientist",
                                    goal = "Research the topics in the most comprehensive way, using search in academic literature and providing sources",
                                    task="Address the research question in the most comprehensive way",
                                    tools = [literature_search])

    # ADDING LOGGER HANDLERS:
    scientist.memory.add_on_message(lambda m: logger.debug(f"SCIENTIST MESSAGE: {m}"))
    scientist.memory.add_on_tool_call(lambda f: logger.debug(f"SCIENTIST FUNCTION: {f}"))
    scientist.memory.add_on_tool_result(lambda m: logger.debug(f"SCIENTIST TOOL result from {m.name} with tool call id {m.tool_call_id} is {m.content}"))


    critic: ChatAgent = ChatAgent(llm_options = LLAMA3,
                              role = "critic",
                              goal = "Evaluate the answer according to the criteria provided",
                              task="Evaluate the answer according to the criteria provided and make recommendations to improve")

    # ADDING CRITICS HANDLERS:
    critic.memory.add_on_message(lambda m: logger.debug(f"CRITIC MESSAGE: {m}"))

    answer = scientist.query(question, output=output / prompt_name / f"{sub_prompt}_initial_answer.txt")
    logger.info(f"INITIAL ANSWER: {answer}")

    for_review = f"""
    The question that scientist asked was: {question}
    The answer that she gave was: {answer}
    """

    review_results = critic.query(for_review, output=output / prompt_name / f"{sub_prompt}_answer_review.txt")
    logger.info(f"REVIEW RESULTS: {review_results}")



if __name__ == "__main__":
    """
    Using typer to add CLI!
    """
    app()