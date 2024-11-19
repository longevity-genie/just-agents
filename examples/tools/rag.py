import pprint
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv

from just_agents.chat_agent import ChatAgent
from just_agents.llm_options import LLAMA3_2
import copy
from just_agents_tools.search import literature_search

import typer
from typer import echo

from just_agents.utils import rotate_env_keys

app = typer.Typer(no_args_is_help=True)

def configure_output(level: str) -> None:
    """Configure the output verbosity."""
    # You might want to implement this based on the level
    pass

@app.command()
def search(query: str, limit: int = 10):
    """
    Command to test if hybrid search works at all
    :param query:
    :return:
    """
    echo(f"QUERY:\n{query}")
    result: list[str] = literature_search(query, limit = limit)

    echo(f"RESULT:\n{result}")
    return result

@app.command()
def rapamycin(prompt_name: str = "rapamycin_case", sub_prompt: str = "with_requirements", output_level: str = "DEBUG"):
    configure_output(output_level)
    load_dotenv()

    # setting up relative paths to define output and load prompts
    current_folder: Path = Path(__file__).parent
    example_prompts = current_folder / "example_prompts.yaml"
    output = Path(__file__).parent.parent / "output" / "examples"
    prompts = yaml.safe_load(example_prompts.open("r"))
    question = prompts[prompt_name][sub_prompt]


    scientist: ChatAgent = ChatAgent(llm_options = LLAMA3_2,
                                    role = "scientist",
                                    goal = "Research the topics in the most comprehensive way, using search in academic literature and providing sources",
                                    task="Address the research question in the most comprehensive way",
                                    tools = [literature_search])

    # ADDING ECHO HANDLERS:
    scientist.memory.add_on_message(lambda m: echo(f"SCIENTIST MESSAGE: {m}"))
    scientist.memory.add_on_tool_call(lambda f: echo(f"SCIENTIST FUNCTION: {f}"))
    scientist.memory.add_on_tool_message(lambda m: echo(f"SCIENTIST TOOL result from {m.name} with tool call id {m.tool_call_id} is {m.content}"))

    answer = scientist.query(question)
    output_path = output / prompt_name / f"{sub_prompt}_initial_answer.txt"
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(answer)
    echo(f"INITIAL ANSWER: {answer}")

    for_review = f"""
    The question that the user asked was: 
    ```
    {question}
    ```
    The answer that the scientist gave was: 
    ```
    {answer}
    ```
    """
    pprint.pprint(for_review)

    critic: ChatAgent = ChatAgent(llm_options = copy.deepcopy(LLAMA3_2),
                                  role = "critic",
                                  goal = "criticise answers to questions, provide evaluations and improvements",
                                  task="evaluate the answer according to the criteria provided and make recommendations to improve")

    # ADDING CRITICS HANDLERS:
    critic.memory.add_on_message(lambda m: echo(f"CRITIC MESSAGE: {m}"))
    #print(f"critic messages: {critic.memory.messages}")

    review_results = critic.query(for_review, output=output / prompt_name / f"{sub_prompt}_answer_review.txt")
    echo(f"REVIEW RESULTS: {review_results}")

if __name__ == "__main__":
    """
    Using typer to add CLI!
    """
    app()