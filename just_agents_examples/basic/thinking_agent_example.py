from pathlib import Path
from dotenv import load_dotenv
from just_agents.llm_options import LLAMA3_2_VISION
from just_agents.thinking import ChainOfThoughtAgent
from just_agents_examples.tools import letter_count
import pprint


basic_examples_dir = Path(__file__).parent.absolute()

"""
This example shows chain of thought reasoning agent.
"""


if __name__ == "__main__":

    load_dotenv(override=True)
    tools = [letter_count]
    agent: ChainOfThoughtAgent = ChainOfThoughtAgent(tools=tools, llm_options=LLAMA3_2_VISION)
    agent.memory.add_on_message(lambda message: print(message))
    (result, chain) = agent.think("Count the number of occurrences of the letter 'L' in the word - 'LOLLAPALOOZA'.")
    #print("Chain of thoughts:")
    #pprint.pprint(chain)
    print("========== Final result: ==========")
    print(result)
