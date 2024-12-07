from pathlib import Path
from dotenv import load_dotenv
from just_agents import llm_options
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
from examples.just_agents.examples.tools import letter_count
import pprint

# Get the directory path where this example is located
basic_examples_dir = Path(__file__).parent.absolute()

"""
This example shows chain of thought reasoning agent.
"""

if __name__ == "__main__":
    # Load environment variables, allowing them to override existing ones
    load_dotenv(override=True)

    # Create a list of tools that the agent can use
    # In this case, only the letter_count tool is provided
    tools = [letter_count]

    # Initialize the Chain of Thought Agent with:
    # - the tools it can use
    # - LLAMA 3.2 Vision as the language model
    agent: ChainOfThoughtAgent = ChainOfThoughtAgent(  # type: ignore
        tools=tools,
        llm_options=llm_options.LLAMA3_3
    )

    # Add a callback to print all messages that the agent processes
    agent.memory.add_on_message(lambda message: print(message))

    # Ask the agent to solve a problem
    # The think() method returns two things:
    # - result: The final answer
    # - chain: The step-by-step reasoning process (list of thoughts)
    (result, chain) = agent.think("Count the number of occurrences of the letter 'L' in the word - 'LOLLAPALOOZA'.")
    
    # Print the final result and the chain of thoughts
    print("==========Final result:==========")
    print(result)
    print("==========Chain of thoughts:==========")
    pprint.pprint(chain)
    file_path = basic_examples_dir / "agent_profiles.yaml"
    agent.save_to_yaml("ChainOfThoughtAgent", file_path=file_path)

