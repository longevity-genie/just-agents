import json
import pprint

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import BaseAgent

load_dotenv(override=True)

"""
This example demonstrates using the PerplexityAI model, which provides citations
for its responses. PerplexityAI is particularly useful for current events and
fact-based queries as it can access recent information and cite sources.
"""

if __name__ == "__main__":
    # Example query about a recent political event
    prompt = "Who won the US election in 2024?"

    # Initialize the agent with PerplexityAI's Llama 3.1 model
    # This model has a 128k context window and can access online information
    agent = BaseAgent(  # type: ignore
        llm_options=llm_options.PERPLEXITY_LLAMA_3_1_SONAR_LARGE_128K_ONLINE
    )

    # Make the first query about the election winner
    winner = agent.query("Who won the US election in 2024?")

    # Add a callback to print all messages in a pretty format
    # This will show the model's responses including citations
    agent.memory.add_on_message(lambda m: pprint.pprint(m, indent=4))
    print(f"WINNER IS: {winner}")

    # Follow-up query comparing policies
    # Note: There's a typo in 'majort' and 'previos' that should be fixed
    diffs = agent.query(f"What are the majort differences between his policies now and in his previos term?")
    print(f"DIFFERENCES: {diffs}")

    # Final query using the original prompt
    result = agent.query(prompt)
    