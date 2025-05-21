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
    prompt = "Who won Romanian presidential election in 2025?"

    # Initialize the agent with PerplexityAI's Llama 3.1 model
    # This model has a 128k context window and can access online information
    agent = BaseAgent(  # type: ignore
        llm_options=llm_options.PERPLEXITY_SONAR_PRO
    )

    # Make the first query about the election winner
    winner = agent.query(prompt)

   
    result = agent.query(prompt)
    print("=================\nWHO WON: \n", result)
    query = """
    Based on the 2025 election campaign, can you collect the memes about Simion? 
    Write down his character profile which will include his personality, style of talking, quotes of his famous speeches and memes, description of how he would behave in debates.
    Search bot in english and in romanian.
    """
    profile_Simion = agent.query(query)
    print("=================\nPROFILE OF SIMION: \n", profile_Simion)
    profile_Nicusor_Dan = agent.query(query.replace("Simion", "Nicusor Dan"))
    print("=================\nPROFILE OF NICUSOR DAN: \n", profile_Nicusor_Dan)
    agent.memory.pretty_print_all_messages()
    