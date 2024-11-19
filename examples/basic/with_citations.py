import json
import pprint

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import BaseAgent

load_dotenv(override=True)

"""
This exampel requires a PerplexityAI API key.
"""
if __name__ == "__main__":

    prompt = "Who won the US election in 2024?"

    load_dotenv(override=True)

    agent = BaseAgent(
        llm_options=llm_options.PERPLEXITY_LLAMA_3_1_SONAR_LARGE_128K_ONLINE
    )
    winner = agent.query("Who won the US election in 2024?")
    agent.memory.add_on_message(lambda m: pprint.pprint(m, indent=4))
    print(f"WINNER IS: {winner}")
    diffs = agent.query(f"What are the majort differences between his policies now and in his previos term?")
    print(f"DIFFERENCES: {diffs}")
    result = agent.query(prompt)
    