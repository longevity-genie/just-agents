import json
import pprint

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import BaseAgent
from just_agents.examples.tools import get_current_weather

load_dotenv(override=True)

"""
This example shows how a function can be used to call a function which potentially can have an external API call.
"""

if __name__ == "__main__":
    # Example prompt asking about weather in multiple cities
    prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"

    # Create an agent instance with:
    # 1. LLAMA3_3 as the language model
    # 2. get_current_weather function as an available tool
    agent = BaseAgent(  # type: ignore
        llm_options=llm_options.GEMINI_2_5_FLASH, #.GEMINI_2_5_FLASH,
        tools=[get_current_weather]
    )
    
   
    # Add a callback to print messages using pprint
    # This will show the internal conversation/function calls
    agent.memory.add_on_message(lambda m: pprint.pprint(m))
    
    # query with memory callback enabled
    result = agent.query(prompt)
    print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    print(result)
