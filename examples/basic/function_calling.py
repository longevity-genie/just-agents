import json
import pprint

from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import BaseAgent

load_dotenv(override=True)

"""
This example shows how a function can be used to call a function which potentially can have an external API call.
"""

def get_current_weather(location: str):
    """Gets the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


if __name__ == "__main__":

    prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"

    load_dotenv(override=True)

    agent = BaseAgent(
        llm_options=llm_options.LLAMA3_2,
        tools=[get_current_weather]
    )
    result = agent.query(prompt)

    agent.memory.add_on_message(lambda m: pprint.pprint(m))
    result = agent.query(prompt)
    print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    print(result)
