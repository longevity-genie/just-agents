#from just_agents.chat_agent import ChatAgent
import asyncio
import json
import pprint

from dotenv import load_dotenv

import just_agents.llm_options
from just_agents.llm_session import LLMSession
from just_agents.utils import rotate_env_keys

load_dotenv(override=True)

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

    llm_options = just_agents.llm_options.LLAMA3_2

    key_getter = rotate_env_keys
    prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"

    load_dotenv(override=True)
    session: LLMSession = LLMSession(
        llm_options=just_agents.llm_options.LLAMA3_2,
        tools=[get_current_weather]
    )
    result = session.query("What's the weather like in San Francisco, Tokyo, and Paris?")


    session.memory.add_on_message(lambda m: pprint.pprint(m) if "content" in m is not None else None)
    result = session.query(prompt)

    """
    print("And now same query but async mode for streaming. Note: we use asyncio.run here to run the stream")
    stream = session.stream(prompt)
    result = asyncio.run(process_stream(stream))
    print("stream finished")
    """
    print("RESULT+++++++++++++++++++++++++++++++++++++++++++++++")
    print(result)
