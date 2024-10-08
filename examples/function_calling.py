#from just_agents.chat_agent import ChatAgent
import asyncio
import json
import pprint

from dotenv import load_dotenv

import just_agents.llm_options
from just_agents.llm_session import LLMSession
from just_agents.utils import rotate_env_keys

load_dotenv()

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

llm_options = just_agents.llm_options.LLAMA3_1
key_getter = rotate_env_keys
prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"


session: LLMSession = LLMSession(
    llm_options=llm_options,
    tools=[get_current_weather]
)
session.memory.add_on_message(lambda m: pprint.pprint(m) if "content" in m is not None else None)
session.query(prompt)

print("And now same query but async mode for streaming. Note: we use asyncio.run here to run the stream")

result = asyncio.run(session.stream_async(prompt))
print("stream finished")