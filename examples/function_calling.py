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

async def process_stream(async_generator):
    collected_data = []
    async for item in async_generator:
        collected_data.append(item)
        # You can also process each item here if needed
    return collected_data

llm_options = just_agents.llm_options.OPENAI_GPT4oMINI
key_getter = rotate_env_keys
prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"


session: LLMSession = LLMSession(
    llm_options=llm_options,
    tools=[get_current_weather]
)
session.memory.add_on_message(lambda m: pprint.pprint(m) if "content" in m is not None else None)
session.query(prompt)

print("And now same query but async mode for streaming. Note: we use asyncio.run here to run the stream")
stream = session.stream(prompt)
result = asyncio.run(process_stream(stream))
print("stream finished")