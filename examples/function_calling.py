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

#llm_options = just_agents.llm_options.OPENAI_GPT4o
#key_getter=lambda: os.getenv("OPENAI_API_KEY")

#QWEN 2 does not work!
#llm_options = just_agents.llm_options.OPEN_ROUTER_Qwen_2_72B_Instruct
#key_getter=lambda: os.getenv("OPEN_ROUTER_KEY")

#llm_options = just_agents.llm_options.DEEPINFRA_Qwen_2_72B_Instruct
#key_getter=lambda:  os.getenv("TOGETHERAI_API_KEY")

#llm_options = just_agents.llm_options.MISTRAL_8x22B
#key_getter=lambda: os.getenv("MISTRAL_API_KEY")

session: LLMSession = LLMSession(
    llm_options=llm_options,
    tools=[get_current_weather]
)
session.memory.add_on_message(lambda m: pprint.pprint(m) if "content" in m is not None else None)
#session.memory.add_on_message(lambda m: pprint.pprint(m.content) if m.content is not None else None)
session.query(prompt)
#for QWEN we get: Message(content='{\n  "function": "get_current_weather",\n  "parameters": {\n    "location": ["San Francisco", "Tokyo", "Paris"]\n  }\n}', role='assistant')


result = asyncio.run(session.stream_async(prompt))
print("stream finished")