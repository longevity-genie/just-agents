#from just_agents.chat_agent import ChatAgent
import json
import pprint

import pytest
from dotenv import load_dotenv

import just_agents.llm_options
from just_agents.llm_session import LLMSession
import asyncio


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

def test_sync_llama_function_calling():
    load_dotenv()
    session: LLMSession = LLMSession(
        llm_options=just_agents.llm_options.LLAMA3_1,
        tools=[get_current_weather]
    )
    result = session.query("What's the weather like in San Francisco, Tokyo, and Paris?")
    assert "72" in result
    assert "22" in result
    assert "10" in result

async def process_stream(async_generator):
    async for item in async_generator:
        pass

def test_stream_llama_function_calling():
    load_dotenv()
    session: LLMSession = LLMSession(
        llm_options=just_agents.llm_options.LLAMA3_1,
        tools=[get_current_weather]
    )
    stream = session.stream("What's the weather like in San Francisco, Tokyo, and Paris?")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_stream(stream))
    result = session.memory.last_message["content"]
    assert "72" in result
    assert "22" in result
    assert "10" in result

@pytest.mark.skip(reason="so far qwen inference we are using has issues with json function calling")
def test_async_gwen2_function_calling():
    load_dotenv()
    session: LLMSession = LLMSession(
        llm_options=just_agents.llm_options.OPEN_ROUTER_Qwen_2_72B_Instruct,
        tools=[get_current_weather]
    )
    result = session.query_all_messages("What's the weather like in San Francisco, Tokyo, and Paris?")
    assert "72" in result
    assert "22" in result
    assert "10" in result
