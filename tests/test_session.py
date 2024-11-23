#from just_agents.chat_agent import ChatAgent
import json

import pytest
from dotenv import load_dotenv

import just_agents.simple.llm_session
from just_agents.simple.llm_session import LLMSession
from just_agents import llm_options
import asyncio
from tests.mock import *


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

def test_sync_function_calling():
    load_dotenv(override=True)
    session: LLMSession = LLMSession(
        llm_options=llm_options.LLAMA3_2_VISION,
        tools=[get_current_weather]
    )
    result = session.query("What's the weather like in San Francisco, Tokyo, and Paris?")
    assert "72" in result
    assert "22" in result
    assert "10" in result

async def process_stream(async_generator):
    async for item in async_generator:
        pass

def test_stream_function_calling():
    load_dotenv(override=True)
    session: LLMSession = LLMSession(
        llm_options=llm_options.LLAMA3_2_VISION,
        tools=[get_current_weather]
    )
    stream = session.stream("What's the weather like in San Francisco, Tokyo, and Paris?")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_stream(stream))
    result = session.memory.last_message["content"]
    assert "72" in result
    assert "22" in result
    assert "10" in result


def test_stream_genetics_function_calling():
    load_dotenv(override=True)
    session: LLMSession = LLMSession(
        llm_options=llm_options.LLAMA3_2_VISION,
        tools=[hybrid_search, rsid_lookup, gene_lookup, pathway_lookup, disease_lookup, sequencing_info, clinical_trails_full_trial]
    )
    stream = session.stream("What is the influence of different alleles in rs10937739?")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_stream(stream))
    result = session.memory.last_message["content"]
    assert "pro-longevity" in result

@pytest.mark.skip(reason="so far qwen inference we are using has issues with json function calling")
def test_async_gwen2_function_calling():
    load_dotenv(override=True)
    session: LLMSession = LLMSession(
        llm_options=llm_options.OPEN_ROUTER_Qwen_2_72B_Instruct,
        tools=[get_current_weather]
    )
    result = session.query_all_messages("What's the weather like in San Francisco, Tokyo, and Paris?")
    assert "72" in result
    assert "22" in result
    assert "10" in result
