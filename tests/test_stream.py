import json
from dotenv import load_dotenv
import pytest
from typing import Callable, Any

from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.base_agent import BaseAgent, BaseAgentWithLogging
from just_agents.llm_options import LLMOptions, LLAMA3_3, LLAMA3_2_VISION, OPENAI_GPT4oMINI
from just_agents.just_tool import JustToolsBus


@pytest.fixture(scope="module", autouse=True)
def load_env():
    load_dotenv(override=True)

def get_current_weather(location: str):
    """
    Gets the current weather in a given location

    Args:
        location (str): The name of the location for which to get the weather.

    Returns:
        str: A JSON-encoded string with the following keys:
            - "location" (str): The location name.
            - "temperature" (str): The temperature value, or "unknown" if not recognized.
            - "unit" (str, optional): The unit of measurement for temperature (e.g., "celsius", "fahrenheit").
    """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def test_auto_prompt_query():
    from tests.tools.toy_tools import get_secret_key,decypher_using_secret_key
    session: BaseAgent = BaseAgentWithLogging(
        llm_options=LLAMA3_3,
        system_prompt="""You are a helpful assistant that can decypher messages using a secret key.  
        You will be given a secret key. You need to decypher the message using the secret key provided by a system prompt.""",
        prompt_tools=[
            (get_secret_key, {"secret_word": "banana"}) #tupple of callable and call_arguments
        ],
        tools=[decypher_using_secret_key]
    )
    secret = session.query("Decipher me the message please - 'JBYEF0QTGV9IPRIAXEpI' ")
    assert "Wake up, Neo..." in secret

def agent_query(prompt: str, options: LLMOptions, **kwargs):
    session: BaseAgent = BaseAgentWithLogging(
        llm_options=options,
        tools=[get_current_weather]
    )
    return session.query(prompt, **kwargs)

def agent_call(prompt: str, options: LLMOptions, reconstruct_chunks: bool, **kwargs):
    session: BaseAgent = BaseAgentWithLogging(
        llm_options=options,
        tools=[get_current_weather]
    )
    chunks = []
    gen = session.stream(prompt, reconstruct_chunks=reconstruct_chunks, **kwargs)
    for sse_event in gen:
        event = SSE.sse_parse(sse_event)
        assert isinstance(event, dict)
        data = event.get("data")
        if isinstance(data, dict):
            delta = session._protocol.message_from_response(data)
            chunk = session._protocol.content_from_delta(delta)
        else:
            continue
        if chunk:
            chunks.append(chunk)

    full_response = ''.join(chunks)
    last = session.memory.last_message_str
    assert full_response.endswith(last)
    return full_response

def test_stream():
    result = agent_call("Why is the sky blue?", OPENAI_GPT4oMINI, False)
    assert "wavelength" in result

def test_stream_amnesia():
    result = agent_call("Why is the sky blue?", OPENAI_GPT4oMINI, False, continue_conversation=False, remember_query=False)
    assert "wavelength" in result

def test_stream_grok():
    result = agent_call("Why is the sky blue?", LLAMA3_3, False)
    assert "wavelength" in result

def test_stream_recon():
    result = agent_call("Why is the grass green?", OPENAI_GPT4oMINI, True)
    assert "chlorophyll" in result

def test_stream_grok_recon():
    result = agent_call("Why is the grass green?", LLAMA3_3, True)
    assert "chlorophyll" in result

def validate_tool_call(call : Callable[[Any,...],str],*args,**kwargs):
    prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"
    bus = JustToolsBus()
    results = []
    result_callback = 'get_current_weather.result'
    def callback(event_name: str, result_interceptor: str, **kwargs):
        assert event_name == result_callback
        results.append(result_interceptor)
    bus.subscribe(result_callback,callback)
    result = call(prompt,*args,**kwargs)
    assert len(results) == 3
    assert "72" in result
    assert "22" in result
    assert "10" in result
    assert any('72' in item for item in results), "San Francisco weather call missing"
    assert any('22' in item for item in results), "Paris weather call missing"
    assert any('10' in item for item in results), "Tokyo weather call missing"

def test_query_tool():
    validate_tool_call(agent_query, OPENAI_GPT4oMINI)

def test_query_tool_amnesia():
    validate_tool_call(agent_query, OPENAI_GPT4oMINI, continue_conversation=False, remember_query=False)

def test_stream_tool():
    validate_tool_call(agent_call, OPENAI_GPT4oMINI, False)

def test_stream_tool_grok():
    validate_tool_call(agent_call, LLAMA3_2_VISION, False)


