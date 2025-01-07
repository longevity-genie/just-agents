import json
from dotenv import load_dotenv
import pytest

from just_agents.base_agent import BaseAgent
from just_agents.llm_options import LLMOptions, LLAMA3_3, OPENAI_GPT4oMINI

@pytest.fixture(scope="module", autouse=True)
def load_env():
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

def agent_query(prompt: str, options: LLMOptions):
    session: BaseAgent = BaseAgent(
        llm_options=options,
        tools=[get_current_weather]
    )
    return session.query(prompt)

def agent_call(prompt: str, options: LLMOptions, reconstruct_chunks: bool):
    session: BaseAgent = BaseAgent(
        llm_options=options,
        tools=[get_current_weather]
    )
    chunks = []
    gen = session.stream(prompt, reconstruct_chunks=reconstruct_chunks)
    for sse_event in gen:
        event = session._protocol.sse_parse(sse_event)
        assert isinstance(event, dict)
        data = event.get("data")
        if isinstance(data, dict):
            delta = data["choices"][0]["delta"]
            chunk = session._protocol.content_from_delta(delta)
        else:
            continue
        chunks.append(chunk)

    full_response = ''.join(chunks)
    last = session.memory.last_message_str
    assert full_response == last
    return full_response

def test_stream():
    result = agent_call("Why is the sky blue?", OPENAI_GPT4oMINI, False)
    print(result)
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

def test_tool_only():
    prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"
    non_stream = agent_query(prompt,OPENAI_GPT4oMINI)
    assert "72" in non_stream
    assert "22" in non_stream
    assert "10" in non_stream

#def test_stream_tool():
#    prompt = "What's the weather like in San Francisco, Tokyo, and Paris?"
#    result = agent_call(prompt, OPENAI_GPT4oMINI, False)
#    assert "72" in result
#    assert "22" in result
#    assert "10" in result

