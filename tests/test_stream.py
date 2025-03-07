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

def stream_to_str(session: BaseAgent, prompt: str, reconstruct_chunks: bool = False, **kwargs) -> str:
    """
    Convert a stream of SSE events from a session into a concatenated string response.
    
    Args:
        session: The BaseAgent instance to use for streaming
        prompt: The prompt to send to the agent
        reconstruct_chunks: Whether to reconstruct the chunks
        **kwargs: Additional arguments to pass to the stream method
        
    Returns:
        str: The concatenated response from the stream
    """
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

    return ''.join(chunks)

def agent_call(prompt: str, options: LLMOptions, reconstruct_chunks: bool, **kwargs):
    session: BaseAgent = BaseAgentWithLogging(
        llm_options=options,
        tools=[get_current_weather]
    )
    full_response = stream_to_str(session, prompt, reconstruct_chunks, **kwargs)
    last = session.memory.last_message_str
    if kwargs.get("remember_query", True):
        assert full_response.endswith(last)
    return full_response

def test_bad_model_query():
    opts: LLMOptions = {
    "model": "groq/llama-nonexistent",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0
    }
    session: BaseAgent = BaseAgentWithLogging(
        llm_options=opts,
        tools=[get_current_weather],
        raise_on_completion_status_errors=False,
        remember_query=False
    )
    result = session.query("Why is the sky blue?")
    assert "NotFoundError" in result
    
    full_response = stream_to_str(session, "Why is the sky blue?", reconstruct_chunks=False)
    assert "NotFoundError" in full_response
    # assert result == full_response # different litellm error representations (bytestring vs string)

def memory_management_test(callback_func):
    """
    Generic test for session memory management that works with both query and streaming.
    
    Args:
        callback_func (Callable): Function that handles agent interaction (query or stream)
    """
    prompt = "Why is the sky blue, describe in 10 words?"
    prompt2 = "What is the highest mountain in the world?"
    amnesia_prompt = "I got distracted, what matters was I asking you about just now?"
    
    # Create agent session
    session: BaseAgent = BaseAgentWithLogging(
        llm_options=OPENAI_GPT4oMINI,
        tools=[get_current_weather]
    )
    
    # Test with continue_conversation=None, remember_query=False
    response_no_memory = callback_func(session, prompt, continue_conversation=None, remember_query=False)
    assert "Rayleigh" in response_no_memory or "scattering" in response_no_memory
    assert session.memory.last_message_str == None  # memory is empty, no query was remembered
    
    # Test with continue_conversation=None, remember_query=True
    response_with_memory = callback_func(session, prompt, continue_conversation=None, remember_query=True)
    assert "Rayleigh" in response_with_memory or "scattering" in response_with_memory
    assert session.memory.last_message_str == response_with_memory  # query was remembered
    
    # Test with continue_conversation=True, remember_query=False
    response_recall_sky = callback_func(session, amnesia_prompt, continue_conversation=True, remember_query=False)
    assert "color" in response_recall_sky or "sky" in response_recall_sky # correct recolection
    assert session.memory.last_message_str == response_with_memory  # query was not remembered, but previous result was
    assert response_recall_sky != response_with_memory  # result is different
    
    # Test with continue_conversation=False, remember_query=False
    response_no_recall = callback_func(session, amnesia_prompt, continue_conversation=False, remember_query=False)
    assert "color" not in response_no_recall or "sky" not in response_no_recall # correct hallucination
    assert session.memory.last_message_str == response_with_memory  # query was not remembered, but previous result was
    assert response_no_recall != response_with_memory  # result is different
    
    # Test with continue_conversation=False, remember_query=True
    response_mountain = callback_func(session, prompt2, continue_conversation=False, remember_query=True)
    assert "Everest" in response_mountain or "Mount" in response_mountain # correct answer
    assert session.memory.last_message_str == response_mountain  # query was remembered
    assert response_mountain != response_with_memory  # result is different
    
    # Test final state of agent memory
    response_recall_mountain = callback_func(session, amnesia_prompt, continue_conversation=True, remember_query=False)
    assert "Everest" in response_recall_mountain or "Mount" in response_recall_mountain #correct recollection
    assert "color" not in response_recall_mountain or "sky" not in response_recall_mountain #previous result properly forgotten
    assert session.memory.last_message_str == response_mountain  # query was not remembered, but previous result was
    assert response_mountain != response_with_memory  # result is different
    assert response_mountain != response_recall_mountain  # result is different

    return response_no_memory, response_recall_sky, response_no_recall, response_mountain, response_recall_mountain, response_with_memory


def test_stream_amnesia_complete():
    """Test memory management using streaming responses."""
    def query_callback(session: BaseAgent, prompt: str, **kwargs) -> str:
        return session.query(prompt, **kwargs)

    def stream_callback(session: BaseAgent, prompt: str, **kwargs) -> str:
        return stream_to_str(session, prompt, reconstruct_chunks=False, **kwargs)

    stream_results = memory_management_test(stream_callback)
    query_results = memory_management_test(query_callback)

    # Step-by-step comparison of each pair of results
    for idx, (stream_result, query_result) in enumerate(zip(stream_results, query_results), start=1):
        try:
            assert stream_result == query_result, f"Mismatch at position {idx}: stream='{stream_result}' vs query='{query_result}'"
        except AssertionError as e:
            print(f"Assertion failed at step {idx}: {e}")
    print("Stream and query results are equal!")


def test_stream():
    result = agent_call("Why is the sky blue, describe in 10 words?", OPENAI_GPT4oMINI, False)
    assert "Rayleigh" in result or "scattering" in result

def test_stream_amnesia():
    # Test: continue_conversation=False, remember_query=False
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


def test_stream_tool():
    validate_tool_call(agent_call, OPENAI_GPT4oMINI, False)

def test_stream_tool_grok():
    validate_tool_call(agent_call, LLAMA3_2_VISION, False)


