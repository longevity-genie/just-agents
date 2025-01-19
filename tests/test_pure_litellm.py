import json
import litellm
import pytest
from dotenv import load_dotenv

def get_current_weather(location: str):
    """
    Gets the current weather in a given location
    """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

@pytest.fixture(scope="module", autouse=True)
def load_env():
    load_dotenv(override=True)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}
    ]
    tools = [{"type": "function",
              "function": litellm.utils.function_to_dict(get_current_weather)}]
    OPENAI_GPT4oMINI = {
        "messages": messages,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "tools": tools,
        "stream": True
    }
    LLAMA3_3 = {
        "messages" : messages,
        "model": "groq/llama-3.3-70b-versatile",
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "tools": tools,
        "stream" : True
    }
    return OPENAI_GPT4oMINI, LLAMA3_3

def execute_completion(opts:dict):
    partial_streaming_chunks = []
    response_gen = litellm.completion(**opts)
    for i, part in enumerate(response_gen):
        partial_streaming_chunks.append(part)
    assembly = litellm.stream_chunk_builder(partial_streaming_chunks)
    print(assembly.choices[0].message.tool_calls)
    assert len(assembly.choices[0].message.tool_calls) == 3, assembly.choices[0].message.tool_calls[0].function.arguments[0]
    print (assembly.choices[0].message.tool_calls)

def test_oai_works(load_env):
    OPENAI_GPT4oMINI,_ = load_env
    execute_completion(OPENAI_GPT4oMINI)

#@pytest.mark.skip(reason="until fixed in https://github.com/BerriAI/litellm/issues/7621")
def test_grok_bug(load_env):
    _, LLAMA3_3 = load_env
    execute_completion(LLAMA3_3)