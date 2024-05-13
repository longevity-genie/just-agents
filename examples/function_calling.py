from just_agents.chat_agent import ChatAgent
import json
from dotenv import load_dotenv
load_dotenv()

llama3 = {
    "model": "groq/llama3-70b-8192",
    "temperature": 0.7,
    "api_base": "https://api.groq.com/openai/v1",
}


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


agent:ChatAgent = ChatAgent(llama3, functions=[get_current_weather])
print(agent("What's the weather like in San Francisco, Tokyo, and Paris?"))
