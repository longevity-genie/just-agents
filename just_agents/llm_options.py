from dataclasses import dataclass, field, replace
from dataclasses import asdict
from typing import Any, Dict, Optional


from dataclasses import dataclass, asdict

OPENAI_GPT4o: Dict = {
    "model": "gpt-4o",
    "temperature": 0.0
}


LLAMA3: Dict = {
    "model": "groq/llama3-groq-70b-8192-tool-use-preview",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

LLAMA3_1: Dict = {
    "model": "groq/llama-3.1-70b-versatile",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_Qwen_2_72B_Instruct = {
    "model": "openrouter/qwen/qwen-2-72b-instruct",
    "temperature": 0.0,
    "tools": []
}

TOGETHER_Qwen_2_72B_Instruct = {
    "model": "together_ai/Qwen/Qwen2-72B-Instruct",
    "temperature": 0.0,
    "tools": []
}
DEEPINFRA_Qwen_2_72B_Instruct = {
    "model" : "deepinfra/Qwen/Qwen2-72B-Instruct",
    "tools": []
}

FIREWORKS_Qwen_2_72B_Instruct = {
    "model": "fireworks_ai/wen/Qwen2-72B-Instruct",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

MISTRAL_8x22B = {
    "model": "mistral/open-mixtral-8x22b",
    "temperature": 0.0
}