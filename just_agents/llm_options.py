from dataclasses import dataclass, field, replace
from dataclasses import asdict
from typing import Any, Dict, Optional


from dataclasses import dataclass, asdict


LLAMA3: Dict = {
    "model": "groq/llama3-70b-8192",
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
    "model": "openrouter/qwen/qwen-2-72b-instruct",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
}

FIREWORKS_Qwen_2_72B_Instruct = {
    "model": "fireworks_ai/wen/Qwen2-72B-Instruct",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
}