from typing import Any, Dict, List, Optional
from pydantic import Field, HttpUrl
from just_agents.types import ModelOptions

LLMOptions = Dict[str, Any]

class LLMOptionsBase(ModelOptions, extra="allow"):
    api_key: Optional[str] = Field(None, examples=["sk-proj-...."])
    api_base : Optional[HttpUrl] = Field(default=None,
        examples=[
            "https://api.groq.com/openai/v1",
            "https://api.openai.com/v1"
        ])
    tools : Optional[List[Any]] = None
    tool_choice : Optional[str] = None

OPENAI_GPT4o: Dict = {
    "model": "gpt-4o",
    "temperature": 0.0
}

OPENAI_GPT4oMINI: Dict = {
    "model": "gpt-4o-mini",
    "temperature": 0.0
}

ANTHROPIC_CLAUDE_3_5_SONNET: Dict = {
    "model": "anthropic/claude-3-5-sonnet-20240620",
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

LLAMA3_2_VISION: Dict = {
    "model": "groq/llama-3.2-11b-vision-preview",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

LLAMA3_2: Dict = {
    "model": "groq/llama-3.2-90b-text-preview",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_Qwen_2_72B_Instruct = {
    "model": "openrouter/qwen/qwen-2-72b-instruct",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_Qwen_2_72B_Instruct_Vision = {
    "model": "openrouter/qwen/qwen-2-vl-72b-instruct",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_LLAMA_3_8B_FREE = {
    "model": "openrouter/meta-llama/llama-3-8b-instruct:free",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_GEMINI_1_5_FLASH_EXP_FREE = {
    "model": "openrouter/google/gemini-flash-1.5-exp",
    "temperature": 0.0,
    "tools": []
}

MISTRAL_8x22B = {
    "model": "mistral/open-mixtral-8x22b",
    "temperature": 0.0
}