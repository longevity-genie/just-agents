from typing import Any, Dict, List, Optional
from pydantic import Field, HttpUrl

from just_agents.core.types import ModelOptions

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


ANTHROPIC_CLAUDE_3_5_SONNET: Dict = {
    "model": "claude-3-5-sonnet-20240620",
    "temperature": 0.0
}

LLAMA3_2_VISION: Dict = {
    #supports both text and vision
    "model": "groq/llama-3.2-90b-vision-preview",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}


MISTRAL_8x22B = {
    "model": "mistral/open-mixtral-8x22b",
    "temperature": 0.0
}

OPENAI_GPT4o: Dict = {
    "model": "gpt-4o",
    "temperature": 0.0
}

OPENAI_GPT4oMINI: Dict = {
    "model": "gpt-4o-mini",
    "temperature": 0.0
}

OPENAI_O1_MINI: Dict = {
    "model": "o1-mini",
    "temperature": 0.0
}

OPENAI_O1_PREVIEW: Dict = {
    "model": "o1-preview",
    "temperature": 0.0
}

PERPLEXITY_LLAMA_3_1_SONAR_LARGE_128K_ONLINE: Dict = {
    "model": "perplexity/llama-3.1-sonar-large-128k-online",
    "temperature": 0.0,
    "return_citations": True,
    "return_related_questions": True
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