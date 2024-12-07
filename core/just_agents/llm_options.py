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


ANTHROPIC_CLAUDE_3_5_SONNET: dict[str, Any] = {
    "model": "claude-3-5-sonnet-20240620",
    "temperature": 0.0
}

LLAMA3_2_VISION: dict[str, Any] = {
    #supports both text and vision
    "model": "groq/llama-3.2-90b-vision-preview",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

LLAMA3_3: dict[str, Any] = {
    "model": "groq/llama-3.3-70b-versatile",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}

LLAMA3_3_specdec: dict[str, Any] = {
    "model": "groq/llama-3.3-70b-specdec",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
    "tools": []
}


MISTRAL_8x22B = {
    "model": "mistral/open-mixtral-8x22b",
    "temperature": 0.0
}

OPENAI_GPT4o: dict[str, Any] = {
    "model": "gpt-4o",
    "temperature": 0.0
}

OPENAI_GPT4oMINI: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "temperature": 0.0
}

OPENAI_O1_MINI: dict[str, Any] = {
    "model": "o1-mini",
    "temperature": 0.0
}

OPENAI_O1_PREVIEW: dict[str, Any] = {
    "model": "o1-preview",
    "temperature": 0.0
}

PERPLEXITY_LLAMA_3_1_SONAR_LARGE_128K_ONLINE: dict[str, Any]= {
    "model": "perplexity/llama-3.1-sonar-large-128k-online",
    "temperature": 0.0,
    "return_citations": True,
    "return_related_questions": True
}

OPEN_ROUTER_Qwen_2_72B_Instruct: dict[str, Any] = {
    "model": "openrouter/qwen/qwen-2-72b-instruct",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_Qwen_2_72B_Instruct_Vision: dict[str, Any] = {
    "model": "openrouter/qwen/qwen-2-vl-72b-instruct",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_LLAMA_3_8B_FREE: dict[str, Any] = {
    "model": "openrouter/meta-llama/llama-3-8b-instruct:free",
    "temperature": 0.0,
    "tools": []
}

OPEN_ROUTER_GEMINI_1_5_FLASH_EXP_FREE: dict[str, Any] = {
    "model": "openrouter/google/gemini-flash-1.5-exp",
    "temperature": 0.0,
    "tools": []
}

DEEPSEEK_CODER: dict[str, Any] = {
    "model": "deepseek/deepseek-coder",
    "temperature": 0.0,
    "tools": []
}

DEEPSEEK_CHAT: dict[str, Any] = {
    "model": "deepseek/deepseek-chat",
    "temperature": 0.0,
    "tools": []
}

def local_vllm_model(model: str = "models/granite-7b-lab.Q4_K_M.gguf", host: str="http://localhost:8000") -> dict[str, Any]:
    return {
        "model": f"hosted_vllm/{model}",
        "temperature": 0.0,
        "api_base": host,
        "tools": []
    }