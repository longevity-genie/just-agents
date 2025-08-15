from typing import Any, Dict, List, Optional, Literal
from pydantic import Field, HttpUrl, BaseModel, ConfigDict

# here we give only the most popular models, 
# but you can specify any model you want, check https://models.litellm.ai/ for models and providers that are supported

#TODO: parse https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

LLMOptions = Dict[str, Any]

class ModelOptions(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        validate_assignment=True,
    )
    model: str = Field(
        ...,
        examples=["groq/llama-3.3-70b-versatile","gpt-4.1-nano"],
        description="LLM model name"
    )
    api_key: Optional[str] = Field(None, examples=["sk-proj-...."])
    api_base : Optional[HttpUrl] = Field(default=None,
        examples=[
            "https://api.groq.com/openai/v1",
            "https://api.openai.com/v1"
        ]
    )
    temperature: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=2.0,
        examples=[0.7],
        description="Sampling temperature, values from 0.0 to 2.0"
    )
    top_p: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1.0,
        examples=[0.9],
        description="Nucleus sampling probability, values from 0.0 to 1.0"
    )
    presence_penalty: Optional[float] = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        examples=[0.6],
        description="Presence penalty, values from -2.0 to 2.0"
    )
    frequency_penalty: Optional[float] = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        examples=[0.5],
        description="Frequency penalty, values from -2.0 to 2.0"
    )
    tool_choice: Optional[Literal["auto", "none", "required"]] = Field(
        "auto",
        description="Whether to automatically select the best tool to use. If set to 'none', the model will not use any tools. If set to 'required', the model will only use tools."
    )

ANTHROPIC_CLAUDE_4_SONNET: LLMOptions = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.0
}

ANTHROPIC_CLAUDE_4_OPUS: LLMOptions = {
    "model": "claude-opus-4-20250514",
    "temperature": 0.0
}


ANTHROPIC_CLAUDE_3_7_SONNET: LLMOptions = {
    "model": "claude-3-7-sonnet",
    "temperature": 0.0
}

ANTHROPIC_CLAUDE_3_5_SONNET: LLMOptions = {
    "model": "claude-3-5-sonnet",
    "temperature": 0.0
}

LLAMA3_3: LLMOptions = {
    "model": "groq/llama-3.3-70b-versatile",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0
}

LLAMA4_MAVERICK: LLMOptions = {
    "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
}

LLAMA4_SCOUT: LLMOptions = {
    "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
}

GPT_OSS_20B: LLMOptions = {
    "model": "groq/openai/gpt-oss-20b",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
}

GPT_OSS_120B: LLMOptions = {
    "model": "groq/openai/gpt-oss-120b",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0,
}


CERABRAS_LLAMA3_3_70B: LLMOptions = {
    "model": "cerebras/meta/llama3-70b-instruct",
    "temperature": 0.0
}

CERABRAS_LLAMA3_3_70B_INSTRUCT: LLMOptions = {
    "model": "cerebras/deepseek-r1-distill-llama-70b",
    "temperature": 0.0
}

GROQ_DEEPSEEK_R1_DISTILL: LLMOptions = {
    "model": "groq/deepseek-r1-distill-llama-70b",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0
}

MISTRAL_LARGE_2: LLMOptions = {
    "model": "mistral/mistral-large-2407",
    "temperature": 0.0
}

MISTRAL_CODESTRAL_LATEST: LLMOptions = {
    "model": "mistral/codestral-latest",
    "temperature": 0.0
}


MISTRAL_CODESTRAL_MAMBA: LLMOptions = {
    "model": "mistral/codestral-mamba-latest",
    "temperature": 0.0
}

MISTRAL_SMALL_LATEST: LLMOptions = {
    "model": "mistral/mistral-small-latest",
    "temperature": 0.0
}

MISTRAL_MEDIUM_LATEST: LLMOptions = {
    "model": "mistral/mistral-medium-latest",
    "temperature": 0.0
}

MISTRAL_LARGE_LATEST: LLMOptions = {
    "model": "mistral/mistral-large-latest",
    "temperature": 0.0
}


MISTRAL_8x22B: LLMOptions = {
    "model": "mistral/open-mixtral-8x22b",
    "temperature": 0.0
}

OPENAI_GPT4o: LLMOptions = {
    "model": "gpt-4o",
    "temperature": 0.0
}

OPENAI_GPT4_1: LLMOptions = {
    "model": "gpt-4.1",
    "temperature": 0.0
}

OPENAI_GPT4_1MINI: LLMOptions = {
    "model": "gpt-4.1-mini",
    "temperature": 0.0
}

OPENAI_GPT4_1NANO: LLMOptions = {
    "model": "gpt-4.1-nano",
    "temperature": 0.0
}

OPENAI_GPT4oMINI: LLMOptions = {
    "model": "gpt-4o-mini",
    "temperature": 0.0
}


OPENAI_GPT5: LLMOptions = {
    "model": "gpt-5",
    "temperature": 0.0
}

OPENAI_GPT5_MINI: LLMOptions = {
    "model": "gpt-5-mini",
    "temperature": 0.0
}

OPENAI_GPT5_NANO: LLMOptions = {
    "model": "gpt-5-nano",
    "temperature": 0.0
}

OPENAI_GPT5_CHAT: LLMOptions = {
    "model": "gpt-5-chat",
    "temperature": 0.0
}

OPENAI_GPT5_CHAT_LATEST: LLMOptions = {
    "model": "gpt-5-chat-latest",
    "temperature": 0.0
}


OPENAI_O1_MINI: LLMOptions = {
    "model": "o1-mini",
    "temperature": 0.0
}


OPENAI_O3_MINI_2025_01_31: LLMOptions = {
    "model": "o3-mini-2025-01-31",
    "temperature": 0.0
}

OPENAI_O3_MINI: LLMOptions = {
    "model": "o3-mini",
    "temperature": 0.0
}

OPENAI_O1: LLMOptions = {
    "model": "o1",
    "temperature": 0.0
}

OPENAI_O1_PREVIEW: LLMOptions = {
    "model": "o1-preview",
    "temperature": 0.0
}

PERPLEXITY_SONAR_PRO: LLMOptions= {
    "model": "perplexity/sonar-pro",
    "temperature": 0.0,
    "return_citations": True,
    "return_related_questions": True
}

PERPLEXITY_SONAR: LLMOptions= {
    "model": "perplexity/sonar",
    "temperature": 0.0,
    "return_citations": True,
    "return_related_questions": True
}

GEMINI_2_FLASH_THINKING_EXP: LLMOptions = {
    "model": "gemini/gemini-2.0-flash-thinking-exp",
    "temperature": 0.0
}

GEMINI_2_FLASH_EXP: LLMOptions = {
    "model": "gemini/gemini-2.0-flash-exp",
    "temperature": 0.0
}

GEMINI_2_FLASH: LLMOptions = {
    "model": "gemini/gemini-2.0-flash",
    "temperature": 0.0
}

GEMINI_2_5_FLASH: LLMOptions = {
    "model": "gemini/gemini-2.5-flash",
    "temperature": 0.0
}

GEMINI_2_5_PRO: LLMOptions = {
    "model": "gemini/gemini-2.5-pro",
    "temperature": 0.0
}

GROK_3: LLMOptions = {
    "model": "xai/grok-3-latest",
    "temperature": 0.0
}


DEEPSEEK_CODER: LLMOptions = {
    "model": "deepseek/deepseek-coder",
    "temperature": 0.0
}

DEEPSEEK_CHAT: LLMOptions = {
    "model": "deepseek/deepseek-chat",
    "temperature": 0.0
}

DEEPSEEK_R1: LLMOptions = {
    "model": "deepseek/deepseek-reasoner",
    "temperature": 0.0
}

def local_vllm_model(model: str = "models/granite-7b-lab.Q4_K_M.gguf", host: str="http://localhost:8000") -> LLMOptions:
    return {
        "model": f"hosted_vllm/{model}",
        "temperature": 0.0,
        "api_base": host
    }