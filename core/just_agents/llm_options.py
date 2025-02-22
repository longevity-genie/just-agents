from typing import Any, Dict, List, Optional
from pydantic import Field, HttpUrl, BaseModel
from pydantic import ConfigDict

# here we give only the most popular models, 
# but you can specify any model you want, check https://models.litellm.ai/ for models and providers that are supported

LLMOptions = Dict[str, Any]

class ModelOptions(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        validate_assignment=True,
    )
    model: str = Field(
        ...,
        examples=["groq/llama-3.3-70b-versatile","gpt-4o-mini"],
        description="LLM model name"
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

class LLMOptionsExt(ModelOptions):
    api_key: Optional[str] = Field(None, examples=["sk-proj-...."])
    api_base : Optional[HttpUrl] = Field(default=None,
        examples=[
            "https://api.groq.com/openai/v1",
            "https://api.openai.com/v1"
        ])
    tools : Optional[List[Any]] = None
    tool_choice : Optional[str] = None


ANTHROPIC_CLAUDE_3_5_SONNET: LLMOptions = {
    "model": "claude-3-5-sonnet-20240620",
    "temperature": 0.0
}

LLAMA3_2_VISION: LLMOptions = {
    #supports both text and vision
    "model": "groq/llama-3.2-90b-vision-preview",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0
}

LLAMA3_3: dict[str, Any] = {
    "model": "groq/llama-3.3-70b-versatile",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0
}

LLAMA3_3_specdec: dict[str, Any] = {
    "model": "groq/llama-3.3-70b-specdec",
    "api_base": "https://api.groq.com/openai/v1",
    "temperature": 0.0
}

CERABRAS_LLAMA3_3_70B: LLMOptions = {
    "model": "cerebras/meta/llama3-70b-instruct",
    "temperature": 0.0
}

GROQ_DEEPSEEK_R1_DISTILL: LLMOptions = {
    "model": "groq/deepseek-r1-distill-llama-70b",
    "api_base": "https://api.groq.com/openai/v1",
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

OPENAI_GPT4oMINI: LLMOptions = {
    "model": "gpt-4o-mini",
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