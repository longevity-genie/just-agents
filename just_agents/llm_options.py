from pydantic import Field
from pydantic import BaseModel


class LLMOptions(BaseModel):
    """
    LLM options to pass to LiteLLM

    """
    model: str = Field(description="the model to be used, for example groq/llama3-70b-8192")
    temperature: float = Field(description="temperature of the model, defined how random it is", default=0.0)
    api_base: str = Field(description="URL that serves API, for example https://api.groq.com/openai/v1")
    tool_choice: str = "auto"

LLAMA3 = LLMOptions(
    model = "groq/llama3-70b-8192",
    temperature = 0.0,
    api_base = "https://api.groq.com/openai/v1"
)