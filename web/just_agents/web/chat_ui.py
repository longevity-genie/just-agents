from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from just_agents.data_classes import ModelPromptExample


class ModelParameters(BaseModel):
    """
    Defines parameters used to configure model responses.
    """
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature to control the creativity of the model's output. Must be between 0.0 and 2.0.",
        examples=[0.0, 0.7, 1.0]
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum number of tokens the model will generate.",
        examples=[128, 256, 512]
    )
    stop: List[str] = Field(
        default_factory=lambda: ["[DONE]"],
        description="List of stop sequences to end generation.",
        examples=[["[DONE]", "###"]]
    )

class ModelEndpoint(BaseModel):
    """
    Describes an endpoint through which the model can be accessed.
    """
    type: str = Field(
        ...,
        description="Type of model endpoint (e.g., 'openai' or 'azure').",
        examples=["openai"]
    )
    baseURL: str = Field(
        ...,
        description="Base URL for the API endpoint.",
        examples=["http://litellm-proxy:4000/v1"]
    )
    apiKey: Optional[str] = Field(
        None,
        description="API key or token used for authentication.",
        examples=["no_key_needed", "sk-XXX"]
    )



class ModelConfig(BaseModel):
    """
    Represents the top-level configuration for a model,
    including parameters, endpoints, and prompt examples.
    """
    model_config = ConfigDict()  # Only needed in the top-level model

    name: str = Field(
        ...,
        description="Internal identifier for the model.",
        examples=["gpt-4o-mini"]
    )
    displayName: str = Field(
        ...,
        description="Human-readable or display name for the model.",
        examples=["proxified-gpt-4o"]
    )
    description: str = Field(
        ...,
        description="Brief description of what this model is or does.",
        examples=["OpenAI gpt-4o-mini model served through cache-proxy"]
    )
    parameters: ModelParameters = Field(
        ...,
        description="Model generation parameters."
    )
    endpoints: List[ModelEndpoint] = Field(
        ...,
        description="List of endpoints configured for this model."
    )
    promptExamples: List[ModelPromptExample] = Field(
        default_factory=list,
        description="List of example prompts demonstrating usage."
    )
