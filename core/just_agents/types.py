from enum import Enum


from typing import Type, TypeVar, Any, List, Union, Optional, Literal, cast, TypeAlias, Sequence, Callable, Dict

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam,ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from pydantic import BaseModel, Field, HttpUrl


######### Common ###########
AbstractMessage = Dict[str, Any]
SupportedMessage = Union[str, AbstractMessage]
SupportedMessages = Union[SupportedMessage, List[SupportedMessage]]
Output = TypeVar('Output')

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    # make it similar to Literal["system", "user", "assistant", tool] while retaining enum convenience

    def __new__(cls, value, *args, **kwargs):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return str(self.value)
    
######### Helper ###########

class ModelOptions(BaseModel):
    model: str = Field(
        ...,
        examples=["gpt-4o-mini"],
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