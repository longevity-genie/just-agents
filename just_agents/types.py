from enum import Enum
from typing import Type, TypeVar, Any, List, Union, Optional, Literal, cast, TypeAlias, Sequence, Callable, Dict
from pathlib import Path

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam,ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from pydantic import BaseModel, Field, HttpUrl

from litellm import ModelResponse

######### Common ###########

BaseT = TypeVar('BaseT', bound=BaseModel)
AbstractMessage = Dict[str,Any] #TODO: proper typing
SupportedMessage = Union[str, AbstractMessage]
SupportedMessages = Union[SupportedMessage, List[SupportedMessage]]

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

class StreamingMode(str, Enum):
    openai = "openai"
    qwen2 = "qwen2"
    chain_of_thought = "chain_of_thought"

    def __new__(cls, value, *args, **kwargs):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return str(self.value)


######### Helper ###########


OAIMessage: TypeAlias = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]

def message_from_response(response: ModelResponse) -> AbstractMessage:
    message = response.choices[0].message.model_dump(
        mode="json",
        exclude_none=True,
        exclude_unset=True,
        by_alias=True,
        exclude={"function_call"} if not response.choices[0].message.function_call else {}  # failsafe
    )
    assert "function_call" not in message
    return message

def extract_common_fields(selected_class: Type[BaseT], instance: BaseModel) -> BaseT:
    """
    Trims and typecasts an instance of a class to only include the fields of the selected class.

    :param selected_class: The class type to trim to.
    :param instance: The instance of the class to be trimmed.
    :return: An instance of the selected class populated with the relevant fields from the provided object.
    """
    # Extract only the fields defined in the base class
    base_fields = {field: getattr(instance, field) for field in selected_class.model_fields}

    # Instantiate and return the base class with these fields
    return selected_class(**base_fields)

def trim_to_parent(instance: BaseT) -> BaseModel:
    """
    Trims an instance of a derived class to only include the fields of its direct parent class.
    :param instance: The instance of the derived class to be trimmed.
    :return: An instance of the parent class populated with the relevant fields from the derived class.
    """
    # Get the direct parent class of the instance
    parent_class = type(instance).__class__.__bases__[0]

    # Instantiate and return the parent class with these fields
    parent_instance = extract_common_fields(parent_class, instance)
    return cast(parent_class, parent_instance)

######### Pydantic models ###########

class DbConnectionArgs(BaseModel):
    host: str = Field("localhost", examples=["local"])
    port: Optional[str] = Field("57687", examples=["57687"])
    db_name: Optional[str] = Field(None, examples=["neo4j"])
    user: Optional[str]=Field(None, examples=["neo4j"])
    password: Optional[str]=Field(None, examples=["neo4j"])

# Content types
class TextContent(BaseModel):
    type: str = Field("text", examples=["text"])
    text: str = Field(..., examples=["What are in these images? Is there any difference between them?"])

class ImageContent(BaseModel):
    type: str = Field("image_url", examples=["image_url"])
    image_url: HttpUrl = Field(..., examples=["https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"])

# Message class - Simple string content or a list of text or image content for vision model
class Message(BaseModel):
    role: Role = Field(..., examples=[Role.assistant])
    content: Union[
        str,  # Simple string content
        List[Union[TextContent, ImageContent]]
    ] = Field(
        ...,
        description="Content can be a simple string, or a list of content items including text or image URLs."
    )


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


class ChatCompletionRequest(ModelOptions):
    messages: List[Message] = Field(..., examples=[[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What drug interactions of rapamycin are you aware of? What are these interactions ?"}
    ]])
    n: Optional[int] = Field(1, ge=1)
    stream: Optional[bool] = Field(default=False, examples=[True])
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    logit_bias: Optional[dict] = Field(None, examples=[None])
    user: Optional[str] = Field(None, examples=[None])


class ResponseMessage(ChatCompletionMessage):
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None

class ChatCompletionChoice(Choice):
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
#    text: Optional[str] = Field(default=None, alias="message.content")
    message : Optional[ResponseMessage]

class ChatCompletionChoiceChunk(ChatCompletionChoice):
    delta: ResponseMessage = Field(default=None)
    message: Optional[ResponseMessage] = Field(default=None, exclude=True) #hax

class ChatCompletionUsage(CompletionUsage):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    pass

#TODO: format responses
class Context(BaseModel):
    mode : str
    context : Any

class ChatCompletionResponse(ChatCompletion):
#    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: Union[int,float]
#    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = Field(default=None)

class ChatCompletionChunkResponse(ChatCompletionResponse):
    choices: List[ChatCompletionChoiceChunk]
