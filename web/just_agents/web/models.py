from typing import Type, TypeVar, Any, List, Union, Optional, Literal, AsyncGenerator, cast
from pydantic import BaseModel, Field, HttpUrl

from just_agents.protocols.litellm_protocol import Message, TextContent
from just_agents.llm_options import ModelOptions

from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage

class ModelOptionsExt(ModelOptions):
    api_key: str = Field(None, examples=["openai_api_key"])

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

class ChatCompletionResponse(ChatCompletion):
#    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: Union[int,float]
#    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = Field(default=None)


class ChatCompletionChunkResponse(ChatCompletionResponse):
    choices: List[ChatCompletionChoiceChunk]

class ErrorResponse(BaseModel):
    class ErrorDetails(BaseModel):
        message: str = Field(...)
        type: str = Field("server_error")
        code: str = Field("internal_server_error")

    error: ErrorDetails
