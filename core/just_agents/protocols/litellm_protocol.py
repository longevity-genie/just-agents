import json
import pprint

from litellm import ModelResponse, CustomStreamWrapper, GenericStreamingChunk, completion, acompletion, stream_chunk_builder
from typing import Optional, Union, Coroutine, ClassVar, Type, Sequence, List, Any, AsyncGenerator
from pydantic import HttpUrl, Field, AliasPath, PrivateAttr, BaseModel, Json, field_validator

from just_agents.types import MessageDict, Role

from just_agents.interfaces.function_call import IFunctionCall, ToolByNameCallback
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from just_agents.protocols.openai_streaming import OpenaiStreamingProtocol
from litellm.types.utils import StreamingChoices, Delta
#from openai.types import CompletionUsage
#from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam,ChatCompletionFunctionMessageParam
#from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage

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

class LiteLLMFunctionCall(BaseModel, IFunctionCall[MessageDict], extra="allow"):
    id: str = Field(...)
    index: Optional[int] = Field(None)
    name: str = Field(..., validation_alias=AliasPath('function', 'name'))
    arguments: Json[dict] = Field(..., validation_alias=AliasPath('function', 'arguments'))
    type: Optional[str] = Field('function')

    @classmethod
    @field_validator('arguments', mode='before')
    def stringify_arguments(cls, value):
        # Convert dict to JSON string if necessary
        if isinstance(value, dict):
            return json.dumps(value)
        return value  # Assume it's already a string

    def execute_function(self, call_by_name: ToolByNameCallback):
        try:
            function_to_call = call_by_name(self.name)
            function_args = self.arguments
            function_response = str(function_to_call(**function_args))
        except Exception as e:
            function_response = str(e)
        message = {"role": "tool", "content": function_response, "name": self.name, "tool_call_id": self.id}
        return message

    @staticmethod
    def reconstruct_tool_call_message(calls: Sequence['LiteLLMFunctionCall']) -> dict:
        tool_calls = []
        for call_params in calls:
            tool_calls.append({"type": "function",
                               "id": call_params.id, "function": {"name": call_params.name, "arguments": str(call_params.arguments)}})
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}


class LiteLLMAdapter(BaseModel, IProtocolAdapter[ModelResponse,MessageDict, CustomStreamWrapper]):
    #Class that describes function convention
    function_convention: ClassVar[Type[IFunctionCall[MessageDict]]] = LiteLLMFunctionCall
    #hooks to agent class
    execute_function_hook:  ExecuteToolCallback[MessageDict] = Field(...)
    _output_streaming: IAbstractStreamingProtocol = PrivateAttr(default_factory=OpenaiStreamingProtocol)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    def completion(self, *args, **kwargs) -> Union[ModelResponse, CustomStreamWrapper]: # for the stream it is CustomStreamWrapper
        return completion(*args, **kwargs)

    async def async_completion(self, *args, **kwargs) \
            -> Coroutine[Any, Any, Union[ModelResponse, CustomStreamWrapper, AsyncGenerator]]:
        return acompletion(*args, **kwargs)
    
    # TODO: what about https://docs.litellm.ai/docs/providers/custom_llm_server ?

    def message_from_response(self, response: ModelResponse) -> MessageDict:
        message = response.choices[0].message.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
            exclude={"function_call"} if not response.choices[0].message.function_call else {}  # failsafe
        )
        if "citations" in response:
            message["citations"] = response.citations #perplexity specific field
        assert "function_call" not in message
        return message

    def delta_from_response(self, response: GenericStreamingChunk) -> MessageDict:
        message = response.choices[0].delta.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
            exclude={"function_call"} if not response.choices[0].delta.function_call else {}  # failsafe
        )
        return message
    
    def content_from_delta(self, delta: MessageDict) -> str:
        return delta.get("content")

    def tool_calls_from_message(self, message: MessageDict) -> List[LiteLLMFunctionCall]:
        # If there are no tool calls or tools available, exit the loop
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return []
        else:
            # Auto-convert each item in tool_calls to a FunctionCall instance with validation
            return [
                LiteLLMFunctionCall(**tool_call)
                for tool_call in tool_calls
            ]
    @staticmethod
    def reenumerate_tool_call_chunks(chunks : List[Any]):
        tool_calls = []
        message = None
        for chunk in chunks:
            if (
                    len(chunk["choices"]) > 0
                    and "tool_calls" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["tool_calls"]
            ):
                message = stream_chunk_builder(chunks=[chunk,chunks[-1]])
                tool_calls.append(message.choices[0].message.tool_calls[0])
        message.choices[0].message.tool_calls = tool_calls
        return message

    def response_from_deltas(self, chunks: List[Any]) -> ModelResponse:
        if "llama" in chunks[-1]["model"] and chunks[-1].choices[0].finish_reason=="tool_calls":
           return self.reenumerate_tool_call_chunks(chunks) # bug fix
        return stream_chunk_builder(chunks=chunks)

