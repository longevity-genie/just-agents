
import json

from litellm import ModelResponse, CustomStreamWrapper, completion, acompletion, stream_chunk_builder
from typing import Optional, Callable, Union, Coroutine, ClassVar, Type, Sequence, List, Any, AsyncGenerator
from pydantic import Field, AliasPath, PrivateAttr, BaseModel, Json, field_validator

from just_agents.core.types import AbstractMessage

from just_agents.streaming.protocols.interfaces.IFunctionCall import IFunctionCall, ToolByNameCallback
from just_agents.streaming.protocols.interfaces.IProtocolAdapter import IProtocolAdapter, ExecuteToolCallback
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol


class OAIFunctionCall(BaseModel, IFunctionCall[AbstractMessage], extra="allow"):
    id: str = Field(...)
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
    def reconstruct_tool_call_message(calls: Sequence['OAIFunctionCall']) -> dict:
        tool_calls = []
        for call_params in calls:
            tool_calls.append({"type": "function",
                               "id": call_params.id, "function": {"name": call_params.name, "arguments": str(call_params.arguments)}})
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}


class OAIAdapter(BaseModel, IProtocolAdapter[ModelResponse,AbstractMessage]):
    #Class that describes function convention
    function_convention: ClassVar[Type[IFunctionCall[AbstractMessage]]] = OAIFunctionCall
    #hooks to agent class
    execute_function_hook:  ExecuteToolCallback[AbstractMessage] = Field(...)
    _output_streaming: AbstractStreamingProtocol = PrivateAttr(default_factory=OpenaiStreamingProtocol)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    def completion(self, *args, **kwargs) -> ModelResponse:
        return completion(*args, **kwargs)

    async def async_completion(self, *args, **kwargs) \
            -> Coroutine[Any, Any, Union[ModelResponse, CustomStreamWrapper, AsyncGenerator]]:
        return acompletion(*args, **kwargs)

    def message_from_response(self, response: ModelResponse) -> AbstractMessage:
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

    def message_from_delta(self, response: ModelResponse) -> AbstractMessage:
        message = response.choices[0].delta.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
            exclude={"function_call"} if not response.choices[0].delta.function_call else {}  # failsafe
        )
        assert "function_call" not in message
        return message

    def content_from_delta(self, delta: AbstractMessage) -> str:
        return delta.get("content")

    def tool_calls_from_message(self, message: AbstractMessage) -> List[OAIFunctionCall]:
        # If there are no tool calls or tools available, exit the loop
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return []
        else:
            # Auto-convert each item in tool_calls to a FunctionCall instance with validation
            return [
                OAIFunctionCall(**tool_call)
                for tool_call in tool_calls
            ]

    def response_from_deltas(self, chunks: List[Any]) -> ModelResponse:
        return stream_chunk_builder(chunks)

