from typing import Optional, Union, Coroutine, ClassVar, Type, Sequence, List, Any, AsyncGenerator
from pydantic import Field, PrivateAttr, BaseModel

from litellm import ModelResponse, CustomStreamWrapper, GenericStreamingChunk, completion, acompletion, stream_chunk_builder
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

from just_agents.interfaces.function_call import IFunctionCall, ToolByNameCallback
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from just_agents.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.protocols.openai_classes import Role, ToolCall

from just_agents.types import MessageDict

class LiteLLMFunctionCall(ToolCall, IFunctionCall[MessageDict]):
    def execute_function(self, call_by_name: ToolByNameCallback):
        function_args = self.arguments or {}
        if isinstance(function_args, str):
            function_response = function_args #error on validation
        else:
            try:
                function_to_call = call_by_name(self.name)
                function_response = str(function_to_call(**function_args))
            except Exception as e:
                function_response = str(e)
        message = {"role": Role.tool.value, "content": function_response, "name": self.name, "tool_call_id": self.id}
        return message

    @staticmethod
    def reconstruct_tool_call_message(calls: Sequence['LiteLLMFunctionCall']) -> dict:
        tool_calls = []
        for call_params in calls:
            tool_calls.append({"type": "function",
                               "id": call_params.id, "function": {"name": call_params.name, "arguments": str(call_params.arguments)}})
        return {"role": Role.assistant.value, "content": None, "tool_calls": tool_calls}


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
    
    # TODO: use https://docs.litellm.ai/docs/providers/custom_llm_server as mock for tests

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

    def delta_from_response(self, response: GenericStreamingChunk) -> MessageDict: #TODO: fix typehint/model choice
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

    def response_from_deltas(self, chunks: List[Any]) -> ModelResponse:
        return stream_chunk_builder(chunks=chunks)

    def get_supported_params(self, model_name: str) -> Optional[list]:
        return get_supported_openai_params(model_name)  # type: ignore

