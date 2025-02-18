from pyexpat.errors import messages
from typing import Optional, Union, Coroutine, ClassVar, Type, Sequence, List, Any, AsyncGenerator, Generator
import time

from pydantic import Field, PrivateAttr, BaseModel
from functools import singledispatchmethod
import litellm
import os
from litellm import CustomStreamWrapper, completion, acompletion, stream_chunk_builder
from litellm._logging import _is_debugging_on
from litellm.types.utils import Delta, Message, ModelResponse, ModelResponseStream
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

from just_agents.interfaces.function_call import IFunctionCall, ToolByNameCallback
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback
from just_agents.data_classes import Role, ToolCall, FinishReason
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.types import MessageDict

SupportedResponse=Union[ModelResponse, ModelResponseStream, MessageDict]

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

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    def completion(self, *args, **kwargs) -> Union[ModelResponse, CustomStreamWrapper]: # for the stream it is CustomStreamWrapper
        return completion(*args, **kwargs)

    async def async_completion(self, *args, **kwargs) \
            -> Coroutine[Any, Any, Union[ModelResponse, CustomStreamWrapper, AsyncGenerator]]:
        return acompletion(*args, **kwargs)
    
    # TODO: use https://docs.litellm.ai/docs/providers/custom_llm_server as mock for tests

    def finish_reason_from_response(self, response: SupportedResponse) -> Optional[FinishReason]:
        if isinstance(response,ModelResponse) or isinstance(response,ModelResponseStream):
            return response.choices[0].finish_reason
        elif isinstance(response, dict):
            if response.get("choices",None):
                return response["choices"][0].get("finish_reason",None)
        else:
            return None

    def enable_logging(self) -> None:
        """
        Enable logging and callbacks for the protocol adapter.
        Sets up Langfuse and Opik callbacks if environment variables are present.
        """
        callbacks = []
        
        # Check if Langfuse credentials are set
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            callbacks.append("langfuse")
            
        # Check if Opik credentials are set    
        if os.environ.get("OPIK_API_KEY") and os.environ.get("OPIK_WORKSPACE"):
            callbacks.append("opik")
            
        # Set unified callbacks if any integrations are enabled
        if callbacks:
            litellm.success_callback = callbacks
            litellm.failure_callback = callbacks
            litellm.callbacks = callbacks

    def debug_enabled(self) -> bool:
        return litellm._is_debugging_on()

    def enable_debug(self) -> None:
        """
        Enable debug mode for the protocol adapter.
        """
        litellm._turn_on_debug()

    def disable_logging(self) -> None:
        """
        Disable logging mode for the protocol adapter by removing all callbacks.
        """
        litellm.callbacks = []  # Reset callbacks to empty list


    @singledispatchmethod
    def message_from_response(self, response: SupportedResponse) -> MessageDict:
        """
        Overrides the abstract method and provides dispatching to specific handlers.
        """
        raise TypeError(f"Unsupported response format: {type(response)}")

    @message_from_response.register
    def delta_from_stream(self, response: ModelResponseStream) -> MessageDict:
        """
        Streaming model contains no message section, only delta.
        """
        return response.choices[0].delta.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
            exclude={"function_call"} if not response.choices[0].delta.function_call else {}  # failsafe
        ) or {}

    @message_from_response.register
    def message_from_model_response(self, response: ModelResponse) -> MessageDict:
        """
        ModelResponse has a required message section.
        """
        message = response.choices[0].message.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
            exclude={"function_call"} if not response.choices[0].message.function_call else {}  # failsafe
        )
        if "citations" in response: #TODO: investigate why not dmped
            message["citations"] = response.citations  # perplexity specific field
        return message or {}

    @message_from_response.register
    def message_from_dict(self, response: dict) -> MessageDict:
        """
        Noting is definite for dict, check everything.
        """
        message = {}
        if response.get("choices", None):
            choice = response["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message", {}) or choice.get("delta", {})
        return message or {}

    def content_from_delta(self, delta: Union[MessageDict,Delta]) -> str:
        if isinstance(delta, Delta):
            return delta.content or ''
        elif isinstance(delta, dict):
            return delta.get("content", '') or ''
        else:
            return ''


    def tool_calls_from_message(self, message: Union[MessageDict,Message]) -> List[LiteLLMFunctionCall]:
        # If there are no tool calls or tools available, exit the loop
        if isinstance(message,Message):
            tool_calls = [ tool.model_dump() for tool in message.tool_calls]
        else:
            tool_calls = message.get("tool_calls")
        if not tool_calls:
            return []
        else:
            # Auto-convert each item in tool_calls to a FunctionCall instance with validation
            return [
                LiteLLMFunctionCall(**tool_call)
                for tool_call in tool_calls
            ]

    def response_from_deltas(self, chunks: List[ModelResponseStream]) -> ModelResponse:
        return stream_chunk_builder(chunks=chunks)

    def get_supported_params(self, model_name: str) -> Optional[list]:
        return get_supported_openai_params(model_name)  # type: ignore

    def message_as_chunk(self, index: int, delta: Union[MessageDict,Delta, str], model: str, role: Optional[str] = None) -> MessageDict:
        if isinstance(delta,str):
            message : dict = {"content": delta}
        elif isinstance(delta, Delta):
            message : dict = {"content": self.content_from_delta(delta)}
        else:
            message : dict = delta
        if role:
            message["role"] = role
        chunk : dict = {
            "id": index,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": model,
            "choices": [{"delta": message}],
        }
        return chunk

    @staticmethod
    def content_from_stream(self, stream_generator: Generator, stop: str) -> str:
        response_content = ""
        for chunk in stream_generator:
            try:
                # Parse the SSE data
                data = SSE.sse_parse(chunk)
                json_data = data.get("data", "{}")
                if json_data == stop:
                    break
                print(json_data)
                if "choices" in json_data and len(json_data["choices"]) > 0:
                    delta = json_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        response_content += delta["content"]
            except Exception as e:
                continue
        return response_content