from typing import Optional, Union, ClassVar, Type, Sequence, List, Dict, Any, AsyncGenerator, Generator, Callable
import time
import openai
import uuid
import json
from pydantic import Field, PrivateAttr, BaseModel
from functools import singledispatchmethod
import litellm
import os

from litellm import CustomStreamWrapper, completion, acompletion, stream_chunk_builder
from litellm.utils import Delta, Message, ModelResponse, ModelResponseStream, function_to_dict
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

from just_agents.interfaces.function_call import IFunctionCall, ToolByNameCallback
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback
from just_agents.data_classes import Role, ToolCall, FinishReason
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.types import MessageDict
from just_agents.just_bus import JustLogBus

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
    _log_bus : JustLogBus = PrivateAttr(default_factory= lambda: JustLogBus())

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    def sanitize_args(self, *args, **kwargs) -> tuple:
        """
        Synchronous method to sanitize and preprocess arguments.
        """
        # Extract and preprocess the model from kwargs
        model = kwargs.get('model')
        if not model:
            raise ValueError("model is required")

        supported_params = self.get_supported_params(model) or []
        if not "response_format" in supported_params:
            kwargs.pop("response_format", None)
            self._log_bus.log_message(
                f"Warning: response_format not supported by model",
                source="litellm_protocol_adapter.sanitize_args",
                action="param_drop",
                model=model
            )
        if not litellm.supports_function_calling(model):
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            self._log_bus.log_message(
                f"Warning: tool calls not supported by model",
                source="litellm_protocol_adapter.sanitize_args",
                action="param_drop",
                model=model
            )

        if kwargs.pop("raise_on_completion_status_errors", None) is not None:
            self._log_bus.log_message(
                f"Warning: raise_on_completion_status_errors not supported by model",
                source="litellm_protocol_adapter.sanitize_args",
                action="param_drop",
                model=model
            )
        
        if kwargs.get("metadata", None) is not None and not "metadata" in supported_params:
            kwargs.pop("metadata", None)
            self._log_bus.log_message(
                f"Warning: metadata not supported by model",
                source="litellm_protocol_adapter.sanitize_args",
                action="param_drop",
                model=model
            )
        # Return sanitized arguments
        return args, kwargs

    def tool_from_function(self, tool: Callable, function_dict: Dict[str, Any] = None, use_litellm: bool = False) -> dict:
        """
        Convert a function to a tool dictionary.
        """

        if function_dict:
            self._log_bus.log_message(
                f"Built-in function_dict for {tool.__name__} provided",
                source="litellm_protocol_adapter.tool_from_function",
                action="input",
                input_dict=function_dict
            )
        litellm_function_dict = ""
        if use_litellm or function_dict is None:
            try:
                self._log_bus.log_message(
                    "Attempting to use fallback LiteLLM implementation",
                    source="litellm_protocol_adapter.tool_from_function",
                    action="numpydoc import"
                )
                litellm_function_dict = function_to_dict(tool) # type: ignore
                self._log_bus.log_message(
                    f"LiteLLM implementation for {tool.__name__}",
                    source="litellm_protocol_adapter.tool_from_function",
                    action="function_to_dict.call",
                    litellm_dict=litellm_function_dict
                )

            except ImportError as e:
                self._log_bus.log_message(
                    f"Warning: Failed to use fallback LiteLLM implementation",
                    source="litellm_protocol_adapter.tool_from_function",
                    action="exception",
                    error=e
                )
        function_dict = litellm_function_dict or function_dict or ""
        return {"type": "function","function": function_dict}

    def completion(self, *args, **kwargs) -> Union[ModelResponse, CustomStreamWrapper, Generator]:
        # Sanitize arguments before calling the completion method
        raise_on_completion_status_errors = kwargs.pop("raise_on_completion_status_errors", False)
        args, kwargs = self.sanitize_args(*args, **kwargs)
        stream = kwargs.get("stream", None)
        
        # Define an error stream generator function outside the try block
        def error_stream_generator(error_msg: str, model: str):
            yield self.create_chunk_from_content(0, error_msg, model, Role.assistant.value)
            yield f"data: {self.stop}\n\n"
        
        try:
            return litellm.completion(*args, **kwargs)
        except openai.APIStatusError as e:
            self._log_bus.log_message(
                "Error in completion",
                source="litellm_protocol_adapter.completion",
                action="exception",
                error=e
            )
            if raise_on_completion_status_errors:
                raise e
            if stream:
                return error_stream_generator(str(e), kwargs.get("model", ""))
            else:
                return ModelResponse(**self.create_response_from_content(
                    str(e), 
                    kwargs.get("model", ""), 
                    Role.assistant.value
                ))


    async def async_completion(self, *args, **kwargs) \
            -> Union[ModelResponse, CustomStreamWrapper, AsyncGenerator[Any, None]]:
        # Sanitize arguments before calling the async_completion method
        raise_on_completion_status_errors = kwargs.pop("raise_on_completion_status_errors", False)
        args, kwargs = self.sanitize_args(*args, **kwargs)
        stream = kwargs.get("stream", None)
        try:
            return await litellm.acompletion(*args, **kwargs)
                
        except Exception as e:
            self._log_bus.log_message(
                "Error in async_completion",
                source="litellm_protocol_adapter.async_completion",
                action="exception",
                error=e
            )
            if raise_on_completion_status_errors:
                raise e
            if stream:
                # For streaming errors, return an async generator that yields error chunks
                async def error_stream_generator():
                    yield self.create_chunk_from_content(0, str(e), kwargs.get("model", ""), Role.assistant.value)
                    yield f"data: {self.stop}\n\n"
                
                return error_stream_generator()
            else:
                # For non-streaming errors, return an error response
                return ModelResponse(**self.create_response_from_content(str(e), kwargs.get("model", ""), Role.assistant.value))

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
        return litellm._is_debugging_on() # type: ignore

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

    @staticmethod
    def content_from_delta(delta: Union[MessageDict, Delta]) -> str:
        """
        Extract content from a delta object.
        
        Args:
            delta: Delta object from model response, can be Delta class, dict, or other
            
        Returns:
            Extracted content as string
        """
        if isinstance(delta, Delta):
            return delta.content or ''
        elif isinstance(delta, dict):
            return delta.get("content", '') or ''
        elif isinstance(delta, str):
            return delta
        else:
            return ''

    @staticmethod
    def tool_calls_from_message(message: Union[MessageDict,Message]) -> List[LiteLLMFunctionCall]:
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

    @staticmethod
    def response_from_deltas(chunks: List[ModelResponseStream]) -> ModelResponse:
        return stream_chunk_builder(chunks=chunks)

    @staticmethod
    def get_supported_params(model_name: str) -> Optional[list]:
        return get_supported_openai_params(model_name)  # type: ignore

    @staticmethod
    def create_chunk_from_content(index: int, delta: Union[MessageDict,Delta, str], model: str, role: Optional[str] = None) -> MessageDict:
        if isinstance(delta,str):
            message : dict = {"content": delta}
        elif isinstance(delta, Delta):
            message : dict = {"content": delta.content or ''}
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
    def create_response_from_content(
        content: str,
        model: str,
        role: str = Role.assistant.value,
        finish_reason: str = FinishReason.stop.value,
        usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Creates an OpenAI-compatible response dictionary from the given content.
        
        Args:
            content: The text content of the response
            model: The model name to include in the response
            role: The role of the message (default: assistant)
            finish_reason: The reason the generation finished (default: stop)
            usage: Optional usage statistics dictionary with keys like 'prompt_tokens', 
                   'completion_tokens', and 'total_tokens'
            
        Returns:
            A dictionary formatted as an OpenAI API response
        """
        # Create the message object
        message = {
            "role": role,
            "content": content
        }
        
        # Create the response dictionary with OpenAI-compatible structure
        response = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason
                }
            ]
        }
        
        # Add usage statistics if provided
        if usage:
            response["usage"] = usage
        
        # self._log_bus.log_message(
        #     f"Created OpenAI-compatible response",
        #     source="litellm_protocol_adapter.create_response",
        #     action="output",
        #     response=response
        # )
        
        return response

    @staticmethod
    def content_from_stream(stream_generator: Generator, stop: str = None) -> str:
        """
        Extract content from a stream generator by parsing SSE data.
        
        Args:
            stream_generator: Generator yielding SSE chunks
            stop: Optional stop token to terminate stream processing
            
        Returns:
            Concatenated content from all delta chunks
        """
        stop = stop or IProtocolAdapter.stop
        response_content = ""
        for chunk in stream_generator:
            try:
                # Parse the SSE data
                data = SSE.sse_parse(chunk)
                json_data = data.get("data", "{}")
                if json_data == stop:
                    break
                # Extract content from delta if available
                json_dict: Dict[str, Any] = {}
                if isinstance(json_data, dict):
                    json_dict = json_data
                elif isinstance(json_data, str):
                    json_dict = json.loads(json_data)
                if "choices" in json_dict and len(json_dict["choices"]) > 0:
                    delta = json_dict["choices"][0].get("delta", {})
                    if "content" in delta:
                        response_content += delta["content"]
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                # Only catch specific exceptions related to parsing
                continue
        return response_content

