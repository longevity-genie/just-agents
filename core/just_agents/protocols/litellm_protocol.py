from typing import Optional, Union, ClassVar, Type, Sequence, List, Dict, Any, AsyncGenerator, Generator, Callable, get_args

from pydantic import Field, PrivateAttr, BaseModel
from functools import singledispatchmethod
from copy import deepcopy
import os
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

from openai import APIStatusError
from openai.types.chat import ChatCompletionToolParam

import litellm
from litellm import CustomStreamWrapper, completion, acompletion, stream_chunk_builder, \
                    supports_function_calling, supports_response_schema, supports_vision, supports_reasoning
from litellm.utils import Delta, Message, ModelResponse, ModelResponseStream, function_to_dict  
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

from just_agents.interfaces.function_call import IFunctionCall, ToolByNameCallback
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback
from just_agents.data_classes import Role, ToolCall, FinishReason, GoogleBuiltInTools
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.types import MessageDict
from just_agents.just_bus import JustLogBus

SupportedResponse=Union[ModelResponse, ModelResponseStream, MessageDict]

class LiteLLMFunctionCall(ToolCall, IFunctionCall[MessageDict]):
    def execute_function(self, call_by_name: ToolByNameCallback):
        function_args = self.arguments or {}
        if isinstance(function_args, str):
            function_response = f"Incorrect arguments received: '{function_args}'" #error on validation
        else:
            try:
                function_to_call = call_by_name(self.name)
                function_response = str(function_to_call(**function_args))
            except Exception as e:
                function_response = f"Error occurred during call: '{str(e)}'"
        message = {"role": Role.tool.value, "content": function_response, "name": self.name, "tool_call_id": self.id}
        return message

    @staticmethod
    def reconstruct_tool_call_message(calls: Sequence['LiteLLMFunctionCall']) -> dict:
        tool_calls = []
        for call_params in calls:
            tool_calls.append({"type": "function",
                               "id": call_params.id, "function": {"name": call_params.name, "arguments": str(call_params.arguments)}})
        return {"role": Role.assistant.value, "content": None, "tool_calls": tool_calls}


def get_valid_models() -> List[str]: #override of litellm_helper, key-independent.
    """
    Returns a list of all available LLM models from litellm, regardless of API key configuration.
    
    Returns:
        A list of all available LLM models
    """
  
    # Get all available providers from litellm
    all_providers = litellm.provider_list
    # List to store all models
    valid_models = []
    
    # For every provider, get its associated models
    for provider in all_providers:
        # For Azure, add a placeholder model name
        if provider == "azure":
            valid_models.append("Azure-LLM")
        else:
            # Get all models for this provider
            models_for_provider = litellm.models_by_provider.get(provider, [])
            valid_models.extend(models_for_provider)
            
    return valid_models
class LiteLLMAdapter(BaseModel, IProtocolAdapter[ModelResponse, MessageDict, Union[CustomStreamWrapper, CustomStreamWrapper]]):
    #Class that describes function convention

    function_convention: ClassVar[Type[IFunctionCall[MessageDict]]] = LiteLLMFunctionCall
    valid_models: ClassVar[List[str]] = get_valid_models()
    log_name: str = Field('anonymous')
    _log_bus : JustLogBus = PrivateAttr(default_factory= lambda: JustLogBus())

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.log_name = f"{self.log_name}.litellm_adapter"

    def sanitize_args(self, *args, **kwargs) -> tuple:
        """
        Synchronous method to sanitize and preprocess arguments.
        """
        source = f"{self.log_name}.sanitize_args"
        # Extract and preprocess the model from kwargs
        model = kwargs.get('model')
        messages = kwargs.get('messages')
        if '/' in model:
            provider = model.split('/')[0]
        else:
            provider = None

        if not model:
            self._log_bus.fatal(
                f"Model is required",
                source=source,
                action="essential_param_error",
                model=model
            )
            raise ValueError("model is required")
        if not messages:
            self._log_bus.error(
                f"Messages are required",
                source=source,
                action="essential_param_error",
                model=model
            )
            raise ValueError("Messages are required")

        for internal_kwarg in ["raise_on_completion_status_errors", "reconstruct_chunks"]:
            if kwargs.pop(internal_kwarg, None) is not None:
                self._log_bus.warn(
                    f"{internal_kwarg} is just-agents internal argument and will be removed from the query",
                    source=source,
                    action="param_drop",
                    model=model
                )

        for key in kwargs.keys():
            if key is None:
                kwargs.pop(key) #remove None keys

        messages = deepcopy(messages) #defnsive copy to avoid mutating original messages
        if "anthropic" in model.lower() or "claude" in model.lower() or "opus" in model.lower() or "sonnet" in model.lower():
            anthropic_reasoning = True
        else:
            anthropic_reasoning = False
        
        for message in messages:
            message["reasoning"] = message.pop("reasoning", None)
            message["reasoning_content"] = message.pop("reasoning_content", None)
            if not anthropic_reasoning: 
                message.pop("thinking_blocks", None)

        api_base = kwargs.get('api_base')

        if model not in self.valid_models:
            if not api_base:
                self._log_bus.warn(
                    f"Model is not supported by litellm by default! Validation is impossible.",
                    source=source,
                    action="validation_canceled",
                    model=model,
                    valid_models=self.valid_models
                )  
                suggestion = None
                if not provider:
                    for valid_model in self.valid_models:
                        if model.lower() in valid_model.lower():
                            suggestion = valid_model
                            break
                if suggestion:
                    self._log_bus.warn(
                        f"Litellm supports {suggestion}. Did you forget to set provider?",
                        source=source,
                        action="param_suggestion",
                        suggestion=suggestion
                    )
            else: #custom api_base provided, assume it's an OpenAi compatible model
                provider = kwargs.get("custom_llm_provider", "openai")
                self._log_bus.info(
                    f"Model is not supported by litellm by default, but custom api_base is set, using {provider} as provider",
                    source=source,
                    action="param_update",
                    model=model,
                    provider=provider
                )
                kwargs["custom_llm_provider"] = provider
            return args, kwargs

        supported_params = self.get_supported_params(model) or []
        if not "response_format" in supported_params:
            kwargs.pop("response_format", None)
            self._log_bus.warn(
                f"response_format not supported by model",
                source=source,
                action="param_drop",
                model=model
            )

        if kwargs.get("response_format", None) and not supports_response_schema(model):
            fallback = {"type": "json_object"}
            self._log_bus.warn(
                f"response_schema is not supported by model, using json_object as fallback",
                source=source,
                action="param_fallback",
                response_format=kwargs["response_format"],
                fallback_response_format=fallback,
                model=model
            )
            kwargs["response_format"] = fallback

        if "tools" in kwargs and not supports_function_calling(model):
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            self._log_bus.warn(
                f"tool calls not supported by model",
                source=source,
                action="param_drop",
                model=model
            )

        if "reasoning_effort" in kwargs and not supports_reasoning(model):
            kwargs.pop("reasoning_effort", None)
            self._log_bus.warn(
                f"reasoning_effort not supported by model",
                source=source,
                action="param_drop",
                model=model
            )
        if not kwargs.get("tools", None):
            kwargs.pop("tool_choice", None)
        
        if kwargs.get("metadata", None) is not None and not "metadata" in supported_params:
            kwargs.pop("metadata", None)
            self._log_bus.log_message(
                f"metadata not supported by model",
                source=source,
                action="param_drop",
                model=model
            )
        # Return sanitized arguments
        return args, kwargs

    def tool_from_function(self, tool: Callable, function_dict: Dict[str, Any] = None, use_litellm: bool = False
    ) -> Union[ChatCompletionToolParam, Dict[str, Any]]:
        """
        Convert a function to a tool dictionary.
        """
        source = f"{self.log_name}.tool_from_function"
        name = getattr(tool, "name", None) or tool.__name__
        #isinstance(tool, JustGoogleBuiltIn) is expensive, plus unncessary coupling
        if name in (GoogleBuiltInTools.search, GoogleBuiltInTools.code):
            self._log_bus.info(
                f"Google built-in tool {name} provided",
                source=source,
                action="google_built_in_tool",
                tool_name=name
            )
            return { name : {} } #special case for google built-in tools
        
        if function_dict:
            self._log_bus.log_message(
                f"Built-in function_dict for {tool.__name__} provided",
                source=source,
                action="input",
                input_dict=function_dict
            )
        litellm_function_dict = ""
        if use_litellm or not function_dict:
            try:
                self._log_bus.log_message(
                    "Attempting to use fallback LiteLLM implementation",
                    source=source,
                    action="numpydoc import"
                )
                litellm_function_dict = function_to_dict(tool) # type: ignore
                self._log_bus.log_message(
                    f"LiteLLM implementation for {tool.__name__}",
                    source=source,
                    action="function_to_dict.call",
                    litellm_dict=litellm_function_dict
                )

            except ImportError as e:
                self._log_bus.log_message(
                    f"Warning: Failed to use fallback LiteLLM implementation",
                    source=source,
                    action="exception",
                    error=e
                )

        function_dict = litellm_function_dict or function_dict or {}

        return ChatCompletionToolParam(
            type="function",
            function=function_dict
        )

    def completion(self, *args, **kwargs) -> Union[ModelResponse, CustomStreamWrapper, Generator]:
        # Sanitize arguments before calling the completion method
        raise_on_completion_status_errors = kwargs.pop("raise_on_completion_status_errors", False)
        args, kwargs = self.sanitize_args(*args, **kwargs)
        stream = kwargs.get("stream", None)
        model = kwargs.get("model", "")
        source = f"{self.log_name}.completion"
        try:
            return completion(*args, **kwargs)
        except APIStatusError as e:
            self._log_bus.error(
                "Error in completion",
                source=source,
                action="exception",
                args=args,
                kwargs=kwargs,
                error=e
            )
            if raise_on_completion_status_errors:
                raise e
            if stream:
                return self.create_streaming_chunks_from_text_wrapper(str(e),model,format_as_sse=False)
            else:
                return self.create_response_from_content(str(e), model)
        except Exception as e:
            self._log_bus.fatal(
                "Unhandled exception in completion!!",
                source=source,
                action="exception",
                args=args,
                kwargs=kwargs,
                error=e
            )
            raise e

    async def async_completion(self, *args, **kwargs) \
            -> Union[ModelResponse, CustomStreamWrapper, AsyncGenerator[Any, None]]:
        # Sanitize arguments before calling the async_completion method
        raise_on_completion_status_errors = kwargs.pop("raise_on_completion_status_errors", False)
        args, kwargs = self.sanitize_args(*args, **kwargs)
        stream = kwargs.get("stream", None)
        source = f"{self.log_name}.async_completion"
        try:
            return await acompletion(*args, **kwargs)
        except APIStatusError as e:
            self._log_bus.log_message(
                "Error in async_completion",
                source=source,
                action="exception",
                error=e
            )
            if raise_on_completion_status_errors:
                raise e
            if stream:
                # For streaming errors, return an async generator that yields error chunks
                async def error_stream_generator(error_msg: str, model: str) -> AsyncGenerator[ModelResponseStream, None]:
                    for chunk in self.create_streaming_chunks_from_text_wrapper(error_msg,model,format_as_sse=False):
                        yield chunk
                return error_stream_generator(str(e), kwargs.get("model", ""))
            else:
                # For non-streaming errors, return an error response
                return self.create_response_from_content(
                    str(e), 
                    kwargs.get("model", ""),
                )

    # TODO: use https://docs.litellm.ai/docs/providers/custom_llm_server as mock for tests

    def finish_reason_from_response(self, response: SupportedResponse) -> Optional[FinishReason]:
        if isinstance(response,ModelResponse) or isinstance(response,ModelResponseStream):
            return response.choices[0].finish_reason
        elif isinstance(response, dict):
            if response.get("choices",None):
                return response["choices"][0].get("finish_reason",None)
        else:
            return None

    def set_logging(self, enable: bool = True) -> None:
        """
        Enable logging and callbacks for the protocol adapter.
        Sets up Langfuse and Opik callbacks if environment variables are present.
        """
        callbacks = []
        
        if enable:
            # Check if Langfuse credentials are set
            if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
                try:
                    is_v3_plus = Version(version("langfuse")).major >= 3
                    if is_v3_plus:
                        callbacks.append("langfuse_otel")
                    else:
                        callbacks.append("langfuse")
                except PackageNotFoundError:
                    pass  # Langfuse not installed, skip
                
            # Check if Opik credentials are set    
            if os.environ.get("OPIK_API_KEY") and os.environ.get("OPIK_WORKSPACE"):
                try:
                    version("opik")  # Just check if installed
                    callbacks.append("opik")
                except PackageNotFoundError:
                    pass  # Opik not installed, skip
            
        # Set unified callbacks if any integrations are enabled
        litellm.success_callback = callbacks
        litellm.failure_callback = callbacks
        litellm.callbacks = callbacks

    def is_debug_enabled(self) -> bool:
        return litellm._is_debugging_on() # type: ignore

    def enable_debug(self) -> None:
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
            warnings='error',
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
            warnings='error',
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
            exclude={"function_call"} if not response.choices[0].message.function_call else {}  # failsafe
        )
        if "citations" in response: #TODO: investigate why not dumped
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
    def re_enumerate_tool_call_chunks(chunks: List[Any]):
        tool_calls = []
        message = None
        for chunk in chunks:
            if (
                    len(chunk["choices"]) > 0
                    and "tool_calls" in chunk["choices"][0]["delta"]
                    and chunk["choices"][0]["delta"]["tool_calls"]
            ):
                message = stream_chunk_builder(chunks=[chunk, chunks[-1]])
                tool_calls.append(message.choices[0].message.tool_calls[0])
        message.choices[0].message.tool_calls = tool_calls
        return message

    @staticmethod
    def response_from_deltas(chunks: List[ModelResponseStream]) -> ModelResponse:
        return stream_chunk_builder(chunks=chunks)

    @staticmethod
    def response_from_deltas_regression_fallback(chunks: List[ModelResponseStream]) -> ModelResponse:
        # litellm regression #https://github.com/BerriAI/litellm/issues/10034
        if "llama" in chunks[-1]["model"] and chunks[-1].choices[0].finish_reason=="tool_calls":
           return LiteLLMAdapter.re_enumerate_tool_call_chunks(chunks) # bug fix
        return LiteLLMAdapter.response_from_deltas(chunks=chunks)

    @staticmethod
    def get_supported_params(model_name: str) -> Optional[list]:
        return get_supported_openai_params(model_name)  

    @staticmethod
    def supports_system_messages(model_name: str) -> Optional[list]:
        return get_supported_openai_params(model_name) 

    @staticmethod
    def supports_reasoning(model_name: str) -> bool:
        return supports_reasoning(model_name)
    
    @staticmethod
    def supports_response_schema(model_name: str) -> bool:
        return supports_response_schema(model_name)
    
    @staticmethod
    def supports_function_calling(model_name: str) -> bool:
        return supports_function_calling(model_name)
    
    @staticmethod
    def supports_vision(model_name: str) -> bool:
        return supports_vision(model_name)

    @staticmethod
    def create_response_from_content(
        content: str,
        model: str,
        role: str = Role.assistant.value,
        finish_reason: str = FinishReason.stop.value,
        usage: Optional[Dict[str, int]] = None
    ) -> ModelResponse:
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
        # Decode content if it's a bytes object
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Use the more comprehensive helper method instead of reimplementing similar functionality
        return ModelResponse(**IProtocolAdapter.create_complete_response(
            content_str=content,
            model=model,
            role=role,
            finish_reason=finish_reason,
            is_streaming=False,
            include_usage=usage is not None,  # Only include usage if it was provided
            usage=usage,
            include_token_details=False
        ))
    

    @staticmethod
    def create_chunk_from_content(
        delta: Union[MessageDict, Delta, str, None], 
        model: str, 
        role: Optional[str] = None,
        chunk_id: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None
    ) -> ModelResponseStream:
        """
        Creates an OpenAI-compatible streaming chunk from the given content.
        
        Args:
            delta: The content delta (can be MessageDict, Delta object, string, or None)
            model: The model name to include in the chunk
            role: Optional role to include in the message
            chunk_id: Optional custom ID for the chunk
            finish_reason: Optional reason the generation finished
            usage: Optional usage statistics
            
        Returns:
            A dictionary formatted as an OpenAI API streaming chunk
        """
        # Process the delta content - keep type handling at this level
        if delta is None:
            message_str = None
        elif isinstance(delta, str):
            message_str = delta
        elif isinstance(delta, Delta):
            message_str = delta.content 
        else:
            message_str = str(delta)

        return ModelResponseStream(**IProtocolAdapter.create_complete_response(
            content_str=message_str,
            model=model,
            role=role,
            response_id=chunk_id,
            finish_reason=finish_reason,
            is_streaming=True,
            include_usage=usage is not None,  # Only include usage if it was provided
            usage=usage,
            include_token_details=False
        ))

    def create_streaming_chunks_from_text_wrapper(
        self,
        content: str,
        model: str,
        prompt_text: str = "",
        role: str = Role.assistant.value,
        finish_reason: str = FinishReason.stop.value,
        response_id: Optional[str] = None,
        include_usage: bool = True,
        include_token_details: bool = False,
        format_as_sse: bool = False
    ) -> Generator[Union[ModelResponseStream, Dict[str, Any]], None, None]:
        """
        Creates a generator that yields a series of chunks mimicking the OpenAI streaming protocol. 
        The sequence follows a standard pattern of content -> finish -> usage.
        """
        # Decode content if it's a bytes object
        if isinstance(content, bytes):
            content = content.decode('utf-8')
            
        for chunk in IProtocolAdapter.create_streaming_chunks_from_text( #wraping low level function
            content=content,
            model=model,
            prompt_text=prompt_text,
            role=role,
            finish_reason=finish_reason,
            response_id=response_id,
            include_usage=include_usage,
            include_token_details=include_token_details,
            format_as_sse=False
        ):
            model_chunk = ModelResponseStream(**chunk) #validate chunk
            if format_as_sse:
                yield SSE.sse_wrap(model_chunk.model_dump())
            else:
                yield model_chunk

            if format_as_sse:
                yield SSE.sse_wrap(IProtocolAdapter.stop)



        


        
   



