from abc import ABC, abstractmethod
import uuid
import time
import json

from pydantic import BaseModel
from typing import Callable, Union, AsyncGenerator, List, Sequence, ClassVar, Type, TypeVar, Generic, Any, Optional, Dict, Generator
from just_agents.protocols.sse_streaming import ServerSentEventsStream as SSE
from just_agents.interfaces.function_call import IFunctionCall

BaseModelResponse = TypeVar('BaseModelResponse', bound=BaseModel)
BaseModelStreamResponse = TypeVar('BaseModelStreamResponse', bound=BaseModel)
AbstractMessage = TypeVar("AbstractMessage")

ModelResponseCallback=Callable[...,BaseModelResponse]
MessageUnpackCallback=Callable[[BaseModelResponse], AbstractMessage]
ExecuteToolCallback=Callable[[Sequence[IFunctionCall]],List[AbstractMessage]]

class IProtocolAdapter(ABC, Generic[BaseModelResponse, BaseModelStreamResponse, AbstractMessage]):
    """
    Class that is required to wrap the model protocol
    """
    stop: ClassVar[str] = "[DONE]"
    
    function_convention: ClassVar[Type[IFunctionCall[Any]]]

    @abstractmethod
    def tool_from_function(self, tool: Callable, **kwargs) -> dict:
        """Convert a function to a tool dictionary."""
        raise NotImplementedError("You need to implement tool_from_function first!")
    
    @abstractmethod
    def sanitize_args(self, *args, **kwargs) -> tuple:
        """Sanitize and preprocess arguments before calling completion methods."""
        raise NotImplementedError("You need to implement sanitize_args first!")
    
    @abstractmethod
    def completion(self, *args, **kwargs) -> Union[BaseModelResponse, Generator[BaseModelStreamResponse, None, None]]:
        """Synchronous completion method."""
        raise NotImplementedError("You need to implement completion first!")

    @abstractmethod
    async def async_completion(self, *args, **kwargs) \
            -> Union[BaseModelResponse, AsyncGenerator[BaseModelStreamResponse, None]]:
        """Asynchronous completion method."""
        raise NotImplementedError("You need to implement async_completion first!")

    @abstractmethod
    def finish_reason_from_response(self, response: Union[AbstractMessage, BaseModelResponse, BaseModelStreamResponse]) -> Any:
        """Extract finish reason from a response."""
        raise NotImplementedError("You need to implement finish_reason_from_response first!")

    @abstractmethod
    def message_from_response(self, response: Union[AbstractMessage, BaseModelResponse, BaseModelStreamResponse]) -> AbstractMessage:
        """Extract message from a response."""
        raise NotImplementedError("You need to implement message_from_response first!")

    @staticmethod
    @abstractmethod
    def content_from_delta(delta: Any) -> str:
        """Extract content from a delta message."""
        raise NotImplementedError("You need to implement content_from_delta first!")

    @staticmethod
    @abstractmethod
    def tool_calls_from_message(message: AbstractMessage) -> List[IFunctionCall[AbstractMessage]]:
        """Extract tool calls from a message."""
        raise NotImplementedError("You need to implement tool_calls_from_message first!")

    @staticmethod
    @abstractmethod
    def response_from_deltas(deltas: List[AbstractMessage]) -> BaseModelResponse:
        """Combine stream deltas into a complete response."""
        raise NotImplementedError("You need to implement response_from_deltas first!")


    @staticmethod
    @abstractmethod
    def create_response_from_content(content: str, model: str, **kwargs) -> Union[BaseModelResponse, AbstractMessage]:
        """Create a response dictionary from content."""
        raise NotImplementedError("You need to implement create_response_from_content first!")

    @staticmethod
    @abstractmethod
    def create_chunk_from_content( delta: Any, model: str, **kwargs) -> Union[BaseModelStreamResponse, AbstractMessage]:
        """Create a chunk from content for streaming."""
        raise NotImplementedError("You need to implement create_chunk_from_content first!")

    @staticmethod
    @abstractmethod
    def get_supported_params( model_name: str) -> Optional[list]:
        """Get supported parameters for a specific model."""
        raise NotImplementedError("You need to implement get_supported_params first!")


    @staticmethod
    @abstractmethod
    def supports_system_messages(model_name: str) -> Optional[list]:
        """Check if a model supports system messages."""
        raise NotImplementedError("You need to implement supports_system_messages first!")


    @staticmethod
    @abstractmethod
    def supports_response_schema(model_name: str) -> bool:
        """Check if a model supports response schema."""
        raise NotImplementedError("You need to implement supports_response_schema first!")

    @staticmethod
    @abstractmethod
    def supports_function_calling(model_name: str) -> bool:
        """Check if a model supports function calling."""
        raise NotImplementedError("You need to implement supports_function_calling first!")

    @staticmethod
    @abstractmethod
    def supports_vision(model_name: str) -> bool:
        """Check if a model supports vision."""
        raise NotImplementedError("You need to implement supports_vision first!")

    @staticmethod
    @abstractmethod
    def create_streaming_chunks_from_text_wrapper(self, content: str, model: str, **kwargs) \
    -> Generator[Union[BaseModelStreamResponse, AbstractMessage], None, None]:
        """Create a generator that yields a series of chunks wrapped in a model object mimicking the OpenAI streaming protocol."""
        raise NotImplementedError("You need to implement create_streaming_chunks_from_text_wraper first!")

    @abstractmethod
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        raise NotImplementedError("You need to implement debug_enabled first!")

    @abstractmethod
    def enable_debug(self) -> None:
        """Enable debug mode."""
        raise NotImplementedError("You need to implement enable_debug first!")

    @abstractmethod
    def set_logging(self, enable: bool = True) -> None:
        """Enable logging."""
        raise NotImplementedError("You need to implement set_logging first!")

    @abstractmethod
    def disable_logging(self) -> None:
        """Disable logging."""
        raise NotImplementedError("You need to implement disable_logging first!")

    @staticmethod
    def get_chat_completion_id() -> Optional[str]:
        return f"chatcmpl-{str(uuid.uuid4())}"

    @staticmethod
    def create_base_response(
        model: str,
        is_chunk: bool = False,
        response_id: Optional[str] = None,
        created_timestamp: Optional[int] = None,
        choices: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Creates the low-level common structure for both chunks and full responses.
        
        Args:
            model: The model name to include in the response
            is_chunk: Whether this is a streaming chunk (True) or full response (False)
            response_id: Optional custom ID (will generate one if not provided)
            created_timestamp: Optional creation timestamp (will use current time if not provided)
            choices: List of choices to include in the response
            usage: Optional usage statistics
            
        Returns:
            A dictionary with the common OpenAI-compatible response structure
        """
        # Create the base response structure
        response = {
            "id": response_id or IProtocolAdapter.get_chat_completion_id(),
            "object": "chat.completion.chunk" if is_chunk else "chat.completion",
            "created": created_timestamp or int(time.time()),
            "model": model,
            "choices": choices or []
        }
        
        # Add usage statistics if provided
        if usage:
            response["usage"] = usage
            
        return response
    
    @staticmethod
    def create_choice(
        message_content: Dict[str, Any],
        index: int = 0,
        finish_reason: Optional[str] = None,
        is_chunk: bool = False
    ) -> Dict[str, Any]:
        """
        Creates a low-level choice object for either a chunk or full response.
        
        Args:
            message_content: The message content as a dictionary
            index: The index of this choice in the array (default: 0)
            finish_reason: Optional reason the generation finished
            is_chunk: Whether this is a streaming chunk (True) or full response (False)
            
        Returns:
            A dictionary representing a choice object compatible with OpenAI API
        """

        # Create the choice object
        choice = {
            "index": index
        }
        
        # Add either delta or message based on whether this is a chunk
        if is_chunk:
            choice["delta"] = message_content
        else:
            choice["message"] = message_content
            
        # Add finish reason if provided
        if finish_reason:
            choice["finish_reason"] = finish_reason
            
        return choice
    
    @staticmethod
    def create_usage(
        prompt_text: str = "",
        completion_text: str = "",
        include_details: bool = False,
        audio_prompt_tokens: Optional[int] = None,
        audio_completion_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Creates an OpenAI-compatible usage statistics dictionary with simple token approximation.
        
        Args:
            prompt_text: The text of the prompt for token estimation
            completion_text: The text of the completion for token estimation
            include_details: Whether to include detailed token breakdown
            audio_prompt_tokens: Optional count of audio tokens in the prompt
            audio_completion_tokens: Optional count of audio tokens in the completion
            cached_tokens: Optional count of cached tokens in the prompt
            reasoning_tokens: Optional count of reasoning tokens in the completion
            
        Returns:
            A dictionary formatted as OpenAI API usage statistics
        """
        # Calculate token counts with simple approximation (len*2)
        prompt_tokens = len(prompt_text) * 2 if prompt_text else 0
        completion_tokens = len(completion_text) * 2 if completion_text else 0
        total_tokens = prompt_tokens + completion_tokens
        
        # Create base usage dictionary
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Add token details if requested
        if include_details:
            # Add prompt details if any detail is provided
            if any(x is not None for x in [audio_prompt_tokens, cached_tokens]):
                prompt_details = {}
                
                if audio_prompt_tokens is not None:
                    prompt_details["audio_tokens"] = audio_prompt_tokens
                    
                if cached_tokens is not None:
                    prompt_details["cached_tokens"] = cached_tokens
                    
                usage["prompt_tokens_details"] = prompt_details
            
            # Add completion details if any detail is provided
            if any(x is not None for x in [audio_completion_tokens, reasoning_tokens]):
                completion_details = {}
                
                if audio_completion_tokens is not None:
                    completion_details["audio_tokens"] = audio_completion_tokens
                    
                if reasoning_tokens is not None:
                    completion_details["reasoning_tokens"] = reasoning_tokens
                    
                usage["completion_tokens_details"] = completion_details
        
        return usage

    @staticmethod
    def create_complete_response(
        content_str: Optional[str],
        model: str,
        prompt_text: str = "",
        is_streaming: bool = False,
        role: str = None,
        finish_reason: Optional[str] = None,
        response_id: Optional[str] = None,
        created_timestamp: Optional[int] = None,
        include_usage: bool = True,
        usage: Optional[Dict[str, int]] = None,
        include_token_details: bool = False
    ) -> Dict[str, Any]:
        """
        Creates a complete OpenAI-compatible response (streaming or full) with a single choice,
        using the three factory methods (create_base_response, create_choice, create_usage).
        
        Args:
            content_str: The content string for the response (can be None for streaming)
            model: The model name to include in the response
            prompt_text: The original prompt text, used for token estimation
            is_streaming: Whether this is a streaming response (chunk) or full response
            role: The role of the message (default: assistant)
            finish_reason: The reason the generation finished (default: stop for full, None for streaming)
            response_id: Optional custom ID for the response
            created_timestamp: Optional creation timestamp
            include_usage: Whether to include usage statistics
            usage: Optional usage statistics dictionary with keys like 'prompt_tokens', 
                   'completion_tokens', and 'total_tokens'
            include_token_details: Whether to include detailed token breakdown
            
        Returns:
            A complete dictionary formatted as an OpenAI API response
        """
        # Process input content based on type
        if is_streaming:
            # For streaming, process delta content
            if content_str is None:
                message_content = {"content": None}
            elif isinstance(content_str, str):
                message_content = {"content": content_str or ''}
            else:
                message_content = {"content": str(content_str)}
                
            # Add role if provided for streaming
            if role:
                message_content["role"] = role
        else:
            # For full response, create message object
            message_content = {
                "role": role,
            }
            if content_str is None:
                message_content["content"] = ""
            elif isinstance(content_str, str):
                message_content["content"] = content_str
            else:
                message_content["content"] = str(content_str)


        # Set appropriate default finish reason if not provided
        if finish_reason is None and not is_streaming:
            finish_reason = "stop"
            
        # Create the choice object using the helper method
        choice = IProtocolAdapter.create_choice(
            message_content=message_content,
            index=0,
            finish_reason=finish_reason,
            is_chunk=is_streaming
        )
        
        # Create usage statistics if requested
        if include_usage and not usage:
            completion_text = ""
            if not is_streaming:  # Only calculate completion tokens for full responses
                completion_text = content_str if content_str else ""
                
            usage = IProtocolAdapter.create_usage(
                prompt_text=prompt_text,
                completion_text=completion_text,
                include_details=include_token_details
            )
        
        # Create the final response using the base response helper
        return IProtocolAdapter.create_base_response(
            model=model,
            is_chunk=is_streaming,
            response_id=response_id,
            created_timestamp=created_timestamp,
            choices=[choice],
            usage=usage
        )


    @staticmethod
    def create_streaming_chunks_from_text(
        content: str,
        model: str,
        prompt_text: str = "",
        role: str = "assistant",
        finish_reason: str = "stop",
        response_id: Optional[str] = None,
        include_usage: bool = True,
        usage: Optional[Dict[str, int]] = None,
        include_token_details: bool = False,
        format_as_sse: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Creates a generator that yields a series of chunks mimicking the OpenAI streaming protocol.
        The sequence follows a standard pattern of content -> finish -> usage.
        
        Args:
            content: The text content to stream
            model: The model name to include in the response
            prompt_text: The original prompt text, used for token estimation
            role: The role of the message (default: assistant)
            finish_reason: The reason the generation finished
            response_id: Optional custom ID for the chunks
            include_usage: Whether to include usage statistics in the final chunk
            usage: Optional usage statistics dictionary with keys like 'prompt_tokens', 
                   'completion_tokens', and 'total_tokens'
            include_token_details: Whether to include detailed token breakdown
            format_as_sse: Whether to format the output as Server-Sent Events
            
        Returns:
            A generator yielding streaming chunks in OpenAI format
        """
        # Generate a response ID to use for all chunks
        chunk_id = response_id or IProtocolAdapter.get_chat_completion_id()
        
        # 1. First chunk: Content with role
        first_chunk = IProtocolAdapter.create_complete_response(
            content_str=content,
            response_id=chunk_id,
            model=model,
            role=role,
            is_streaming=True,
            include_usage=False,
            finish_reason=None,
            include_token_details=include_token_details
        )
        
        if format_as_sse:
            yield SSE.sse_wrap(first_chunk)
        else:
            yield first_chunk
        
        # 2. Second chunk: Null content with finish reason
        finish_chunk = IProtocolAdapter.create_complete_response(
            content_str=None,
            model=model,
            role=None,
            is_streaming=True,
            include_usage=False,
            include_token_details=include_token_details,
            finish_reason=finish_reason,
            response_id=chunk_id
        )
        
        if format_as_sse:
            yield SSE.sse_wrap(finish_chunk)
        else:
            yield finish_chunk
        
        # 3. Final chunk with usage if requested
        if include_usage:
            # Create usage dictionary
            usage = usage or IProtocolAdapter.create_usage(
                prompt_text=prompt_text,
                completion_text=content,
                include_details=include_token_details
            )
            
            # Create final chunk with usage
            usage_chunk = IProtocolAdapter.create_complete_response(
                content_str=None,
                model=model,
                role=None,
                is_streaming=True,
                include_usage=True,
                include_token_details=include_token_details,
                finish_reason=None,
                response_id=chunk_id,
                usage=usage
            )
            
            if format_as_sse:
                yield SSE.sse_wrap(usage_chunk)
            else:
                yield usage_chunk
        
        # 4. Optional stop signal if using SSE format
        if format_as_sse:
            yield SSE.sse_wrap(IProtocolAdapter.stop)



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

