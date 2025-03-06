from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Callable, Coroutine, Union, AsyncGenerator, List, Sequence, ClassVar, Type, TypeVar, Generic, Any, Optional, Dict, Generator

from just_agents.interfaces.function_call import IFunctionCall

BaseModelResponse = TypeVar('BaseModelResponse', bound=BaseModel)
BaseModelStreamResponse = TypeVar('BaseModelStreamResponse', bound=BaseModel)
AbstractMessage = TypeVar("AbstractMessage")

ModelResponseCallback=Callable[...,BaseModelResponse]
MessageUnpackCallback=Callable[[BaseModelResponse], AbstractMessage]
ExecuteToolCallback=Callable[[Sequence[IFunctionCall]],List[AbstractMessage]]

class IProtocolAdapter(ABC, Generic[BaseModelResponse, AbstractMessage, BaseModelStreamResponse]):
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
    def completion(self, *args, **kwargs) -> Union[BaseModelResponse, BaseModelStreamResponse, Generator]:
        """Synchronous completion method."""
        raise NotImplementedError("You need to implement completion first!")

    @abstractmethod
    async def async_completion(self, *args, **kwargs) \
            -> Union[BaseModelResponse, BaseModelStreamResponse, AsyncGenerator]:
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
    def create_chunk_from_content( index: int, delta: Any, model: str, **kwargs) -> AbstractMessage:
        """Create a chunk from content for streaming."""
        raise NotImplementedError("You need to implement create_chunk_from_content first!")

    @staticmethod
    @abstractmethod
    def get_supported_params( model_name: str) -> Optional[list]:
        """Get supported parameters for a specific model."""
        raise NotImplementedError("You need to implement get_supported_params first!")
    
    @abstractmethod
    def create_response_from_content(self, content: str, model: str, **kwargs) -> Dict[str, Any]:
        """Create a response dictionary from content."""
        raise NotImplementedError("You need to implement create_response_from_content first!")

    @staticmethod
    @abstractmethod
    def content_from_stream(stream_generator: Generator, stop: str = None) -> str:
        """Extract content from a stream generator."""
        raise NotImplementedError("You need to implement content_from_stream first!")

    @abstractmethod
    def debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        raise NotImplementedError("You need to implement debug_enabled first!")

    @abstractmethod
    def enable_debug(self) -> None:
        """Enable debug mode."""
        raise NotImplementedError("You need to implement enable_debug first!")

    @abstractmethod
    def enable_logging(self) -> None:
        """Enable logging."""
        raise NotImplementedError("You need to implement enable_logging first!")

    @abstractmethod
    def disable_logging(self) -> None:
        """Disable logging."""
        raise NotImplementedError("You need to implement disable_logging first!")
