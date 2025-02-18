from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Callable, Coroutine, Union, AsyncGenerator, List, Sequence, ClassVar, Type, TypeVar, Generic, Any, Optional

from just_agents.interfaces.function_call import IFunctionCall

BaseModelResponse = TypeVar('BaseModelResponse', bound=BaseModel)
BaseModelStreamResponse = TypeVar('BaseModelStreamResponse', bound=BaseModel)
AbstractMessage = TypeVar("AbstractMessage")
ResponseTypes=Union[AbstractMessage,BaseModelResponse, BaseModelStreamResponse]

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
    def completion(self, *args, **kwargs) -> BaseModelResponse:
        raise NotImplementedError("You need to implement completion first!")

    async def async_completion(self, *args, **kwargs) \
            -> Coroutine[Any,Any,Union[BaseModelResponse, AsyncGenerator, Any]]:
        raise NotImplementedError("You need to implement async_completion first!")

    @abstractmethod
    def finish_reason_from_response(self, response: ResponseTypes) -> Any:
        raise NotImplementedError("You need to implement finish_reason_from_response first!")

    @abstractmethod
    def message_from_response(self, response: ResponseTypes) -> AbstractMessage:
        raise NotImplementedError("You need to implement message_from_response first!")

    @abstractmethod
    def content_from_delta(self, delta: AbstractMessage) -> str:
        raise NotImplementedError("You need to implement content_from_delta first!")

    @abstractmethod
    def tool_calls_from_message(self, message: AbstractMessage) -> List[IFunctionCall[AbstractMessage]]:
        raise NotImplementedError("You need to implement tool_calls_from_response first!")

    @abstractmethod
    def response_from_deltas(self, deltas: List[AbstractMessage]) -> BaseModelResponse:
        raise NotImplementedError("You need to implement message_from_deltas first!")

    @abstractmethod
    def message_as_chunk(self, index: int, delta: AbstractMessage, model: str, role: Optional[str] = None) -> AbstractMessage:
        raise NotImplementedError("You need to implement message_as_chunk first!")

    @abstractmethod
    def get_supported_params(self, model_name: str) -> Optional[list]:
        raise NotImplementedError("You need to implement get_supported_params first!")

    @property
    @abstractmethod
    def debug_enabled(self) -> bool:
        raise NotImplementedError("You need to implement debug_enabled first!")

    @abstractmethod
    def enable_debug(self) -> None:
        raise NotImplementedError("You need to implement enable_debug first!")

    @abstractmethod
    def enable_logging(self) -> None:
        raise NotImplementedError("You need to implement enable_logging first!")

    @abstractmethod
    def disable_logging(self) -> None:
        raise NotImplementedError("You need to implement disable_logging first!")
