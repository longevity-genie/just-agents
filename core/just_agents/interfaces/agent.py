from abc import ABC, abstractmethod
import ast
import json
import re
from typing import Type, Union, Generator, AsyncGenerator, Any, TypeVar, Generic, List, Optional, Callable, Coroutine, \
    Protocol, ParamSpec
from pydantic import BaseModel
import sys
from ..just_schema import ModelHelper

# Define generic types for inputs and outputs
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar("Self")

AbstractQueryInputType = TypeVar("AbstractQueryInputType")
AbstractQueryResponseType = TypeVar("AbstractQueryResponseType")
AbstractStreamingChunkType = TypeVar("AbstractStreamingChunkType")

AbstractAgentInputType = TypeVar('AbstractAgentInputType', bound=BaseModel)
AbstractAgentOutputType = TypeVar('AbstractAgentOutputType', bound=BaseModel)

# Define the type that represents streaming responses
AbstractStreamingGeneratorResponseType = Union[
    Coroutine[Any, Any, AbstractQueryResponseType],
    Coroutine[Any, Any, AsyncGenerator[AbstractStreamingChunkType, None]],
    Generator[AbstractStreamingChunkType, None ,None],
    AsyncGenerator[AbstractStreamingChunkType, None]
]

# Signature for a query function
ResponseFunction = Callable[...,AbstractQueryResponseType]
StreamingResponseFunction = Callable[...,AbstractStreamingGeneratorResponseType]

class IAgent(ABC, Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType]):

    shortname: str # must have an identifier for the agent

    @abstractmethod
    def query(self, query_input: AbstractQueryInputType, **kwargs) -> Optional[AbstractQueryResponseType]:
        raise NotImplementedError("You need to implement query() abstract method first!")
    
    @abstractmethod
    def stream(self, query_input: AbstractQueryInputType) -> Optional[AbstractStreamingGeneratorResponseType]:
        raise NotImplementedError("You need to implement stream() abstract method first!")
    
    @abstractmethod
    def query_structural(
        self, 
        query_input: AbstractQueryInputType, 
        parser: Type[BaseModel] = BaseModel,
        response_format: Optional[str] = None,
        **kwargs
    ) -> Union[dict, BaseModel]:
        raise NotImplementedError("You need to implement query_structural() abstract method first!")
    
    def completion(self, query_input: AbstractQueryInputType, **kwargs) -> Optional[Union[AbstractQueryResponseType, AbstractStreamingGeneratorResponseType]]:
        stream = kwargs.get("stream", False)
        if stream:
            return self.stream(query_input, **kwargs)
        else:
            if kwargs.get("response_format", None) or kwargs.get("parser", None):
                return self.query_structural(query_input, **kwargs)
            else:
                return self.query(query_input, **kwargs)
        
VariArgs = ParamSpec('VariArgs')

# Signatures for listener templates
class QueryListener(Protocol[AbstractQueryInputType]):
    def __call__(self, input_query: AbstractQueryInputType, action: str, source: str, *args:VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        ...

class ResponseListener(Protocol[AbstractQueryResponseType]):
    def __call__(self, response: AbstractQueryResponseType, action: str, source: str) -> None:
        ...

# Define IAgentWithInterceptors with methods to manage interceptors
class IAgentWithInterceptors(
        IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
        ABC,
        Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType]
    ):

    _on_query: List[QueryListener[AbstractQueryInputType]]
    _on_response: List[ResponseListener[AbstractQueryResponseType]]

    # Methods to manage on_query listeners
    def handle_on_query(self, input_query: AbstractQueryResponseType, *args:VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        for handler in self._on_query:
            handler(input_query, *args, **kwargs)

    def add_on_query_listener(self, listener: QueryListener[AbstractQueryInputType]) -> None:
        self._on_query.append(listener)

    def remove_on_query_listener(self, listener: QueryListener[AbstractQueryInputType]) -> None:
        self._on_query.remove(listener)

    # Methods to manage on_response listeners
    def handle_on_response(self, query_result: AbstractQueryResponseType, *args, **kwargs) -> None:
        for handler in self._on_response:
            handler(query_result, *args, **kwargs)

    def add_on_response_listener(
            self,
            listener: ResponseListener[AbstractQueryResponseType]
    ) -> None:
        self._on_response.append(listener)

    def remove_on_response_listener(
            self,
            listener: ResponseListener[AbstractQueryResponseType]
    ) -> None:
        self._on_response.remove(listener)
