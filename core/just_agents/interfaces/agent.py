from abc import ABC, abstractmethod
import ast
import json
from typing import Type, Union, Generator, AsyncGenerator, Any, TypeVar, Generic, List, Optional, Callable, Coroutine, \
    Protocol, ParamSpec, ParamSpecArgs, ParamSpecKwargs
from pydantic import BaseModel, ConfigDict
import sys

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

    @abstractmethod
    def query(self, query_input: AbstractQueryInputType, **kwargs) -> Optional[AbstractQueryResponseType]:
        raise NotImplementedError("You need to implement query() abstract method first!")
    
    def query_structural(
        self, 
        query_input: AbstractQueryInputType, 
        parser: Type[BaseModel] = BaseModel,
        **kwargs
    ) -> Union[dict, BaseModel]:
        """
        Query the agent and parse the response according to the provided parser.
        Attempts multiple parsing strategies in the following order:
        1. Direct dict usage if already a dict
        2. Standard json parsing
        3. AST literal eval as final fallback
        """
        raw_response = self.query(query_input, **kwargs)

        # If already a dict, no parsing needed
        if isinstance(raw_response, dict):
            return raw_response if parser == dict else parser.model_validate(raw_response)

        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        # Check for and fix common double-escaping issues
        if '\\\\' in raw_response:
            raw_response = raw_response.replace('\\\\', '\\')

        parsing_errors = []
        
        # Try standard json first
        try:
            response_dict = json.loads(raw_response)
            return response_dict if parser == dict else parser.model_validate(response_dict)
        except json.JSONDecodeError as e:
            parsing_errors.append(f"Standard JSON parsing failed: {str(e)}")

        # Final fallback: AST literal eval
        try:
            response_dict = ast.literal_eval(raw_response)
            if isinstance(response_dict, dict):
                return response_dict if parser == dict else parser.model_validate(response_dict)
            parsing_errors.append("AST parsing succeeded but result was not a dict")
        except (ValueError, SyntaxError) as e:
            parsing_errors.append(f"AST literal_eval parsing failed: {str(e)}")

        # If all parsing attempts failed, raise detailed error
        raise ValueError(
            f"Failed to parse response using multiple methods:\n"
            f"{chr(10).join(parsing_errors)}\n"
            f"Original response:\n{raw_response}"
        )

    
    @abstractmethod
    def stream(self, query_input: AbstractQueryInputType) -> Optional[AbstractStreamingGeneratorResponseType]:
        raise NotImplementedError("You need to implement stream() abstract method first!")


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
