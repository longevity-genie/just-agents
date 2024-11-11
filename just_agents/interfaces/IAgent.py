from abc import ABC, abstractmethod
from typing import Union, AsyncGenerator, Any, TypeVar, Generic, List, Optional, Callable, Coroutine, Protocol
from functools import wraps

# Define generic types for inputs and outputs
Self = TypeVar("Self") # 3.11+ only, replacement for 3.8+ compatibility
AbstractQueryInputType = TypeVar("AbstractQueryInputType")
AbstractQueryResponseType = TypeVar("AbstractQueryResponseType")
AbstractStreamingResponseType = TypeVar("AbstractStreamingResponseType")

# Define the type that represents streaming responses
AbstractStreamingGeneratorResponseType = Union[
    Coroutine[Any, Any, AsyncGenerator[AbstractStreamingResponseType, None]],
    AsyncGenerator[AbstractStreamingResponseType, None]
]

# Signature for a query function
QueryFunction = Callable[[Self,AbstractQueryInputType,...],Any]
ResponseFunction = Callable[...,AbstractQueryResponseType]
StreamingResponseFunction = Callable[...,AbstractStreamingGeneratorResponseType]

# Signatures for listener templates
class QueryListener(Protocol[AbstractQueryResponseType]):
    def __call__(self, input_query: AbstractQueryResponseType, *args: Any, **kwargs: Any) -> None:
        ...

class ResponseListener(Protocol[AbstractQueryResponseType]):
    def __call__(self, result: AbstractQueryResponseType, *args: Any, **kwargs: Any) -> None:
        ...

class StreamingResponseListener(Protocol[AbstractStreamingResponseType]):
    def __call__(self, result: AbstractStreamingGeneratorResponseType, *args: Any, **kwargs: Any) -> None:
        ...

class IAgent(ABC, Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingResponseType]):

    @abstractmethod
    def query(self, query_input: AbstractQueryInputType) -> Optional[AbstractQueryResponseType]:
        raise NotImplementedError("You need to implement query() abstract method first!")

    @abstractmethod
    def stream(self, query_input: AbstractQueryInputType) -> Optional[AbstractStreamingGeneratorResponseType]:
        raise NotImplementedError("You need to implement stream() abstract method first!")

# Define IAgentWithInterceptors with methods to manage interceptors
class IAgentWithInterceptors(
        IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingResponseType],
        Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingResponseType]
    ):

    @property
    @abstractmethod
    def on_query(self) -> List[
        QueryListener[AbstractQueryInputType]
    ]: # Assuming on_query is a list of handlers
        # In a concrete class, this can be a simple list or custom input processing logic
        raise NotImplementedError("You need to implement _on_input abstract property first!")

    @property
    @abstractmethod
    def on_response(self) -> List[
        ResponseListener[AbstractQueryResponseType]
    ]: # Assuming on_response is a list of handlers
        # In a concrete class, this can be a simple list or custom response processing logic
        raise NotImplementedError("You need to implement _on_response abstract property first!")

    @property
    @abstractmethod
    def on_streaming_response(self) -> List[
        StreamingResponseListener[AbstractStreamingGeneratorResponseType]
    ]: # Assuming on_streaming_response is a list of handlers
        # In a concrete class, this can be a simple list or custom response processing logic
        raise NotImplementedError("You need to implement _on_response abstract property first!")

    @staticmethod
    def query_handler(query_func: QueryFunction):
        @wraps(query_func)
        def _pre_wrapper(self, *args, **kwargs):
            # Extract `query_input` from either args or kwargs
            query_input = kwargs.get("query_input")
            if query_input is None and args:
                query_input = args[0]  # Assuming `query_input` is the first positional arg
                args = args[1:]  # Remove the first element from args
            # Iteratively pre-process input arguments by permutation
            for handler in self.on_query:
                handler(query_input, *args, **kwargs)
                # Proceed to the main function, passing `self`
                return query_func(self, query_input, *args, **kwargs)
        return _pre_wrapper

    @staticmethod
    def response_handler(query_func: ResponseFunction):
        @wraps(query_func)
        def _post_wrapper(self, *args, **kwargs):
            # Execute the main function
            query_result: AbstractQueryResponseType = query_func(self, *args, **kwargs)
            # Call the abstract post-processing logic
            for handler in self.on_response:
                handler(query_result, *args, **kwargs)
            return query_result

        return _post_wrapper

    # Decorator to attach streaming listeners
    @staticmethod
    def streaming_response_handler(query_func: StreamingResponseFunction):
        @wraps(query_func)
        def _post_streaming_wrapper(self, *args, **kwargs):
            # Execute the main function
            query_result: AbstractStreamingGeneratorResponseType = query_func(self, *args, **kwargs)
            # Call the abstract post-processing logic
            for handler in self.on_streaming_response:
                handler(query_result, *args, **kwargs)
            return query_result

        return _post_streaming_wrapper

    # Methods to manage on_query listeners
    def add_on_query_listener(self, listener: QueryListener[AbstractQueryInputType]) -> None:
        self.on_query.append(listener)

    def remove_on_query_listener(self, listener: QueryListener[AbstractQueryInputType]) -> None:
        self.on_query.remove(listener)

    # Methods to manage on_response listeners
    def add_on_response_listener(
            self,
            listener: ResponseListener[AbstractQueryResponseType]
    ) -> None:
        self.on_response.append(listener)

    def remove_on_response_listener(
            self,
            listener: ResponseListener[AbstractQueryResponseType]
    ) -> None:
        self.on_response.remove(listener)

    # Methods to manage on_streaming_response listeners
    def add_on_streaming_response_listener(
            self,
            listener: StreamingResponseListener[AbstractStreamingGeneratorResponseType]
    ) -> None:
        self.on_streaming_response.append(listener)

    def remove_on_streaming_response_listener(
            self,
            listener: StreamingResponseListener[AbstractStreamingGeneratorResponseType]
    ) -> None:
        self.on_streaming_response.remove(listener)

