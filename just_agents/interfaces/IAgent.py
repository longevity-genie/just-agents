from abc import ABC, abstractmethod, abstractproperty
import ast
import json
from typing import Type, Union, Generator, AsyncGenerator, Any, TypeVar, Generic, List, Optional, Callable, Coroutine, Protocol, Tuple
from litellm import Field
from pydantic import BaseModel, ConfigDict

# Add this near the top of the file, before any class definitions
import pydantic
pydantic.TypeAdapter.validate_python.__globals__['ConfigDict'] = ConfigDict(
    max_error_message_length=None
)

# Define generic types for inputs and outputs
Self = TypeVar("Self") # 3.11+ only, replacement for 3.8+ compatibility
AbstractQueryInputType = TypeVar("AbstractQueryInputType")

AbstractQueryResponseType = TypeVar("AbstractQueryResponseType")
AbstractStreamingChunkType = TypeVar("AbstractStreamingChunkType")

AbstractAgentInputType = TypeVar('AbstractAgentInputType', bound=BaseModel)
AbstractAgentOutputType = TypeVar('AbstractAgentOutputType', bound=BaseModel)

# New TypeVar for Thought
THOUGHT_TYPE = TypeVar('THOUGHT_TYPE', bound='IThought')

# Define the type that represents streaming responses
AbstractStreamingGeneratorResponseType = Union[
    Coroutine[Any, Any, AbstractQueryResponseType],
    Coroutine[Any, Any, AsyncGenerator[AbstractStreamingChunkType, None]],
    Generator[AbstractStreamingChunkType, None ,None],
    AsyncGenerator[AbstractStreamingChunkType, None]
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

class IAgent(ABC, Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType]):

    @abstractmethod
    def query(self, query_input: AbstractQueryInputType) -> Optional[AbstractQueryResponseType]:
        raise NotImplementedError("You need to implement query() abstract method first!")
    
    def query_structural(
        self, 
        query_input: AbstractQueryInputType, 
        parser: Type[BaseModel] = dict
    ) -> Union[dict, BaseModel]:
        """
        Query the agent and parse the response according to the provided parser.
        
        Args:
            query_input: Input messages for the query
            parser: A pydantic model class or dict to parse the response (default: dict)
            
        Returns:
            Parsed response as either a dictionary or pydantic model instance
        """
        response = self.query(query_input)
        # Convert Python-style string to valid JSON dict
        try:
            # First try parsing as valid JSON
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            # If that fails, use ast.literal_eval to handle Python dict format
            response_dict = ast.literal_eval(response)

        if parser == dict:
            return response_dict
        else:
            # Use model_validate instead of model_validate_json since we already have a dict
            return parser.model_validate(response_dict)


    @abstractmethod
    def stream(self, query_input: AbstractQueryInputType) -> Optional[AbstractStreamingGeneratorResponseType]:
        raise NotImplementedError("You need to implement stream() abstract method first!")

class IThought(BaseModel):
    @abstractmethod
    def is_final(self) -> bool:
        raise NotImplementedError("You need to implement is_final() abstract method first!")
  
    
  
class ITypedAgent(
    IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
    Generic[
        AbstractQueryInputType,
        AbstractQueryResponseType,
        AbstractStreamingChunkType,
        AbstractAgentInputType,
        AbstractAgentOutputType
    ]
):
    """
    A typed agent is an agent that can query and stream typed inputs and outputs.
    It can recieve typed inputs from which it will extract query for the LLM and then return typed outputs.
    """
    

    @abstractmethod
    def query_from_input(self, input: AbstractAgentInputType) -> AbstractQueryInputType:
        """
        This method is used to convert the input to the query input type for self.query(message) call
        """
        raise NotImplementedError("You need to implement query_from_input() abstract method first!")
    
    @abstractmethod
    def output_from_response(self, response: AbstractQueryResponseType) -> AbstractAgentOutputType:
        """
        This method is used to convert the query response to the output type.
        
        Args:
            response: The raw response from the query
            
        Returns:
            The response converted to the agent's output type
        """
        # Get the concrete output type from the class's type parameters
        output_type = self.__class__.__orig_bases__[0].__args__[4]  # Gets AbstractAgentOutputType
        return self.query_structural(response, parser=output_type)
        
    
    def query_typed(self, input: AbstractAgentInputType) -> AbstractAgentOutputType:
        """
        This method is used to query the agent with a typed input and get a typed output.
        """
        query = self.query_from_input(input)
        response = self.query(query)
        output = self.output_from_response(response)
        return output


class IThinkingAgent(
    IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
    Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType, THOUGHT_TYPE]
):
    
    @abstractmethod
    def thought_query(self, response: AbstractQueryResponseType) -> THOUGHT_TYPE:
        raise NotImplementedError("You need to implement thought_from_response() abstract method first!")

    def think(self, query: AbstractQueryInputType, max_iter: int = 3, chain: Optional[list[THOUGHT_TYPE]] = None) -> tuple[Optional[THOUGHT_TYPE], Optional[list[THOUGHT_TYPE]]]:
        """
        This method will continue to query the agent until the final thought is not None or the max_iter is reached.
        Returns a tuple of (final_thought, thought_chain)
        """
        current_chain = chain or []
        thought = self.thought_query(query) #queries itself with thought as expected output
        new_chain = [*current_chain, thought] #updates chain with the new thought
        if thought.is_final() or max_iter <= 0:
            return (self.thought_query(query), new_chain) #returns the final thought and the chain that preceded it
        else:
            return self.think(query, max_iter - 1, new_chain) #continues the thought process
            



# Define IAgentWithInterceptors with methods to manage interceptors
class IAgentWithInterceptors(
        IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
        ABC,
        Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType]
    ):

    _on_query: List[QueryListener[AbstractQueryInputType]]
    _on_response: List[ResponseListener[AbstractQueryResponseType]]

    # Methods to manage on_query listeners
    def handle_on_query(self, input_query: AbstractQueryResponseType, *args, **kwargs) -> None:
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
