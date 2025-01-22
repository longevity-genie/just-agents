from abc import ABC, abstractmethod
from typing import  Callable, Sequence, Any, TypeVar, Generic

ToolByNameCallback=Callable[[str],Callable]
AbstractMessage = TypeVar("AbstractMessage")

class IFunctionCall(ABC, Generic[AbstractMessage]):
    id: str
    type: str

    @abstractmethod
    def execute_function(self, call_by_name: ToolByNameCallback) -> AbstractMessage:
        raise NotImplementedError("You need to implement execute_function() abstract method first!")

    @staticmethod
    @abstractmethod
    def reconstruct_tool_call_message(calls: Sequence['IFunctionCall']) -> AbstractMessage:
        raise NotImplementedError("You need to implement reconstruct_tool_call_message() abstract method first!")

