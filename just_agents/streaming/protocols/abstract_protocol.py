from typing import Any
from abc import ABC, abstractmethod

class AbstractStreamingProtocol(ABC):
    @abstractmethod
    def get_chunk(self, index:int, delta:str, options:dict) -> Any:
        raise NotImplementedError()
        return ""

    @abstractmethod
    def done(self) -> str:
        raise NotImplementedError()
        return ""