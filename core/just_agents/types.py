from enum import Enum
from typing import TypeVar, Any, List, Union, Dict

######### Common ###########
AbstractMessage = Dict[str, Any]
SupportedMessage = Union[str, AbstractMessage]
SupportedMessages = Union[SupportedMessage, List[SupportedMessage]]
Output = TypeVar('Output')

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    # make it similar to Literal["system", "user", "assistant", tool] while retaining enum convenience

    def __new__(cls, value, *args, **kwargs):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return str(self.value)
    


