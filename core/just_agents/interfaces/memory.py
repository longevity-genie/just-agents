from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Callable, Optional, Dict

Memorable = TypeVar("Memorable")
MemoryKey = TypeVar("MemoryKey")
from rich.panel import Panel
HandlerType = Callable[[Memorable], None]

class IMemory(ABC, Generic[MemoryKey, Memorable]):
    """
    An abstract base class for memory management.
    """
    messages: List[Memorable]
    _on_message: Dict[MemoryKey, List[HandlerType]]

    @property
    def last_message(self) -> Optional[Memorable]:
        """Returns the last message in the memory."""
        return self.messages[-1] if self.messages else None

    @property
    @abstractmethod
    def last_message_str(self) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def add_message(self, message: Memorable) -> None:
        """Adds a message to the memory."""
        raise NotImplementedError

    @abstractmethod
    def handle_message(self, message: Memorable) -> None:
        """Handles a message by calling all relevant handlers."""
        raise NotImplementedError

    def add_messages(self, messages: List[Memorable]) -> None:
        """Adds multiple messages to the memory."""
        for message in messages:
            self.add_message(message)

    def clear_messages(self) -> None:
        """Clears all messages from the memory."""
        self.messages.clear()

    # Additional methods for handling handlers and roles
    def add_on_message(self, handler: HandlerType) -> None:
        for selector in self._on_message:
            self.add_on_message_handler(selector, handler)

    def add_on_message_handler(self, selector: MemoryKey, handler: HandlerType) -> None:
        self._on_message[selector].append(handler)

    def remove_on_message(self, handler: HandlerType) -> None:
        for selector in self._on_message:
            self.remove_on_message_handler(selector, handler)

    def remove_on_message_handler(self, selector: MemoryKey, handler: HandlerType) -> None:
        if handler in self._on_message[selector]:
            self._on_message[selector].remove(handler)

    def clear_all_on_message(self) -> None:
        for selector in self._on_message:
            self._on_message[selector].clear()


    @abstractmethod
    def deepcopy(self) -> 'IMemory':
        """Return a deep copy"""
        raise NotImplementedError


class IMessageFormatter(ABC):
    @abstractmethod
    def pretty_print_message(self, msg: Memorable) -> Panel:
        pass

    @abstractmethod
    def pretty_print_all_messages(self):
        pass