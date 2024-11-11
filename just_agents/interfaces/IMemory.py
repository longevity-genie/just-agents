from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Callable, Optional, Dict

Memorable = TypeVar("Memorable")
Selector = TypeVar("Selector")
HandlerType = Callable[[Memorable], None]

class IMemory(ABC, Generic[Memorable,Selector]):
    """
    An abstract base class for memory management.
    """

    @property
    @abstractmethod
    def conversation(self) -> List[Memorable]:
        """List of stored messages."""
        raise NotImplementedError

    @property
    @abstractmethod
    def on_message(self) -> Dict[Selector,List[HandlerType]]:
        """Dictionary of handlers Lists to process messages."""
        raise NotImplementedError

    @property
    def last_message(self) -> Optional[Memorable]:
        """Returns the last message in the memory."""
        return self.conversation[-1] if self.conversation else None

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
        self.conversation.clear()

    # Additional methods for handling handlers and roles
    def add_on_message(self, handler: HandlerType) -> None:
        for selector in self.on_message:
            self.add_on_message_handler(selector, handler)

    def add_on_message_handler(self, selector: Selector, handler: HandlerType) -> None:
        self.on_message[selector].append(handler)

    def remove_on_message(self, handler: HandlerType) -> None:
        for selector in self.on_message:
            self.remove_on_message_handler(selector, handler)

    def remove_on_message_handler(self, selector: Selector, handler: HandlerType) -> None:
        if handler in self.on_message[selector]:
            self.on_message[selector].remove(handler)

    def clear_all_on_message(self) -> None:
        for selector in self.on_message:
            self.on_message[selector].clear()

