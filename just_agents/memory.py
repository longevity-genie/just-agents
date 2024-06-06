import pprint
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, TypeAlias, Callable, List, Optional
from litellm.utils import Message

OnMessageCallable = Callable[[Message], None]

@dataclass
class Memory:

    on_message: list[OnMessageCallable] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    def add_on_message(self, handler: OnMessageCallable):
        self.on_message.append(handler)

    def remove_on_message(self, handler: OnMessageCallable):
        self.on_message = [m for m in self.on_message if m == handler]


    def add_message(self, message: Message, run_callbacks: bool = True):
        """
        adds message to the memory
        :param message:
        :param run_callbacks:
        :return:
        """
        self.messages.append(message)

        if run_callbacks:
            for handler in self.on_message:
                handler(message)

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if len(self.messages) > 0 else None


    def add_messages(self, messages: list, run_callbacks: bool = True):
        for message in messages:
            msg = Message(content=message["content"], role=message["role"])
            self.messages.append(msg)
            if run_callbacks:
                for handler in self.on_message:
                    handler(message)
