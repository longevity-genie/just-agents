import pprint
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, TypeAlias, Callable, List, Optional, Union
from litellm.utils import Message, ChatCompletionMessageToolCall, Function

OnMessageCallable = Callable[[Message], None]
OnFunctionCallable = Callable[[Function], None]

@dataclass
class Memory:

    on_message: list[OnMessageCallable] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    def add_on_message(self, handler: OnMessageCallable):
        self.on_message.append(handler)

    def add_on_tool_result(self, handler: OnMessageCallable):
        self.add_on_message(lambda m: handler(m) if m.role == "tool" else None)

    def add_on_tool_call(self, fun: OnFunctionCallable):
        """
        Adds handler only to function calls to track what exactly was called
        :param fun:
        :return:
        """
        def tool_handler(message: Message) -> None:
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                for call in message.tool_calls:
                    #if call.function is Function:
                    fun(call.function)
        self.add_on_message(tool_handler)



    def remove_on_message(self, handler: OnMessageCallable):
        self.on_message = [m for m in self.on_message if m == handler]

    def add_system_message(self, prompt: str, run_callbacks: bool = True):
        return self.add_message(Message(role="system", content=prompt), run_callbacks=run_callbacks)

    def add_user_message(self, prompt: str, run_callbacks: bool = True):
        return self.add_message(Message(role="user", content=prompt), run_callbacks=run_callbacks)


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
