from dataclasses import dataclass, field
from typing import Callable, Optional
from litellm.types.utils import Function

OnMessageCallable = Callable[[dict], None]
OnFunctionCallable = Callable[[Function], None]

TOOL = "tool"
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"

class Memory:
    def __init__(self):
        self.on_message:dict[str, list] = {TOOL:[], USER:[], ASSISTANT:[], SYSTEM:[]}
        self.messages: list[dict] = []


    def add_on_message(self, handler: OnMessageCallable):
        for role in self.on_message:
            self.on_message[role].append(handler)


    def add_on_tool_message(self, handler: OnMessageCallable):
        self.on_message[TOOL].append(handler)

    def add_on_user_message(self, handler: OnMessageCallable):
        self.on_message[USER].append(handler)

    def add_on_assistant_message(self, handler: OnMessageCallable):
        self.on_message[ASSISTANT].append(handler)

    def add_on_system_message(self, handler: OnMessageCallable):
        self.on_message[SYSTEM].append(handler)

    def add_on_tool_call(self, fun: OnFunctionCallable):
        """
        Adds handler only to function calls to track what exactly was called
        :param fun:
        :return:
        """
        def tool_handler(message: dict) -> None:
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                for call in message.tool_calls:
                    #if call.function is Function:
                    fun(call.function)
        self.add_on_assistant_message(tool_handler)


    def _remove_on_message(self, handler: OnMessageCallable, role:str):
        if handler in self.on_message[role]:
            self.on_message[role].remove(handler)


    def remove_on_message(self, handler: OnMessageCallable):
        for role in self.on_message:
            self._remove_on_message(handler, role)


    def clear_all_on_message(self):
        for role in self.on_message:
            self.on_message[role].clear()


    def remove_on_tool_message(self, handler: OnMessageCallable):
        self._remove_on_message(handler, TOOL)

    def remove_on_user_message(self, handler: OnMessageCallable):
        self._remove_on_message(handler, USER)

    def remove_on_assistant_message(self, handler: OnMessageCallable):
        self._remove_on_message(handler, ASSISTANT)

    def remove_on_system_message(self, handler: OnMessageCallable):
        self._remove_on_message(handler, SYSTEM)

    def add_system_message(self, prompt: str):
        return self.add_message({"role":"system", "content":prompt})

    def add_user_message(self, prompt: str):
        return self.add_message({"role":"user", "content":prompt})

    def add_message(self, message: dict):
        """
        adds message to the memory
        :param message:
        :param run_callbacks:
        :return:
        """
        self.messages.append(message)
        role = message["role"]
        for handler in self.on_message[role]:
            handler(message)

    @property
    def last_message(self) -> Optional[dict]:
        return self.messages[-1] if len(self.messages) > 0 else None


    def add_messages(self, messages: list[dict]):
        for message in messages:
            self.add_message(message)


    def clear_messages(self):
        self.messages.clear()
