from dataclasses import dataclass, field
from typing import Callable, Optional
from litellm.types.utils import Function

OnMessageCallable = Callable[[dict], None]
OnFunctionCallable = Callable[[Function], None]

@dataclass
class Memory:

    on_message: list[OnMessageCallable] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

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
        def tool_handler(message: dict) -> None:
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                for call in message.tool_calls:
                    #if call.function is Function:
                    fun(call.function)
        self.add_on_message(tool_handler)



    def remove_on_message(self, handler: OnMessageCallable):
        self.on_message = [m for m in self.on_message if m == handler]

    def add_system_message(self, prompt: str, run_callbacks: bool = True):
        return self.add_message({"role":"system", "content":prompt}, run_callbacks=run_callbacks)

    def add_user_message(self, prompt: str, run_callbacks: bool = True):
        return self.add_message({"role":"user", "content":prompt}, run_callbacks=run_callbacks)

    def add_message(self, message: dict, run_callbacks: bool = True):
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
    def last_message(self) -> Optional[dict]:
        return self.messages[-1] if len(self.messages) > 0 else None


    def add_messages(self, messages: list[dict], run_callbacks: bool = True):
        for message in messages:
            self.messages.append(message)
            if run_callbacks:
                for handler in self.on_message:
                    handler(message)

    def clear_messages(self):
        self.messages.clear()
