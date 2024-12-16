from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Callable, List, Dict, Union
from functools import singledispatchmethod
from just_agents.core.interfaces.IMemory import IMemory
from just_agents.core.types import Role, AbstractMessage, SupportedMessages, SupportedMessage
from litellm.types.utils import Function
from abc import ABC

OnMessageCallable = Callable[[AbstractMessage], None]
OnFunctionCallable = Callable[[Function], None]

class IBaseMemory(BaseModel, IMemory[Role, AbstractMessage], ABC):
    """
    Abstract Base Class to fulfill Pydantic schema requirements for concrete-attributes.
    """

    messages: List[AbstractMessage] = Field(default_factory=list, validation_alias='conversation')

    # Private dict of message handlers for each role
    _on_message: Dict[Role, List[OnMessageCallable]] = PrivateAttr(default_factory=lambda: {
        Role.assistant: [],
        Role.tool: [],
        Role.user: [],
        Role.system: [],
    })

    def deepcopy(self) -> 'IBaseMemory':
        return self.model_copy(deep=True)

class BaseMemory(IBaseMemory):
    """
    The Memory class provides storage and handling of messages for a language model session.
    It supports adding, removing, and handling different types of messages and
    function calls categorized by roles: assistant, tool, user, and system.
    """

    def handle_message(self, message: AbstractMessage) -> None:
        """
        Implements the abstract method to handle messages based on their roles.
        """
        role: Optional[Role] = message.get("role")
        if role is None:
            raise ValueError("Message does not have a role")
        for handler in self._on_message.get(role, []):
            handler(message)


    # Overriding add_message with specific implementations
    @singledispatchmethod
    def add_message(self, message: SupportedMessages) -> None:
        """
        Overrides the abstract method and provides dispatching to specific handlers.
        """
        raise TypeError(f"Unsupported message format: {type(message)}")

    @add_message.register
    def _add_abstract_message(self, message: dict) -> None:
        """
        Handles AbstractMessage instances.
        """
        self.messages.append(message)
        self.handle_message(message)

    @add_message.register
    def _add_message_str(self, message: str) -> None:
        """
        Handles string messages.
        """
        self.add_message({"role": Role.user, "content": message})

    @add_message.register
    def _add_message_list(self, messages: list) -> None:
        """
        Handles lists of messages.
        """
        self.add_messages(messages)

    # Methods to add messages of specific roles
    def add_system_message(self, prompt: str) -> None:
        self.add_message({"role": Role.system, "content": prompt})

    def add_user_message(self, prompt: str) -> None:
        self.add_message({"role": Role.user, "content": prompt})

    @property
    def last_message_str(self) -> Optional[str]:
        message_str = None
        last_message = self.last_message
        if last_message:
            message_str = last_message["content"] if "content" in last_message else str(last_message)
        return message_str

    def add_on_tool_call(self, fun: OnFunctionCallable) -> None:
        """
        Adds a handler to track function calls.
        """

        def tool_handler(message: AbstractMessage) -> None:
            tool_calls = message.get('tool_calls', [])
            for call in tool_calls:
                function_name = call.get('function')
                if function_name:
                    fun(function_name)
                else:
                    raise ValueError("Function name is None")

        self.add_on_message_handler(Role.assistant, tool_handler)


    def add_on_tool_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for tool messages.

        :param handler: The callable to be executed when a tool message is added.
        """
        self.add_on_message_handler(Role.tool, handler)

    def add_on_user_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for user messages.
        
        :param handler: The callable to be executed when a user message is added.
        """
        self.add_on_message_handler(Role.user, handler)

    def add_on_assistant_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for assistant messages.
        
        :param handler: The callable to be executed when an assistant message is added.
        """
        self.add_on_message_handler(Role.assistant, handler)

    def add_on_system_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for system messages.
        
        :param handler: The callable to be executed when a system message is added.
        """
        self.add_on_message_handler(Role.system, handler)


    def remove_on_tool_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for tool messages.
        
        :param handler: The handler to be removed.
        """
        self.remove_on_message_handler(Role.tool, handler)

    def remove_on_user_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for user messages.
        
        :param handler: The handler to be removed.
        """
        self.remove_on_message_handler(Role.user, handler)

    def remove_on_assistant_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for assistant messages.
        
        :param handler: The handler to be removed.
        """
        self.remove_on_message_handler(Role.assistant, handler)

    def remove_on_system_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for system messages.
        
        :param handler: The handler to be removed.
        """
        self._remove_on_message(handler, Role.system)

