from pydantic import BaseModel
from just_agents.types import Role, AbstractMessage, OnMessageCallable, OnFunctionCallable
from typing import Optional, ClassVar, List, Dict

class Memory(BaseModel):
    """
    The Memory class provides storage and handling of messages for a language model session.
    It supports adding, removing, and handling different types of messages and 
    function calls categorized by roles: assistant, tool, user, and system.
    """

    ASSISTANT: ClassVar[Role] = Role.assistant
    TOOL: ClassVar[Role] = Role.tool
    USER: ClassVar[Role] = Role.user
    SYSTEM: ClassVar[Role] = Role.system

    on_message: Dict[Role, List] = {
        ASSISTANT: [],
        TOOL: [],
        USER: [],
        SYSTEM: [],
    }

    messages: List[AbstractMessage] = []

    def add_on_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for any message from any role.
        
        :param handler: The callable to be executed when a message is added.
        """
        for role in self.on_message:
            self.on_message[role].append(handler)

    def add_on_tool_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for tool messages.
        
        :param handler: The callable to be executed when a tool message is added.
        """
        self.on_message[Role.tool].append(handler)

    def add_on_user_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for user messages.
        
        :param handler: The callable to be executed when a user message is added.
        """
        self.on_message[Role.user].append(handler)

    def add_on_assistant_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for assistant messages.
        
        :param handler: The callable to be executed when an assistant message is added.
        """
        self.on_message[Role.assistant].append(handler)

    def add_on_system_message(self, handler: OnMessageCallable) -> None:
        """
        Adds a handler to be called for system messages.
        
        :param handler: The callable to be executed when a system message is added.
        """
        self.on_message[Role.system].append(handler)

    def add_on_tool_call(self, fun: OnFunctionCallable) -> None:
        """
        Adds a handler to be called for function calls to track which functions are called.
        
        :param fun: The callable to be executed when a function call is detected.
        """

        def tool_handler(message: dict) -> None:  # TODO:Verity code
            if hasattr(message,
                       'tool_calls') and message.tool_calls is not None:  # Hasattr for dict? Maybe BaseModel type or dataclass assumed here?
                for call in message.tool_calls:
                    fun(call.function)

        self.add_on_assistant_message(tool_handler)

    def _remove_on_message(self, handler: OnMessageCallable, role: Role) -> None:
        """
        Removes a specific handler for a given role.
        
        :param handler: The handler to be removed.
        :param role: The role from which the handler will be removed.
        """
        if handler in self.on_message[role]:
            self.on_message[role].remove(handler)

    def remove_on_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for all roles.
        
        :param handler: The handler to be removed from all roles.
        """
        for role in self.on_message:
            self._remove_on_message(handler, role)

    def clear_all_on_message(self) -> None:
        """
        Clears all handlers for all roles.
        """
        for role in self.on_message:
            self.on_message[role].clear()

    def remove_on_tool_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for tool messages.
        
        :param handler: The handler to be removed.
        """
        self._remove_on_message(handler, Role.tool)

    def remove_on_user_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for user messages.
        
        :param handler: The handler to be removed.
        """
        self._remove_on_message(handler, Role.user)

    def remove_on_assistant_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for assistant messages.
        
        :param handler: The handler to be removed.
        """
        self._remove_on_message(handler, Role.assistant)

    def remove_on_system_message(self, handler: OnMessageCallable) -> None:
        """
        Removes a specific handler for system messages.
        
        :param handler: The handler to be removed.
        """
        self._remove_on_message(handler, Role.system)

    def add_system_message(self, prompt: str) -> None:
        """
        Adds a system message to the memory.
        
        :param prompt: The content of the system message.
        :return: The added message.
        """
        return self.add_message({"role": Role.system, "content": prompt})

    def add_user_message(self, prompt: str) -> None:
        """
        Adds a user message to the memory.
        
        :param prompt: The content of the user message.
        :return: The added message.
        """
        return self.add_message({"role": Role.user, "content": prompt})

    def add_message(self, message: AbstractMessage) -> None:
        """
        Adds a message to the memory and triggers the handlers associated with the message's role.
        
        :param message: The message to add.
        """
        self.messages.append(message)
        role: Role = message["role"]
        for handler in self.on_message[role]:
            handler(message)

    @property
    def last_message(self) -> Optional[AbstractMessage]:
        """
        Returns the last message in the memory.
        
        :return: The last message or None if there are no messages.
        """
        return self.messages[-1] if len(self.messages) > 0 else None

    def add_messages(self, messages: List[AbstractMessage]) -> None:
        """
        Adds multiple messages to the memory.
        
        :param messages: The list of messages to add.
        """
        for message in messages:
            self.add_message(message)

    def clear_messages(self) -> None:
        """
        Clears all messages from the memory.
        """
        self.messages.clear()
