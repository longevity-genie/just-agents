from pydantic import BaseModel, Field, PrivateAttr
from typing import Callable, List, Dict
from functools import singledispatchmethod
from just_agents.interfaces.memory import IMemory, IMessageFormatter
from just_agents.data_classes import Message, Role, ToolCall
from just_agents.types import MessageDict, SupportedMessages

from abc import ABC

from typing import Optional
from rich.console import Console
from rich.text import Text
from rich.panel import Panel


class MessageFormatter(IMessageFormatter):
    messages: List[MessageDict] = Field(default_factory=list, validation_alias='conversation')

    def pretty_print_message(self, msg: MessageDict) -> Panel:
        role = msg.get('role', 'unknown').capitalize()
    
        # If the role is an enum, extract its value
        if isinstance(role, str) and '<Role.' in role: #TODO: this should not be firing with use_enum_values=True,
            role = role.split('.')[-1].replace('>', '').capitalize()

        # Define role-specific colors
        role_colors = {
            'User': 'green',
            'Assistant': 'blue',
            'System': 'yellow',
            'Function': 'magenta',
            'Tool': 'magenta',
        }
        border_colors = {
            'User': 'bright_green',
            'Assistant': 'bright_blue',
            'System': 'bright_yellow',
            'Function': 'bright_magenta',
            'Tool': 'bright_magenta',
        }
        
        # Get colors for the role (default to cyan/bright_yellow if role not found)
        role_color = role_colors.get(role, 'cyan')
        border_color = border_colors.get(role, 'bright_yellow')

        # Create a title with bold text for the role
        role_title = Text(f"[{role}]", style=f"bold {role_color}")

        # Process tool call details if present
        if 'tool_calls' in msg:
            for tool_call in msg['tool_calls']:
                tool_name = tool_call.get('function', {}).get('name', 'unknown tool')
                arguments = tool_call.get('function', {}).get('arguments', '{}')
                return Panel(
                    f"Tool Call to [bold magenta]{tool_name}[/bold magenta]:\n{arguments}",
                    title=role_title,
                    border_style=role_color,
                )
        elif 'tool_call_id' in msg:
            tool_name = msg.get('name', 'unknown tool')
            tool_result = msg.get('content', 'no content')
            return Panel(
                f"Response from [bold magenta]{tool_name}[/bold magenta]:\n{tool_result}",
                title=role_title,
                border_style=border_color,
            )
        else:
            # Standard message
            return Panel(
                f"{msg.get('content', '')}",
                title=role_title,
                border_style=border_color,
            )

    def pretty_print_all_messages(self):
        if not self.messages:
            return
            
        console = Console()
        for msg in self.messages:
            panel = self.pretty_print_message(msg)
            console.print(panel)

OnMessageCallable = Callable[[MessageDict], None]
OnToolCallable = Callable[[ToolCall], None]

class IBaseMemory(BaseModel, IMemory[Role, MessageDict], IMessageFormatter, ABC):
    """
    Abstract Base Class to fulfill Pydantic schema requirements for concrete-attributes.
    """

    messages: List[MessageDict] = Field(default_factory=list, validation_alias='conversation')

    # Private dict of message handlers for each role
    _on_message: Dict[Role, List[OnMessageCallable]] = PrivateAttr(default_factory=lambda: {
        Role.assistant: [],
        Role.tool: [],
        Role.user: [],
        Role.system: [],
    })

    # Methods to add messages of specific roles
    def add_system_message(self, prompt: str) -> None:
        self.add_message({"role": Role.system, "content": prompt})
        self.messages.sort(key=lambda msg: msg.get("role","user") != Role.system)

    def add_user_message(self, prompt: str) -> None:
        self.add_message({"role": Role.user, "content": prompt})

    def get_message_by_role(self, role: Role) -> List[MessageDict]:
        """
        Retrieves all messages that match the given role.

        :param role: The role to filter messages by.
        :return: A list of messages matching the selector.
        """
        return [message for message in self.messages if message.get("role","user") == role.value]

    @property
    def prompt_messages(self) -> List[MessageDict]:
        return self.get_message_by_role(Role.system)

    def clear_system_messages(self, clear_non_empty: bool = True) -> None:
        for sys_prompt in self.prompt_messages:
            if not clear_non_empty and Message(**sys_prompt).get_text():
                continue
            self.messages.remove(sys_prompt)
        if clear_non_empty and self.prompt_messages:
            raise ValueError("Failed to clear system prompts")

    def deepcopy(self) -> 'IBaseMemory':
        return self.model_copy(deep=True)

    # Role-specific message handlers
    def add_on_tool_call(self, fun: OnToolCallable) -> None:
        """
        Adds a handler to track function calls.
        """

        def tool_handler(message: MessageDict) -> None:
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


class BaseMemory(IBaseMemory, MessageFormatter):
    """
    The Memory class provides storage and handling of messages for a language model session.
    It supports adding, removing, and handling different types of messages and
    function calls categorized by roles: assistant, tool, user, and system.
    """

    def handle_message(self, message: MessageDict) -> None:
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
    def _add_message_container(self, message: Message) -> None:
        """
        Converts Message Pydantic model to plain dictionary, enforces text-only format
        """
        self.add_message(message.text_format().model_dump(mode="json", exclude_none=True, exclude_defaults=False))

    @add_message.register
    def _add_message_dict(self, message: dict) -> None:
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

    @property
    def last_message_str(self) -> Optional[str]:
        message_str = None
        last_message = self.last_message
        if last_message:
            message_str = last_message["content"] if "content" in last_message else str(last_message)
        return message_str


