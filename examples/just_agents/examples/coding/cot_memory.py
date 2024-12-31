from pydantic import Field, PrivateAttr
from typing import Optional, Callable, List, Dict, Literal
from functools import singledispatchmethod
from just_agents.interfaces.memory import IMemory
from just_agents.just_serialization import JustSerializable

from just_agents.patterns.interfaces.IThinkingAgent import IThought
from abc import ABC



class ActionableThought(IThought):
    """
    This is a thought object that is used to represent a thought in the chain of thought agent.
    """

    agent: Optional[str] = Field("Error", description="Agent's name")

    title: str = Field("Final Thought", description="Represents the title/summary of the current thinking step")
    content: str = Field(..., description="The detailed explanation/reasoning for this thought step")
    next_action: Literal["continue", "final_answer"] = Field(default="final_answer", description="Indicates whether to continue thinking or provide final answer")

    code: Optional[str] = Field(None, description="Optional field containing script code")
    console: Optional[str] = Field(None, description="Optional field containing console outputs")

    def is_final(self) -> bool:
        # Helper method to check if this is the final thought in the chain
        return self.next_action == "final_answer"

OnThoughtCallable = Callable[[ActionableThought], None]

class IBaseThoughtMemory(JustSerializable, IMemory[str, ActionableThought], ABC):
    """
    Abstract Base Class to fulfill Pydantic schema requirements for concrete-attributes.
    """

    messages: List[ActionableThought] = Field(default_factory=list)

    # Private dict of message handlers for each role
    _on_message: Dict[str, List[OnThoughtCallable]] = PrivateAttr(default_factory=dict)

    def deepcopy(self) -> 'IBaseThoughtMemory':
        return self.model_copy(deep=True)


class BaseThoughtMemory(IBaseThoughtMemory):
    """
    The Memory class provides storage and handling of messages for a language model session.
    It supports adding, removing, and handling different types of messages and
    function calls categorized by roles: assistant, tool, user, and system.
    """

    def handle_message(self, message: ActionableThought) -> None:
        """
        Implements the abstract method to handle messages based on their roles.
        """
        if hasattr(message, "agent") and message.agent is not None:
            name: str = message.agent
        else:
            name: str = "Error"
        for handler in self._on_message.get(name, []):
            handler(message)

    # Overriding add_message with specific implementations
    @singledispatchmethod
    def add_message(self, message: ActionableThought) -> None:
        """
        Overrides the abstract method and provides dispatching to specific handlers.
        """
        raise TypeError(f"Unsupported message format: {type(message)}")

    @add_message.register
    def _add_abstract_message(self, message: ActionableThought) -> None:
        """
        Handles AbstractMessage instances.
        """
        self.messages.append(message)
        self.handle_message(message)

    @add_message.register
    def _add_abstract_message(self, message: IThought) -> None:
        """
        Handles AbstractMessage instances.
        """
        thought = ActionableThought.model_validate(message.model_dump()) #reinstantiate
        self.messages.append(thought)
        self.handle_message(thought)

    @add_message.register
    def _add_abstract_message(self, message: dict) -> None:
        """
        Handles AbstractMessage instances.
        """
        thought = ActionableThought.model_validate(message)
        self.messages.append(thought)
        self.handle_message(thought)

    @add_message.register
    def _add_message_list(self, messages: list) -> None:
        """
        Handles lists of messages.
        """
        self.add_messages(messages)

    @property
    def last_message_str(self) -> Optional[str]:
        last_message = self.last_message
        if last_message:
            return last_message.content
        else:
            return None

    @property
    def final_thoughts(self) -> Optional[List[ActionableThought]]:
        thoughts = [thought for thought in self.final_thoughts if thought.next_action == "final_answer"]
        if thoughts:
            return thoughts
        else:
            return None



