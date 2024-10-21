from pathlib import Path
from pydantic import Field
from typing import Optional, List, ClassVar, Tuple
from just_agents.just_serialization import JustSerializable

class JustAgentProfile(JustSerializable):
    """
    A Pydantic model representing an agent profile with extended attributes.
    """
    DEFAULT_GENERIC_PROMPT: ClassVar[str] = "You are a helpful AI assistant"
    DEFAULT_DESCRIPTION: ClassVar[str] = "Generic all-purpose AI agent"
    CONFIG_PARENT_SECTION: ClassVar[str] = 'agent_profiles'
    CONFIG_PATH: ClassVar[Path] = Path('config/agent_profiles.yaml')

    system_prompt: str = Field(
        DEFAULT_GENERIC_PROMPT,
        description="System prompt of the agent")
    """System prompt of the agent."""

    description: str = Field(
        DEFAULT_DESCRIPTION,
        description="Short description of what the agent does")
    """Short description of what the agent does."""

    TO_REFRESH: ClassVar[Tuple[str, ...]] = ('shortname', 'description')
    """Fields to force-renew using LLM"""

    role: Optional[str] = Field(
        None,
        description="Role of the agent")
    """Role of the agent."""

    goal: Optional[str] = Field(
        None,
        description="Goal of the agent")
    """Goal of the agent."""

    task: Optional[str] = Field(
        None,
        description="Tasks of the agent")
    """Tasks of the agent."""

    expertise_domain: Optional[str] = Field(
        None,
        description="Agent's field of expertise")
    """Agent's field of expertise."""

    limitations: Optional[str] = Field(
        None,
        description="Agent's known limitations")
    """Agent's known limitations."""

    backstory: Optional[str] = Field(
        None,
        description="Backstory of the agent")
    """Backstory of the agent."""

    llm_model_name: Optional[str] = Field(
        None,
        description="The name of the preferred model to use for inference",
        alias="model_name" #model_name conflicts with pydantic
    )
    """The name of the preferred model to use for inference"""

    tool_names: Optional[List[str]] = Field(
        None,
        description="A List[str] of the tools names available to the agent")
    """List of the tools available to the agent"""

    knowledge_sources: Optional[List[str]] = Field(
        None,
        description="A List[str] of of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc.")
    """List of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc."""

    def __str__(self) -> str:
        """
        Returns the 'description' field when the instance is converted to a string.
        """
        return self.description


