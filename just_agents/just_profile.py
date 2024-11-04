from pathlib import Path
from pydantic import Field, model_validator
from typing import Optional, List, ClassVar, Tuple, Sequence, Callable, Self, Any
from just_agents.just_serialization import JustSerializable
from just_agents.just_tool import JustTool, JustTools

class JustAgentProfile(JustSerializable):
    """
    A Pydantic model representing an agent profile with extended attributes.
    """
    DEFAULT_GENERIC_PROMPT: ClassVar[str] = "You are a helpful AI assistant"
    DEFAULT_DESCRIPTION: ClassVar[str] = "Generic all-purpose AI agent"
    DEFAULT_PARENT_SECTION: ClassVar[str] = 'agent_profiles'
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path('config/agent_profiles.yaml')
    config_parent_section: Optional[Path] = Field(DEFAULT_PARENT_SECTION, exclude=True)

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

    tools: JustTools = Field(
        None,
        description="A List[Callable] of tools s available to the agent and their descriptions")
    """A List[Callable] of tools s available to the agent and their descriptions"""

    knowledge_sources: Optional[List[str]] = Field(
        None,
        description="A List[str] of of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc.")
    """List of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc."""

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        """Converts callables to JustTool instances and refreshes them before validation."""
        if not self.tools:
            return self
        if not isinstance(self.tools, Sequence):
            raise TypeError("The 'tools' field must be a sequence of callables or JustTool instances.")
        elif not {x for x in self.tools if not isinstance(x, JustTool)}: #all converted
            return self
        new_tools = []
        for item in self.tools:
            if isinstance(item, JustTool):
                new_tools.append(item.refresh())
            elif callable(item):
                new_tools.append(JustTool.from_callable(item))
            else:
                raise TypeError("Items in 'tools' must be callables or JustTool instances.")
        setattr(self, 'tools', new_tools)
        return self

    def get_tools_callables(self) -> Optional[List[Callable]]:
        """Retrieves the list of callables from the tools."""
        if self.tools is None:
            return None
        return [tool.refresh().get_callable() for tool in self.tools]

    @staticmethod
    def auto_load(
                section_name: str,
                parent_section: Optional[str] = DEFAULT_PARENT_SECTION,
                file_path: Path = DEFAULT_CONFIG_PATH,
        ) -> Any:
        """
        Creates an instance from a YAML file.

        This function reads configuration data from a specified YAML file, section name,
        and parent section name. If the configuration data contains a `class_qualname` field,
        it dynamically imports and instantiates the corresponding class. Otherwise, returns None.

        Args:
            section_name (str): The section name in the YAML file.
            parent_section (Optional[str]): The parent section name in the YAML file.
            file_path (Path): The path to the YAML file.

        Returns:
            Any: An instance of the dynamically imported class if `class_qualname` is found in the
                 configuration data; otherwise, returns None.
        """
        return JustSerializable.from_yaml_auto(section_name, parent_section, file_path)

    def __str__(self) -> str:
        """
        Returns the 'description' field when the instance is converted to a string.
        """
        return self.description


