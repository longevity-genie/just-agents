from pathlib import Path
from pydantic import Field, model_validator, BaseModel
from typing import Optional, List, ClassVar, Tuple, Sequence, Callable, Dict, Union, Type

from just_agents.just_serialization import JustSerializable
from just_agents.data_classes import ModelPromptExample
from just_agents.just_tool import JustTool, JustTools, SubscriberCallback


class JustAgentProfile(JustSerializable):
    """
    A Pydantic model representing an agent profile
    """
    DEFAULT_DESCRIPTION: ClassVar[str] = "Generic all-purpose AI agent"
    DEFAULT_GENERIC_PROMPT: ClassVar[str] = "You are a helpful AI assistant"
    DEFAULT_PARENT_SECTION: ClassVar[str] = None#'agent_profiles'
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path('./config/agent_profiles.yaml')
    config_parent_section: Optional[str] = Field(DEFAULT_PARENT_SECTION, exclude=True)

    system_prompt: str = Field(
        DEFAULT_GENERIC_PROMPT,
        description="System prompt of the agent")
    """System prompt of the agent."""

    description: str = Field(
        DEFAULT_DESCRIPTION,
        description="Short description of what the agent does")
    """Short description of what the agent does."""

    tools: Optional[JustTools] = Field(
        None,
        description="A List[Callable] of tools s available to the agent and their descriptions")
    """A List[Callable] of tools s available to the agent and their descriptions"""

    def _add_tool(self, fun: callable):
        """
        Adds a tool to the agent's tools dictionary.
        """
        tool = JustTool.from_callable(fun)
        if self.tools is None:
            self.tools = {
                tool.name: tool
            }
        else:
            self.tools[tool.name] = tool

    
    @model_validator(mode='after')
    def validate_model(self) -> 'JustAgentProfile':
        """Converts callables to JustTool instances and refreshes them before validation."""
        if not self.tools:
            return self
        if isinstance(self.tools, Dict):
            return self
        elif not isinstance(self.tools, Sequence):
            raise TypeError("The 'tools' field must be a sequence of callables or JustTool instances.")
        # elif not {x for x in self.tools if not isinstance(x, JustTool)}: #no items that are not JustTools
        #     return self

        new_tools = {}
        for item in self.tools:
            if isinstance(item, JustTool):
                if item.auto_refresh:
                    item=item.refresh()
                new_tools[item.name]= item
            elif callable(item):
                new_tool = JustTool.from_callable(item)
                new_tools[new_tool.name] = new_tool
            else:
                raise TypeError("Items in 'tools' must be callables or JustTool instances.")
        setattr(self, 'tools', new_tools)
        return self

    def get_tools_callables(self) -> Optional[List[Callable]]:
        """Retrieves the list of callables from the tools."""
        if self.tools is None:
            return None
        return [tool.get_callable(refresh=tool.auto_refresh) for tool in self.tools]
    
    def subscribe_to_tool_call(self, callback: SubscriberCallback) -> None:
        if self.tools:
            for name, tool in self.tools.items():
                tool.subscribe_to_call(callback)

    def subscribe_to_tool_result(self, callback: SubscriberCallback) -> None:
        if self.tools:
            for name, tool in self.tools.items():
                tool.subscribe_to_result(callback)

    def subscribe_to_tool_error(self, callback: SubscriberCallback) -> None:
        if self.tools:
            for name, tool in self.tools.items():
                tool.subscribe_to_error(callback)

    @classmethod
    def auto_load(
                cls,
                section_name: str,
                parent_section: Optional[str] = None,
                file_path: Path = None,
        ) -> JustSerializable:
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
        if parent_section is None:
            parent_section = cls.DEFAULT_PARENT_SECTION
        if file_path is None:
            file_path = cls.DEFAULT_CONFIG_PATH

        return cls.from_yaml_auto(section_name, parent_section, file_path)

    @staticmethod
    def load_legacy_schema(
            file_path: Path = DEFAULT_CONFIG_PATH,
            class_hint: str = "just_agents.base_agent.BaseAgent",
        ) -> JustSerializable:
        """
        Creates an instance from a YAML file.

        This function reads configuration data from a specified YAML file with agent schema,
        it dynamically imports and instantiates the corresponding class. Otherwise, returns None.

        Args:
            file_path (Path): The path to the YAML file.
            class_hint (str): A substitute class to use if not specified by legacy schema

        Returns:
            Any: An instance of the dynamically imported class if `class_qualname` is found in the
                 configuration data; otherwise, returns None.
        """
        return JustSerializable.from_yaml_auto("", "", file_path, class_hint=class_hint)
    

    @staticmethod
    def convert_from_legacy(
            legacy_file_path: Path,
            output_file_path: Optional[Path] = None,
            class_hint:  Optional[Union[Type|str]] = 'just_agents.base_agent.BaseAgent',
            section_name: Optional[str] = None,
            parent_section: Optional[str] = DEFAULT_PARENT_SECTION,
            exclude_defaults: bool = False,
            exclude_unset: bool = False
        ) -> 'JustAgentProfile':
        """
        Converts a legacy agent schema file to the new format and saves it.

        Args:
            legacy_file_path (Path): Path to the legacy schema YAML file
            output_file_path (Path): Path where the converted file should be saved
            section_name (str): Name of the section to save the converted agent under
            parent_section (str): Parent section name to save the converted agent under
            class_hint (str): A substitute class to use if not specified by legacy schema
            exclude_defaults (bool): Whether to exclude fields with the default values from the output.
            exclude_unset (bool): Whether to exclude unset fields from the output.

        Returns:
            JustAgentProfile: The converted agent profile instance
        """
        # Load the legacy schema
        agent = JustAgentProfile.load_legacy_schema(
            file_path=legacy_file_path,
            class_hint=class_hint
        )
        file_path = output_file_path or legacy_file_path
        
        # Save in new format
        agent.save_to_yaml(section_name=section_name, 
                           parent_section=parent_section, 
                           file_path=file_path, 
                           exclude_defaults=exclude_defaults, 
                           exclude_unset=exclude_unset)
        
        return agent

class JustAgentProfileChatMixin(BaseModel):
    role: Optional[str] = Field(
        default=None,
        description="Defines the agent's persona or identity")
    """Defines the agent's persona or identity."""

    goal: Optional[str] = Field(
        default=None,
        description="Specifies the agent's broader objective")
    """Specifies the agent's broader objective"""
    task: Optional[str] = Field(
        default=None,
        description="Describes the specific task the agent is responsible for")
    """TDescribes the specific task the agent is responsible for"""

    format: Optional[str] = Field(
        default=None,
        description="Describes the specific format the agent is responsible for")
    """Describes the specific format the agent is responsible for"""

    backstory: Optional[str] = Field(
        None,
        description="Detailed narrative of the agent's background, personality traits, and experiences that shape its behavior and responses")
    """Detailed narrative of the agent's background, personality traits, and experiences that shape its behavior and responses."""
    
class JustAgentProfileWebMixin(BaseModel):

    DEFAULT_DISPLAY_NAME: ClassVar[str] = "🦙 A simple Web AI agent"
    DEFAULT_PROMPT_EXAMPLE: ClassVar[ModelPromptExample] = ModelPromptExample(
            title = "Is aging a disease?",
            prompt = "Explain why biological aging can be classified as a disease"
        )
    
    display_name: Optional[str] = Field(
        None,
        description="A fancy one-line name of the agent, replaces shortname in UIs, may include spaces, emoji and other stuff")

    assistant_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Non-negative value that specifies model position in Chat UI models list, highest is default")

    examples: Optional[List[ModelPromptExample]] = Field(
        None,
        description="A List[dict] of model prompt examples, each example is structured as {\"title\":\"Letter counting\", \"prompt\":\"How many letters...\"} "
    )

    hidden: bool = Field(
        False,
        description="Whether to hide the agent from the UI")
    """Whether to hide the agent from the UI"""

    def __str__(self):
        name = self.display_name or self.shortname
        return f"{name}"

class JustAgentProfileSpecializationMixin(BaseModel):
    preferred_model_name: Optional[str] = Field(
        None,
        description="The name of the preferred model to use for inference",
        alias="model_name" #model_name conflicts with pydantic
    )
    """The name of the preferred model to use for inference"""
    
    expertise_domain: Optional[str] = Field(
        None,
        description="Agent's field of expertise")
    """Agent's field of expertise."""

    limitations: Optional[str] = Field(
        None,
        description="Agent's known limitations")
    """Agent's known limitations."""

class JustAgentProfileRagMixin(BaseModel):

    knowledge_sources: Optional[List[str]] = Field(
        None,
        description="A List[str] of of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc.")
    """List of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc."""

class JustAgentFullProfile(JustAgentProfile, JustAgentProfileChatMixin, JustAgentProfileWebMixin, JustAgentProfileSpecializationMixin, JustAgentProfileRagMixin):
    """
    A Pydantic model representing an agent profile with all extended attributes.
    """

    TO_REFRESH: ClassVar[Tuple[str, ...]] = ('shortname', 'description')
    """Fields to force-renew using LLM"""
