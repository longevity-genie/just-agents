from pathlib import Path
from pydantic import Field, model_validator, BaseModel
from typing import Optional, List, ClassVar, Tuple, Sequence, Callable, Dict, Union, Type, Any

from just_agents.just_serialization import JustSerializable
from just_agents.data_classes import ModelPromptExample
from just_agents.just_tool import JustTool, JustTools, SubscriberCallback, JustPromptTool, JustPromptTools, JustToolBase, JustToolFactory


class JustAgentProfileToolsetMixin(BaseModel):
    """
    A mixin class that provides tool-related functionality for agent profiles.
    """
    litellm_tool_description: Optional[bool] = Field(
        False, exclude=True,
        description="Whether to use the litellm tool description utility fallback, requires numpydoc")
    """Whether to use the litellm tool description utility fallback, requires numpydoc"""

    tools: Optional[JustTools] = Field(
        None,
        description="A List[Callable] of tools available to the agent and their descriptions")
    """A List[Callable] of tools available to the agent and their descriptions"""

    prompt_tools: Optional[JustPromptTools] = Field(
        None,
        description="A List[Callable] of tools that will be executed to append results to the prompt before completion call")
    """A List[Callable] of tools that will be executed to append results to the prompt before completion call"""

    def add_tool(self, fun: callable) -> None:
        """
        Adds a tool to the agent's tools dictionary.

        Args:
            fun (callable): The function to add as a tool
        """
        tool = JustToolFactory.create_tool(fun)

        if self.tools is None:
            self.tools = {tool.name: tool}
        else:
            self.tools[tool.name] = tool

    def add_prompt_tool(self, fun: callable, call_arguments: Dict[str, Any]) -> None:
        """
        Adds a tool to the agent's prompt_tools dictionary with input parameters.

        Args:
            fun (callable): The function to add as a prompt tool
            call_arguments (Dict[str, Any]): Arguments to call the function with
        """
        # Ensure input parameters are JSON serializable
        try:
            import json
            json.dumps(call_arguments)
        except (TypeError, OverflowError):
            raise ValueError("Input parameters must be JSON serializable")

        prompt_tool = JustToolFactory.create_prompt_tool((fun, call_arguments))

        if self.prompt_tools is None:
            self.prompt_tools = {prompt_tool.name: prompt_tool}
        else:
            self.prompt_tools[prompt_tool.name] = prompt_tool

    def _process_tools_field(self) -> None:
        """
        Process the tools field to convert callables to JustTool instances
        """
        if not self.tools:
            return
        
        if isinstance(self.tools, dict):
            if all(isinstance(tool, JustTool) for tool in self.tools.values()):
                return
        elif not isinstance(self.tools, Sequence):
            raise TypeError("The 'tools' field must be a sequence of callables or JustTool instances.")
        self.tools = JustToolFactory.create_tools_dict(self.tools, type_hint=JustTool)

    def _process_prompt_tools_field(self) -> None:
        """
        Process the prompt_tools field to convert callables with input parameters to JustPromptTool instances
        """
        if not self.prompt_tools:
            return

        if isinstance(self.prompt_tools, dict):
            if all(isinstance(tool, JustPromptTool) for tool in self.prompt_tools.values()):
                return
        elif not isinstance(self.prompt_tools, Sequence):
            raise TypeError(
                "The 'prompt_tools' field must be a sequence of (callable, input_params) tuples or JustPromptTool instances.")
        self.prompt_tools = JustToolFactory.create_prompt_tools_dict(self.prompt_tools)

    @model_validator(mode='after')
    def validate_agent_profile(self) -> 'JustAgentProfileToolsetMixin':
        """
        Converts callables to JustTool instances and refreshes them before validation.

        Returns:
            JustAgentProfileToolsetMixin: The validated instance
        """
        self._process_tools_field()
        self._process_prompt_tools_field()
        return self

    def get_tools_callables(self) -> Optional[List[Callable]]:
        """
        Retrieves the list of callables from the tools.

        Returns:
            Optional[List[Callable]]: List of callable functions or None if no tools
        """
        if not self.tools:
            return None
        return [tool.get_callable(refresh=tool.auto_refresh) for tool in self.tools.values()]

    def get_prompt_tools_with_inputs(self) -> Optional[List[Tuple[Callable, Dict[str, Any]]]]:
        """
        Retrieves the list of callables from the prompt_tools along with their input parameters.

        Returns:
            Optional[List[Tuple[Callable, Dict[str, Any]]]]: List of (callable, call_arguments) tuples or None if no prompt_tools
        """
        if not self.prompt_tools:
            return None
        return [(tool.get_callable(refresh=tool.auto_refresh), tool.call_arguments)
                for tool in self.prompt_tools.values()]

    def subscribe_to_tool_call(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to tool call events.

        Args:
            callback (SubscriberCallback): The callback function to be called when a tool is called
        """
        if self.tools:
            for tool in self.tools.values():
                tool.subscribe_to_call(callback)

    def subscribe_to_tool_result(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to tool result events.

        Args:
            callback (SubscriberCallback): The callback function to be called when a tool returns a result
        """
        if self.tools:
            for tool in self.tools.values():
                tool.subscribe_to_result(callback)

    def subscribe_to_tool_error(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to tool error events.

        Args:
            callback (SubscriberCallback): The callback function to be called when a tool raises an error
        """
        if self.tools:
            for tool in self.tools.values():
                tool.subscribe_to_error(callback)

    def subscribe_to_prompt_tool_call(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to prompt tool call events.

        Args:
            callback (SubscriberCallback): The callback function to be called when a prompt tool is called
        """
        if self.prompt_tools:
            for tool in self.prompt_tools.values():
                tool.subscribe_to_call(callback)

    def subscribe_to_prompt_tool_result(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to prompt tool result events.

        Args:
            callback (SubscriberCallback): The callback function to be called when a prompt tool returns a result
        """
        if self.prompt_tools:
            for tool in self.prompt_tools.values():
                tool.subscribe_to_result(callback)

    def subscribe_to_prompt_tool_error(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to prompt tool error events.

        Args:
            callback (SubscriberCallback): The callback function to be called when a prompt tool raises an error
        """
        if self.prompt_tools:
            for tool in self.prompt_tools.values():
                tool.subscribe_to_error(callback)

class JustAgentProfile(JustSerializable, JustAgentProfileToolsetMixin):
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

class JustAgentFullProfile(JustAgentProfile, JustAgentProfileToolsetMixin, JustAgentProfileChatMixin, JustAgentProfileWebMixin, JustAgentProfileSpecializationMixin, JustAgentProfileRagMixin):
    """
    A Pydantic model representing an agent profile with all extended attributes.
    """

    TO_REFRESH: ClassVar[Tuple[str, ...]] = ('shortname', 'description')
    """Fields to force-renew using LLM"""
