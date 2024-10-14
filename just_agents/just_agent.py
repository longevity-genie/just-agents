from pathlib import Path
import importlib

from abc import ABC, abstractmethod

from attr.validators import instance_of
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Any, Dict, AsyncGenerator
from just_agents.llm_session import LLMSession
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
from just_agents.just_yaml import JustYaml as Yml

class JustAbstractAgent(ABC):

    @abstractmethod
    def stream(
        self,
        prompt: str,
        run_callbacks: bool = True,
        output: Optional[Path] = None
    ) -> AsyncGenerator[Any, None]:
        raise NotImplementedError()

    @abstractmethod
    def query(
        self,
        prompt: str,
        run_callbacks: bool = True,
        output: Optional[Path] = None
    ) -> str:
        raise NotImplementedError()

#class JustAgent(JustAbstractAgent, BaseModel, arbitrary_types_allowed=True):
class JustAgent(JustAbstractAgent, BaseModel):
    DEFAULT_GENERIC_PROMPT : str = "You are a helpful AI assistant"
    DEFAULT_DESCRIPTION : str = "A general-purpose AI assistant."

    llm_options: Dict[str, Any]
    tools: Optional[Any] = None
    autoload_from_yaml : bool = False
    config_path : Optional[Path] = None
    agent_name: Optional[str] = Field(None)
    system_prompt: Optional[str] = Field(DEFAULT_GENERIC_PROMPT)
    description: Optional[Any] = Field(DEFAULT_DESCRIPTION) # JustProfile extension class can be placed here
    #output_streaming: AbstractStreamingProtocol = Field(default_factory=OpenaiStreamingProtocol)
    _session: LLMSession = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._session = LLMSession(llm_options=self.llm_options, tools=self.tools)
        config_path = self.config_path or Yml.DEFAULT_CONFIG_PATH
        if self.agent_name is None:
            self.agent_name = self.__class__.__name__
        if self.autoload_from_yaml:
            config_data = Yml.read_yaml_data_safe(config_path, self.agent_name)
            if config_data:
                # Load from config, if available and if not customized on creation
                if self.system_prompt == self.DEFAULT_GENERIC_PROMPT:
                    self.system_prompt = config_data.get('system_prompt', self.system_prompt)
                if self.description == self.DEFAULT_DESCRIPTION:
                    self.description = config_data.get('description', self.description)

                # Dynamically load the class specified in 'profile_type'
                profile_type = config_data.get('profile_type')
                if profile_type:
                    try:
                        # Splits into `module.submodule` and `ClassName` for dynamic import
                        module_name, class_name = profile_type.rsplit(".",1)
                        module = importlib.import_module(module_name)
                        cls = getattr(module, class_name)
                        # Dynamic instantiation of `JustAgentProfile` or whatever class is specified
                        extension_class = cls.from_json(config_data)
                        self.description = extension_class
                    except Exception as e:
                        print(f"Exception occurred: {str(e)}")
                        self.description = config_data.get('description', self.description)

        if self.system_prompt is not None:
            self._session.instruct(self.system_prompt)

    def stream(
        self,
        prompt: str,
        run_callbacks: bool = True,
        output: Optional[Path] = None
    ) -> AsyncGenerator[Any, None]:
        return self._session.stream(prompt, run_callbacks, output)

    def query(
        self,
        prompt: str,
        run_callbacks: bool = True,
        output: Optional[Path] = None
    ) -> str:
        return self._session.query(prompt, run_callbacks, output)

    def save_config(
        self,
        config_path: Optional[Path] = None,
        section_name: Optional[str] = None,
        parent_section: Optional[str] = None,
    ) -> None:
        config_path = config_path or self.config_path or Yml.DEFAULT_CONFIG_PATH
        section_name = section_name or self.agent_name or self.__class__.__name__
        parent_section = parent_section or Yml.DEFAULT_AGENT_PROFILES_SECTION
        data = {}
        try:
            if isinstance(self.description, BaseModel):
                if hasattr(self.description, 'model_dump_with_extras'):
                    data = self.description.to_json()
                elif hasattr(self.description, 'model_dump'):
                    data = self.description.model_dump()
                elif hasattr(self.description, 'to_dict'):
                    data = self.description.to_dict()
            elif self.description is instance_of(str):
                data.update({'description':self.description})
            if self.system_prompt:
                data.update({'system_prompt': self.system_prompt})
        finally:
            if data:
                Yml.save_to_yaml(
                    config_path,
                    data,
                    section_name,
                    parent_section )

    def last_message(self):
        return self._session.memory.last_message






