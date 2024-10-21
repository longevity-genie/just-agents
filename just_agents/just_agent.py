from pathlib import Path
from pydantic import  Field, PrivateAttr
from typing import Optional, Any, Dict, AsyncGenerator, Union, Sequence, ClassVar, Set
from just_agents.interfaces.IAgent import IAgent
from just_agents.llm_session import LLMSession
from just_agents.llm_options import OPENAI_GPT4oMINI
from just_agents.just_profile import JustAgentProfile

class JustAgent(IAgent, JustAgentProfile):
    llm_options: Dict[str, Any] = Field(default=OPENAI_GPT4oMINI)
    tools: Optional[Sequence[Any]] = Field(None)
    autoload_from_yaml : bool = Field(False,exclude=True)
    config_path : Optional[Path] = Field(JustAgentProfile.CONFIG_PATH,exclude=True)
    config_parent_section: Optional[Path] = Field(JustAgentProfile.CONFIG_PARENT_SECTION,exclude=True)

    _session: LLMSession = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._session = LLMSession(llm_options=self.llm_options, tools=self.tools)
        #Change default name to clss name:
        if self.shortname == JustAgentProfile.DEFAULT_SECTION_NAME:
            self.shortname = self.__class__.__name__
        if self.autoload_from_yaml:
            profile  = JustAgentProfile.from_yaml_auto(
                self.shortname,
                parent_section=self.config_parent_section,
                file_path=self.config_path,
            )
            # Loaded some data, parameters set in init take precedence
            if profile and isinstance(profile, JustAgentProfile):
               self.update(
                   profile.to_json(
                       by_alias=False
                   ),
                   overwrite=True
               )
        self._session.instruct(self.system_prompt)


    def stream(
        self,
        prompt: Union[str, Dict | Sequence[Dict]],

    ) -> AsyncGenerator[Any, None]:
        return self._session.stream(prompt)

    def query(
        self,
        prompt: Union[str, Dict | Sequence[Dict]],
    ) -> str:
        return self._session.query(prompt)

    def save_profile_to_yaml(
        self,
        config_path: Optional[Path] = None,
        section_name: Optional[str] = None,
        parent_section: Optional[str] = None,
    ) -> None:
        section_name = section_name or self.shortname
        config_path = config_path or self.config_path
        parent_section = parent_section or self.config_parent_section
        self.save_to_yaml(
            section_name=section_name,
            parent_section=parent_section,
            file_path=config_path
        )

    def last_message(self):
        return self._session.memory.last_message


    def save_to_yaml(
            self,
            section_name: str = None,
            parent_section: str = None,
            file_path: Path = None,
            include_extras: bool = True,
            include: Optional[Set[str]] = None,
            exclude: Optional[Set[str]] = None,
            by_alias: bool = True,
            exclude_none: bool = True,
            serialize_as_any: bool = True,
    ):
        if not file_path:
            file_path = self.CONFIG_PATH
        if not parent_section:
            parent_section = self.CONFIG_PARENT_SECTION
        if not section_name:
            section_name = self.shortname

        super().save_to_yaml(
            section_name=section_name,
            parent_section=parent_section,
            file_path=file_path,
            include_extras=include_extras,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_none=exclude_none,
            serialize_as_any=serialize_as_any,)



