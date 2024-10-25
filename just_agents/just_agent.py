from pydantic import  Field, PrivateAttr
from typing import Any, Dict, AsyncGenerator, Union, Sequence
from just_agents.interfaces.IAgent import IAgent
from just_agents.llm_session import LLMSession
from just_agents.llm_options import OPENAI_GPT4oMINI
from just_agents.just_profile import JustAgentProfile

class JustAgent(IAgent, JustAgentProfile):
    llm_options: Dict[str, Any] = Field(default=OPENAI_GPT4oMINI)
    _session: LLMSession = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._session = LLMSession(llm_options=self.llm_options, tools=self.get_tools_callables())
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