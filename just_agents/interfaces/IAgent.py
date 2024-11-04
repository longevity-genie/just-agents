from pathlib import Path
from typing import AsyncGenerator, Any, Union, Sequence, Dict
from abc import ABC, abstractmethod
from just_agents.utils import resolve_agent_schema

def build_agent(agent_schema: str | Path | dict):
    from just_agents.cot_agent import ChainOfThoughtAgent
    from just_agents.llm_session import LLMSession
    agent_schema = resolve_agent_schema(agent_schema)
    class_name = agent_schema.get("class", None)
    if class_name is None or class_name == "LLMSession":
        return LLMSession(agent_schema=agent_schema)
    elif class_name == "ChainOfThoughtAgent":
        return ChainOfThoughtAgent(agent_schema=agent_schema)

class IAgent(ABC):
    @abstractmethod
    def stream(self, query_input: Union[str, Dict, Sequence[Dict]]) -> AsyncGenerator[Any, None]:
        raise NotImplementedError("You need to implement stream() first!")

    @abstractmethod
    def query(self, query_input: Union[str, Dict, Sequence[Dict]]) -> str:
        raise NotImplementedError("You need to implement query() first!")
