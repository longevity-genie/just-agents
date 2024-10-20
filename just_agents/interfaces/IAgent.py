from pathlib import Path
from typing import AsyncGenerator, Any
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

class IAgent:
    def stream(self, input: str | dict | list[dict]) -> AsyncGenerator[Any, None]:
        raise NotImplementedError("You need to impelement stream() first!")

    def query(self, input: str | dict | list[dict]) -> str:
        raise NotImplementedError("You need to impelement query() first!")