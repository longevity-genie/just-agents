from typing import Optional

from just_agents.chat_agent import ChatAgent
from dataclasses import dataclass, field

from just_agents.utils import load_config


@dataclass
class ThinkingAgent(ChatAgent):
    thinking_prompt: Optional[str] = field(default_factory=lambda: load_config("agent_prompts.yaml"))