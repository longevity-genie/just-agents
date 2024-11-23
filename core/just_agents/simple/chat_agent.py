from typing import Any, Dict, Optional

from just_agents.simple.llm_session import LLMSession
from just_agents.simple.utils import load_config


class ChatAgent(LLMSession):
    """
    Default agent class, it assumes that each agent has a role, goal, and background story.
    So far this class is in a refactoring process, use LLMSession instead.
    """
    def __init__(self, role: str, goal: Optional[str] = None, task: Optional[str] = None,
                 backstory: Optional[str] = None, config: Dict[str, Any] = None,
                 agent_prompt: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.goal = goal
        self.task = task
        self.backstory = backstory
        self.config = config or load_config("agent_prompts.yaml")
        self.agent_prompt = agent_prompt

        agent_template: str = str(self.config["agent_prompt"])
        self.agent_prompt = self._render_template(agent_template)
        if self.agent_prompt is not None:
            self.instruct(self.agent_prompt)

    def _render_template(self, template: str):
        """
        renders agent template
        :param template:
        :return:
        """
        return template.format(role=self.role,
                               goal=self.goal,
                               backstory=self.backstory,
                               task=self.task)
