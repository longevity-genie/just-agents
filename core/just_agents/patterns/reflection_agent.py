from typing import Union, Dict, Sequence, AsyncGenerator, Any
from just_agents.interfaces.agent import IAgent

ITERATIONS = "iterations"
CRITIC_PROMPT = "critic_prompt"
STOP_WORD = "stop_word"

class ReflectionAgent(IAgent):
    """
    This agent is used to reflect on the output of another agent.
    This implementation is deprecated, we will rewrite it
    """

    def __init__(self, agent_schema: dict = None, author: IAgent = None, critic: IAgent = None):
        self.author: IAgent = author
        self.critic: IAgent = critic
        self.agent_schema: dict = agent_schema
        if self.agent_schema is None:
            self.agent_schema = dict()

    def query(self, query_input: Union[str, Dict, Sequence[Dict]]) -> str:
        solution = self.author.query(query_input)
        iterations = self.agent_schema.get(ITERATIONS, 3)
        critic_prompt = self.agent_schema.get(CRITIC_PROMPT, "Review this output and provide specific constructive criticism: ")
        stop_word = self.agent_schema.get(STOP_WORD, "done")
        for _ in range(iterations):
            feedback = self.critic.query(critic_prompt + "'''"+solution+"'''")
            if feedback.lower().strip() == stop_word:
                break
            solution = self.author.query(feedback)

        return solution

    def stream(self, query_input: Union[str, Dict, Sequence[Dict]]) -> AsyncGenerator[Any, None]:
        pass