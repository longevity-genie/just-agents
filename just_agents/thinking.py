
from typing import ClassVar, Literal
from just_agents.base_agent import BaseAgent
from pydantic import BaseModel, Field

from just_agents.interfaces.IAgent import IThinkingAgent, IThought
from just_agents.types import Output, SupportedMessages


class Thought(IThought):
    """
    This is a thought object that is used to represent a thought in the chain of thought agent.
    """
    title: str
    content: str
    next_action: Literal["continue", "final_answer"]
    
    def is_final(self) -> bool:
        return self.next_action == "final_answer"


class ChainOfThoughtAgent(BaseAgent, IThinkingAgent[SupportedMessages, SupportedMessages, SupportedMessages, Thought]):

    DEFAULT_SYSTEM_PROMPT: ClassVar[str] = """
You are an expert AI assistant that explains your reasoning step by step. 
  For each step, provide a title that describes what you're doing in that step, along with the content. 
  Decide if you need another step or if you're ready to give the final answer. 
  Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. 
  Make sure you send only one JSON step object. 
  USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. 
  BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. 
  IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. 
  CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. 
  FULLY TEST ALL OTHER POSSIBILITIES. 
  YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. 
  DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
  
              Example of a valid JSON response:
              ```json
              {
                  "title": "Identifying Key Information",
                  "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
                  "next_action": "continue"
              }```
"""

    system_prompt: str = Field(
        DEFAULT_SYSTEM_PROMPT,
        description="System prompt of the agent")
    
    def thought_query(self, response: SupportedMessages) -> Thought: # type: ignore
        return self.query_structural(response, parser=Thought)

