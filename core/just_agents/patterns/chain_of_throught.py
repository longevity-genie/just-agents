from typing import ClassVar, Literal, Any
from just_agents.base_agent import BaseAgent
from pydantic import Field
from just_agents.types import SupportedMessages
from just_agents.base_memory import IBaseMemory
from just_agents.patterns.interfaces.IThinkingAgent import IThinkingAgent, IThought


class Thought(IThought):
    """
    This is a thought object that is used to represent a thought in the chain of thought agent.
    """
    # Represents the title/summary of the current thinking step
    title: str
    # The detailed explanation/reasoning for this thought step
    content: str
    # Indicates whether to continue thinking or provide final answer
    next_action: Literal["continue", "final_answer"]
    
    def is_final(self) -> bool:
        # Helper method to check if this is the final thought in the chain
        return self.next_action == "final_answer"


class ChainOfThoughtAgent(BaseAgent, IThinkingAgent[SupportedMessages, SupportedMessages, SupportedMessages, Thought]):
    # Default prompt that instructs the agent to:
    # 1. Explain reasoning step by step
    # 2. Use at least 3 steps
    # 3. Consider limitations and alternative answers
    # 4. Use multiple methods to verify answers
    # 5. Format response as JSON with specific fields
    DEFAULT_COT_PROMPT: ClassVar[str] = """ You are an expert AI assistant that explains your reasoning step by step. 
    """
    RESPONSE_FORMAT: ClassVar[str] = """
  For each step, provide a title that describes what you're doing in that step, along with the content. 
  Decide if you need another step or if you're ready to give the final answer. 
  Respond in JSON format with "title", "content", and "next_action" (either "continue" or "final_answer") keys. 
  Make sure you send only one JSON step object AND NOTHING ELSE. 
  You response should be a valid JSON object. 
  In the JSON use Use Triple Quotes for Multi-line Strings.
  USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 2. 
  BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. 
  IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. 
  CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. 
  FULLY TEST ALL OTHER POSSIBILITIES. 
  YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. 
  DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
  
              Example of a valid JSON response:
              {
                  "title": "Identifying Key Information",
                  "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
                  "next_action": "continue"
              }
"""

    system_prompt: str = Field(
        DEFAULT_COT_PROMPT,
        description="System prompt of the agent")

    response_format: str = Field(
        RESPONSE_FORMAT,
        description="Response format of the agent")

    max_steps: int = Field(IThinkingAgent.MAX_STEPS, ge=1, description="Maximum number of reasoning steps")
    append_response_format: bool = Field(True,
                                         description="Whether to append default COT prompt of this agent to the provided")

    def instruct(self, prompt: str, memory: IBaseMemory = None):
        # override how system prompt is applied:
        super().instruct(prompt, memory)
        if self.append_response_format:
            super().instruct(self.response_format, memory) # don't modify self system prompt to avoid saving it into profile

    def thought_query(self, query: SupportedMessages, **kwargs) -> Thought:
        # Parses the LLM response into a structured Thought object
        if self.supports_response_format and "gpt-4" in self.llm_options["model"]: # despite what they declare only openai does support reponse format right
           return self.query_structural(query, parser=Thought, response_format={"type": "json_object"}, **kwargs) # type: ignore
        else:
            return self.query_structural(query, parser=Thought, **kwargs) # type: ignore
