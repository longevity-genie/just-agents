from typing import ClassVar, Optional, Any, Dict
from just_agents.base_agent import BaseAgent
from pydantic import Field, PrivateAttr
from just_agents.just_tool import JustToolsBus

from just_agents.examples.coding.cot_memory import ActionableThought, IBaseThoughtMemory, BaseThoughtMemory
from just_agents.types import SupportedMessages
from just_agents.patterns.interfaces.IThinkingAgent import IThinkingAgent




class ChainOfThoughtDevAgent(BaseAgent, IThinkingAgent[SupportedMessages, SupportedMessages, SupportedMessages, ActionableThought]):
    """
    Agen uses default prompt that instructs the agent to:
    1. Explain reasoning step by step
    2. Use at least 3 steps
    3. Consider limitations and alternative answers
    4. Use multiple methods to verify answers
    5. Format response as JSON with specific fields

    This prompt may be appended after the other custom prompt to introduce COT pattern
    """
    RESPONSE_FORMAT: ClassVar[str] = """
RESPONSE FORMAT:

Your input may contain 'final_answer' entries, consider these answers of other agents.   
For each step, provide a title that describes what you're doing in that step, along with the content.
Decide if you need another step or if you're ready to give the final answer. 
Respond in JSON format with 'title', 'content', 'code', 'console', and 'next_action' (either 'continue' or 'final_answer') keys.
Make sure you send only one JSON step object. You response should be a valid JSON object. In the JSON use Use Triple Quotes for Multi-line Strings.

USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. 
BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. 
IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. 
CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. 
FULLY TEST ALL OTHER POSSIBILITIES. 
YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. 
DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example 1 of a valid JSON response:
```json
{
  "title": "Identifying Key Information",
  "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
  "next_action": "continue"
}```
Example 2 of a valid JSON response:
```json
{
  "title": "Code to solve the problem",
  "content": "This code is expected to ... As a result the following should be produced: ...",
  "code": "\"\"
        import numpy as np
        ...
  \"\"",
  "next_action": "final_answer"
}```
Example 3 of a valid JSON response:
```json
{
  "title": "Code execution observations",
  "content": "Code execution failed during ... , root cause of the problem likely is ..."
  "code": " "
  "console": "\"\"
      Traceback (most recent call last):
  \"\"",
  "next_action": "final_answer"
}```
Example 1 of INVALID response including multiple JSON objects instead of one, DO NOT do that:
```json
{
  "title": "Some thinking",
  "content": "...",
  "next_action": "continue"
}
{
  "title": "Final thought!",
  "content": "I got an answer already",
  "next_action": "final_answer"
}
Example 2 of INVALID response including multiple JSON objects instead of one, DO NOT do that:
```json
{
  "title": "Some thinking",
  "content": "...",
  "next_action": "continue"
}
{
  "title": "Some more thinking in same step",
  "content": "...",
  "next_action": "continue"
}
```

"""

    # Allow customization of the system prompt while maintaining the default as fallback
    DEFAULT_COT_PROMPT: ClassVar[str] = """ You are an expert AI assistant that explains your reasoning step by step. 
    """

    CODE_OK: ClassVar[str] = "Code syntax is correct"

    system_prompt: str = Field(
        DEFAULT_COT_PROMPT,
        description="System prompt of the agent")

    response_format: str = Field(
        RESPONSE_FORMAT,
        description="System prompt of the agent")

    thoughts: IBaseThoughtMemory = Field(default_factory= BaseThoughtMemory, exclude=True, description="Memory of thought chains")

    max_steps: int = Field(IThinkingAgent.MAX_STEPS, ge=1, description="Maximum number of reasoning steps")
    append_response_format: bool = Field(True, description="Whether to append default COT prompt of this agent to the provided")

    _event_bus : JustToolsBus = PrivateAttr(default_factory= JustToolsBus)
    _code_buffer: Dict[str,str]  = PrivateAttr(default_factory=dict)
    _console_buffer: str = PrivateAttr("")

    def model_post_init(self, __context: Any) -> None:
        # Call parent class's post_init first (from JustAgentProfile)
        super().model_post_init(__context)
        if self.append_response_format:
            system_prompt  = self.system_prompt + "\n\n" + self.response_format
            self.memory.clear_messages()
            self.instruct(system_prompt) # don't modify self system prompt to avoid saving it into profile

        # Subscribe handlers to events
        self.event_bus.subscribe("submit_code.call", self.handle_submit_code)
        self.event_bus.subscribe("submit_console_output.call", self.handle_submit_console_output)


    def thought_query(self, query: SupportedMessages, **kwargs) -> ActionableThought:
        # Parses the LLM response into a structured ActionableThought object
        if self.supports_response_format and "gpt-4" in self.llm_options["model"]:  # despite what they declare only openai does support reponse format right
            return self.query_structural(query, parser=ActionableThought, response_format={"type": "json_object"}, **kwargs)
        else:
            return self.query_structural(query, parser=ActionableThought, **kwargs)

    def handle_submit_code(self, code: str, filename: str, result: str) -> None:
        if result == self.CODE_OK:
            # Prepare JSON with proper escaping
            self._code_buffer[filename] = code
            # escaped_json = json.dumps(output, indent=4)

    def handle_submit_console_output(self, output: str, append: bool) -> None:
        previous = ""
        if append:
            previous = self._console_buffer
        self._console_buffer = previous + '\n' + output

    def think(
        self,
        query: SupportedMessages,
        max_iter: Optional[int] = None,
        chain: Optional[list[ActionableThought]] = None,
        **kwargs
    ) -> tuple[Optional[ActionableThought], Optional[list[ActionableThought]]]:
        if not max_iter:
            max_iter = self.max_steps
        final_result : Optional[ActionableThought] = None
        current_chain : Optional[list[ActionableThought]] = None
        (final_result , current_chain) = super().think(query,max_iter=max_iter,chain=None,**kwargs)
        for thought in current_chain:
            thought.agent = self.shortname
        if final_result and isinstance(final_result, ActionableThought):
            final_result.agent = self.shortname

        self.thoughts.add_messages([current_chain,final_result]) #remember individual thougths
        return (
            final_result,
            [*(chain or []), current_chain]
        )




