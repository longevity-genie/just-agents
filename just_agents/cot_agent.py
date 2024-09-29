from just_agents.llm_session import LLMSession
import json
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol

FINAL_PROMPT = "Please provide the final answer based solely on your reasoning above."
DEFAULT_SYSTEM_PROMPT = """You are an expert AI assistant that explains your reasoning step by step. 
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

class ChainOfThoughtAgent():

    def __init__(self, llm_options, tools = None, system_prompt:str = DEFAULT_SYSTEM_PROMPT, output_streaming:AbstractStreamingProtocol = OpenaiStreamingProtocol()):
        self.session: LLMSession = LLMSession(llm_options=llm_options, tools=tools)
        if system_prompt is not None:
            self.session.instruct(system_prompt)
        self.output_streaming: AbstractStreamingProtocol = output_streaming


    def stream(self, prompt, max_steps: int = 25, thought_max_tokes:int = 300, final_max_tokens:int = 1200, final_prompt:str = FINAL_PROMPT):
        self.session.update_options("max_tokens", thought_max_tokes)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(prompt))
        content = step_data['content'] + "\n"
        yield self.output_streaming.get_chunk(0, content, self.session.llm_options)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data['content'] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, self.session.llm_options)
            if step_data['next_action'] == 'final_answer':
                break

        self.session.update_options("max_tokens", final_max_tokens)
        final_data = json.loads(self.session.query(final_prompt))
        yield self.output_streaming.get_chunk(step_count + 1, final_data['content'], self.session.llm_options)
        yield self.output_streaming.done()


    def query(self, prompt, max_steps: int = 25, thought_max_tokes:int = 300, final_max_tokens:int = 1200, final_prompt:str = FINAL_PROMPT):
        self.session.update_options("max_tokens", thought_max_tokes)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(prompt))
        content = step_data['content'] + "\n"
        thoughts:str = content
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data['content'] + "\n"
            thoughts += content
            if step_data['next_action'] == 'final_answer':
                break

        self.session.update_options("max_tokens", final_max_tokens)
        final_data = json.loads(self.session.query(final_prompt))
        return final_data['content'], thoughts

    def last_message(self):
        return self.session.memory.last_message