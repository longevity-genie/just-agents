from just_agents.interfaces.IAddAllMessages import IAddAllMessages
from just_agents.llm_session import LLMSession
import json
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
from pathlib import Path, PurePath
import yaml
from just_agents.utils import resolve_agent_schema, resolve_llm_options, resolve_system_prompt

class ChainOfThoughtAgent():

    def __init__(self, llm_options: dict = None, agent_schema: str | Path | dict | None = None,
                 tools: list = None, init_system_prompt: bool = True, output_streaming:AbstractStreamingProtocol = OpenaiStreamingProtocol()):
        self.agent_schema: dict = resolve_agent_schema(agent_schema, "ChainOfThoughtAgent", "cot_agent_prompt.yaml")
        self.session: LLMSession = LLMSession(llm_options=resolve_llm_options(self.agent_schema, llm_options),
                                              agent_schema=self.agent_schema.get("llm_session", None), tools=tools)
        if init_system_prompt:
            system_prompt = resolve_system_prompt(self.agent_schema)
            if system_prompt is not None:
                self.session.instruct(system_prompt)
        self.output_streaming: AbstractStreamingProtocol = output_streaming


    def stream(self, input: str | dict | list):
        thought_max_tokes = self.agent_schem.get("thought_max_tokes", 300)
        self.session.update_options("max_tokens", thought_max_tokes)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(input))
        CONTENT = self.agent_schem.get("content", "content")
        content = step_data[CONTENT] + "\n"
        yield self.output_streaming.get_chunk(0, content, self.session.llm_options)
        max_steps = self.agent_schem.get("max_steps", 25)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[CONTENT] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, self.session.llm_options)
            if step_data[self.agent_schem.get("next_action", "next_action")] == self.agent_schem.get("action_final", "final_answer"):
                break

        final_max_tokens = self.agent_schem.get("final_max_tokens", 1200)
        self.session.update_options("max_tokens", final_max_tokens)
        FINAL_PROMPT = self.agent_schem.get("final_prompt", "Please provide the final answer based solely on your reasoning above.")
        final_data = json.loads(self.session.query(FINAL_PROMPT))
        yield self.output_streaming.get_chunk(step_count + 1, final_data[CONTENT], self.session.llm_options)
        yield self.output_streaming.done()


    def query(self, input:  str | dict | list):
        thought_max_tokes = self.agent_schem.get("thought_max_tokes", 300)
        self.session.update_options("max_tokens", thought_max_tokes)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(input))
        CONTENT = self.agent_schem.get("content", "content")
        content = step_data[CONTENT] + "\n"
        thoughts:str = content
        max_steps = self.agent_schem.get("max_steps", 25)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[CONTENT] + "\n"
            thoughts += content
            if step_data[self.agent_schem.get("next_action", "next_action")] == self.agent_schem.get("action_final", "final_answer"):
                break

        final_max_tokens = self.agent_schem.get("final_max_tokens", 1200)
        self.session.update_options("max_tokens", final_max_tokens)
        FINAL_PROMPT = self.agent_schem.get("final_prompt", "Please provide the final answer based solely on your reasoning above.")
        final_data = json.loads(self.session.query(FINAL_PROMPT))
        return final_data[CONTENT], thoughts

    def last_message(self):
        return self.session.memory.last_message