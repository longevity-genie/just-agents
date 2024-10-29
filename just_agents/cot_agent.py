from just_agents.interfaces.IAgent import IAgent
from just_agents.llm_session import LLMSession
import json
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
from pathlib import Path, PurePath
import yaml
from just_agents.utils import _resolve_agent_schema, resolve_llm_options, resolve_system_prompt, resolve_tools

# schema parameters:
LLM_SESSION = "llm_session"
THOUGHT_MAX_TOKES = "thought_max_tokes"
CONTENT_NAME = "content"
MAX_STEPS = "max_steps"
NEXT_ACTION = "next_action"
ACTION_FINAL = "action_final"
FINAL_MAX_TOKENS = "final_max_tokens"
FINAL_PROMPT = "final_prompt"


class ChainOfThoughtAgent(IAgent):

    def __init__(self, llm_options: dict = None, agent_schema: str | Path | dict | None = None,
                 tools: list = None, output_streaming:AbstractStreamingProtocol = OpenaiStreamingProtocol()):
        self.agent_schema: dict = _resolve_agent_schema(agent_schema, "cot_agent_prompt.yaml")
        if tools is None:
            tools = resolve_tools(self.agent_schema)
        self.session: LLMSession = LLMSession(llm_options=resolve_llm_options(self.agent_schema, llm_options),
                                              system_prompt=resolve_system_prompt(self.agent_schema),
                                              agent_schema=self.agent_schema.get(LLM_SESSION, None), tools=tools)

        self.output_streaming: AbstractStreamingProtocol = output_streaming


    def stream(self, input: str | dict | list):
        thought_max_tokes = self.agent_schema.get(THOUGHT_MAX_TOKES, 500)
        self.session.update_options("max_tokens", thought_max_tokes)
        if not "claude" in self.session.llm_options["model"]:
            self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(input))
        CONTENT = self.agent_schema.get(CONTENT_NAME, "content")
        content = step_data[CONTENT] + "\n"
        yield self.output_streaming.get_chunk(0, content, self.session.llm_options)
        max_steps = self.agent_schema.get(MAX_STEPS, 25)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[CONTENT] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, self.session.llm_options)
            if step_data[self.agent_schema.get(NEXT_ACTION, "next_action")] == self.agent_schema.get(ACTION_FINAL, "final_answer"):
                break

        final_max_tokens = self.agent_schema.get(FINAL_MAX_TOKENS, 1200)
        self.session.update_options("max_tokens", final_max_tokens)
        final_prompt = self.agent_schema.get(FINAL_PROMPT, "Please provide the final answer based solely on your reasoning above.")
        final_data = json.loads(self.session.query(final_prompt))
        yield self.output_streaming.get_chunk(step_count + 1, final_data[CONTENT], self.session.llm_options)
        yield self.output_streaming.done()


    def query(self, input:  str | dict | list):
        thought_max_tokes = self.agent_schema.get(THOUGHT_MAX_TOKES, 500)
        self.session.update_options("max_tokens", thought_max_tokes)
        if not "claude" in self.session.llm_options["model"]:
            self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(input))
        CONTENT = self.agent_schema.get(CONTENT_NAME, "content")
        content = step_data[CONTENT] + "\n"
        thoughts:str = content
        max_steps = self.agent_schema.get(MAX_STEPS, 25)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[CONTENT] + "\n"
            thoughts += content
            if step_data[self.agent_schema.get(NEXT_ACTION, "next_action")] == self.agent_schema.get(ACTION_FINAL, "final_answer"):
                break

        final_max_tokens = self.agent_schema.get(FINAL_MAX_TOKENS, 1200)
        self.session.update_options("max_tokens", final_max_tokens)
        final_prompt = self.agent_schema.get(FINAL_PROMPT, "Please provide the final answer based solely on your reasoning above.")
        final_data = json.loads(self.session.query(final_prompt))
        return final_data[CONTENT], thoughts

    def last_message(self):
        return self.session.memory.last_message