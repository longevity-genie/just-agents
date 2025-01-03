from just_agents.interfaces.agent import IAgent
from just_agents.simple.llm_session import LLMSession
import json
from just_agents.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from pathlib import Path
from just_agents.simple.utils import resolve_and_validate_agent_schema, resolve_llm_options, resolve_system_prompt, resolve_tools

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
                 tools: list = None, output_streaming:IAbstractStreamingProtocol = OpenaiStreamingProtocol()):
        self.agent_schema: dict = resolve_and_validate_agent_schema(agent_schema, "cot_agent_prompt.yaml")
        if tools is None:
            tools = resolve_tools(self.agent_schema)
        self.session: LLMSession = LLMSession(llm_options=resolve_llm_options(self.agent_schema, llm_options),
                                              system_prompt=resolve_system_prompt(self.agent_schema),
                                              agent_schema=self.agent_schema.get(LLM_SESSION, None), tools=tools)

        self.output_streaming: IAbstractStreamingProtocol = output_streaming


    def _process_initial_query(self, input: str | dict | list) -> tuple[dict, str]:
        thought_max_tokens = self.agent_schema.get(THOUGHT_MAX_TOKES, 300)
        self.session.update_options("max_tokens", thought_max_tokens)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(input))
        content_key = self.agent_schema.get(CONTENT_NAME, "content")
        content = f"{step_data[content_key]}\n"
        max_steps = self.agent_schema.get(MAX_STEPS, 25)

        return content, content_key, max_steps


    def _get_final_answer(self, content_key:str) -> str:
        final_max_tokens = self.agent_schema.get(FINAL_MAX_TOKENS, 1200)
        self.session.update_options("max_tokens", final_max_tokens)

        final_prompt = self.agent_schema.get(
            FINAL_PROMPT,
            "Please provide the final answer based solely on your reasoning above."
        )
        final_data = json.loads(self.session.query(final_prompt))
        return final_data[content_key]


    def stream(self, input: str | dict | list):
        content, content_key, max_steps = self._process_initial_query(input)
        yield self.output_streaming.get_chunk(0, content, self.session.llm_options)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[content_key] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, self.session.llm_options)
            if step_data[self.agent_schema.get(NEXT_ACTION, "next_action")] == self.agent_schema.get(ACTION_FINAL, "final_answer"):
                break

        content = self._get_final_answer(content_key)
        yield self.output_streaming.get_chunk(step_count + 1, content, self.session.llm_options)
        yield self.output_streaming.done()


    def query(self, input:  str | dict | list):
        content, content_key, max_steps = self._process_initial_query(input)
        thoughts:str = content
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[content_key] + "\n"
            thoughts += content
            if step_data[self.agent_schema.get(NEXT_ACTION, "next_action")] == self.agent_schema.get(ACTION_FINAL, "final_answer"):
                break

        content = self._get_final_answer(content_key)
        return content, thoughts


    def last_message(self):
        return self.session.memory.last_message