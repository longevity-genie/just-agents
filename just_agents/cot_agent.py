from just_agents.llm_session import LLMSession
import json
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
from pathlib import Path, PurePath
import yaml

class ChainOfThoughtAgent():

    def __init__(self, llm_options, prompt_schema: any = None, tools: list = None, output_streaming:AbstractStreamingProtocol = OpenaiStreamingProtocol()):
        self.session: LLMSession = LLMSession(llm_options=llm_options, tools=tools)
        if prompt_schema is None:
            prompt_schema = Path(Path(__file__).parent, "config", "cot_agent_prompt.yaml")
        if isinstance(prompt_schema, str):
            prompt_schema = Path(prompt_schema)
        if isinstance(prompt_schema, Path):
            if not prompt_schema.exists():
                raise ValueError(
                    f"In ChainOfThoughtAgent constructor prompt_schema path is not exists: ({prompt_schema})!")
            with open(prompt_schema) as f:
                prompt_schema = yaml.full_load(f)
        if not isinstance(prompt_schema, dict):
            raise ValueError("In ChainOfThoughtAgent constructor prompt_schema parameter should be None, string, Path or dict!")

        self.prompt_schema: dict = prompt_schema
        self.session.instruct(self.prompt_schema["system_prompt"])
        self.output_streaming: AbstractStreamingProtocol = output_streaming


    def get_param(self, key: str, default: any):
        just_agents = self.session.llm_options.get("just-agents", None)
        if just_agents is None:
            return default
        return just_agents.get(key, default)


    def stream(self, prompt):
        thought_max_tokes = self.get_param("thought_max_tokes", 300)
        self.session.update_options("max_tokens", thought_max_tokes)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(prompt))
        content = step_data[self.prompt_schema["content"]] + "\n"
        yield self.output_streaming.get_chunk(0, content, self.session.llm_options)
        max_steps = self.get_param("max_steps", 25)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[self.prompt_schema["content"]] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, self.session.llm_options)
            if step_data[self.prompt_schema["next_action"]] == self.prompt_schema["action_final"]:
                break

        final_max_tokens = self.get_param("final_max_tokens", 1200)
        self.session.update_options("max_tokens", final_max_tokens)
        final_data = json.loads(self.session.query(self.prompt_schema["final_prompt"]))
        yield self.output_streaming.get_chunk(step_count + 1, final_data[self.prompt_schema["content"]], self.session.llm_options)
        yield self.output_streaming.done()


    def query(self, prompt):
        thought_max_tokes = self.get_param("thought_max_tokes", 300)
        self.session.update_options("max_tokens", thought_max_tokes)
        self.session.update_options("response_format", {"type": "json_object"})
        step_data = json.loads(self.session.query(prompt))
        content = step_data[self.prompt_schema["content"]] + "\n"
        thoughts:str = content
        max_steps = self.get_param("max_steps", 25)
        for step_count in range(1, max_steps):
            step_data = json.loads(self.session.proceed())
            content = step_data[self.prompt_schema["content"]] + "\n"
            thoughts += content
            if step_data[self.prompt_schema["next_action"]] == self.prompt_schema["action_final"]:
                break

        final_max_tokens = self.get_param("final_max_tokens", 1200)
        self.session.update_options("max_tokens", final_max_tokens)
        final_data = json.loads(self.session.query(self.prompt_schema["final_prompt"]))
        return final_data[self.prompt_schema["content"]], thoughts

    def last_message(self):
        return self.session.memory.last_message