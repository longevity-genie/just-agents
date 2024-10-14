import copy
import json
from pathlib import Path
from typing import Any, AsyncGenerator

import litellm
from litellm import ModelResponse, completion
from litellm.utils import Choices

from just_agents.llm_options import LLAMA3
from just_agents.memory import Memory
from dataclasses import dataclass, field
from typing import Callable, Optional
from just_agents.streaming.abstract_streaming import AbstractStreaming
from just_agents.streaming.openai_streaming import AsyncSession
# from just_agents.utils import rotate_completion
from just_agents.interfaces.IAddAllMessages import IAddAllMessages
from just_agents.utils import resolve_agent_schema, resolve_llm_options, resolve_system_prompt
from just_agents.rotate_keys import RotateKeys

OnCompletion = Callable[[ModelResponse], None]


class LLMSession():
    available_tools: dict[str, Callable] = dict()
    memory: Memory = Memory()
    streaming: AbstractStreaming = None
    key_getter: RotateKeys = None


    def __init__(self, llm_options: dict[str, Any] = None, agent_schema: str | Path | dict | None = None, init_system_prompt: bool = True,  tools: list[Callable] = None, on_response: list[OnCompletion] = []):

        self.agent_schema = resolve_agent_schema(agent_schema, "LLMSession", "llm_session_schema.yaml")
        self.llm_options: dict[str, Any] = resolve_llm_options(self.agent_schema, llm_options)
        if self.agent_schema.get("key_getter", None) is not None:
            self.key_getter = RotateKeys(self.agent_schema["key_getter"])
        self.tools: list[Callable] = tools
        self.on_response: list[OnCompletion] = on_response

        if self.llm_options is not None:
            self.llm_options = copy.deepcopy(self.llm_options) #just a satefy requirement to avoid shared dictionaries
            if (self.key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
                print("Warning api_key will be rewriten by key_getter. Both are present in llm_options.")

        if init_system_prompt:
            system_prompt = resolve_system_prompt(self.agent_schema)
            if system_prompt is not None:
                self.session.instruct(system_prompt)

        streaming_method = self.agent_schema.get("streaming_method", None)
        if streaming_method is None or streaming_method == "openai":
            self.streaming = AsyncSession()
        elif streaming_method.lower() == "qwen2":
            from just_agents.streaming.qwen2_streaming import Qwen2AsyncSession
            self.streaming = Qwen2AsyncSession()
        elif streaming_method.lower() == "chain_of_thought":
            from just_agents.streaming.chain_of_thought import ChainOfThought
            self.streaming = ChainOfThought()
        else:
            raise ValueError("just_streaming_method is incorrect. "
                             "It should be one of this ['qwen2', 'chain_of_thought']")

        if self.tools is not None:
            self._prepare_tools(self.tools)


    def rotate_completion(self, stream: bool, remove_key_on_error: bool = True,
                          max_tries: int = 2) -> ModelResponse:
        opt = self.llm_options.copy()
        if self.key_getter is not None:
            if max_tries < 1:
                max_tries = self.key_getter.len()
            else:
                if remove_key_on_error:
                    max_tries = min(max_tries, self.key_getter.len())
            last_exception = None
            for _ in range(max_tries):
                opt["api_key"] = self.key_getter()
                try:
                    response = completion(messages=self.memory.messages, stream=stream, **opt)
                    return response
                except Exception as e:
                    last_exception = e
                    if remove_key_on_error:
                        self.key_getter.remove(opt["api_key"])

            backup_opt: dict = self.agent_schema.get("backup_options", None)
            if backup_opt:
                return completion(messages=self.memory.messages, stream=stream, **backup_opt)
            if last_exception:
                raise last_exception
            else:
                raise Exception(
                    f"Run out of tries to execute completion. Check your keys! Keys {self.key_getter.len()} left.")
        else:
            return completion(messages=self.memory.messages, stream=stream, **opt)


    def _process_response(self, response: ModelResponse):
        """
        Running handlers to process model responses
        :param response:
        :return:
        """
        for handler in self.on_response:
            handler(response)


    @staticmethod
    def message_from_response(response: ModelResponse) -> dict:
        choice: Choices = response.choices[0]
        message: dict = choice.message.to_dict(exclude_none=True, exclude_unset=True) #converting to dictionary and deleting nonw
        if "function_call" in message and message["function_call"] is None:
            del message["function_call"]
        return message


    def instruct(self, prompt: str):
        """
        Ads system instructions
        :param prompt:
        :return:
        """
        system_instruction =  {"content":prompt, "role":"system"}
        self.memory.add_message(system_instruction, True)
        return system_instruction


    def update_options(self, key:str, value:Any):
        self.llm_options[key] = value


    def _add_to_memory(self, input: str | dict | list[dict]):
        if isinstance(input, list):
            self.memory.add_messages(input)
        else:
            if isinstance(input, str):
                input = {"role": "user", "content": input}
            self.memory.add_message(input)


    def query(self, input: str | dict | list[dict]) -> str:
        """
        Query large language model
        :param input:
        :return:
        """
        self._add_to_memory(input)
        return self.proceed()


    def stream(self, input: str | dict | list[dict]) -> AsyncGenerator[Any, None]: # -> ContentStream:
        """
        streaming method
        :input prompt, message, list of messages:
        :return:
        """
        self._add_to_memory(input)
        return self.proceed_stream()


    def proceed_stream(self) -> AsyncGenerator[Any, None]: # -> ContentStream:
        return self.streaming.resp_async_generator(self.memory, self.llm_options, self.available_tools)


    def proceed(self) -> str:
        response: ModelResponse = self.rotate_completion(stream=False)
        self._process_response(response)
        response = self._process_function_calls(response)
        answer = self.message_from_response(response)
        self.memory.add_message(answer)
        result: str = self.memory.last_message["content"] if "content" in self.memory.last_message else str(self.memory.last_message)
        return result


    def _process_function_calls(self, response: ModelResponse) -> Optional[ModelResponse]:
        """
        processes function calls in the response
        :param response_message:
        :return:
        """
        proceed = True
        while proceed:
            proceed = False
            response_message = response.choices[0].message
            tool_calls = response_message.get("tool_calls")

            if tool_calls and (self.tools is not None):
                proceed = True
                message = self.message_from_response(response)
                self.memory.add_message(message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.available_tools[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    try:
                        function_response = str(function_to_call(**function_args))
                    except Exception as e:
                        function_response = str(e)
                    result = {"role":"tool", "content":function_response, "name":function_name, "tool_call_id":tool_call.id}
                    self.memory.add_message(result)
                response = self.rotate_completion(stream=False)
                self._process_response(response)
        return response

    def _prepare_tools(self, functions: list[Any]):
        """
        Prepares functions as tools that LLM can call.
        Note, the functions should have comments explaining LLM how to use them
        :param functions:
        :return:
        """
        tools = []
        self.available_tools = {}
        for fun in functions:
            function_description = litellm.utils.function_to_dict(fun)
            self.available_tools[function_description["name"]] = fun
            tools.append({"type": "function", "function": function_description})
        self.llm_options["tools"] = tools
        self.llm_options["tool_choice"] = "auto"