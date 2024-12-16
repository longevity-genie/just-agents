import copy
import json
from pathlib import Path
from typing import Any, AsyncGenerator

import litellm
from litellm import ModelResponse, completion
from litellm.utils import Choices
from just_agents.interfaces.agent import IAgent
from just_agents.simple.memory import Memory
from typing import Callable, Optional
from just_agents.simple.streaming import AbstractStreaming
from just_agents.simple.streaming.openai_streaming import AsyncSession
from just_agents.simple.utils import resolve_and_validate_agent_schema, resolve_llm_options, resolve_system_prompt, resolve_tools
from just_agents.rotate_keys import RotateKeys

OnCompletion = Callable[[ModelResponse], None]

# schema parameters:
KEY_LIST_PATH = "key_list_path"
STREAMING_METHOD = "streaming_method"
BACKUP_OPTIONS = "backup_options"
COMPLETION_REMOVE_KEY_ON_ERROR = "completion_remove_key_on_error"
COMPLETION_MAX_TRIES = "completion_max_tries"

#streaming methods:
OPENAI = "openai"
QWEN2 = "qwen2"
CHAIN_OF_THOUGHT = "chain_of_thought"


class LLMSession(
    IAgent[
        str | dict | list[dict],
        str,
        AsyncGenerator[Any, None]
    ]
):

    def __init__(self, llm_options: Optional[dict[str, Any]] = None,
                 system_prompt: Optional[str] = None,
                 agent_schema: Optional[str | Path | dict[str, Any]] = None,
                 tools: Optional[list[Callable]] = None):
        self.on_response = []
        self.available_tools: Optional[dict[str, Callable]] = {}
        self.memory: Memory = Memory()
        self.streaming: Optional[AbstractStreaming] = None
        self.key_getter: Optional[RotateKeys] = None
        self.on_response: list[OnCompletion] = []

        self.agent_schema = resolve_and_validate_agent_schema(agent_schema, "llm_session_schema.yaml")
        self.llm_options: dict[str, Any] = resolve_llm_options(self.agent_schema, llm_options)
        if self.agent_schema.get(KEY_LIST_PATH, None) is not None:
            self.key_getter = RotateKeys(self.agent_schema[KEY_LIST_PATH])
        self.tools: list[Callable] = tools
        if self.tools is None:
            self.tools = resolve_tools(self.agent_schema)

        if self.llm_options is not None:
            self.llm_options = copy.deepcopy(self.llm_options) #just a satefy requirement to avoid shared dictionaries
            if (self.key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
                print("Warning api_key will be rewriten by key_getter. Both are present in llm_options.")

        if system_prompt is None:
            system_prompt = resolve_system_prompt(self.agent_schema)
        if system_prompt is not None:
            self.instruct(system_prompt)

        streaming_method = self.agent_schema.get(STREAMING_METHOD, None)
        if streaming_method is None or streaming_method == OPENAI:
            self.streaming = AsyncSession(self)
        elif streaming_method.lower() == QWEN2:
            from just_agents.simple.streaming import Qwen2AsyncSession
            self.streaming = Qwen2AsyncSession(self)
        elif streaming_method.lower() == CHAIN_OF_THOUGHT:
            from just_agents.simple.streaming.chain_of_thought import ChainOfThought
            self.streaming = ChainOfThought(self)
        else:
            raise ValueError("just_streaming_method is incorrect. "
                             "It should be one of this ['qwen2', 'chain_of_thought']")

        if self.tools is not None:
            self._prepare_tools(self.tools)


    def add_on_response_listener(self, listener:OnCompletion):
        self.on_response.append(listener)


    def remove_on_response_listener(self, listener:OnCompletion):
        if listener in self.on_response:
            self.on_response.remove(listener)


    def _rotate_completion(self, stream: bool) -> ModelResponse:
        """
        Uses key rotation to call LLMs in case of errors
        """
        opt = self.llm_options.copy()
        max_tries = self.agent_schema.get(COMPLETION_MAX_TRIES, 2)
        if self.key_getter is not None:
            if max_tries < 1:
                max_tries = self.key_getter.len()
            else:
                if self.agent_schema.get(COMPLETION_REMOVE_KEY_ON_ERROR, True):
                    max_tries = min(max_tries, self.key_getter.len())
            last_exception = None
            for _ in range(max_tries):
                opt["api_key"] = self.key_getter()
                try:
                    response = completion(messages=self.memory.messages, stream=stream, **opt)
                    return response
                except Exception as e:
                    last_exception = e
                    if self.agent_schema.get(COMPLETION_REMOVE_KEY_ON_ERROR, True):
                        self.key_getter.remove(opt["api_key"])

            backup_opt: dict = self.agent_schema.get(BACKUP_OPTIONS, None)
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
        message: dict[str, object] = choice.message.to_dict(exclude_none=True, exclude_unset=True) #converting to dictionary and deleting nonw
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
        self.memory.add_message(system_instruction)
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
        return self.streaming.resp_async_generator()


    def proceed(self) -> str:
        response: ModelResponse = self._rotate_completion(stream=False)
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
                response = self._rotate_completion(stream=False)
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