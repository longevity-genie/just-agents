
import json
from litellm import ModelResponse, completion
from litellm.utils import Choices

from pydantic import Field, PrivateAttr
from typing import Callable, Optional, Dict, List, Sequence, Union, Any, AsyncGenerator, Coroutine

from just_agents.interfaces.IAgent import IAgent
from just_agents.streaming.abstract_streaming import AbstractStreaming
from just_agents.streaming.openai_streaming import AsyncSession

from just_agents.memory import Memory
from just_agents.just_profile import JustAgentProfile
from just_agents.just_tool import JustTool
from just_agents.rotate_keys import RotateKeys
from just_agents.types import StreamingMode, OnCompletion

class BaseAgent(IAgent, JustAgentProfile):

    llm_options: Dict[str, Any] = Field(...,
        description="options that will be passed to the LLM, see https://platform.openai.com/docs/api-reference/completions/create for more details")
    available_tools: Optional[dict[str, Callable]] = Field(None, deprecated=True, exclude=True,
        description="dictionary of tools that LLM can call. ")
    completion_remove_key_on_error: bool = Field(True,
        description="In case of using list of keys removing key from the list after error call with this key")
    completion_max_tries: Optional[int]  = Field(2, ge=0,
        description="maximum number of completion retries before giving up")
    streaming_method: StreamingMode = Field(StreamingMode.openai,
        description="protocol to handle llm format for function calling")
    backup_options: Optional[Dict] = Field(None,
        description="options that will be used after we give up with main options, one more completion call will be done with backup options")
    key_list_path: Optional[str] = Field(None,
        description="path to text file with list of api keys, one key per line")
    drop_params: bool = Field(True,
        description=" drop params from the request, useful for some models that do not support them")

    on_response : List[OnCompletion] = Field([])
    memory: Memory = Field(default_factory=Memory, exclude=True, repr=False)

    _streaming: Optional[AbstractStreaming] = PrivateAttr()
    _key_getter: Optional[RotateKeys] = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.instruct(self.system_prompt)
        if self.tools is not None: #TODO: eliminate
            self._prepare_tools()
        if self.key_list_path is not None:
            self._key_getter = RotateKeys(self.key_list_path)

        if (self._key_getter is not None) and (self.llm_options.get("api_key", None) is not None):
            print("Warning api_key will be rewriten by key_getter. Both are present in llm_options.")
        if self.streaming_method == StreamingMode.openai:
            self._streaming = AsyncSession(self)
        elif self.streaming_method == StreamingMode.qwen2:
            from just_agents.streaming.qwen2_streaming import Qwen2AsyncSession
            self._streaming = Qwen2AsyncSession(self)
        elif self.streaming_method == StreamingMode.chain_of_thought:
            from just_agents.streaming.chain_of_thought import ChainOfThought
            self._streaming = ChainOfThought(self)
        else:
            raise ValueError("just_streaming_method is incorrect. "
                             "It should be one of this ['openai', 'qwen2', 'chain_of_thought']")

    def add_on_response_listener(self, listener:OnCompletion):
        self.on_response.append(listener)


    def remove_on_response_listener(self, listener:OnCompletion):
        if listener in self.on_response:
            self.on_response.remove(listener)


    def _rotate_completion(self, stream: bool) -> ModelResponse:
        opt = self.llm_options.copy()
        max_tries = self.completion_max_tries
        if self._key_getter is not None:
            if max_tries < 1:
                max_tries = self._key_getter.len()
            else:
                if self.completion_remove_key_on_error:
                    max_tries = min(max_tries, self._key_getter.len())
            last_exception = None
            for _ in range(max_tries):
                opt["api_key"] = self._key_getter()
                try:
                    response = completion(messages=self.memory.messages, stream=stream, **opt)
                    return response
                except Exception as e:
                    last_exception = e
                    if self.completion_remove_key_on_error:
                        self._key_getter.remove(opt["api_key"])

            if self.backup_options:
                return completion(messages=self.memory.messages, stream=stream, **self.backup_options)
            if last_exception:
                raise last_exception
            else:
                raise Exception(
                    f"Run out of tries to execute completion. Check your keys! Keys {self._key_getter.len()} left.")
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


    def query(self, query_input: Union[str, Dict, Sequence[Dict]]) -> str:
        """
        Query large language model
        :param query_input:prompt, message, list of messages:
        :return:
        """
        self._add_to_memory(query_input)
        return self.proceed()


    def stream(self, query_input: Union[str, Dict, Sequence[Dict]]) -> AsyncGenerator[Any, None]: # -> ContentStream:
        """
        streaming method
        :param query_input: prompt, message, list of messages:
        :return:
        """
        self._add_to_memory(query_input)
        return self.proceed_stream()

    def proceed_stream(self) -> Union[AsyncGenerator[Any, None], Coroutine[Any, Any, AsyncGenerator]]: # -> ContentStream:
        return self._streaming.resp_async_generator()


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
        :param response:
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

    def _prepare_tools(self): #TODO: remove this 3x redundancy in tools listings
        """
        Prepares functions as tools that LLM can call.
        Note, the functions should have comments explaining LLM how to use them
        :return:
        """

        tools = []
        self.available_tools = {}
        for tool in self.tools:
            if not isinstance(tool, JustTool):
                raise TypeError(f"The tool {str(tool)} is not an instance of JustTool")
            self.available_tools[tool.function] = tool.get_callable()
            tools.append({"type": "function", "function": tool.litellm_description})
        self.llm_options["tools"] = tools
        self.llm_options["tool_choice"] = "auto"