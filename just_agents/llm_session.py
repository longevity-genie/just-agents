import copy
import json
from pathlib import Path
from typing import Any, AsyncGenerator

import litellm
from litellm import ModelResponse, completion
from litellm.utils import Choices

from just_agents.llm_options import LLAMA3
from just_agents.memory import *
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming
from just_agents.streaming.openai_streaming import AsyncSession
from just_agents.streaming.qwen_streaming import QwenStreaming

OnCompletion = Callable[[ModelResponse], None]
GetKey = Callable[[], str] #useful for key rotation





@dataclass(kw_only=True)
class LLMSession:
    llm_options: Dict[str, Any] = field(default_factory=lambda: LLAMA3)
    tools: List[Callable] = field(default_factory=list)
    available_tools: Dict[str, Callable] = field(default_factory=lambda: {})

    on_response: list[OnCompletion] = field(default_factory=list)
    memory: Memory = field(default_factory=lambda: Memory())
    streaming: AbstractStreaming = None

    def __post_init__(self):
        if self.llm_options is not None:
            self.llm_options = copy.deepcopy(self.llm_options) #just a satefy requirement to avoid shared dictionaries
        if "qwen" in self.llm_options["model"].lower():
            self.streaming = QwenStreaming()
        else:
            self.streaming = AsyncSession()

        if self.tools is not None:
            self._prepare_tools(self.tools)

    def _process_response(self, response: ModelResponse):
        """
        Running handlers to process model responses
        :param response:
        :return:
        """
        for handler in self.on_response:
            handler(response)

    @staticmethod
    def message_from_response(response: ModelResponse):
        choice: Choices = response.choices[0]
        message: Message = choice.message
        return message


    def instruct(self, prompt: str):
        """
        Ads system instructions
        :param prompt:
        :return:
        """
        system_instruction =  Message(content = prompt, role = "system")
        self.memory.add_message(system_instruction, True)
        return system_instruction

    def query(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None, key_getter: Optional[GetKey] = None) -> str:
        """
        Query large language model
        :param prompt:
        :param run_callbacks:
        :param output:
        :return:
        """

        question = Message(role="user", content=prompt)
        self.memory.add_message(question, run_callbacks)
        return self._query(run_callbacks, output, key_getter=key_getter)


    def query_all_messages(self, messages: list[dict], run_callbacks: bool = True, output: Optional[Path] = None, key_getter: Optional[GetKey] = None) -> str:
        self.memory.add_messages(messages, run_callbacks)
        return self._query(run_callbacks, output, key_getter=key_getter)


    def stream_all(self, messages: list, run_callbacks: bool = True): # -> ContentStream:
        #TODO this function is super-dangerous as it does not seem to clean memory!
        #TODO: should we add memory cleanup?
        self.memory.add_messages(messages, run_callbacks)
        return self._stream()

    async def stream_async(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None, key_getter: Callable[[], str] = None) -> List[Any]:
        """temporary function that allows testing the stream function which Alex wrote but I do not fully understand"""
        collected_data = []
        async for item in self.stream(prompt, run_callbacks, output, key_getter=key_getter):
            collected_data.append(item)
            # You can also process each item here if needed
        return collected_data


    def stream(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None, key_getter: Callable[[], str] = None) -> AsyncGenerator[Any, None]: # -> ContentStream:
        """
        streaming method
        :param prompt:
        :param run_callbacks:
        :param output:
        :return:
        """
        question = Message(role="user", content=prompt)
        self.memory.add_message(question, run_callbacks)

        # Start the streaming process
        content_stream = self._stream(key_getter=key_getter)

        # If output file is provided, write the stream to the file
        if output is not None:
            try:
                with output.open('w') as file:
                    if True: #if isinstance(content_stream, ContentStream):
                        #looks like ContentStream is only used for typehinting
                        # while it brings pretty heavy starlette dependency
                        # let's temporally comment it out
                        for content in content_stream:
                            file.write(content)
                    else:
                        raise TypeError("ContentStream expected from self._stream()")
            except Exception as e:
                print(f"Error writing to file: {e}")

        return content_stream



    def _stream(self, key_getter: Optional[GetKey] = None) -> AsyncGenerator[Any, None]: # -> ContentStream:
        return self.streaming.resp_async_generator(self.memory, self.llm_options, self.available_tools, key_getter=key_getter )


    def _query(self, run_callbacks: bool = True, output: Optional[Path] = None, key_getter: Optional[GetKey] = None) -> str:
        api_key = key_getter() if key_getter is not None else None
        response: ModelResponse = completion(messages=self.memory.messages, stream=False, api_key=api_key, **self.llm_options)
        self._process_response(response)
        executed_response = self._process_function_calls(response)
        if executed_response is not None:
            response = executed_response
            self._process_response(response)
        answer = self.message_from_response(response)
        self.memory.add_message(answer, run_callbacks)
        result: str = self.memory.last_message.content if self.memory.last_message is not None and self.memory.last_message.content is not None else str(
            self.memory.last_message)
        if output is not None:
            if not output.parent.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result)
        return result


    def _process_function_calls(self, response: ModelResponse) -> Optional[ModelResponse]:
        """
        processes function calls in the response
        :param response_message:
        :return:
        """
        response_message = response.choices[0].message
        tool_calls = response_message.get("tool_calls")

        if tool_calls and (self.tools is not None):
            message = self.message_from_response(response)
            self.memory.add_message(message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                try:
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = str(e)
                result = Message(role="tool", content=function_response, name=function_name, tool_call_id=tool_call.id)
                self.memory.add_message(result)
            return completion(messages=self.memory.messages, stream=False, **self.llm_options)
        return None

    def _prepare_tools(self, functions: List[Any]):
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