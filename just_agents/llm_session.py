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
from just_agents.utils import rotate_completion

OnCompletion = Callable[[ModelResponse], None]


@dataclass(kw_only=True)
class LLMSession:
    llm_options: dict[str, Any] = field(default_factory=lambda: LLAMA3)
    tools: list[Callable] = field(default_factory=list)
    available_tools: dict[str, Callable] = field(default_factory=lambda: {})

    on_response: list[OnCompletion] = field(default_factory=list)
    memory: Memory = field(default_factory=lambda: Memory())
    streaming: AbstractStreaming = None

    def __post_init__(self):
        if self.llm_options is not None:
            self.llm_options = copy.deepcopy(self.llm_options) #just a satefy requirement to avoid shared dictionaries
            if (self.llm_options.get("key_getter") is not None) and (self.llm_options.get("api_key") is not None):
                print("Warning api_key will be rewriten by key_getter. Both are present in llm_options.")

        if "qwen2" in self.llm_options["model"].lower():
            from just_agents.streaming.qwen2_streaming import Qwen2AsyncSession
            self.streaming = Qwen2AsyncSession()
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

    def query(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        """
        Query large language model
        :param prompt:
        :param run_callbacks:
        :param output:
        :return:
        """

        question = {"role": "user", "content": prompt}
        self.memory.add_message(question, run_callbacks)
        return self._query(run_callbacks, output)


    def query_add_all(self, messages: list[dict], run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        self.memory.add_messages(messages, run_callbacks)
        return self._query(run_callbacks, output)


    def stream_all(self, messages: list, run_callbacks: bool = True): # -> ContentStream:
        self.memory.add_messages(messages, run_callbacks)
        return self._stream()

    async def stream_async(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None) -> list[Any]:
        """temporary function that allows testing the stream function which Alex wrote but I do not fully understand"""
        collected_data = []
        async for item in self.stream(prompt, run_callbacks, output):
            collected_data.append(item)
            # You can also process each item here if needed
        return collected_data


    def stream(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None) -> AsyncGenerator[Any, None]: # -> ContentStream:
        """
        streaming method
        :param prompt:
        :param run_callbacks:
        :param output:
        :return:
        """
        question = {"role":"user", "content":prompt}
        self.memory.add_message(question, run_callbacks)

        # Start the streaming process
        content_stream = self._stream()

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



    def _stream(self) -> AsyncGenerator[Any, None]: # -> ContentStream:
        return self.streaming.resp_async_generator(self.memory, self.llm_options, self.available_tools)


    def _query(self, run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        response: ModelResponse = rotate_completion(messages=self.memory.messages, stream=False, options=self.llm_options)
        self._process_response(response)
        response = self._process_function_calls(response)
        answer = self.message_from_response(response)
        self.memory.add_message(answer, run_callbacks)
        result: str = self.memory.last_message["content"] if "content" in self.memory.last_message else str(self.memory.last_message)
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
                        function_response = function_to_call(**function_args)
                    except Exception as e:
                        function_response = str(e)
                    result = {"role":"tool", "content":function_response, "name":function_name, "tool_call_id":tool_call.id}
                    self.memory.add_message(result)
                response = rotate_completion(messages=self.memory.messages, stream=False, options=self.llm_options)
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