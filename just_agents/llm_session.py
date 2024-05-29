import pprint
from dataclasses import field, dataclass

from litellm import ModelResponse, completion
from typing import Any, Dict, List, Optional, Callable
import litellm
import json
from loguru import logger

import pathlib

from just_agents.llm_options import LLAMA3

OnMessageCallable = Callable[[Dict[str, Any]], None]
OnCompletion = Callable[[ModelResponse], None]

@dataclass(kw_only=True)
class LLMSession:
    llm_options: Dict[str, Any] = field(default_factory=lambda: LLAMA3)
    functions: List[Callable] = None
    available_functions: Dict[str, Callable] = field(default_factory=lambda: {})
    messages: List[Dict[str, str]] = field(default_factory=list)
    on_message: list[OnMessageCallable] = field(default_factory=list)
    on_response: list[OnCompletion] = field(default_factory=list)

    def __post_init__(self):
        if self.functions is not None:
            self._prepare_tools(self.functions)

    def add_on_message(self, handler: OnMessageCallable):
        self.on_message.append(handler)

    def remove_on_message(self, handler: OnMessageCallable):
        self.on_message = [m for m in self.on_message if m == handler]


    def _post_init(self) -> None:
        """
        post init method of the model
        :return:
        """
        if self.functions is not None:
            self._prepare_tools(self.functions)

    def _process_response(self, response: ModelResponse):
        """
        Running handlers to process model responses
        :param response:
        :return:
        """
        for handler in self.on_response:
            handler(response)


    def add_message(self, message: Dict[str, str], run_callbacks: bool = True):
        self.messages.append(message)
        if run_callbacks:
            for handler in self.on_message:
                handler(message)

    def add_message_from_response(self, response: ModelResponse, run_callbacks: bool = True):
        """
        extract messages from response and logs them
        :param response:
        :param run_callbacks:
        :return:
        """
        content: str = response.choices[0].message.get("content")
        answer = {"role": "assistant", "content": content}
        self.add_message(answer, run_callbacks)
        return content


    def instruct(self, prompt: str):
        """
        Ads system instructions
        :param prompt:
        :return:
        """
        system_instruction = {"role": "system", "content": prompt}
        self.add_message(system_instruction, True)
        return system_instruction

    def query(self, prompt: str, stream: bool = False, run_callbacks: bool = True) -> str:
        """
        Query large language model
        :param prompt:
        :param stream:
        :param run_callbacks:
        :return:
        """
        question: Dict = {"role": "user", "content": prompt}
        self.add_message(question)
        options: Dict = self.llm_options
        response: ModelResponse = completion(messages=self.messages, stream=stream, **options)
        self._process_response(response)
        executed_response = self._process_function_calls(response)
        if executed_response is not None:
            response = executed_response
            self._process_response(response)
        return self.add_message_from_response(response, run_callbacks)

    def _process_function_calls(self, response: ModelResponse) -> Optional[ModelResponse]:
        """
        processes function calls in the response
        :param response:
        :return:
        """
        response_message = response.choices[0].message
        tool_calls = response_message.get("tool_calls")

        if tool_calls and (self.functions is not None):
            self.messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                try:
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = str(e)

                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            return completion(messages=self.messages, stream=False, **self.llm_options)
        return None

    def _prepare_tools(self, functions: List[Any]):
        """
        Prepares functions as tools that LLM can call.
        Note, the functions should have comments explaining LLM how to use them
        :param functions:
        :return:
        """
        tools = []
        self.available_functions = {}
        for fun in functions:
            function_description = litellm.utils.function_to_dict(fun)
            self.available_functions[function_description["name"]] = fun
            tools.append({"type": "function", "function": function_description})
        self.llm_options["tools"] = tools
        self.llm_options["tool_choice"] = "auto"
