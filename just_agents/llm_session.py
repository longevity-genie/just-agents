from pathlib import Path

from litellm import ModelResponse, completion, Message
from typing import Any, Dict, List, Optional, Callable
import litellm
import json
from just_agents.memory import *
from litellm.utils import Choices

from just_agents.llm_options import LLAMA3
from just_agents.memory import Memory

OnCompletion = Callable[[ModelResponse], None]

@dataclass(kw_only=True)
class LLMSession:
    llm_options: Dict[str, Any] = field(default_factory=lambda: LLAMA3)
    tools: List[Callable] = None
    available_tools: Dict[str, Callable] = field(default_factory=lambda: {})

    on_response: list[OnCompletion] = field(default_factory=list)
    memory: Memory = field(default_factory=lambda: Memory())

    def __post_init__(self):
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

    def query(self, prompt: str, stream: bool = False, run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        """
        Query large language model
        :param prompt:
        :param stream:
        :param run_callbacks:
        :return:
        """
        question = Message(role="user", content=prompt)
        self.memory.add_message(question)
        options: Dict = self.llm_options
        response: ModelResponse = completion(messages=self.memory.messages, stream=stream, **options)
        self._process_response(response)
        executed_response = self._process_function_calls(response)
        if executed_response is not None:
            response = executed_response
            self._process_response(response)
        answer = self.message_from_response(response)
        self.memory.add_message(answer, run_callbacks)
        result = self.memory.last_message.content if self.memory.last_message is not None and self.memory.last_message.content is not None else str(self.memory.last_message)
        if output is not None:
            output.write_text(result)
        return result


    def _process_function_calls(self, response: ModelResponse) -> Optional[ModelResponse]:
        """
        processes function calls in the response
        :param response:
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
                result = Message(role="tool", content=function_response, name=function_name, tool_call_id=tool_call.id) #TODO need to track arguemnts , arguments=function_args
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
