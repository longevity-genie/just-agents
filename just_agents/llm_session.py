from litellm import ModelResponse, completion
from typing import Any, Dict, List, Optional, Callable
import litellm
import json
from loguru import logger
from pydantic import BaseModel, Field

from just_agents.llm_options import LLMOptions
import pathlib

OnMessageCallable = Callable[[Dict[str, Any]], None]

class LLMSession(BaseModel):

    llm_options: Dict[str, Any] | LLMOptions
    functions: List[Callable] = None
    available_functions: Dict[str, Callable] = {}
    messages: List[Dict[str, str]] = []
    on_message: Optional[OnMessageCallable] = Field(description="Callback called on any message the agent resieves", default=lambda dic: None)

    def __post_init__(self):
       if self.functions is not None:
            self._prepare_tools(self.functions)

    def instruct(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})


    def query(self, prompt:str, stream: bool = False) -> str:
        question: Dict = {"role": "user", "content": prompt}
        self.messages.append(question)
        self.on_message(question)

        options: Dict = self.llm_options.dict() if isinstance(self.llm_options, BaseModel) else self.llm_options
        response: ModelResponse = completion(messages=self.messages, stream=stream, **options)
        response = self._process_function_calls(response) or response
        content: str = response.choices[0].message.get("content")
        answer = {"role": "assistant", "content": content}
        self.messages.append(answer)
        self.on_message(answer)
        return content


    def _process_function_calls(self, response:ModelResponse) -> Optional[ModelResponse]:
        response_message = response.choices[0].message
        tool_calls = response_message.get("tool_calls")

        if tool_calls and (self.functions is not None):
            self.messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.functions[function_name]
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


