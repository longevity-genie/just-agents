from litellm import ModelResponse, completion
from typing import Any, Dict, List
import litellm
import json

class LLMSession:

    def __init__(self, llm_options:Dict[str, Any],  system:str = "", functions:List[callable] = None):
        if llm_options is None:
            raise Exception("llm_options should not be None")
        self.llm_options:Dict[str, Any] = llm_options.copy()
        self.system:str = system
        self.messages:List[Dict[str, str]] = [{"role": "system", "content": system}]
        if functions is None:
            self.available_functions = None
        else:
            self._prepare_tools(functions)


    def query(self, prompt:str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response:ModelResponse = completion(messages=self.messages, stream=False, **self.llm_options)
        response = self._process_function_calls(response) or response
        content:str = response.choices[0].message.get("content")
        self.messages.append({"role": "assistant", "content": content})

        return content


    def _process_function_calls(self, response:ModelResponse) -> ModelResponse:
        response_message = response.choices[0].message
        tool_calls = response_message.get("tool_calls")

        if tool_calls and (self.available_functions is not None):
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
        tools = []
        available_functions = {}
        for fun in functions:
            function_description = litellm.utils.function_to_dict(fun)
            available_functions[function_description["name"]] = fun
            tools.append({"type": "function", "function": function_description})
        self.available_functions = available_functions
        self.llm_options["tools"] = tools
        self.llm_options["tool_choice"] = "auto"


