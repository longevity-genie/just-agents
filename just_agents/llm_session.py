from pathlib import Path

from litellm.utils import ChatCompletionMessageToolCall, Function
from litellm import ModelResponse, completion, acompletion, Message
from typing import Any, Dict, List, Optional, Callable
import litellm
import json
from just_agents.memory import *
from litellm.utils import Choices

from just_agents.llm_options import LLAMA3
from just_agents.memory import Memory
from starlette.responses import ContentStream
import time

OnCompletion = Callable[[ModelResponse], None]

class FunctionParser:
    id:str = ""
    name:str = ""
    arguments:str = ""

    def __init__(self, id:str):
        self.id = id

    def parsed(self, name:str, arguments:str):
        if name:
            self.name += name
        if arguments:
            self.arguments += arguments
        if len(self.name) > 0 and len(self.arguments) > 0 and self.arguments.endswith("}"):
            return True
        return False


def get_chunk(i:int, delta:str, options: Dict):
    chunk = {
        "id": i,
        "object": "chat.completion.chunk",
        "created": time.time(),
        "model": options["model"],
        "choices": [{"delta": {"content": delta}}],
    }
    return json.dumps(chunk)


def process_function(parser:FunctionParser, available_tools: Dict[str, Callable]):
    function_args = json.loads(parser.arguments)
    function_to_call = available_tools[parser.name]
    try:
        function_response = function_to_call(**function_args)
    except Exception as e:
        function_response = str(e)
    message = Message(role="tool", content=function_response, name=parser.name,
                     tool_call_id=parser.id)  # TODO need to track arguemnts , arguments=function_args
    return message


def get_tool_call_message(parsers:list[FunctionParser]) -> Message:
    tool_calls = []
    for parser in parsers:
        tool_calls.append({"type":"function",
            "id":parser.id, "function":{"name":parser.name, "arguments":parser.arguments}})
    return Message(role="assistant", content=None, tool_calls=tool_calls)


async def _resp_async_generator(memory: Memory, options: Dict, available_tools: Dict[str, Callable]):
    response: ModelResponse = completion(messages=memory.messages, stream=True, **options)
    parser:FunctionParser = None
    function_response = None
    tool_calls_message = None
    tool_messages:list[Message] = []
    parsers:list[FunctionParser] = []
    deltas:list[str] = []
    for i, part in enumerate(response):
        delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
        if delta:
            deltas.append(delta)
            yield f"data: {get_chunk(i, delta, options)}\n\n"

        tool_calls = part["choices"][0]["delta"].get("tool_calls")
        if tool_calls and (available_tools is not None):
            if not parser:
                parser = FunctionParser(id = tool_calls[0].id)
            if parser.parsed(tool_calls[0].function.name, tool_calls[0].function.arguments):
                tool_messages.append(process_function(parser, available_tools))
                parsers.append(parser)
                parser = None

    if len(tool_messages) > 0:
        memory.add_message(get_tool_call_message(parsers))
        for message in tool_messages:
            memory.add_message(message)
        response = completion(messages=memory.messages, stream=True, **options)
        deltas = []
        for i, part in enumerate(response):
            delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
            if delta:
                deltas.append(delta)
                yield f"data: {get_chunk(i, delta, options)}\n\n"
        memory.add_message(Message(role="assistant", content="".join(deltas)))
    elif len(deltas) > 0:
        memory.add_message(Message(role="assistant", content="".join(deltas)))

    yield "data: [DONE]\n\n"


@dataclass(kw_only=True)
class LLMSession:
    llm_options: Dict[str, Any] = field(default_factory=lambda: LLAMA3)
    tools: List[Callable] = field(default_factory=list)
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

    def query(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        """
        Query large language model
        :param prompt:
        :param run_callbacks:
        :param output:
        :return:
        """

        question = Message(role="user", content=prompt)
        self.memory.add_message(question, run_callbacks)
        return self._query(run_callbacks, output)


    def query_all(self, messages: list, run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        self.memory.add_messages(messages, run_callbacks)
        return self._query(run_callbacks, output)


    def stream_all(self, messages: list, run_callbacks: bool = True) -> ContentStream:
        self.memory.add_messages(messages, run_callbacks)
        return self._stream()


    def stream(self, prompt: str, run_callbacks: bool = True, output: Optional[Path] = None) -> ContentStream:
        question = Message(role="user", content=prompt)
        self.memory.add_message(question, run_callbacks)
        return self._stream()


    def _stream(self) -> ContentStream:
        return _resp_async_generator(self.memory, self.llm_options, self.available_tools)


    def _query(self, run_callbacks: bool = True, output: Optional[Path] = None) -> str:
        options: Dict = self.llm_options
        response: ModelResponse = completion(messages=self.memory.messages, stream=False, **options)
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
