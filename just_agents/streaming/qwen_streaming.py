import json
from dataclasses import dataclass
from typing import Dict, Callable, Optional, AsyncGenerator
from enum import Enum, auto

from litellm import ModelResponse, completion, Message
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser

class ParserState(Enum):
    """
    States for the Qwen parser
    """
    CLEARED = auto()
    STOPPED = auto()
    WAITING = auto()
    UNDETERMINED = auto()
    PARSING = auto()

@dataclass
class QwenFunctionParser:
    """
    Qwen has differences in formats of how it streams stuff
    """
    name: str = ""
    arguments: str = ""
    buffer: str = ""
    state: ParserState = ParserState.WAITING

    def parsing(self, token: str) -> bool:
        if self.state in {ParserState.STOPPED, ParserState.CLEARED}:
            return False

        self.buffer += token

        if self.state == ParserState.PARSING:
            return True

        if self.state == ParserState.WAITING and token.startswith("{"):
            self.state = ParserState.UNDETERMINED
            return True

        if self.state == ParserState.UNDETERMINED and len(self.buffer) < 12:
            return True

        if self.state == ParserState.UNDETERMINED and len(self.buffer) >= 12:
            if "function" in self.buffer:
                self.state = ParserState.PARSING
                return True
            else:
                self.state = ParserState.STOPPED
                return False

    def need_cleared(self) -> bool:
        return self.state == ParserState.STOPPED

    def clear(self) -> str:
        self.state = ParserState.CLEARED
        return self.buffer

    def is_ready(self):
        return self.state == ParserState.PARSING

    def get_function_parsers(self):
        if self.state == ParserState.PARSING:
            res = []
            data = self.buffer.replace("\n", "")
            data = "[" + data.replace("}}", "}},")[:-1] + "]"
            functions = json.loads(data)
            for func in functions:
                res.append(FunctionParser(name=func["function"], arguments=json.dumps(func["parameters"])))
            return res
        return []

@dataclass
class QwenStreaming(AbstractStreaming):

    def _process_function(self, parser: FunctionParser, available_tools: Dict[str, Callable]):
        function_args = json.loads(parser.arguments)
        function_to_call = available_tools[parser.name]
        try:
            function_response = function_to_call(**function_args)
        except Exception as e:
            function_response = str(e)
        message = Message(role="function", content=function_response, name=parser.name,
                          tool_call_id=parser.id)  # TODO need to track arguments , arguments=function_args
        return message

    async def resp_async_generator(self, memory: Memory, options: Dict, available_tools: Dict[str, Callable], key_getter: Callable[[], str] = None) -> AsyncGenerator[str, None]:
        """
        parses and streams results of the function
        :param memory:
        :param options:
        :param available_tools:
        :return:
        """
        api_key = key_getter() if key_getter is not None else None
        if api_key is None:
            api_key = options.pop("api_key", None)

        response: ModelResponse = completion(messages=memory.messages, stream=True, api_key=api_key, **options)
        parser: QwenFunctionParser = QwenFunctionParser()
        deltas: list[str] = []
        tool_messages: list[Message] = []

        for i, part in enumerate(response):
            delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
            if delta:
                if parser.parsing(delta):
                    continue
                if parser.need_cleared():
                    delta = parser.clear()
                deltas.append(delta)
                yield f"data: {self._get_chunk(i, delta, options)}\n\n"

        parsers = parser.get_function_parsers()
        for pars in parsers:
            tool_messages.append(self._process_function(pars, available_tools))

        if len(tool_messages) > 0:
            memory.add_message(self._get_tool_call_message(parsers))
            for message in tool_messages:
                memory.add_message(message)
            response = completion(messages=memory.messages, stream=True, **options)
            deltas = []
            for i, part in enumerate(response):
                delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
                if delta:
                    deltas.append(delta)
                    yield f"data: {self._get_chunk(i, delta, options)}\n\n"
            memory.add_message(Message(role="assistant", content="".join(deltas)))
        elif len(deltas) > 0:
            memory.add_message(Message(role="assistant", content="".join(deltas)))

        yield "data: [DONE]\n\n"