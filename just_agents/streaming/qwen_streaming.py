import json
from dataclasses import dataclass
from typing import Dict, Callable, Optional

from litellm import ModelResponse, completion, Message

from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser

CLEARED:int = -2
STOPED:int = -1
WAITING:int = 0
UNDETERMIND:int = 1
PARSING:int = 2

@dataclass
class QwenFunctionParser:
    name:str = ""
    arguments:str = ""
    buffer:str = ""
    state:int = WAITING

    def parsing(self, token:str) -> bool:
        if self.state == STOPED or self.state == CLEARED:
            return False

        self.buffer += token

        if self.state == PARSING:
            return True

        if self.state == WAITING and token.startswith("{"):
            self.state = UNDETERMIND
            return True

        if self.state == UNDETERMIND and len(self.buffer) < 12:
            return True

        if self.state == UNDETERMIND and len(self.buffer) >= 12:
            if str(self.buffer).find("function") != -1:
                self.state = PARSING
                return True
            else:
                self.state = STOPED
                return False

    def need_cleared(self) -> bool:
        return self.state == STOPED


    def clear(self) -> str:
        self.state = CLEARED
        return self.buffer

    def is_ready(self):
        return self.state == PARSING

    def get_function_parsers(self):
        if self.state == PARSING:
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


    async def resp_async_generator(self, memory: Memory, options: Dict, available_tools: Dict[str, Callable]):
        response: ModelResponse = completion(messages=memory.messages, stream=True, **options)
        parser: QwenFunctionParser = QwenFunctionParser()
        deltas: list[str] = []
        tool_messages: list[Message] = []
        parsers: list[FunctionParser] = []

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
