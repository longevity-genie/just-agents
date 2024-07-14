import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Callable, AsyncGenerator, Optional

from just_agents.memory import Memory

@dataclass
class FunctionParser:
    id: str = ""
    name: str = ""
    arguments: str = ""

    def parsed(self, name: str, arguments: str):
        if name:
            self.name += name
        if arguments:
            self.arguments += arguments
        if len(self.name) > 0 and len(self.arguments) > 0 and self.arguments.strip().endswith("}"):
            return True
        return False


class AbstractStreaming(ABC):
    """
    Class that is required to implement the streaming logic
    """


    @abstractmethod
    async def resp_async_generator(
            self,
            memory: Memory,
            options: Dict,
            available_tools: Dict[str, Callable]
    ) -> AsyncGenerator:
        """
        Async generator that fills memory with streaming data
        :param memory:
        :param options:
        :param available_tools:
        :return:
        """
        pass

    def _process_function(self, parser: FunctionParser, available_tools: Dict[str, Callable]):
        function_args = json.loads(parser.arguments)
        function_to_call = available_tools[parser.name]
        try:
            function_response = function_to_call(**function_args)
        except Exception as e:
            function_response = str(e)
        message = {"role":"tool", "content":function_response, "name":parser.name,
                         "tool_call_id":parser.id}  # TODO need to track arguments , arguments=function_args
        return message

    def _get_tool_call_message(self, parsers: list[FunctionParser]) -> dict:
        tool_calls = []
        for parser in parsers:
            tool_calls.append({"type":"function",
                "id": parser.id, "function": {"name": parser.name, "arguments": parser.arguments}})
        return {"role":"assistant", "content":None, "tool_calls":tool_calls}

    def _get_chunk(self, i: int, delta: str, options: Dict):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": options["model"],
            "choices": [{"delta": {"content": delta}}],
        }
        return json.dumps(chunk)