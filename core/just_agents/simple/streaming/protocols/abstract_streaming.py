import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Callable, AsyncGenerator
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol


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
    output_streaming: IAbstractStreamingProtocol

    def __init__(self, llm_session):
        self.session = llm_session

    @abstractmethod
    async def resp_async_generator(self) -> AsyncGenerator:
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