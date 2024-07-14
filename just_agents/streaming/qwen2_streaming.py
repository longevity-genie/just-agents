from typing import AsyncGenerator
from litellm import ModelResponse, completion
from typing import Callable, Optional
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser
from just_agents.utils import rotate_completion
import json


class Qwen2AsyncSession(AbstractStreaming):

    def _process_function(self, parser: FunctionParser, available_tools: dict[str, Callable]):
        function_args = json.loads(parser.arguments)
        function_to_call = available_tools[parser.name]
        try:
            function_response = function_to_call(**function_args)
        except Exception as e:
            function_response = str(e)
        message = {"role":"function", "content":function_response, "name":parser.name,
                         "tool_call_id":parser.id}  # TODO need to track arguments , arguments=function_args
        return message

    async def resp_async_generator(self, memory: Memory,
                                   options: dict,
                                   available_tools: dict[str, Callable]
                                   ) -> AsyncGenerator[str, None]:
        response: ModelResponse = rotate_completion(messages=memory.messages, stream=True, options=options)
        parser: Optional[FunctionParser] = None
        tool_messages: list[dict] = []
        parsers: list[FunctionParser] = []
        deltas: list[str] = []
        for i, part in enumerate(response):
            delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
            if delta:
                deltas.append(delta)
                yield f"data: {self._get_chunk(i, delta, options)}\n\n"

            tool_calls = part["choices"][0]["delta"].get("tool_calls")
            if tool_calls and (available_tools is not None):
                if not parser:
                    parser = FunctionParser(id = tool_calls[0].id)
                if parser.parsed(tool_calls[0].function.name, tool_calls[0].function.arguments):
                    tool_messages.append(self._process_function(parser, available_tools))
                    parsers.append(parser)
                    parser = None #maybe Optional?

        if len(tool_messages) > 0:
            memory.add_message(self._get_tool_call_message(parsers))
            for message in tool_messages:
                memory.add_message(message)
            response = rotate_completion(messages=memory.messages, stream=True, options=options)
            deltas = []
            for i, part in enumerate(response):
                delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
                if delta:
                    deltas.append(delta)
                    yield f"data: {self._get_chunk(i, delta, options)}\n\n"
            memory.add_message({"role":"assistant", "content":"".join(deltas)})
        elif len(deltas) > 0:
            memory.add_message({"role":"assistant", "content":"".join(deltas)})

        yield "data: [DONE]\n\n"