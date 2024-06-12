from typing import AsyncGenerator

from litellm import ModelResponse, completion

from just_agents.memory import *
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser


class AsyncSession(AbstractStreaming):

    async def resp_async_generator(self, memory: Memory,
                                   options: Dict,
                                   available_tools: Dict[str, Callable],
                                   key_getter: Callable[[], str] = None
                                   ) -> AsyncGenerator[str, None]:
        api_key = key_getter() if key_getter is not None else None
        response: ModelResponse = completion(messages=memory.messages, stream=True, api_key=api_key, **options)
        parser: Optional[FunctionParser] = None
        tool_messages: list[Message] = []
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