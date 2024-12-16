from typing import AsyncGenerator

from litellm import ModelResponse
from typing import Optional

from just_agents.simple.streaming.protocols.abstract_streaming import AbstractStreaming, FunctionParser
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from just_agents.simple.streaming.protocols.openai_streaming import OpenaiStreamingProtocol


class AsyncSession(AbstractStreaming):

    def __init__(self, llm_session, output_streaming: IAbstractStreamingProtocol = OpenaiStreamingProtocol()):
        super().__init__(llm_session)
        self.output_streaming = output_streaming


    async def resp_async_generator(self) -> AsyncGenerator[str, None]:
        from just_agents.simple.llm_session import LLMSession
        llm_session: LLMSession = self.session
        proceed = True
        while proceed:
            proceed = False
            response: ModelResponse = llm_session._rotate_completion(stream=True)
            parser: Optional[FunctionParser] = None
            tool_messages: list[dict] = []
            parsers: list[FunctionParser] = []
            deltas: list[str] = []
            for i, part in enumerate(response):
                delta: str = part["choices"][0]["delta"].get("content")  # type: ignore
                if delta:
                    deltas.append(delta)
                    yield self.output_streaming.get_chunk(i, delta, llm_session.llm_options)

                tool_calls = part["choices"][0]["delta"].get("tool_calls")
                if tool_calls and (llm_session.available_tools is not None):
                    if not parser:
                        parser = FunctionParser(id = tool_calls[0].id)
                    if parser.parsed(tool_calls[0].function.name, tool_calls[0].function.arguments):
                        tool_messages.append(self._process_function(parser, llm_session.available_tools))
                        parsers.append(parser)
                        parser = None #maybe Optional?

            if len(tool_messages) > 0:
                proceed = True
                llm_session.memory.add_message(self._get_tool_call_message(parsers))
                for message in tool_messages:
                    llm_session.memory.add_message(message)

            if len(deltas) > 0:
                llm_session.memory.add_message({"role":"assistant", "content":"".join(deltas)})

        yield self.output_streaming.done()