from typing import AsyncGenerator
from typing import Callable

from just_agents.simple.streaming.protocols.abstract_streaming import AbstractStreaming
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from just_agents.simple.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
import json
from qwen_agent.llm import get_chat_model # deprecated, so far we do not use qwen_agent
import litellm


class Qwen2AsyncSession(AbstractStreaming):

    def __init__(self, llm_session, output_streaming: IAbstractStreamingProtocol = OpenaiStreamingProtocol()):
        super().__init__(llm_session)
        self.output_streaming = output_streaming


    def _process_function(self, name: str, arguments: str, available_tools: dict[str, Callable]):
        function_args = json.loads(arguments)
        function_to_call = available_tools[name]
        print(function_to_call, function_args)
        try:
            function_response = function_to_call(**function_args)
        except Exception as e:
            function_response = str(e)
        message = {
            'role': 'function',
            'name': name,
            'content': function_response,
        }  # TODO need to track arguments , arguments=function_args
        return message

    async def resp_async_generator(self
                                   ) -> AsyncGenerator[str, None]:
        from just_agents.simple.llm_session import LLMSession
        llm_session: LLMSession = self.session
        llm = get_chat_model(llm_session.llm_options)
        functions = None
        if llm_session.available_tools:
            functions = []
            for fun_name in llm_session.available_tools:
                functions.append(litellm.utils.function_to_dict(llm_session.available_tools[fun_name]))

        proceed = True
        while proceed:
            proceed = False
            responces = llm.chat(messages=llm_session.memory.messages, functions=functions, stream=True,
                                 extra_generate_cfg=dict(parallel_function_calls=True))
            prev_len = 0
            messages = []
            for i, msg in enumerate(responces):
                messages = msg
                content = msg[0]["content"]
                if len(content) > 0:
                    delta = content[prev_len:]
                    prev_len = len(content)
                    yield self.output_streaming.get_chunk(i, delta, llm_session.llm_options)

            fncall_msgs = [rsp for rsp in messages if rsp.get('function_call', None)]
            llm_session.memory.add_messages(messages)
            # function_call = messages[-1].get('function_call', None)
            if fncall_msgs and len(fncall_msgs) > 0 and llm_session.available_tools:
                proceed = True
                for msg in fncall_msgs:
                    function_call = msg['function_call']
                    llm_session.memory.add_message(self._process_function(function_call["name"], function_call['arguments'], llm_session.available_tools))
        yield self.output_streaming.done()