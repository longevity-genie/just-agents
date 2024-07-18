from typing import AsyncGenerator
from litellm import ModelResponse, completion
from typing import Callable, Optional
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser
from just_agents.utils import rotate_completion
import json
from qwen_agent.llm import get_chat_model
import litellm


class Qwen2AsyncSession(AbstractStreaming):

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

    async def resp_async_generator(self, memory: Memory,
                                   options: dict,
                                   available_tools: dict[str, Callable]
                                   ) -> AsyncGenerator[str, None]:
        llm = get_chat_model(options)
        functions = None
        if available_tools:
            functions = []
            for fun_name in available_tools:
                functions.append(litellm.utils.function_to_dict(available_tools[fun_name]))

        proceed = True
        while proceed:
            proceed = False
            responces = llm.chat(messages=memory.messages, functions=functions, stream=True,
                                 extra_generate_cfg=dict(parallel_function_calls=True))
            prev_len = 0
            messages = []
            for i, msg in enumerate(responces):
                messages = msg
                content = msg[0]["content"]
                if len(content) > 0:
                    delta = content[prev_len:]
                    prev_len = len(content)
                    yield f"data: {self._get_chunk(i, delta, options)}\n\n"

            fncall_msgs = [rsp for rsp in messages if rsp.get('function_call', None)]
            memory.add_messages(messages)
            # function_call = messages[-1].get('function_call', None)
            if fncall_msgs and len(fncall_msgs) > 0 and available_tools:
                proceed = True
                for msg in fncall_msgs:
                    function_call = msg['function_call']
                    memory.add_message(self._process_function(function_call["name"], function_call['arguments'], available_tools))
        yield "data: [DONE]\n\n"