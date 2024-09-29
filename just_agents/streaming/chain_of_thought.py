from typing import AsyncGenerator

from litellm import ModelResponse, completion
from typing import Callable, Optional
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser
from just_agents.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
from just_agents.utils import rotate_completion
import time
import json

class ChainOfThought(AbstractStreaming):

    def __init__(self, output_streaming: AbstractStreamingProtocol = OpenaiStreamingProtocol()):
        self.output_streaming = output_streaming

    async def resp_async_generator(self, memory: Memory,
                                   options: dict,
                                   available_tools: dict[str, Callable]
                                   ) -> AsyncGenerator[str, None]:
        print("This method depricated use cot_agent instead.")

        max_steps = 25

        opt = options.copy()
        opt["max_tokens"] = 300
        opt["response_format"] = {"type": "json_object"}
        for step_count in range(1, max_steps):
            response = rotate_completion(messages=memory.messages, stream=False, options=opt)
            step_data = json.loads(response.choices[0].message.content)
            memory.add_message({"role": "assistant", "content": json.dumps(step_data)})
            print(step_count, " ", step_data)
            content = step_data['content'] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, opt)
            if step_data[
                'next_action'] == 'final_answer':  # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                break

        memory.add_message({"role": "user",
                            "content": "Please provide the final answer based solely on your reasoning above."})

        opt["max_tokens"] = 1200
        response = rotate_completion(messages=memory.messages, stream=False, options=opt)
        final_data = json.loads(response.choices[0].message.content)
        # yield steps, total_thinking_time
        print("Final: ", final_data)
        yield self.output_streaming.get_chunk(step_count + 1, final_data['content'], opt)
        yield self.output_streaming.done()