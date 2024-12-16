from typing import AsyncGenerator

from just_agents.simple.streaming.protocols.abstract_streaming import AbstractStreaming
from just_agents.simple.streaming.protocols.openai_streaming import OpenaiStreamingProtocol
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
import json

class ChainOfThought(AbstractStreaming):

    def __init__(self, llm_session, output_streaming: IAbstractStreamingProtocol = OpenaiStreamingProtocol()):
        super().__init__(llm_session)
        self.output_streaming = output_streaming


    async def resp_async_generator(self) -> AsyncGenerator[str, None]:
        print("This method depricated use cot_agent instead.")
        from just_agents.simple.llm_session import LLMSession
        llm_session: LLMSession = self.session
        max_steps = 25

        opt = llm_session.llm_options.copy()
        opt["max_tokens"] = 500
        #if not "claude" in opt["model"]:
        #    opt["response_format"] = {"type": "json_object"}
        for step_count in range(1, max_steps):
            response = llm_session._rotate_completion(stream=False)
            step_data = json.loads(response.choices[0].message.content)
            llm_session.memory.add_message({"role": "assistant", "content": json.dumps(step_data)})
            print(step_count, " ", step_data)
            content = step_data['content'] + "\n"
            yield self.output_streaming.get_chunk(step_count, content, opt)
            if step_data[
                'next_action'] == 'final_answer':  # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                break

        llm_session.memory.add_message({"role": "user",
                            "content": "Please provide the final answer based solely on your reasoning above."})

        opt["max_tokens"] = 1200
        response = llm_session._rotate_completion(stream=False)
        final_data = json.loads(response.choices[0].message.content)
        # yield steps, total_thinking_time
        print("Final: ", final_data)
        yield self.output_streaming.get_chunk(step_count + 1, final_data['content'], opt)
        yield self.output_streaming.done()