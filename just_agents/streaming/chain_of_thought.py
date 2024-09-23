from typing import AsyncGenerator

from litellm import ModelResponse, completion
from typing import Callable, Optional
from just_agents.memory import Memory
from just_agents.streaming.abstract_streaming import AbstractStreaming, FunctionParser
from just_agents.utils import rotate_completion
import time
import json

class ChainOfThought(AbstractStreaming):

    async def resp_async_generator(self, memory: Memory,
                                   options: dict,
                                   available_tools: dict[str, Callable]
                                   ) -> AsyncGenerator[str, None]:
        memory.add_message({"role": "assistant",
             "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        )
        step_count = 0

        while True:
            step_count += 1
            step_data = self.make_api_call(memory.messages, options, max_tokens = 300)
            memory.add_message({"role": "assistant", "content": json.dumps(step_data)})
            # Yield after each step for Streamlit to update
            print(step_count, " ", step_data['content'])
            content = step_data['content'] + "\n"
            yield f"data: {self._get_chunk(step_count, content, options)}\n\n"
            if step_data[
                'next_action'] == 'final_answer' or step_count > 25:  # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                break

        # Generate final answer
        memory.add_message({"role": "user",
                         "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."})
        final_data = self.make_api_call(memory.messages, options, max_tokens = 1200, is_final_answer=True)
        # yield steps, total_thinking_time
        print("Final: ", final_data)
        yield f"data: {self._get_chunk(step_count + 1, final_data, options)}\n\n"
        yield "data: [DONE]\n\n"


    def make_api_call(self, messages, options:dict, max_tokens, is_final_answer=False):
        opt = options.copy()
        for attempt in range(3):
            try:
                if is_final_answer:
                    opt["max_tokens"] = max_tokens
                    response = rotate_completion(messages=messages, stream=False, options=opt, max_tries = 1)
                    return response.choices[0].message.content
                else:
                    opt["max_tokens"] = max_tokens
                    opt["response_format"] = {"type": "json_object"}
                    response = rotate_completion(messages=messages, stream=False, options=opt, max_tries = 1)
                    return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    if is_final_answer:
                        return {"title": "Error",
                                "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                    else:
                        return {"title": "Error",
                                "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                                "next_action": "final_answer"}
                time.sleep(1)  # Wait for 1 second before retrying