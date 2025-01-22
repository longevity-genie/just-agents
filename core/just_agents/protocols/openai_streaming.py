from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
import time
from typing import Generator

DEFAULT_OPENAI_STOP="[DONE]"

class OpenaiStreamingProtocol(IAbstractStreamingProtocol):
    def __init__(self, stop: str = DEFAULT_OPENAI_STOP):
        self.stop = stop

    def get_chunk(self, index: int, delta: str, options: dict):
        chunk : dict = {
            "id": index,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": options["model"],
            "choices": [{"delta": {"content": delta}}],
        }
        return self.sse_wrap(chunk)
    #    return f"data: {json.dumps(chunk)}\n\n"


    def done(self):
        return self.sse_wrap(self.stop)
    #    return "\ndata: [DONE]\n\n"

    def response_from_stream(self, stream_generator: Generator) -> str:
        response_content = ""
        for chunk in stream_generator:
            try:
                # Parse the SSE data
                data = IAbstractStreamingProtocol.sse_parse(chunk)
                json_data = data.get("data", "{}")
                if json_data == self.stop:
                    break
                print(json_data)
                if "choices" in json_data and len(json_data["choices"]) > 0:
                    delta = json_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        response_content += delta["content"]
            except Exception as e:
                continue
        return response_content


