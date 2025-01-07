from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol

import time

class OpenaiStreamingProtocol(IAbstractStreamingProtocol):

    stop: str = "[DONE]"

    def get_chunk(self, index: int, delta: str, options: dict):
        chunk : dict = {
            "id": index,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": options["model"],
            "choices": [{"delta": {"content": delta}}],
        }
        return self.sse_wrap(chunk)


    def done(self):
        return self.sse_wrap(self.stop)