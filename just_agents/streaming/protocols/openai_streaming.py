from just_agents.streaming.protocols.abstract_protocol import AbstractStreamingProtocol
import json
import time


class OpenaiStreamingProtocol(AbstractStreamingProtocol):

    def get_chunk(self, index: int, delta: str, options: dict):
        chunk = {
            "id": index,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": options["model"],
            "choices": [{"delta": {"content": delta}}],
        }
        return f"data: {json.dumps(chunk)}\n\n"


    def done(self):
        return "data: [DONE]\n\n"