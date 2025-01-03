from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
import json
import time

class OpenaiStreamingProtocol(IAbstractStreamingProtocol):
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
        # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
        return "data: [DONE]\n\n"