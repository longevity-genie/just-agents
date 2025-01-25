from enum import Enum
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback


class StreamingMode(str, Enum):
    openai = "openai"
    echo = "echo"

    def __new__(cls, value, *args, **kwargs):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        return str(self.value)

class ProtocolAdapterFactory:
    @staticmethod
    def get_protocol_adapter(
            mode: StreamingMode,
            **kwargs
    ) -> IProtocolAdapter:
        if mode == StreamingMode.openai:
            from just_agents.protocols.litellm_protocol import LiteLLMAdapter
            return LiteLLMAdapter(**kwargs)
        elif mode == StreamingMode.echo:
            from just_agents.protocols.echo_protocol import EchoProtocolAdapter
            return EchoProtocolAdapter(**kwargs)
        else:
            raise ValueError("Unknown streaming method")


