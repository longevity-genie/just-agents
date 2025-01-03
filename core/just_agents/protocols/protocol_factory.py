from enum import Enum
from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback

class StreamingMode(str, Enum):
    openai = "openai"
    echo = "echo"
    qwen2 = "qwen2"

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
            execute_functions: ExecuteToolCallback,

    ) -> IProtocolAdapter:
        if mode == StreamingMode.openai:
            from just_agents.protocols.litellm_protocol import LiteLLMAdapter
            return LiteLLMAdapter(
                execute_function_hook=execute_functions,
            )
        elif mode == StreamingMode.echo:
            from just_agents.protocols.mock_protocol import EchoProtocolAdapter
            return EchoProtocolAdapter(
                execute_function_hook=execute_functions,
            )
        elif mode == StreamingMode.qwen2:
            # todo: implement
            raise NotImplementedError("Qwen2 streaming is not yet implemented for pydantic-based agents")
        else:
            raise ValueError("Unknown streaming method")


