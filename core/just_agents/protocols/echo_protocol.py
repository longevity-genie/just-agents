import random
from pydantic import BaseModel, Field, PrivateAttr
from typing import Any, List, Union, Dict, Sequence, Coroutine, AsyncGenerator, Generator, ClassVar, Type, Callable, Optional

from just_agents.interfaces.protocol_adapter import IProtocolAdapter, ExecuteToolCallback, AbstractMessage
from just_agents.interfaces.function_call import IFunctionCall, ToolByNameCallback
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from just_agents.protocols.openai_streaming import OpenaiStreamingProtocol

class NoopFunctionCall(BaseModel, IFunctionCall[AbstractMessage]):
    """
    A no-op function call implementation that performs no actual work.
    """
    id: str = Field("noop_id", description="Function call identifier.")
    name: str = Field("noop_function", description="The name of the no-op function.")
    arguments: Any = Field(None, description="Arguments for the no-op function.")

    def execute_function(self, call_by_name: ToolByNameCallback) -> AbstractMessage:
        """
        Execute the no-op function call.

        Args:
            call_by_name: A callback to retrieve a tool by name.

        Returns:
            AbstractMessage: A simple tool message indicating no execution was done.
        """
        return {"role": "tool", "content": "Noop function execution.", "name": self.name, "tool_call_id": self.id}

    @staticmethod
    def reconstruct_tool_call_message(calls: Sequence['NoopFunctionCall']) -> AbstractMessage:
        """
        Reconstruct a message from multiple no-op calls.

        Args:
            calls: A sequence of no-op function calls.

        Returns:
            AbstractMessage: A simple assistant message indicating no-op calls.
        """
        return {"role": "assistant", "content": "Noop calls reconstructed."}


class EchoModelResponse(BaseModel):
    """
    A response model that simply echoes the input content.
    """
    content: Union[str, Any] = Field(..., description="The echoed content.")
    role: str = Field("assistant", description="The role associated with this response.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the response.")

class StreamedEchoModelResponse(EchoModelResponse, populate_by_name=True):
    """
    A streamed response model that includes a delta field.
    'content' is synchronized with 'delta' by using alias and populate_by_name=True.
    """
    delta: str = Field(..., alias="content", description="Streamed delta content.")
    role: str = Field("assistant", description="The role associated with this response.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the response.")


class EchoProtocolAdapter(BaseModel, IProtocolAdapter[EchoModelResponse, AbstractMessage]):
    """
    An echo protocol adapter that returns the input as output,
    serving as a stub for testing without actual model calls.
    """
    function_convention: ClassVar[Type[IFunctionCall[Any]]] = NoopFunctionCall
    _output_streaming: IAbstractStreamingProtocol = PrivateAttr(default_factory=OpenaiStreamingProtocol)

    def _completion(self, prompt: str) -> EchoModelResponse:
        """
        Produce a complete EchoModelResponse.

        Args:
            prompt: The input prompt.

        Returns:
            EchoModelResponse: The echoed prompt.
        """
        return EchoModelResponse(content=prompt)

    def _streaming_completion(self, prompt: str) -> Generator[StreamedEchoModelResponse, None, None]:
        """
        Produce a streaming response by splitting the prompt into random-sized chunks
        and yielding them as StreamedEchoModelResponse.

        Args:
            prompt: The input prompt.

        Yields:
            StreamedEchoModelResponse: Each chunk of the prompt as a streamed delta.
        """
        data = prompt
        start = 0
        while start < len(data):
            chunk_size = random.randint(2, 5)
            chunk = data[start:start + chunk_size]
            start += chunk_size

            # Use the streaming protocol to simulate a chunk
            yield self._output_streaming.get_chunk(index=start, delta=chunk, options={"model": "echo-model"})

        yield self._output_streaming.done()

    def completion(self, *args, **kwargs) -> Union[EchoModelResponse, Generator[StreamedEchoModelResponse, None, None]]:
        """
        Return either a full EchoModelResponse or a streaming generator of StreamedEchoModelResponse
        depending on 'streaming' flag in kwargs.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments, including 'prompt' and 'streaming'.

        Returns:
            Union[EchoModelResponse, Generator[StreamedEchoModelResponse]]:
                If 'streaming' is True, returns a generator for streaming chunks;
                otherwise, returns a full EchoModelResponse.
        """
        prompt = kwargs.get("prompt", "")
        streaming = kwargs.get("streaming", False)
        if streaming:
            return self._streaming_completion(prompt)
        else:
            return self._completion(prompt)

    async def async_completion(self, *args, **kwargs) \
            -> Coroutine[Any, Any, Union[EchoModelResponse, AsyncGenerator[StreamedEchoModelResponse, None]]]:
        """
        An async wrapper around completion.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The same result as completion(), but awaited if necessary.
        """
        # Wrap the synchronous completion in an async context
        # Since completion might return either a generator or a single response,
        # we just await in a trivial manner. If it's a generator, it's yielded upon iteration.
        result = self.completion(*args, **kwargs)
        return result

    def message_from_response(self, response: EchoModelResponse) -> AbstractMessage:
        """
        Convert EchoModelResponse to an abstract message dict.

        Args:
            response: The EchoModelResponse instance.

        Returns:
            AbstractMessage: A dictionary with role, content, and metadata.
        """
        return {
            "role": response.role,
            "content": response.content,
            "metadata": response.metadata
        }

    def delta_from_response(self, response: EchoModelResponse) -> AbstractMessage:
        """
        Convert a delta EchoModelResponse to an abstract message.

        Args:
            response: The delta EchoModelResponse.

        Returns:
            AbstractMessage: The same abstract message structure as a full response.
        """
        return self.message_from_response(response)

    def content_from_delta(self, delta: AbstractMessage) -> str:
        """
        Extract textual content from a delta message.

        Args:
            delta: The abstract message dict.

        Returns:
            str: The content string from the delta.
        """
        return delta.get("content", "")

    def tool_calls_from_message(self, message: AbstractMessage) -> List[IFunctionCall[AbstractMessage]]:
        """
        Extract tool calls from the given message. This echo protocol does not produce tool calls.

        Args:
            message: The abstract message dict.

        Returns:
            List[IFunctionCall]: An empty list as no tool calls are produced.
        """
        return []

    def response_from_deltas(self, deltas: List[EchoModelResponse]) -> EchoModelResponse:
        """
        Combine all deltas into a single EchoModelResponse.

        Args:
            deltas: A list of EchoModelResponse chunks.

        Returns:
            EchoModelResponse: A single response with concatenated content.
        """
        combined_content = "".join(delta.content for delta in deltas)
        return EchoModelResponse(content=combined_content)

    # get_chunk and done are inherited from IProtocolAdapter and rely on _output_streaming
