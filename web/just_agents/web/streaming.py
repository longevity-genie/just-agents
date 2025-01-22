import asyncio
import time
from uuid import uuid4

from typing import Optional, Any, AsyncGenerator, Generator
from just_agents.protocols.openai_streaming import OpenaiStreamingProtocol, DEFAULT_OPENAI_STOP
from just_agents.protocols.litellm_protocol import Role
from just_agents.web.models import (
    ChatCompletionRequest, ChatCompletionChoice, ChatCompletionUsage,
    ChatCompletionChoiceChunk, ChatCompletionChunkResponse,
    ChatCompletionResponse, ResponseMessage
)


def messages_content_to_text(request: ChatCompletionRequest) -> ChatCompletionRequest:
    modified_request = request.model_copy(deep=True)
    for message in modified_request.messages:
        if message.role == Role.user:
            message.content = message.get_text()
    return modified_request

def remove_system_messages(request: ChatCompletionRequest) -> ChatCompletionRequest:
    modified_request = request.model_copy(deep=True)
    for message in modified_request.messages:
        if message.role == Role.system:
            modified_request.messages.remove(message)
    return modified_request

def has_system_prompt(request: ChatCompletionRequest) -> Optional[str]:
    text = ""
    for message in request.messages:
        if message.role == Role.system:
            text = text  + message.get_text() + '\n'
    return text if text else None

def get_completion_response(
        model : Optional[str]="SYSTEM",
        text : Optional[str]=None,
        usage: Optional[ChatCompletionUsage] = None
) -> ChatCompletionResponse:

    if not text:
        text = "Something went wrong with response!!"
    message = ResponseMessage(
        role=Role.assistant,
        content=text
    )
    choice = ChatCompletionChoice(
        index=0,
        finish_reason="stop",
        message=message,
    )
    response = ChatCompletionResponse(
        id = "chatcmpl-"+str(uuid4()),
        object = "chat.completion",
        created=time.time(),
        model = model,
        choices = [choice],
        usage=usage
    )
    return response

async def async_wrap(response: Generator) -> AsyncGenerator[Any, None]:
    for chunk in response:
        yield chunk
        await asyncio.sleep(0.001)
    yield f"\n\n"


def response_from_stream(stream_generator: Generator, stop: Optional[str] = DEFAULT_OPENAI_STOP) -> str:
    return OpenaiStreamingProtocol(stop=stop).response_from_stream(stream_generator)
# generator function to mimic yield ChatCompletionChunk chunks
async def generate_response_chunks(response: ChatCompletionResponse, stop: Optional[str] = DEFAULT_OPENAI_STOP) -> \
        AsyncGenerator[str, None]:
    # logger.info("Imitating generation")
    # logger.trace(f"Given {str(response)}")
    for choice in response.choices:
        delta = ChatCompletionChoiceChunk(
            index=choice.index,
            delta=choice.message,
            finish_reason=None
        )
        # logger.trace(f"choice {str(delta)}")
        chunk = ChatCompletionChunkResponse(
            object="chat.completion.chunk",
            id=response.id,
            choices=[delta],
            created=int(time.time()),
            model=response.model,
        )
        chunk = chunk.model_dump_json()
        # logger.debug(f"chunk {str(chunk)}")
        yield f"data: {chunk}\n\n"

        await asyncio.sleep(1)
    final_chunk = ChatCompletionChunkResponse(
        object="chat.completion.chunk",
        id=response.id,
        created=int(time.time()),
        model=response.model,
        choices=[ChatCompletionChoiceChunk(
            index=0,
            delta=ResponseMessage(
                role=None,
                content=None
            ),
            finish_reason="stop"
        )],
        usage=response.usage,
    )
    final_chunk = final_chunk.model_dump_json()
    # logger.debug(f"final_chunk {str(final_chunk)}")
    yield f"data: {final_chunk}\n\n"
    yield f"data: {stop}\n\n"
    await asyncio.sleep(1)