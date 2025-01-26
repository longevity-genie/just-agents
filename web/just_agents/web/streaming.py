import asyncio
import aiofiles
import time
import sys
from uuid import uuid4

from typing import Union, Optional, Any, AsyncGenerator, Generator
from just_agents.protocols.openai_streaming import OpenaiStreamingProtocol, DEFAULT_OPENAI_STOP
from just_agents.data_classes import Role
from just_agents.base_agent import BaseAgent
from just_agents.web.models import (
    ChatCompletionRequest, ChatCompletionChoice, ChatCompletionUsage,
    ChatCompletionChoiceChunk, ChatCompletionChunkResponse,
    ChatCompletionResponse, ResponseMessage
)
from fastapi.responses import StreamingResponse
from eliot import log_message, start_task, log_call

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
            text = text + message.get_text()
        if text and text [-1] != '\n':
            text = text + '\n'

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
        created=int(time.time()),
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

def chunk_to_bytes(chunk: Any) -> bytes:
    """
    Reads a stream byte by byte and dumps it into a file for debugging.
    """

    try:
        if isinstance(chunk, str):
            chunk_bytes = chunk.encode('utf-8')  # Convert str to bytes
        elif isinstance(chunk, bytes):
            chunk_bytes = chunk
        else:
            chunk_bytes = bytes(chunk)
        return chunk_bytes
    except Exception as e:
        return bytes(str(e).encode('utf-8'))

def dump_sync_stream_to_file(stream, filename: str):
    """
    Dumps a synchronous generator to a file, handling chunk-to-bytes conversion.
    """
    try:
        with open(filename, 'wb') as f:
            for chunk in stream:
                chunk_bytes = chunk_to_bytes(chunk)  # Use helper function
                f.write(chunk_bytes)
                sys.stdout.buffer.write(chunk_bytes)  # Optional: output to console
                yield chunk
    except Exception as e:
        print(f"Error while dumping sync stream: {e}")

async def dump_async_stream_to_file(stream, filename: str):
    """
    Dumps an asynchronous generator to a file, handling chunk-to-bytes conversion.
    """
    try:
        async with aiofiles.open(filename, 'wb') as f:
            async for chunk in stream:
                chunk_bytes = chunk_to_bytes(chunk)  # Use helper function
                await f.write(chunk_bytes)
                sys.stdout.buffer.write(chunk_bytes)  # Optional: output to console
                yield chunk
    except Exception as e:
        print(f"Error while dumping async stream: {e}")

def process_request_with_debugging(request: ChatCompletionRequest) -> StreamingResponse:
    """
    Processes a request with debugging logic and logs for potential future use.

    :param request: The incoming request object containing messages and model details.
    :return: StreamingResponse with the generated response chunks.
    """
    prompt = has_system_prompt(request)
    if prompt:
        start_task(action_type="chat_completions").log(
            message_type="call_with_prompt",
            prompt=prompt
        )

        # Define LLM options
        OPENAI_GPT4oMINI_prox = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "api_base": "http://127.0.0.1:14000"
        }

        # Debugging code (can be used for diagnostics)
        rsp = generate_response_chunks(
            get_completion_response(
                model=request.model,
                text=response_from_stream(
                    dump_sync_stream_to_file(
                        BaseAgent(llm_options=OPENAI_GPT4oMINI_prox).stream(
                            request.messages,
                            enforce_agent_prompt=False
                        ),
                        "bad.txt"
                    )
                )
            )
        )

        return StreamingResponse(
            # Original debugging line, commented out for reference
            # dump_async_stream_to_file(rsp, "working.txt"),
            BaseAgent(llm_options=OPENAI_GPT4oMINI_prox).stream(
                request.messages,
                enforce_agent_prompt=False
            ),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

# Call the function as needed
# response = process_request_with_debugging(your_request_object)
