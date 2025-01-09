import asyncio
import time

from typing import Optional, List, Dict, Any, Union, AsyncGenerator, Generator

from just_agents.base_agent import BaseAgent
from just_agents.web.models import (
    ChatCompletionRequest, TextContent, ChatCompletionChoiceChunk, ChatCompletionChunkResponse,
    ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage, ResponseMessage, ErrorResponse
)

async def async_wrap(response: Generator) -> AsyncGenerator[Any, None]:
    for chunk in response:
        yield chunk
        await asyncio.sleep(0.001)
    yield f"\n\n"

# generator function to mimick yield ChatCompletionChunk chunks
async def generate_response_chunks(response: ChatCompletionResponse, stop: Optional[str] = "[DONE]") -> \
AsyncGenerator[str, None]:
    #logger.info("Imitating generation")
    #logger.trace(f"Given {str(response)}")
    for choice in response.choices:
        delta = ChatCompletionChoiceChunk(
            index=choice.index,
            delta=choice.message,
            finish_reason=None
        )
        #logger.trace(f"choice {str(delta)}")
        chunk = ChatCompletionChunkResponse(
            object="chat.completion.chunk",
            id=response.id,
            choices=[delta],
            created=int(time.time()),
            model=response.model,
        )
        chunk = chunk.model_dump_json()
        #logger.debug(f"chunk {str(chunk)}")
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
    #logger.debug(f"final_chunk {str(final_chunk)}")
    yield f"data: {final_chunk}\n\n"
    yield f"data: {stop}\n\n"
    await asyncio.sleep(1)