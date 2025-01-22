import base64
import hashlib
import mimetypes
import os
import json
import time
import asyncio

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, AsyncGenerator

from just_agents.base_agent import BaseAgent
from just_agents.web.streaming import async_wrap
from just_agents.web.models import (
    ChatCompletionRequest, TextContent, ChatCompletionChoiceChunk, ChatCompletionChunkResponse,
    ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage, ResponseMessage, ErrorResponse
)
from dotenv import load_dotenv
from just_agents.interfaces.streaming_protocol import IAbstractStreamingProtocol
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from eliot import log_call, log_message



class AgentRestAPI(FastAPI):

    def __init__(
        self,
        *,
        agent_config: Optional[Path | str] = None,
        agent_section: Optional[str] = None,
        agent_parent_section: Optional[str] = None,
        debug: bool = False,
        title: str = "Just-Agent endpoint",
        description: str = "OpenAI-compatible API endpoint for Just-Agents",
        version: str = "1.0.0",
        openapi_url: str = "/openapi.json",
        openapi_tags: Optional[List[Dict[str, Any]]] = None,
        servers: Optional[List[Dict[str, Union[str, Any]]]] = None,
        docs_url: str = "/docs",
        redoc_url: str = "/redoc",
        terms_of_service: Optional[str] = None,
        contact: Optional[Dict[str, Union[str, Any]]] = None,
        license_info: Optional[Dict[str, Union[str, Any]]] = None,
        remove_system_prompt: Optional[bool] = None
    ) -> None:
        """Initialize the AgentRestAPI with FastAPI parameters.
        
        Args:
            debug: Enable debug mode
            title: API title shown in documentation
            description: API description shown in documentation
            version: API version
            openapi_url: URL for OpenAPI schema
            openapi_tags: List of tags to be included in the OpenAPI schema
            servers: List of servers to be included in the OpenAPI schema
            docs_url: URL for API documentation
            redoc_url: URL for ReDoc documentation
            terms_of_service: URL to the terms of service
            contact: Contact information in the OpenAPI schema
            license_info: License information in the OpenAPI schema
        """
        super().__init__(
            debug=debug,
            title=title,
            description=description,
            version=version,
            openapi_url=openapi_url,
            openapi_tags=openapi_tags,
            servers=servers,
            docs_url=docs_url,
            redoc_url=redoc_url,
            terms_of_service=terms_of_service,
            contact=contact,
            license_info=license_info
        )

        load_dotenv(override=True)
        if remove_system_prompt is None:
            self.remove_system_prompt = os.getenv('REMOVE_SYSTEM_PROMPT', False)
        else:
            self.remove_system_prompt = remove_system_prompt

        self._agent_related_config(agent_config, agent_section, agent_parent_section)
        self._routes_config()

    def _agent_related_config(self, agent_config: Path | str, agent_section: Optional[str] = None, agent_parent_section: Optional[str] = None):
        if agent_config is None:
            # Load from environment variable or use default
            agent_config = os.getenv('AGENT_CONFIG_PATH', 'agent_profiles.yaml')
        self.agent: BaseAgent = BaseAgent.from_yaml(file_path=agent_config, section_name=agent_section, parent_section=agent_parent_section)

      

    def _routes_config(self):
          # Add CORS middleware
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Register routes
        self.get("/")(self.default)
        self.post("/v1/chat/completions", description="OpenAI compatible chat completions")(self.chat_completions)


    def _clean_messages(self, request: ChatCompletionRequest):
        for message in request.messages:
            if message.role == "user":
                content = message.content
                if type(content) is list:
                    if len(content) > 0:
                        if isinstance(content[0],TextContent):
                            message.content = content[0].text

    def _remove_system_prompt(self, request: ChatCompletionRequest):
        if request.messages[0].role == "system":
            request.messages = request.messages[1:]
    
    def default(self):
        return f"This is default page for the {self.title}"

    def sha256sum(self, content_str: str):
        hash = hashlib.sha256()
        hash.update(content_str.encode('utf-8'))
        return hash.hexdigest()

    def save_files(self, request: dict):
        for file in request.get("file_params", []):
            file_name = file.get("name")
            file_content_base64 = file.get("content")
            file_checksum = file.get("checksum")
            file_mime = file.get("mime")

            if self.sha256sum(file_content_base64) != file_checksum:
                raise Exception("File checksum does not match")

            extension = mimetypes.guess_extension(file_mime)
            file_content = base64.urlsafe_b64decode(file_content_base64.encode('utf-8'))
            full_file_name = file_name + extension

            file_path = Path('/tmp', full_file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)



#    @log_call(action_type="chat_completions", include_result=False)
    async def chat_completions(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, Any, ErrorResponse]:
        try:
            agent = self.agent
            self._clean_messages(request)
            if self.remove_system_prompt:
                self._remove_system_prompt(request)

            if "file_params" in request:
                params = request.file_params
                if params != []:
                    self.save_files(request.model_dump())

            #Done by FastAPI+pydantic under the hood! Just supply schema...

            # if not request.messages:
            #     log_message(
            #         message_type="validation_error",
            #         error="No messages provided in request"
            #     )
            #     return {
            #         "error": {
            #             "message": "No messages provided in request",
            #             "type": "invalid_request_error",
            #             "param": "messages",
            #             "code": "invalid_request_error"
            #         }
            #     }, 400
            #
            # # Validate required fields
            # if "model" not in request:
            #     log_message(
            #         message_type="validation_error",
            #         error="model is required"
            #     )
            #     return {
            #         "error": {
            #             "message": "model is required",
            #             "type": "invalid_request_error",
            #             "param": "model",
            #             "code": "invalid_request_error"
            #         }
            #     }, 400

            is_streaming = request.stream
            messages = [message.model_dump(mode='json') for message in request.messages] # todo: support pydantic model!!!
            stream_generator = agent.stream(
                messages
            )

            if is_streaming:
                return StreamingResponse(
                    async_wrap(stream_generator),
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream"
                    }
                )
            else:
                # Collect all chunks into final response
                response_content = ""
                for chunk in stream_generator:

                    if chunk == "[DONE]":
                        break
                    try:
                        # Parse the SSE data
                        data = IAbstractStreamingProtocol.sse_parse(chunk)
                        json_data = data.get("data", "{}")
                        print(json_data)
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            delta = json_data["choices"][0].get("delta", {})
                            if "content" in delta:
                                response_content += delta["content"]
                    except Exception:
                        continue

                return ChatCompletionResponse(
                        id=f"chatcmpl-{time.time()}",
                        object="chat.completion",
                        created=int(time.time()),
                        model=request.model,
                        choices=[ChatCompletionChoice(
                            index=0,
                            message=ResponseMessage(
                                role= "assistant",
                                content= response_content
                            ),
                            finish_reason="stop"
                        )],
                        usage=ChatCompletionUsage(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0
                       ))

        except Exception as e:
            # log_message(
            #     message_type="chat_completion_error",
            #     error=str(e),
            #     error_type=type(e).__name__,
            #     request_details={
            #         "model": request.model,
            #         "message_count": len(request.messages),
            #         "streaming": request.stream
            #     }
            # )

            error_response = ErrorResponse(
                error=ErrorResponse.ErrorDetails(
                    message=str(e)
                )
            )
            return error_response

