import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI
from just_agents.base_agent import BaseAgent
from just_agents.interfaces.agent import IAgent
from starlette.responses import StreamingResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pycomfort.logging import log_function
import yaml
import os
from pycomfort.logging import log_function
from eliot import log_call, log_message
import json


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
        license_info: Optional[Dict[str, Union[str, Any]]] = None
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
        self.post("/v1/chat/completions")(self.chat_completions)




    def _clean_messages(self, request: dict):
        for message in request["messages"]:
            if message["role"] == "user":
                content = message["content"]
                if type(content) is list:
                    if len(content) > 0:
                        if type(content[0]) is dict:
                            if content[0].get("type", "") == "text":
                                if type(content[0].get("text", None)) is str:
                                    message["content"] = content[0]["text"]

    def _remove_system_prompt(self, request: dict):
        if request["messages"][0]["role"] == "system":
            request["messages"] = request["messages"][1:]
    
    def default(self):
        return f"This is default page for the {self.title}"

    @log_call(action_type="chat_completions", include_result=False)
    def chat_completions(self, request: dict):
        try:
            agent = self.agent
            self._clean_messages(request)
            self._remove_system_prompt(request)
            
            if not request["messages"]:
                log_message(
                    message_type="validation_error",
                    error="No messages provided in request"
                )
                return {
                    "error": {
                        "message": "No messages provided in request",
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": "invalid_request_error"
                    }
                }, 400

            # Validate required fields
            if "model" not in request:
                log_message(
                    message_type="validation_error",
                    error="model is required"
                )
                return {
                    "error": {
                        "message": "model is required",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "invalid_request_error"
                    }
                }, 400

            is_streaming = request.get("stream", False)
            stream_generator = agent.stream(request["messages"])

            if is_streaming:
                return StreamingResponse(
                    stream_generator,
                    media_type="text/event-stream",
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
                        data = json.loads(chunk.decode().split("data: ")[1])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                response_content += delta["content"]
                    except Exception:
                        continue

                return {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", "unknown"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }

        except Exception as e:
            log_message(
                message_type="chat_completion_error",
                error=str(e),
                error_type=type(e).__name__,
                request_details={
                    "model": request.get("model"),
                    "message_count": len(request.get("messages", [])),
                    "streaming": request.get("stream", False)
                }
            )
            return {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_server_error"
                }
            }, 500