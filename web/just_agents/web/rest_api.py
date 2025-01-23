import base64
import hashlib
import mimetypes
import os
import time
from pydantic import BaseModel
import yaml

from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from just_agents.web.web_agent import WebAgent
from just_agents.web.streaming import response_from_stream

from just_agents.web.models import (
    Role, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage, ResponseMessage, ErrorResponse
)
from rich.pretty import pprint
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from eliot import log_message, start_task


class Model(BaseModel):
    id: str
    created: int
    object: str = "model"
    owned_by: str = "organization"
    permission: Optional[List[Dict[str, Any]]] = None  # Array of permissions objects
    root: Optional[str] = None  # The model this is derived from, if applicable
    parent: Optional[str] = None  # The parent model

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]



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
            self.remove_system_prompt = os.getenv('REMOVE_SYSTEM_PROMPT', True)
        else:
            self.remove_system_prompt = remove_system_prompt

        self.agents = {}  # Dictionary to store multiple agents
        self._agent_related_config(agent_config, agent_section, agent_parent_section)
        self._routes_config()

    def _agent_related_config(self, agent_config: Path | str, agent_section: Optional[str] = None, agent_parent_section: Optional[str] = None):
        if agent_config is None:
            # Load from environment variable or use default
            agent_config = os.getenv('AGENT_CONFIG_PATH', 'agent_profiles.yaml')

        if agent_section:
            # Load single agent
            agent = WebAgent.from_yaml(agent_section, agent_parent_section, agent_config)
            agent.enforce_agent_prompt = self.remove_system_prompt
            self.agents[agent_section] = agent
            self.agent = agent  # Keep default agent for backward compatibility
        else:
            # Load all agents using from_yaml_dict
            self.agents = WebAgent.from_yaml_dict(agent_config, agent_parent_section)
            # Set enforce_agent_prompt for all agents
            for agent in self.agents.values():
                agent.enforce_agent_prompt = self.remove_system_prompt
            
            # Set the first agent as default if any were loaded
            if self.agents:
                self.agent = next(iter(self.agents.values()))

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
        self.get("/v1/models", description="List available models")(self.list_models)
        self.post("/v1/chat/completions", description="OpenAI compatible chat completions")(self.chat_completions)


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
        with start_task(action_type="chat_completions") as action:
            try:
                action.log(
                    message_type=f"chat_completions for {request.model}",
                    request = str(request.model_dump())
                )
                # Get the agent based on the model name
                agent = self.agents.get(request.model)
                if not agent:
                    available_models = list(self.agents.keys())
                    action.log(
                        message_type="invalid_model_requested",
                        model=request.model,
                        available_models=available_models,
                        action="error_no_model"
                    )
                    if len(self.agents) == 0:
                        raise ValueError(f"No agents found. Available models: {available_models}")
                    else:
                        action.log(
                            message_type=f"no_agent_found for {request.model} with {request.messages[0].content}",
                            available_models=available_models,
                            action="error_no_model",
                            error="No agent found, using the first one"
                        )
                        agent = next(iter(self.agents.values()))

                if "file_params" in request:
                    params = request.file_params
                    if params != []:
                        self.save_files(request.model_dump())

                is_streaming = request.stream
                stream_generator = agent.stream(
                    request.messages
                )

                if is_streaming:
                    return StreamingResponse(
                        stream_generator,
                        media_type="application/x-ndjson",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Content-Type": "text/event-stream"
                        }
                    )
                else:
                    # Collect all chunks into final response
                    response_content = response_from_stream(stream_generator)

                    return ChatCompletionResponse(
                            id=f"chatcmpl-{time.time()}",
                            object="chat.completion",
                            created=int(time.time()),
                            model=request.model,
                            choices=[ChatCompletionChoice(
                                index=0,
                                message=ResponseMessage(
                                    role= Role.assistant,
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
                action.log(
                message_type="CHAT_COMPLETION_ERROR",
                    error=str(e),
                    error_type=type(e).__name__,
                    request_details={
                        "model": request.model,
                        "message_count": len(request.messages),
                        "streaming": request.stream
                    }
                )
                error_response = ErrorResponse(
                    error=ErrorResponse.ErrorDetails(
                        message=str(e)
                    )
                )
                return error_response

    def list_models(self) -> ModelList:
        """List the available models."""
        models = []
        # Create a Model object for each agent in self.agents
        for agent_name in self.agents:
            models.append(Model(
                id=agent_name,  # Uses the agent name as the model ID
                created=int(time.time()),
                object="model",
                owned_by="organization"
            ))
        print(models)

        
        return ModelList(data=models, object="list")

