import base64
import hashlib
import mimetypes
import time

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Type, ClassVar
from just_agents.just_bus import JustLogBus
from just_agents.base_agent import BaseAgent
from just_agents.web.models import Model, ModelList
from just_agents.web.web_agent import WebAgent
from just_agents.web.config import WebAgentConfig
from just_agents.web.streaming import (
    response_from_stream, 
    get_completion_response, 
    async_wrap, 
    has_system_prompt
)

from just_agents.web.models import (
     ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage, ErrorResponse
)
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from eliot import start_task




class AgentRestAPI(FastAPI):
    """FastAPI implementation providing OpenAI-compatible endpoints for Just-Agents.
    This class extends FastAPI to provide endpoints that mimic OpenAI's API structure,
    allowing Just-Agents to be used as a drop-in replacement for OpenAI's API.
    """
    AGENT_CLASS: ClassVar[Type[BaseAgent]] = WebAgent

    def __init__(
        self,
        *,
        agent_config: Optional[Path | str] = None,  # Path to agent configuration file
        agent_section: Optional[str] = None,        # Specific section in config to load
        agent_parent_section: Optional[str] = None, # Parent section for inheritance
        debug: bool = False,                       # Enable debug mode
        title: str = "Just-Agent endpoint",        # API title for documentation
        description: str = "OpenAI-compatible API endpoint for Just-Agents",
        version: str = "1.1.0",
        openapi_url: str = "/openapi.json",
        openapi_tags: Optional[List[Dict[str, Any]]] = None,
        servers: Optional[List[Dict[str, Union[str, Any]]]] = None,
        docs_url: str = "/docs",
        redoc_url: str = "/redoc",
        terms_of_service: Optional[str] = None,
        contact: Optional[Dict[str, Union[str, Any]]] = None,
        license_info: Optional[Dict[str, Union[str, Any]]] = None,
        agents: Optional[Dict[str, BaseAgent]] = None, # We can set up agents explicitly here instead of loading them from yaml
        use_proxy:Optional[bool] = None,
        proxy_address:Optional[str] = None,

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

        # Initialize WebAgentConfig
        self._initialize_config()
        if use_proxy is not None:
            self.config.use_proxy = use_proxy

        if proxy_address is not None:
            self.config.proxy_address = proxy_address

        self.agents = {} if agents is None else agents # Dictionary to store multiple agents
        self._agent_related_config(agent_config, agent_section, agent_parent_section, self.AGENT_CLASS)
        self._routes_config()

    def _initialize_config(self):
        self.config = WebAgentConfig()
        if Path(self.config.env_keys_path).resolve().absolute().exists():
            load_dotenv(self.config.env_keys_path, override=True)

    def _agent_related_config(
            self, 
            agent_config: Path | str, 
            agent_section: Optional[str] = None, 
            agent_parent_section: Optional[str] = None,
            agent_class: Optional[Type[BaseAgent]] = None
        ):
        with start_task(action_type="agent_related_configs") as action:
            if agent_class is None:
                agent_class = WebAgent

            if agent_config is None:
                # Load from environment variable or use default
                agent_config = self.config.agent_config_path
            
            # Store loaded agents separately
            loaded_agents: Dict[str, BaseAgent] = {}
            
            # Check if config file exists - handle both string and Path types
            agent_path = agent_config if isinstance(agent_config, Path) else Path(agent_config)
            agent_path = agent_path.resolve()
            if not agent_path.exists():
                action.log(
                    message_type="Config file not found",
                    path=str(agent_path),
                    action="config_not_found"
                )
                if self.agents == {}:
                    action.log(
                        message_type="No agents loaded",
                        action="error_no_agents_loaded"
                    )
                    raise ValueError("No agents loaded neither from config file nor from explicit agents")
                return  # Return early if no config file exists
           
            loaded_agents = agent_class.from_yaml_dict(
                agent_config, 
                agent_parent_section, 
                section=agent_section,
                fail_on_any_error=self.config.agent_failfast,
                use_proxy=self.config.use_proxy,
                proxy_address=self.config.proxy_address,
            )

            if not loaded_agents:
                action.log(
                    message_type="No agents loaded",
                    action="error_no_agents_loaded"
                )
                raise ValueError("No agents loaded successfully")

            if agent_section:
                # Load single agent
                if agent_section in loaded_agents:
                    self.agent = loaded_agents[agent_section]
                    loaded_agents = {agent_section: self.agent}
                    action.log(
                        message_type=f"Single agent successfully loaded and selected",
                        name=self.agent.shortname,
                        action="agent_load_success"
                    )
                else:
                    action.log(
                        message_type=f"Requested agent {agent_section} not loaded, cause in previous messages",
                        name=agent_section,
                        action="agent_load_error",
                    )
                    raise ValueError(f"Requested agent {agent_section} not loaded due to errors")
            else:
                for agent in loaded_agents.values():
                    action.log(
                        message_type=f"Added agent",
                        name=agent.shortname,
                        action="agent_load_success"
                    )

            # Merge loaded agents with existing agents
            self.agents.update(loaded_agents)

            # Set the first agent as default if no default agent exists
            if not hasattr(self, 'agent') and self.agents:
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


    def default(self) -> str:
        return f"This is default page for the {self.title}"

    def sha256sum(self, content_str: str) -> str:
        hash = hashlib.sha256()
        hash.update(content_str.encode('utf-8'))
        return hash.hexdigest()

    def save_files(self, request: dict) -> None:
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

    async def preprocess_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if "file_params" in request:
            params = request.file_params
            if params != []:
                self.save_files(request.model_dump())
        return request
    
    async def get_override_agent(self, request: ChatCompletionRequest, selected_agent:Optional[WebAgent], available_models: List[str]) -> Optional[WebAgent]:
        with start_task(action_type="get_override_agent") as action:
            override_agent = None
            if not selected_agent:
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
                    override_agent = next(iter(self.agents.values()))
           
            return override_agent 


#    @log_call(action_type="chat_completions", include_result=False)
    async def chat_completions(self, request: ChatCompletionRequest, authorization: Optional[str] = Header(None)) -> Union[ChatCompletionResponse, Any, ErrorResponse]:
        with start_task(action_type="chat_completions") as action:
            try:
                action.log(
                    message_type=f"chat_completions for {request.model}",
                    request = str(request.model_dump())
                )
                request = await self.preprocess_request(request)
                # Get the agent based on the model name
                available_models = list(self.agents.keys())
                selected_agent = self.agents.get(request.model)

                agent = await self.get_override_agent(request, selected_agent, available_models) or selected_agent
                action.log(
                    message_type=f"Agent selected: {str(selected_agent)}",
                    requested_agent=request.model,
                    selected_agent=selected_agent,
                    override_agent=agent,
                    available_models=available_models,
                    action="agent_select"
                )

                is_streaming = request.stream
                
                # Extract API key from Authorization header if available
                header_api_key: Optional[str] = None
                if authorization and authorization.startswith("Bearer "):
                    header_api_key = authorization.replace("Bearer ", "").strip()
                
                # Use header API key first, fallback to request API key if present
                api_key = header_api_key or request.api_key

                input_kwargs : dict = {}

                action.log(
                    message_type=f"Completion api_key",
                    request_api_key=JustLogBus.mask_api_key(request.api_key),
                    header_api_key=JustLogBus.mask_api_key(header_api_key),
                    security_key=JustLogBus.mask_api_key(self.config.security_api_key),
                    keys_match=(api_key == self.config.security_api_key),
                    action="completion_auth",
                )

                if self.config.security_api_key:
                    if api_key != self.config.security_api_key: #only accept requests with the correct security key
                        raise ValueError("Invalid API key")
                else:
                    if api_key and len(api_key) > 8: # drop "sk-proj-" and other empty prefixes
                        input_kwargs["api_key"] = api_key #accept any requests, but if api_key is provided, add it to the input

                if is_streaming:
                    stream_generator = agent.stream(
                        request.messages,
                        **input_kwargs
                    )
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
                    # response_content = response_from_stream(stream_generator, request.stop)
                    # return get_completion_response(
                    #     model=request.model,
                    #     text=response_content,
                    #     usage = ChatCompletionUsage(
                    #         prompt_tokens=0,
                    #         completion_tokens=0,
                    #         total_tokens=0
                    #     )
                    # )
                    return agent.query(
                        request.messages,
                        **input_kwargs
                    )

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

