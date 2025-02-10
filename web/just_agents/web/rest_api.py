import base64
import hashlib
import mimetypes
import os
import time
import json
import glob


from pathlib import Path
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, List, Dict, Any, Union, Type, ClassVar

from just_agents.base_agent import BaseAgent
from just_agents.web.models import Model, ModelList
from just_agents.web.chat_ui import ModelConfig
from just_agents.web.web_agent import WebAgent
from just_agents.web.chat_ui_agent import ChatUIAgent
from just_agents.web.config import WebAgentConfig, ChatUIAgentConfig
from just_agents.web.streaming import response_from_stream, get_completion_response, async_wrap, has_system_prompt

from just_agents.web.models import (
     ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage, ErrorResponse
)
from dotenv import load_dotenv, find_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
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
        
        self.agents = {}  # Dictionary to store multiple agents
        self._agent_related_config(agent_config, agent_section, agent_parent_section, self.AGENT_CLASS)
        self._routes_config()

    def _initialize_config(self):
        self.config = WebAgentConfig()

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
            
            # Load all agents using from_yaml_dict
           
            if agent_section:
                # Load single agent
                self.agents: Dict[str, BaseAgent] = agent_class.from_yaml_dict(agent_config, agent_parent_section, section=agent_section) # requirements-aware autoload

                if agent_section in self.agents:
                    self.agent = self.agents[agent_section]
                    self.agents = {agent_section:self.agent}
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
                self.agents: Dict[str, BaseAgent] = agent_class.from_yaml_dict(agent_config, agent_parent_section) # requirements-aware autoload
                if not self.agents:
                    action.log(
                        message_type="No agents loaded",
                        action="error_no_agents_loaded"
                    )
                    raise ValueError("No agents loaded successfully")

                for agent in self.agents.values():
                    #agent.enforce_agent_prompt = self.remove_system_prompt
                    action.log(
                        message_type=f"Added agent",
                        name=agent.shortname,
                        action="agent_load_success"
                    )

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
    async def chat_completions(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, Any, ErrorResponse]:
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
                stream_generator = agent.stream(
                    request.messages
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
                    response_content = response_from_stream(stream_generator)
                    return get_completion_response(
                        model=request.model,
                        text=response_content,
                        usage = ChatCompletionUsage(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0
                        )
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

class ChatUIAgentRestAPI(AgentRestAPI):
    AGENT_CLASS: ClassVar[Type[BaseAgent]] = ChatUIAgent

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ChatUIAgentRestAPI with FastAPI parameters."""
        # Initialize ChatUIAgentConfig
        super().__init__(*args, **kwargs)
        self._prepare_model_jsons()
        self._prepare_dot_env()

    def _initialize_config(self):
        super()._initialize_config()
        self.config = ChatUIAgentConfig()
        if Path(self.config.env_keys_path).resolve().absolute().exists():
            load_dotenv(self.config.env_keys_path, override=True)

    def get_override_agent(self, request: ChatCompletionRequest, selected_agent: Optional[WebAgent], available_models: List[str]) -> Optional[WebAgent]:
        with start_task(action_type="get_override_agent") as action:
            override_agent = super().get_override_agent(request, selected_agent, available_models)
            if self.config.trap_summarization:
                prompt = has_system_prompt(request)
                if prompt and prompt.startswith(
                    "You are a summarization AI. Summarize the user's request into a single short sentence of four words or less."
                ):
                    override_agent = self.agents.get("chat_naming_agent")
                    action.log(
                        message_type=f"Agent overridden for chat naming request to {request.model} with {str(override_agent)}",
                        available_models=available_models,
                        action="agent_override"
                    )
            return override_agent

    def _prepare_model_jsons(self):
        with start_task(action_type="prepare_model_jsons") as action:
            # Remove unlisted config files if flag is set
            if self.config.remove_dd_configs:
                for config_file in Path(self.config.models_dir).glob(self.config.json_file_pattern):
                    config_file.unlink()
            for agent in self.agents.values():
                agent.address = self.config.agent_host or "127.0.0.1"
                agent.port = self.config.agent_port or 8088
                agent.write_model_config_to_json(models_dir=Path(self.config.models_dir))
                action.log(
                    message_type=f"Config saved for agent",
                    displayname=agent.display_name,
                    name=agent.shortname,
                    address=agent.address,
                    port=agent.port,
                    enforce_prompt=agent.enforce_agent_prompt,
                    action="agent_config_success"
            )

    def _prepare_env_file(self, env_file_name: str) -> List[str]:
        """
        Ensures that the .env file exists at the specified path. If it doesn't exist,
        creates it with the provided environment variables.

        """
        with start_task(action_type="prepare_env_file") as action:
            env_file_path = Path(env_file_name).resolve().absolute()

            if not env_file_path.exists():
                try:
                    action.log(
                        message_type=f"Warning: {str(env_file_path)} does not exist.",
                        env_local_path=str(env_file_path),
                        action="error_no_env_file"
                    )
                    with open(env_file_path, 'w') as env_file:
                        for key, value in self.AGENT_CLASS.DEFAULT_ENV_VARS.items():
                                env_file.write(f"{key}={value}\n")
                except Exception as e:
                    action.log(
                        message_type=f"Error: {str(env_file_path)} can not be written.",
                        error=str(e),
                        env_local_path=str(env_file_path),
                        action="error_env_file_not_writable"
                    )
                    raise e
                action.log(
                    message_type=f".env file created at {str(env_file_path)} with default values.",
                    env_local_path=str(env_file_path),
                    action="env_file_created"
                )
            else:
                action.log(
                    message_type=f".env file already exists at {str(env_file_path)}.",
                    env_local_path=str(env_file_path),
                    action="env_file_exists"
                )
            
            # Read the existing .env.local file
            with open(env_file_path, 'r') as f:
                lines = f.readlines()

            return lines

    def _prepare_dot_env(self):
        with start_task(action_type="prepare_dot_env") as action:
            models_dir = self.config.models_dir
            lines = self._prepare_env_file(self.config.env_keys_path)

            # Remove existing MODELS variable (including multi-line definitions)
            new_lines = []
            skip = False

            for line in lines:
                stripped_line = line.strip()
                if not skip:
                    if stripped_line.startswith('MODELS='):
                        if '`' in stripped_line:
                            # Check if it's a single-line MODELS definition
                            if stripped_line.count('`') == 2:
                                continue  # Skip the entire line
                            else:
                                skip = True  # Start skipping lines
                        else:
                            skip = True  # Start skipping lines
                    else:
                        new_lines.append(line)
                else:
                    if '`' in stripped_line:
                        skip = False  # Found closing backtick, stop skipping
                    continue  # Skip lines within MODELS definition

            # Load and validate JSON files from models.d
            model_files = glob.glob(os.path.join(models_dir, '*.json'))
            model_files.sort()  # Sort the list in place based on file names
            models = []

            if not model_files:
                action.log(
                    message_type=f"No JSON files found in {models_dir}.",
                    models_dir=models_dir,
                    action="error_no_json_models"
                )
                raise ValueError(f"No JSON files found in {models_dir}.")

            for model_file in model_files:
                try:
                    with open(model_file, 'r') as f:
                        data = json.load(f)
                        try:
                            ModelConfig.model_validate(data)
                        except ValidationError as e:
                            action.log(
                                message_type="Syntax problem in {model_file}",
                                data=data,
                                error=str(e),
                                action="json_not_validated_error"
                            )
                            raise ValueError(f"Syntax problem in {model_file}")
                        models.append(data)
                except Exception as e:
                    action.log(
                        message_type=f"Error loading {model_file}",
                        model_file=model_file,
                        error=str(e),
                        action="json_not_loaded_error"
                    )
                    raise ValueError(f"Error loading {model_file}: {str(e)}")

            try:
                # Write the updated .env.local file
                with open(self.config.env_keys_path, 'w') as f:
                    f.writelines(new_lines)
                    # Serialize the models list as a JSON string and add it to MODELS=
                    models_json = json.dumps(models, ensure_ascii=False, indent=4)
                    f.write("MODELS=`\n")
                    f.write(f"{models_json}\n")
                    f.write("`\n")
                action.log(
                    message_type=f"Updated .env.local file at {self.config.env_keys_path}",
                    action="env_file_updated"
                )
            except Exception as e:
                action.log(
                    message_type=f"Error writing {self.config.env_keys_path}",
                    error=str(e),
                    action="error_env_file_not_writable"
                )
                raise e
            # Output the list of models using 'displayName' field
            for model in models:
                action.log(
                    message_type=f"- {model.get('displayName')}",
                    action="model_loaded"
                )
