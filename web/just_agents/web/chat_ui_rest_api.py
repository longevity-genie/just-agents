import os
import json
import glob
import asyncio

from pathlib import Path
from pydantic import Field, ValidationError
from typing import Optional, List, Union, Type, ClassVar

from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui import ModelConfig
from just_agents.web.web_agent import WebAgent
from just_agents.web.chat_ui_agent import ChatUIAgent
from just_agents.web.config import ChatUIAgentConfig
from just_agents.web.streaming import response_from_stream, get_completion_response, async_wrap, has_system_prompt

from just_agents.web.models import (
     ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage, ErrorResponse
)

from eliot import start_task
from just_agents.web.rest_api import AgentRestAPI


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
 

    async def get_override_agent(self, request: ChatCompletionRequest, selected_agent: Optional[WebAgent], available_models: List[str]) -> Optional[WebAgent]:
        with start_task(action_type="get_override_agent") as action:
            override_agent = await super().get_override_agent(request, selected_agent, available_models)
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
            
            # Compute index overrides for all agents based on existing values and natural order.
            index_overrides: dict[str, int] = self._compute_index_overrides()

            # Iterate over agents (using items so we can look up the override by shortname)
            for key, agent in self.agents.items():
                if agent.hidden:
                    # Log and skip hidden agents.
                    action.log(
                        message_type="Skipping hidden agent",
                        displayname=agent.display_name,
                        name=agent.shortname,
                        action="agent_config_skip"
                    )
                    continue

                # Set network address/port for the agent.
                agent.address = self.config.agent_host or "127.0.0.1"
                agent.port = self.config.agent_port or 8088

                # Write JSON config using the computed index override.
                agent.write_model_config_to_json(
                    models_dir=Path(self.config.models_dir),
                    index_override=index_overrides[agent.shortname],
                    api_key=self.config.security_api_key
                )

                action.log(
                    message_type="Config saved for agent",
                    displayname=agent.display_name,
                    name=agent.shortname,
                    address=agent.address,
                    port=agent.port,
                    index=index_overrides[agent.shortname],
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
            lines = self._prepare_env_file(self.config.env_models_path)

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
                with open(self.config.env_models_path, 'w') as f:
                    f.writelines(new_lines)
                    # Serialize the models list as a JSON string and add it to MODELS=
                    models_json = json.dumps(models, ensure_ascii=False, indent=4)
                    f.write("MODELS=`\n")
                    f.write(f"{models_json}\n")
                    f.write("`\n")
                action.log(
                    message_type=f"Updated .env.local file at {self.config.env_models_path}",
                    action="env_file_updated"
                )
            except Exception as e:
                action.log(
                    message_type=f"Error writing {self.config.env_models_path}",
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

    def _compute_index_overrides(self) -> dict[str, int]:
        """
        Compute index overrides for agents that will be used when saving
        their model configurations.

        For each agent:
         - If `assistant_index` is already set (not None), then use that value.
         - Otherwise, assign the next available index (starting from 1) while ensuring that
           the index does not collide with an already assigned index.
           If a collision is detected, increment until a free value is found or
           use 99 if the candidate has reached 99.

        Returns:
            A dictionary mapping each agent's shortname to its computed index override.
        """
        from typing import Set
        
        used_indexes: Set[int] = set()  # Set of indexes already assigned (explicitly or by previous agents)
        overrides: dict[str, int] = {}  # Mapping from agent shortname to override index
        
        # Candidate for assignment of agents without an index (must be > 0)
        candidate: int = 1

        # Iterate over agents in natural (dictionary insertion) order
        for key, agent in self.agents.items():
            if agent.assistant_index is not None:
                # Use the agent's configured index and mark it as used
                index_val: int = agent.assistant_index
                used_indexes.add(index_val)
                overrides[agent.shortname] = index_val
          

        # Iterate over agents in natural (dictionary insertion) order
        for key, agent in self.agents.items():
            if agent.assistant_index is  None:        
                # Find the next free candidate index not already used
                while candidate in used_indexes and candidate < 99:
                    candidate += 1
                if candidate >= 99:
                    index_val = 99
                else:
                    index_val: int = candidate
                    candidate += 1  # Increment candidate for the next agent
                used_indexes.add(index_val)
                overrides[agent.shortname] = index_val

        return overrides
