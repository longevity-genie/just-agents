from pathlib import Path
import json
import os
from typing import List, ClassVar, Optional, Dict
from just_agents.just_profile import JustAgentFullProfile
from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui import ModelConfig, ModelParameters, ModelEndpoint, ModelPromptExample
from pydantic import BaseModel, Field, ConfigDict
from just_agents.protocols.openai_streaming import DEFAULT_OPENAI_STOP
import yaml
from eliot import start_action

class WebAgent(BaseAgent, JustAgentFullProfile):
    
    DEFAULT_PROMPT_EXAMPLE: ClassVar[ModelPromptExample] = ModelPromptExample(
                title = "Is aging a disease?",
                prompt = "Explain why biological aging can be classified as a disease"
            )
    DEFAULT_DESCRIPTION: ClassVar[str] = "Generic all-purpose Web AI agent"
    DEFAULT_DISPLAY_NAME: ClassVar[str] = "🦙 A simple Web AI agent"
    DEFAULT_ADDRESS: ClassVar[str] = "http://172.17.0.1"

    description: str = Field(
        DEFAULT_DESCRIPTION,
        description="Short description of what the agent does")

    display_name: Optional[str] = Field(
        DEFAULT_DISPLAY_NAME,
        description="A fancy one-line name of the agent, replaces shortname in UIs, may include spaces, emoji and other stuff")

    enforce_agent_prompt: bool = Field(
        default=False,
        description="Queries containing 'system' messages fall back to completion")

    continue_conversation: bool = Field(
        default=False,
        description="Concatenate memory messages and query messages ")

    remember_query: bool = Field(
        default=False,
        description="Add new query messages to memory")

    assistant_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Non-negative value that specifies model position in Chat UI models list, highest is default")

    address: str = Field(DEFAULT_ADDRESS, description="Http address of the REST endpoint hosting the agent")
    port: int = Field(8088 ,ge=1000, lt=65535, description="Port of the REST endpoint hosting the agent")


    def compose_model_config(self, proxy_address: str = None) -> dict:
        """
        Creates a ModelConfig instance populated with reasonable defaults.
        """
        # Create a default list of prompt examples
        prompt_examples = self.examples or [self.BLUE_SKY]
        # Create a default parameters object
        params = ModelParameters(
            temperature=self.llm_options.get("temperature",0.0),
            max_new_tokens=self.llm_options.get("max_new_tokens",4096),
            stop=self.llm_options.get("stop",[DEFAULT_OPENAI_STOP]),
        )
        # Create a default list of endpoints
        if proxy_address:
            baseurl = proxy_address
        else:
            baseurl = f"{self.address}:{self.port}/v1"
        endpoints = [
            ModelEndpoint(
                type="openai",
                baseURL=baseurl
            )
        ]
        # Compose the top-level ModelConfig
        model_config = ModelConfig(
            name=self.shortname,
            displayName=self.display_name or self.shortname,
            description=self.description,
            parameters=params,
            endpoints=endpoints,
            promptExamples=prompt_examples
        )

        return model_config.model_dump(
            mode='json',
            exclude_defaults=False,
            exclude_unset=False,
            exclude_none=True,
        )


    
    def write_model_config_to_json(self, models_dir: Path, filename: str = None):
        """
        Writes a sample ModelConfig instance to a JSON file in the specified test directory.

        Args:
            models_dir (Path): Directory where the JSON file will be saved.
            filename (str): Name of the JSON file. Defaults to "model_config.json".

        Returns:
            Path: The path to the written JSON file.
        """
        with start_action(action_type="model_config.write") as action:
        # Create the sample ModelConfig instance
            model_config = self.compose_model_config()
            models_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(models_dir, 0o777)

            # Define the file path
            if filename is None:
                filename = f"{self.assistant_index:02d}_{self.shortname}_config.json"
            file_path = models_dir / filename

            # Write the JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=4)
            os.chmod(models_dir, 0o777)

            action.log(message_type="model_config.write", file_path=str(file_path))

        return file_path

    @staticmethod
    def from_yaml_dict(
        yaml_path: Path | str,
        parent_section: Optional[str] = "agent_profiles"
    ) -> Dict[str, 'WebAgent']:
        """
        Creates a dictionary of WebAgent instances from a YAML file.
        """
        with start_action(action_type="agent.load") as action:
            if isinstance(yaml_path, str):
                yaml_path = Path(yaml_path)

            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")

            with yaml_path.open('r') as f:
                config_data = yaml.safe_load(f) or {}

            agents = {}
            
            # Get the correct section data
            if parent_section:
                sections = config_data.get(parent_section, {})
            else:
                # If no parent_section specified, try common section names or use root
                if "agent_profiles" in config_data:
                    sections = config_data["agent_profiles"]
                    parent_section = "agent_profiles"
                elif "agents" in config_data:
                    sections = config_data["agents"]
                    parent_section = "agents"
                else:
                    sections = config_data

            # Process each section
            for section_name, section_data in sections.items():
                agent = WebAgent.from_yaml(
                    section_name,
                    parent_section,
                    yaml_path
                )
                agents[section_name] = agent
                action.log(message_type="agent.load", section_name=section_name, parent_section=parent_section)

            return agents
