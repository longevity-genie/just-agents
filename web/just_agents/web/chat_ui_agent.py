from pathlib import Path
from typing import ClassVar, Optional, Dict, Any, Callable, Literal, Type, Union
from just_agents.base_agent import BaseAgent, BaseAgentWithLogging, VariArgs, LogFunction
from pydantic import Field,BaseModel,PrivateAttr
import json
import os
from eliot import start_action
from just_agents.web.chat_ui import ModelParameters, ModelEndpoint, ModelConfig
from just_agents.just_profile import JustAgentProfileWebMixin
from just_agents.web.web_agent import WebAgent

class ChatUIAgent(WebAgent, JustAgentProfileWebMixin):
    """
    A WebChatUIAgent is a WebAgent flavor that can be used in a HuggingFace Chat UI.
    """
    REQUIRED_CLASS: ClassVar[Type['ChatUIAgent']] = None # override to force self-forward-reference as the required class
    DEFAULT_ADDRESS: ClassVar[str] = "http://172.17.0.1"
    # Define the environment variables as a dictionary
    DEFAULT_ENV_VARS: ClassVar[Dict[str, str]] = {
        "MONGODB_URL": "mongodb://genie:super-secret-password@chat-mongo:27017",
        "ALLOW_INSECURE_COOKIES": "true",
        "PUBLIC_APP_NAME": "Just a ChatUI LLM-agent Server",
        "PUBLIC_APP_ASSETS": "chatui",
        "PUBLIC_APP_COLOR": "green",
        "PUBLIC_APP_DESCRIPTION": "A HuggingChat demonstrator of chat + JustAgent",
        "PUBLIC_APP_DATA_SHARING": "1",
        "PUBLIC_APP_DISCLAIMER": "0",
        "MODELS": "\`\n[\n]\n\`"
    }

    address: str = Field(DEFAULT_ADDRESS, description="Http address of the REST endpoint hosting the agent")
    port: int = Field(8088, ge=1000, lt=65535, description="Port of the REST endpoint hosting the agent")

    def compose_model_config(self, proxy_address: str = None) -> dict:
        """
        Creates a ModelConfig instance populated with reasonable defaults.
        """
        # Create a default list of prompt examples
        prompt_examples = self.examples or [self.DEFAULT_PROMPT_EXAMPLE]
        # Create a default parameters object
        params = ModelParameters(
            temperature=self.llm_options.get("temperature", 0.0),
            max_new_tokens=self.llm_options.get("max_new_tokens", 4096),
            stop=self.llm_options.get("stop", [self._protocol.stop]),
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

    def write_model_config_to_json(self, models_dir: Union[Path, str], filename: str = None, index_override: int = None):
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
            if isinstance(models_dir, str):
                models_dir = Path(models_dir).resolve().absolute()
            model_config = self.compose_model_config()
            models_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(models_dir, 0o777)

            # Define the file path
            if filename is None:
                index = index_override or self.assistant_index or 99
                filename = f"{index:02d}_{self.shortname}_config.json"
            file_path = models_dir / filename

            # Write the JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=4)
            os.chmod(models_dir, 0o777)

            action.log(message_type="model_config.write", file_path=str(file_path))

        return file_path




