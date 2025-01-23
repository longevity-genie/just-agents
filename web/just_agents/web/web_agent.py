from pathlib import Path
import json
from typing import List, ClassVar
from just_agents.base_agent import BaseAgent
from just_agents.web.chat_ui import ModelConfig, ModelParameters, ModelEndpoint, ModelPromptExample
from pydantic import BaseModel, Field, ConfigDict
from just_agents.protocols.openai_streaming import DEFAULT_OPENAI_STOP
class WebAgent(BaseAgent):
    BLUE_SKY: ClassVar[ModelPromptExample] = ModelPromptExample(
                title = "Why is the sky blue?",
                prompt = "Explain in 10 words why the sky is blue"
            )
    DEFAULT_DESCRIPTION: ClassVar[str] = "Generic all-purpose Web AI agent"

    description: str = Field(
        DEFAULT_DESCRIPTION,
        description="Short description of what the agent does")

    examples: List[ModelPromptExample] = Field(
        default_factory=list,
        description="List of model prompt examples"
    )
    enforce_agent_prompt: bool = Field(
        default=False,
        description="Queries containing 'system' messages fall back to completion")
    address:str = Field("http://172.17.0.1", description="Http address of the REST endpoint hosting the agent")
    port: int = Field(8088 ,ge=1000, lt=65535, description="Port of the REST endpoint hosting the agent")


    def compose_model_config(self) -> dict:
        """
        Creates a ModelConfig instance populated with reasonable defaults.
        """
        # Create a default list of prompt examples
        prompt_examples = self.examples or [self.BLUE_SKY]
        # Create a default parameters object
        params = ModelParameters(
            temperature=self.llm_options.get("temperature",0.5),
            max_new_tokens=self.llm_options.get("max_new_tokens",4096),
            stop=self.llm_options.get("stop",[DEFAULT_OPENAI_STOP]),
        )
        # Create a default list of endpoints
        endpoints = [
            ModelEndpoint(
                type="openai",
                baseURL=f"{self.address}:{self.port}/v1",
                apiKey="no_key_needed"
            )
        ]
        # Compose the top-level ModelConfig
        model_config = ModelConfig(
            name=self.class_qualname,
            displayName=self.shortname,
            description=self.description,
            parameters=params,
            endpoints=endpoints,
            promptExamples=prompt_examples
        )

        return model_config.model_dump(
            mode='json',
            exclude_defaults=False,
            exclude_unset=False,
            exclude_none=False,
        )



    def write_model_config_to_json(self, models_dir: Path, filename: str = "00_model_config.json"):
        """
        Writes a sample ModelConfig instance to a JSON file in the specified test directory.

        Args:
            models_dir (Path): Directory where the JSON file will be saved.
            filename (str): Name of the JSON file. Defaults to "model_config.json".

        Returns:
            Path: The path to the written JSON file.
        """
        # Create the sample ModelConfig instance
        model_config = self.compose_model_config()
        models_dir.mkdir(parents=True, exist_ok=True)

        # Define the file path
        file_path = models_dir / filename

        # Write the JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, ensure_ascii=False, indent=4)

        print(f"ModelConfig JSON written to {file_path}")

        return file_path
