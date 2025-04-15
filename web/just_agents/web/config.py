import os
from pydantic import BaseModel, Field
from typing import Optional

class WebAgentConfig(BaseModel):
    """
    Configuration for agent server settings loaded from environment variables.
    
    Provides flexible configuration options for running a Just-Agent server, 
    with defaults and environment variable overrides.
    """
    host: str = Field(
        default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"),
        description="Host address to bind the server",
        examples=["0.0.0.0", "127.0.0.1"]
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("APP_PORT", "8088")),
        description="Port number for the server",
        ge=1024,  # Recommended to use ports above 1024 for non-root users
        le=65535,
        examples=[8088, 8000, 5000]
    )
    workers: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_WORKERS", "1")),
        description="Number of worker processes to run",
        ge=1,
        examples=[1, 2, 4]
    )
    title: str = Field(
        default_factory=lambda: os.getenv("AGENT_TITLE", "Just-Agent endpoint"),
        description="Title of the API endpoint",
        examples=["Just-Agent endpoint", "My AI Service"]
    )
    section: Optional[str] = Field(
        default_factory=lambda: os.getenv("AGENT_SECTION"),
        description="Specific configuration section to load",
        examples=["production", "development", "testing"]
    )
    parent_section: Optional[str] = Field(
        default_factory=lambda: os.getenv("AGENT_PARENT_SECTION"),
        description="Parent configuration section for inheritance",
        examples=["base_config", "global_settings"]
    )
    agent_failfast: bool = Field(
        default_factory=lambda: os.getenv("AGENT_FAILFAST", "true").lower() == "true",
        description="Fail multiple agents loading on any error",
        examples=[True, False]
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("AGENT_DEBUG", "true").lower() == "true",
        description="Enable debug mode for additional logging and error details",
        examples=[True, False]
    )
    remove_system_prompt: bool = Field(
        default_factory=lambda: os.getenv("AGENT_REMOVE_SYSTEM_PROMPT", "false").lower() == "true",
        description="Option to remove system prompts from agent interactions",
        examples=[False, True]
    )
    agent_config_path: str = Field(
        default_factory=lambda: os.getenv('AGENT_CONFIG_PATH', 'agent_profiles.yaml'),
        description="Path to the agent configuration file",
        examples=['agent_profiles.yaml', 'config/agent_profiles.yaml']
    )
    env_keys_path: str = Field(
        default_factory=lambda: os.getenv('ENV_KEYS_PATH', "env/.env.keys"),
        description="Path to environment keys file",
        examples=["env/.env.secrets", "config/.env.keys"]
    )
    app_dir: str = Field(
        default_factory=lambda: os.getenv('APP_DIR', "/app"),
        description="Path to temporary directory",
        examples=["/app", "/opt/app"]
    )
    tmp_dir: str = Field(
        default_factory=lambda: os.getenv('TMP_DIR', "tmp"),
        description="Path to temporary directory",
        examples=["tmp", "/temp"]
    )
    log_dir: str = Field(
        default_factory=lambda: os.getenv('LOG_DIR', "logs"),
        description="Path to log directory",
        examples=["logs", "/app/logs"]
    )
    data_dir: str = Field(
        default_factory=lambda: os.getenv('DATA_DIR', "data"),
        description="Path to log directory",
        examples=["logs", "/app/data"]
    )
    use_proxy: bool = Field(
        default_factory=lambda: os.getenv('AGENT_USE_PROXY', "false").lower() == "true",
        description="Whether to use a proxy to connect to the internet",
        examples=[True, False]
    )
    proxy_address: str = Field(
        default_factory=lambda: os.getenv('AGENT_PROXY_ADDRESS', "http://172.17.0.1:4000/v1"),
        description="The address of the proxy to use",
        examples=["http://172.17.0.1:4000/v1", "http://localhost:4000/v1"]
    )
    security_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv('SECURITY_API_KEY', None),
        description="The security API key to protect the API from unauthorized access",
        examples=["None","security_api_key"]
    )
    


class ChatUIAgentConfig(WebAgentConfig):
    """
    Configuration for Chat UI Agent settings loaded from environment variables.
    
    Extends WebAgentConfig to include additional settings specific to Chat UI Agents.
    """
    models_dir: str = Field(
        default_factory=lambda: os.getenv('MODELS_DIR', "models.d"),
        description="Directory containing model configs",
        examples=["models.d", "configs/models"]
    )

    env_models_path: str = Field(
        default_factory=lambda: os.getenv('ENV_MODELS_PATH', "env/.env.local"),
        description="Path to environment models file specific for Chat-Ui",
        examples=["env/.env.local", "config/.env.models"]
    )
    remove_dd_configs: bool = Field(
        default_factory=lambda: os.getenv('REMOVE_DD_CONFIGS', "true").lower() == "true",
        description="Whether to remove DD configs",
        examples=[True, False]
    )
    trap_summarization: bool = Field(
        default_factory=lambda: os.getenv("TRAP_CHAT_NAMES", "true").lower() == "true",
        description="Whether to trap summarization requests",
        examples=[True, False]
    )
    agent_host: str = Field(
        default_factory=lambda: os.getenv("AGENT_HOST", "http://127.0.0.1"),
        description="Host address for the API",
        examples=["http://127.0.0.1", "http://localhost"]
    )
    agent_port: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_PORT", "8088")),
        description="Port number for the API",
        ge=1024,
        le=65535,
        examples=[8088, 8000, 5000]
    )
    json_file_pattern: str = Field(
        default_factory=lambda: os.getenv("JSON_FILE_PATTERN", "[0123456789][0123456789]_*.json"),
        description="Pattern for JSON files to be removed",
        examples=["[0123456789][0123456789]_*.json"]
    )