from pathlib import Path
from typing import Optional, Type
from just_agents.web.config import ChatUIAgentConfig
from just_agents.web.rest_api import AgentRestAPI
from just_agents.web.chat_ui_rest_api import ChatUIAgentRestAPI
import uvicorn
import typer
from eliot import start_action, start_task

env_config = ChatUIAgentConfig()
app = typer.Typer()

def validate_agent_config(
    config: Optional[Path] = None, 
    section: Optional[str] = None, 
    parent_section: Optional[str] = None,
    api_class: Type[AgentRestAPI] = AgentRestAPI,
    debug: bool = True,
) -> AgentRestAPI:
    """
    Validate the agent configuration and return an AgentRestAPI instance.
    
    Args:
        config: Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory
        section: Optional section name in the config file
        parent_section: Optional parent section name in the config file
        debug: Debug mode
        api_class: AgentRestAPI or ChatUIAgentRestAPI
        
    Returns:
        AgentRestAPI: Validated API instance
    """
    if config is None:
        config = Path("agent_profiles.yaml")
    
    if not config.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config}. Please provide a valid config file path "
            "or ensure 'agent_profiles.yaml' exists in the current directory."
        )
    
    return api_class(
        agent_config=config,
        title="Just-Agent endpoint",
        agent_section=section,
        agent_parent_section=parent_section,
        debug=debug
    )

def run_agent_server(
    config: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 8088,
    workers: int = 1,
    title: str = "Just-Agent endpoint",
    section: Optional[str] = None,
    parent_section: Optional[str] = None,
    debug: bool = True,
    api_class: Type[AgentRestAPI] = AgentRestAPI
) -> None:
    """
    Run the FastAPI server with the given configuration.
    
    Args:
        config: Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory
        host: Host to bind the server to
        port: Port to run the server on
        workers: Number of worker processes
        title: Title for the API endpoint
        section: Optional section name in the config file
        parent_section: Optional parent section name in the config file
        debug: Debug mode
        api_class: AgentRestAPI or ChatUIAgentRestAPI

    """
    #to_nice_stdout()

    # Initialize the API class with the updated configuration
    api = api_class(
        agent_config=config,
        agent_section=section,
        agent_parent_section=parent_section,
        debug=debug,
        title=title
    )
    
    uvicorn.run(
        api,
        host=host,
        port=port,
        workers=workers
    )

@app.command()
def run_server_command(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory"
    ),
    host: str = typer.Option(env_config.host, help="Host to bind the server to"),
    port: int = typer.Option(env_config.port, help="Port to run the server on"),
    workers: int = typer.Option(env_config.workers, help="Number of worker processes"),
    title: str = typer.Option(env_config.title, help="Title for the API endpoint"),
    section: Optional[str] = typer.Option(env_config.section, help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(env_config.parent_section, help="Optional parent section name in the config file"),
    debug: bool = typer.Option(env_config.debug, help="Debug mode"),

) -> None:
    """Run the FastAPI server with the given configuration."""
    with start_task(action_type="run_agent_server"):
        run_agent_server(
            config=config,
            host=host,
            port=port,
            workers=workers,
            title=title,
            section=section,
            parent_section=parent_section,
            debug=debug,
            api_class=AgentRestAPI
        )

@app.command()
def validate_config(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory"
    ),
    section: Optional[str] = typer.Option(env_config.section, help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(env_config.parent_section, help="Optional parent section name in the config file"),
    debug: bool = typer.Option(env_config.debug, help="Debug mode"),

) -> None:
    """Validate the agent configuration without starting the server."""
    with start_action(action_type="validate_agent_config.write") as action:
        validate_agent_config(
            config=config,
            section=section,
            parent_section=parent_section,
            debug=debug,
            api_class=AgentRestAPI
        )
        action.log(
            message_type=f"Configuration validation successful!",
            action="validate_config_success"
        )

@app.command()
def run_chat_ui_server_command(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory"
    ),
    host: str = typer.Option(env_config.host, help="Host to bind the server to"),
    port: int = typer.Option(env_config.port, help="Port to run the server on"),
    workers: int = typer.Option(env_config.workers, help="Number of worker processes"),
    title: str = typer.Option(env_config.title, help="Title for the API endpoint"),
    section: Optional[str] = typer.Option(env_config.section, help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(env_config.parent_section, help="Optional parent section name in the config file"),
    debug: bool = typer.Option(env_config.debug, help="Debug mode"),

) -> None:
    """Run the FastAPI server for ChatUIAgentRestAPI with the given configuration."""
    run_agent_server(
        config=config,
        host=host,
        port=port,
        workers=workers,
        title=title,
        section=section,
        parent_section=parent_section,
        debug=debug,
        api_class=ChatUIAgentRestAPI
    )

@app.command()
def validate_chat_ui_config(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to the YAML configuration file. Defaults to 'agent_profiles.yaml' in current directory"
    ),
    section: Optional[str] = typer.Option(env_config.section, help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(env_config.parent_section, help="Optional parent section name in the config file"),
    debug: bool = typer.Option(env_config.debug, help="Debug mode"),
) -> None:
    """Validate the ChatUIAgentRestAPI configuration without starting the server."""
    validate_agent_config(
        config=config,
        section=section,
        parent_section=parent_section,
        debug=debug,
        api_class=ChatUIAgentRestAPI
    )

if __name__ == "__main__":
    # Run the Typer app which will show help by default
    app()
