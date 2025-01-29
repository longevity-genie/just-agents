from pathlib import Path
from typing import Optional
from just_agents.web.rest_api import AgentRestAPI
import uvicorn
import typer
import os
from pycomfort.logging import to_nice_stdout
from eliot import start_action, start_task

app = typer.Typer()

def run_agent_server(
    config: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 8088,
    workers: int = 1,
    title: str = "Just-Agent endpoint",
    section: Optional[str] = None,
    parent_section: Optional[str] = None,
    debug: bool = True,
    remove_system_prompt: bool = False
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
        remove_system_prompt: Whether to remove system prompt
    """
    to_nice_stdout()

    if config is None:
        config = Path("agent_profiles.yaml")
    
    if not config.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config}. Please provide a valid config file path "
            "or ensure 'agent_profiles.yaml' exists in the current directory."
        )
    
    api = AgentRestAPI(
        agent_config=config,
        title=title,
        agent_section=section,
        agent_parent_section=parent_section,
        debug=debug,
        remove_system_prompt=remove_system_prompt
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
    host: str = typer.Option(os.getenv("APP_HOST", "0.0.0.0"), help="Host to bind the server to"),
    port: int = typer.Option(int(os.getenv("APP_PORT", 8088)), help="Port to run the server on"),
    workers: int = typer.Option(int(os.getenv("AGENT_WORKERS", 1)), help="Number of worker processes"),
    title: str = typer.Option(os.getenv("AGENT_TITLE", "Just-Agent endpoint"), help="Title for the API endpoint"),
    section: Optional[str] = typer.Option(os.getenv("AGENT_SECTION", None), help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(os.getenv("AGENT_PARENT_SECTION", None), help="Optional parent section name in the config file"),
    debug: bool = typer.Option(os.getenv("AGENT_DEBUG", "true").lower() == "true", help="Debug mode"),
    remove_system_prompt: bool = typer.Option(os.getenv("AGENT_REMOVE_SYSTEM_PROMPT", "false").lower() == "true", help="Remove system prompt")
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
            remove_system_prompt=remove_system_prompt
        )

if __name__ == "__main__":
    app()
