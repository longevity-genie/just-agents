from pathlib import Path
from typing import Optional
from just_agents.web.rest_api import AgentRestAPI
import uvicorn
import typer
from pycomfort.logging import to_nice_stdout
from eliot import start_action, start_task

app = typer.Typer()

def run_server(
    config: Path,
    host: str = "0.0.0.0",
    port: int = 8088,
    workers: int = 1,
    title: str = "Just-Agent endpoint",
    section: Optional[str] = None,
    parent_section: Optional[str] = None
) -> None:
    """
    Run the FastAPI server with the given configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        host: Host to bind the server to
        port: Port to run the server on
        workers: Number of worker processes
        title: Title for the API endpoint
        agent_section: Optional section name in the config file
        agent_parent_section: Optional parent section name in the config file
    """
    to_nice_stdout()
    api = AgentRestAPI(
        agent_config=config,
        title=title,
        agent_section=section,
        agent_parent_section=parent_section,
        debug=True
    )
    
    uvicorn.run(
        api,
        host=host,
        port=port,
        workers=workers
    )

@app.command()
def run_server_command(
    config: Path = typer.Argument(..., help="Path to the YAML configuration file"),
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8088, help="Port to run the server on"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    title: str = typer.Option("Just-Agent endpoint", help="Title for the API endpoint"),
    section: Optional[str] = typer.Option(None, help="Optional section name in the config file"),
    parent_section: Optional[str] = typer.Option(None, help="Optional parent section name in the config file")
) -> None:
    """Run the FastAPI server with the given configuration."""
    with start_task(action_type="run_server"):
        run_server(
            config=config,
            host=host,
            port=port,
            workers=workers,
            title=title,
            section=section,
            parent_section=parent_section
        )

if __name__ == "__main__":
    app()