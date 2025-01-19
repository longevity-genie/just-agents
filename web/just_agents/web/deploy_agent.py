import typer
import subprocess
import shutil
from pathlib import Path

app = typer.Typer()

@app.command()
def deploy(
    agent_file: Path,
    name: str = "chat-server",
    output_path: Path = typer.Option(
        Path("./"), "--output", "-o", 
        help="Path where to clone the repository"
    )
):
    """
    Clone the chat-server repository and set up configuration files

    Args:
        agent_file: Path to the agent profiles file
        name: Name of the target folder (defaults to "chat-server")
        output_path: Path where to clone the repository (defaults to "./")
    """
    try:
        # Verify agent_file exists
        if not agent_file.exists():
            typer.echo(f"Error: Agent file {agent_file} not found", err=True)
            raise typer.Exit(1)

        # Construct full folder path
        folder_name = output_path / name
        
        # Clone the repository
        typer.echo(f"Cloning chat-server repository into {folder_name}...")
        subprocess.run(
            ["git", "clone", "https://github.com/longevity-genie/chat-server.git", str(folder_name)],
            check=True
        )

        # Change to the cloned directory
        repo_path = folder_name
        if not repo_path.exists():
            raise typer.Exit("Repository directory not found after cloning")

        # Copy configuration files
        config_files = [
            (".env.local.template", ".env.local"),
            (".env.proxy.template", ".env.proxy"),
            (".env.db.template", ".env.db")
        ]

        for template, target in config_files:
            template_path = repo_path / template
            target_path = repo_path / target
            
            if template_path.exists():
                shutil.copy2(template_path, target_path)
                typer.echo(f"Created {target} from template")
            else:
                typer.echo(f"Warning: Template file {template} not found")

        # Copy agent file to the repository
        target_agent_file = repo_path / agent_file.name
        shutil.copy2(agent_file, target_agent_file)
        typer.echo(f"Copied {agent_file.name} to repository")

        typer.echo("Setup completed successfully!")

    except subprocess.CalledProcessError as e:
        typer.echo(f"Error during git clone: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during setup: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
