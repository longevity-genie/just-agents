import time
from pathlib import Path
from fastapi import FastAPI
from just_agents.interfaces.agent import IAgent
from just_agents.simple.utils import build_agent
from starlette.responses import StreamingResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import loguru
import yaml


"""
Module that can wrap the agent into OpenAI API endpoint
"""

def clean_messages(request: dict):
    for message in request["messages"]:
        if message["role"] == "user":
            content = message["content"]
            if type(content) is list:
                if len(content) > 0:
                    if type(content[0]) is dict:
                        if content[0].get("type", "") == "text":
                            if type(content[0].get("text", None)) is str:
                                message["content"] = content[0]["text"]


def remove_system_prompt(request: dict):
    if request["messages"][0]["role"] == "system":
        request["messages"] = request["messages"][1:]


def get_agent(request):
    with open("endpoint_options.yaml") as f:
        agent_schema = yaml.full_load(f).get(request["model"])

    return build_agent(agent_schema)



def create_app(config_path: Path, title: str = "Just-Agent endpoint") -> FastAPI:
    """
    Create and configure the FastAPI application with the given config path.
    
    Args:
        config_path: Path to the YAML configuration file (str or Path object)
        title: Title for the API endpoint (defaults to "Just-Agent endpoint")
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(title=title)

    load_dotenv(override=True)
    
    # Verify config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {str(config_path)}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Move the config loading here so it's available throughout the app lifetime
    with config_path.open() as f:
        agent_config = yaml.full_load(f)

    def get_agent(request):
        return build_agent(agent_config.get(request["model"]))

    # ... existing clean_messages and remove_system_prompt functions ...

    @app.get("/")
    def default():
        return f"This is default page for the {title}"

    @app.post("/v1/chat/completions")
    def chat_completions(request: dict):
        try:
            loguru.logger.debug(request)
            agent: IAgent = get_agent(request)
            clean_messages(request)
            remove_system_prompt(request)
            if request["messages"]:
                if request.get("stream") and str(request.get("stream")).lower() != "false":
                    return StreamingResponse(
                        agent.stream(request["messages"]), media_type="application/x-ndjson"
                    )
                resp_content = agent.query(request["messages"])
            else:
                resp_content = "Something goes wrong, request did not contain messages!!!"
        except Exception as e:
            loguru.logger.error(str(e))
            resp_content = str(e)
        return {
            "id": "1",
            "object": "chat.completion",
            "created": time.time(),
            "model": request["model"],
            "choices": [{"message": {"role": "assistant", "content": resp_content}}],
        }

    return app

def run_server(config_path: Path, host: str = "0.0.0.0", port: int = 8088, workers: int = 10):
    """
    Run the FastAPI server with the given configuration.
    
    Args:
        config_path: Path to the YAML configuration file (str or Path object)
        host: Host to bind the server to
        port: Port to run the server on
        workers: Number of worker processes
    """
    import uvicorn
    app = create_app(config_path)
    return uvicorn.run(app, host=host, port=port) 
    """TODO: to let workers work we need to have something like this:
      return uvicorn.run(
        "just_agents_web.web:create_app(config_path)", 
        host=host, 
        port=port, 
        workers=workers,
        factory=True
    )
    """