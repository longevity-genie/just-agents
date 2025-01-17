# just-agents-web

A web API wrapper for just-agents that provides OpenAI-compatible endpoints. This package allows you to expose just-agents functionality through a REST API that follows OpenAI's API conventions, making it easy to integrate with existing applications that use OpenAI's API format.

## Installation

```bash
pip install just-agents-web
```

## Features

- 🔄 OpenAI-compatible REST API endpoints
- 🤖 Wrap any just-agents agent as an API service
- 🔌 Drop-in replacement for OpenAI API clients
- 🛠️ Built on just-agents-core

## Dependencies

- just-agents-core
- FastAPI
- pydantic

## Quick Start

The project subpublish run-agent script, so you can run any agent yaml by running:

```bash
run-agent path/to/agent.yaml
```

You can also do it with few lines of python code:
```python
from just_agents.web.run_agent import run_agent_server

run_agent_server(Path("agent_profiles.yaml"))
```

We also provide AgentRestAPI class that you can extend and add your methods to agent endpoint.