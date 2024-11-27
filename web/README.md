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

```python
from just_agents_web import create_app
from just_agents.simple.chat_agent import ChatAgent

agent = ChatAgent(...)
app = create_app(agent)
```

For detailed documentation and examples, visit our [main repository](https://github.com/longevity-genie/just-agents).
