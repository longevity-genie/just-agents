# just-agents
[![Python CI](https://github.com/longevity-genie/just-agents/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/longevity-genie/just-agents/actions/workflows/run_tests.yaml)
[![PyPI version](https://badge.fury.io/py/just-agents-core.svg)](https://badge.fury.io/py/just-agents-core)
[![Documentation Status](https://readthedocs.org/projects/just-agents/badge/?version=latest)](https://just-agents.readthedocs.io/en/latest/?badge=latest)

A lightweight, straightforward library for LLM agents - no over-engineering, just simplicity!

## Quick Start
```bash
pip install just-agents-core
```

WARNING: we are reorganizing the package structure right now so the published package is a bit different than the code in this repository.

## 🎯 Motivation

Most of the existing agentic libraries are extremely over-engineered either directly or by using over-engineered libraries under the hood, like langchain and llamaindex.
In reality, interactions with LLMs are mostly about strings, and you can write your own template by just using f-strings and python native string templates.
There is no need in complicated chain-like classes and other abstractions, in fact popular libraries create complexity just to sell you their paid services for LLM calls monitoring because it is extremely hard to understand what exactly is sent to LLMs.

It is way easier to reason about the code if you separate your prompting from python code to a separate easily readable files (like yaml files).

We wrote this libraries while being pissed of by high complexity and wanted something controlled and simple.
Of course, you might comment that we do not have the ecosystem like, for example, tools and loaders. In reality, most of langchain tools are just very simple functions wrapped in their classes, you can always quickly look at them and write a simple function to do the same thing that just-agents will pick up easily.

## ✨ Key Features
- 🪶 Lightweight and simple implementation
- 📝 Easy-to-understand agent interactions
- 🔧 Customizable prompts using YAML files
- 🤖 Support for various LLM models through litellm
- 🔄 Chain of Thought reasoning with function calls

## 📚 Documentation & Tutorials

### Interactive Tutorials (Google Colab)
- [Basic Agents Tutorial](https://github.com/longevity-genie/just-agents/blob/main/examples/notebooks/01_just_agents_colab.ipynb)
- [Database Agent Tutorial](https://github.com/longevity-genie/just-agents/blob/main/examples/notebooks/02_sqlite_example.ipynb)
- [Coding Agent Tutorial](https://github.com/longevity-genie/just-agents/blob/main/examples/notebooks/03_coding_agent.ipynb)

### Example Code
Browse our [examples](https://github.com/longevity-genie/just-agents/tree/main/examples) directory for:
- 🔰 Basic usage examples
- 💻 Code generation and execution
- 🛠️ Tool integration examples
- 👥 Multi-agent interactions


## 🚀 Installation

### Quick Install
```bash
pip install just-agents-core
```

### Development Setup
1. Clone the repository:
```bash
git clone git@github.com:longevity-genie/just-agents.git
cd just-agents
```

2. Set up the environment:
We use Poetry for dependency management. First, [install Poetry](https://python-poetry.org/docs/#installation) if you haven't already.

```bash
# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
```
## 🏗️ Architecture

### Core Components
1. **BaseAgent**: A thin wrapper around litellm for LLM interactions
3. **ChainOfThoughtAgent**: Extended agent with reasoning capabilities


### ChatAgent

The `ChatAgent` class is the core of our library. 
It represents an agent with a specific role, goal, and task. Here's an example of a moderated debate between political figures:

```python
from just_agents.base_agent import ChatAgent
from just_agents.llm_options import LLAMA3_3, LLAMA3_3_specdec:

# Initialize agents with different roles
Harris = ChatAgent(
    llm_options=LLAMA3_3, 
    role="You are Kamala Harris in a presidential debate",
    goal="Win the debate with clear, concise responses",
    task="Respond briefly and effectively to debate questions"
)

Trump = ChatAgent(
    llm_options=LLAMA3_3_specdec,
    role="You are Donald Trump in a presidential debate",
    goal="Win the debate with your signature style",
    task="Respond briefly and effectively to debate questions"
)

Moderator = ChatAgent(
    llm_options={
        "model": "groq/mixtral-8x7b-32768",
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "tools": []
    },
    role="You are a neutral debate moderator",
    goal="Ensure a fair and focused debate",
    task="Generate clear, specific questions about key political issues"
)

exchanges = 2

# Run the debate
for _ in range(exchanges):
    question = Moderator.query("Generate a concise debate question about a current political issue.")
    print(f"\nMODERATOR: {question}\n")

    trump_reply = Trump.query(question)
    print(f"TRUMP: {trump_reply}\n")

    harris_reply = Harris.query(f"Question: {question}\nTrump's response: {trump_reply}")
    print(f"HARRIS: {harris_reply}\n")

# Get debate summary
debate = str(Harris.memory.messages)
summary = Moderator.query(f'Summarise the following debate in less than 30 words: {debate}')
print(f"SUMMARY:\n {summary}")
```

This example demonstrates how multiple agents can interact in a structured debate format, each with their own role, 
goal, and task. The moderator agent guides the conversation while two political figures engage in a debate.

All prompts that we use are stored in yaml files that you can easily overload.

## Chain of Thought Agent with Function Calls

The `ChainOfThoughtAgent` class extends the capabilities of our agents by allowing them to use reasoning steps and call functions. 
Here's an example:

```python
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
from just_agents import llm_options

def count_letters(character: str, word: str) -> str:
    """ Returns the number of character occurrences in the word. """
    count = 0
    for char in word:
        if char == character:
            count += 1
    print(f"Function: {character} occurs in {word} {count} times.")
    return str(count)

# Initialize agent with tools and LLM options
agent = ChainOfThoughtAgent(
    tools=[count_letters],
    llm_options=llm_options.LLAMA3_3
)

# Optional: Add callback to see all messages
agent.memory.add_on_message(lambda message: print(message))

# Get result and reasoning chain
result, chain = agent.think("Count the number of occurrences of the letter 'L' in the word - 'LOLLAPALOOZA'.")
```

This example shows how a Chain of Thought agent can use a custom function to count letter occurrences in a word. The agent can 
reason about the problem and use the provided tool to solve it.

## 📦 Package Structure
- `just_agents`: Core library
- `just_agents_coding`: Sandbox containers and code execution agents
- `just_agents_examples`: Usage examples
- `just_agents_tools`: Reusable agent tools
- `just_agents_web`: OpenAI-compatible REST API endpoints

## 🔒 Sandbox Execution

The `just_agents_coding` package provides secure containers for code execution:
- 📦 Sandbox container
- 🧬 Biosandbox container
- 🌐 Websandbox container

Mount `/input` and `/output` directories to easily manage data flow and monitor generated code.

## 🌐 Web Deployment Features

### Quick API Deployment
With a single command `run-agent`, you can instantly serve any just-agents agent as an OpenAI-compatible REST API endpoint. This means:
- 🔌 Instant OpenAI-compatible endpoint
- 🔄 Works with any OpenAI client library
- 🛠️ Simple configuration through YAML files
- 🚀 Ready for production use

### Full Chat UI Deployment
Using the `deploy-agent` command, you can deploy a complete chat interface with all necessary infrastructure:
- 💬 Modern Hugging Face-style chat UI
- 🔄 LiteLLM proxy for model management
- 💾 MongoDB for conversation history
- ⚡ Redis for response caching
- 🐳 Complete Docker environment

### Benefits
1. **Quick Time-to-Production**: Deploy agents from development to production in minutes
2. **Standard Compatibility**: OpenAI-compatible API ensures easy integration with existing tools
3. **Scalability**: Docker-based deployment provides consistent environments
4. **Security**: Proper isolation of services and configuration
5. **Flexibility**: Easy customization through YAML configurations