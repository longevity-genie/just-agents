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
It represents an agent with a specific role, goal, and task. Here's a simple example of two agents talking to each other.

```python
from dotenv import load_dotenv

from just_agents.simple.chat_agent import ChatAgent
from just_agents.simple.llm_options import LLAMA3_2_VISION
load_dotenv(override=True)

customer: ChatAgent = ChatAgent(llm_options = LLAMA3_2_VISION, role = "customer at a shop",
                                goal = "Your goal is to order what you want, while speaking concisely and clearly",
                                task="Find the best headphones!")
storekeeper: ChatAgent = ChatAgent(llm_options = LLAMA3_2_VISION,
                                    role = "helpful storekeeper",
                                    goal="earn profit by selling what customers need",
                                    task="sell to the customer")


exchanges: int = 3 # how many times the agents will exchange messages
customer.memory.add_on_message(lambda m: logger.info(f"Customer: {m}") if m.role == "user" else logger.info(f"Storekeeper: {m}"))

customer_reply = "Hi."
for _ in range(exchanges):
    storekeeper_reply = storekeeper.query(customer_reply)
    customer_reply = customer.query(storekeeper_reply)
```

This example demonstrates how two agents (a customer and a storekeeper) can interact with each other, each with their own role, 
goal, and task. The agents exchange messages for a specified number of times, simulating a conversation in a shop.

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
    llm_options=llm_options.LLAMA3_2_VISION
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
- `just_agents_coding`: Sandbox containers and code execution
- `just_agents_examples`: Usage examples
- `just_agents_tools`: Reusable agent tools

## 🔒 Sandbox Execution

The `just_agents_sandbox` package provides secure containers for code execution:
- 📦 Sandbox container
- 🧬 Biosandbox container
- 🌐 Websandbox container

Mount `/input` and `/output` directories to easily manage data flow and monitor generated code.