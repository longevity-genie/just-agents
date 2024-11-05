# just-agents
[![Python CI](https://github.com/longevity-genie/just-agents/actions/workflows/python-ci.yaml/badge.svg)](https://github.com/longevity-genie/just-agents/actions/workflows/python-ci.yaml)
[![PyPI version](https://badge.fury.io/py/just-agents.svg)](https://badge.fury.io/py/just-agents)
[![Documentation Status](https://readthedocs.org/projects/just-agents/badge/?version=latest)](https://just-agents.readthedocs.io/en/latest/?badge=latest)


LLM agents done right, no over-engineering and redundant complexity! 

# Motivation

Most of the existing agentic libraries are extremely over-engineered either directly or by using over-engineered libraries under the hood, like langchain and llamaindex.
In reality, interactions with LLMs are mostly about strings, and you can write your own template by just using f-strings and python native string templates.
There is no need in complicated chain-like classes and other abstractions, in fact popular libraries create complexity just to sell you their paid services for LLM calls monitoring because it is extremely hard to understand what exactly is sent to LLMs.

It is way easier to reason about the code if you separate your prompting from python code to a separate easily readable files (like yaml files).

We wrote this libraries while being pissed of by high complexity and wanted something controlled and simple.
Of course, you might comment that we do not have the ecosystem like, for example, tools and loaders. In reality, most of langchain tools are just very simple functions wrapped in their classes, you can always quickly look at them and write a simple function to do the same thing that just-agents will pick up easily.

## Key Features

- Simple and lightweight implementation
- Easy-to-understand agent interactions
- Customizable prompts using YAML files
- Support for various LLM models through litellm
- Chain of Thought reasoning with function calls

## Examples

We provide examples of using Just Agents for various tasks in the [examples](examples) directory, including:

* basic - basic examples of using Just Agents
* coding - generating and executing code and commands (note: require docker to be installed)
* tools - tools used by examples
* multiagent - multiagent examples

# How it works

We use litellm library to interact with LLMs.
LLMSession class is a thin wrapper around litellm that allows you to interact with varios LLMs and add function as tools that it can call.
Agent classes either inherit or use LLMSession inside of them and also inherit from just_serialization class which allows you to save and load agents to yaml files.


## ChatAgent

The `ChatAgent` class is the core of our library. 
It represents an agent with a specific role, goal, and task. Here's a simple example of two agents talking to each other.

```python
from dotenv import load_dotenv

from just_agents.chat_agent import ChatAgent
from just_agents.llm_options import LLAMA3_2
load_dotenv(override=True)

customer: ChatAgent = ChatAgent(llm_options = LLAMA3_2, role = "customer at a shop",
                                goal = "Your goal is to order what you want, while speaking concisely and clearly",
                                task="Find the best headphones!")
storekeeper: ChatAgent = ChatAgent(llm_options = LLAMA3_2,
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

This example demonstrates how two agents (a customer and a storekeeper) can interact with each other, each with their own role, goal, and task. The agents exchange messages for a specified number of times, simulating a conversation in a shop.

All prompts that we use are stored in yaml files that you can easily overload.

## Chain of Thought Agent with Function Calls

The `ChainOfThoughtAgent` class extends the capabilities of our agents by allowing them to use reasoning steps and call functions. Here's an example:

```python
from just_agents.cot_agent import ChainOfThoughtAgent

def count_letters(character:str, word:str) -> str:
    """ Returns the number of character occurrences in the word. """
    count:int = 0
    for char in word:
        if char == character:
            count += 1
    print("Function: ", character, " occurres in ", word, " ", count, " times.")
    return str(count)

opt = just_agents.llm_options.OPENAI_GPT4oMINI.copy()
agent: ChainOfThoughtAgent = ChainOfThoughtAgent(opt, tools=[count_letters])
result, thoughts = agent.query("Count the number of occurrences of the letter 'L' in the word - 'LOLLAPALOOZA'.")
```

This example shows how a Chain of Thought agent can use a custom function to count letter occurrences in a word. The agent can reason about the problem and use the provided tool to solve it.

# Package structure

* just_agents - core library
* just_agents_coding - contains sandbox containers for safe code executions and code running agents
* just_agents_examples - contains examples of just-agents usage
* just_agents_tools - contains useful tools that you can use in your agents

# Examples

Examples is one of the easiest ways to get started with just-agents.
Start with basic examples and then move on to multiagent and coding examples.
For coding examples you might need to have docker installed to be able to run the containers.


# Installation

Note: the instructions below assume that you are running the code on Linuxor under WSL2 on Windows.
If you use plain Windows you may use user interfaces instead of some of the commands below.
Also, we recommend using virtual environment and give examples with micromamba (which is same as conda or anaconda in API) however you can use any other environment tool

## Using pip
If you want to install as pip package use:
```
pip install just-agents
```

## Local Development

To contribute or develop locally:

1. Clone the repository (SSH recommended for contributors):
```bash
# Using SSH (recommended for contributors)
git clone git@github.com:longevity-genie/just-agents.git

# Or using HTTPS
git clone https://github.com/longevity-genie/just-agents.git
```

2. Change to the project directory:
```bash
cd just-agents
```

3. Set up the development environment using micromamba or other anaconda:
```bash
micromamba create -f environment.yaml
micromamba activate just-agents
```

4. (Optional) Install the package in editable mode:
```bash
pip install -e .
```

Create configuration file:
   ```bash
   cp .env.example .env
   ```

Edit the configuration file:

   This refers to API keys for various LLMs. You'll need to edit these files to customize your setup.

Edit `.env`:
      ```bash
      nano .env
      ```
      This file contains API keys for various services. Add your keys here:
      - `OPENAI_API_KEY=your_openai_api_key_here`
      - `GROQ_API_KEY=your_groq_api_key_here`
      - Add any other API keys required for models you plan to use




# Running the code by coding agents

You can allow agents to install dependencies and run code by using the sandbox, biosandbox or websandbox containers.

We provide a package just_agents_sandbox that contains the sandbox and biosandbox containers as well as micromamba session to run the containers.

In some of the examples we also provide mounting /input and /output directories to the containers to make it easier to pass data to and from the agents and see which code and bash commands are being generated.