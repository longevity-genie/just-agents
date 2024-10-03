# just-agents
[![Tests](https://github.com/longevity-genie/just-agents/actions/workflows/tests.yml/badge.svg)](https://github.com/longevity-genie/just-agents/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/just-agents.svg)](https://badge.fury.io/py/just-agents)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM agents done right, no over-engineering and redundant complexity! 

# Motivation

Most of the existing agentic libraries are extremely over-engineered either directly or by using over-engineered libraries under the hood, like langchain and llamaindex.
In reality, interactions with LLMs are mostly about strings, and you can write your own template by just using f-strings and python native string templates. 
There is no need in complicated chain-like classes and other abstractions, in fact popular libraries create complexity just to sell you their paid services for LLM calls monitoring because it is extremely hard to understand what exactly is sent to LLMs.

We wrote this libraries while being pissed of by high complexity and wanted something controlled and simple.
Of course, you might comment that we do not have the ecosystem like, for example, tools and loaders. In reality, most of langchain tools are just very simple functions wrapped in their classes, you can always quickly look at them and re-implement them easier.

## Key Features

- Simple and lightweight implementation
- Easy-to-understand agent interactions
- Customizable prompts using YAML files
- Support for various LLM models through litellm
- Chain of Thought reasoning with function calls

# How it works

We use litellm library to interact with LLMs. 

The `ChatAgent` class is the core of our library. It represents an agent with a specific role, goal, and task. Here's a simple example of two agents talking to each other.

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

The only complex (but not mandatory) dependency that we use is Mako for prompt templates.

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

# Installation

If you want to install as pip package use:
```
pip install just-agents
```

If you want to contribute to the project you can use micromamba or other anaconda to install the environment
```
micromamba create -f environment.yaml
micromamba activate just-agents
```
then you can edit the library. Optionally you can install it locally with:
```
pip install -e .
```
