# just-agents
LLM agents done right, no over-engineering and redundant complexity!

# Motivation

Most of the existing agentic libraries are extremely over-engineered either directly or by using over-engineered libraries under the hood, like langchain and llamaindex.
In reality, interactions with LLMs are mostly about strings, and you can write your own template by just using f-strings and python native string templates. 
There is no need in complicated chains and other abstractions, in fact popular libraries create complexity just to sell you their paid services for LLM calls monitoring because it is extremely hard to understand what exactly is sent to LLMs.

We wrote this libraries while being pissed of by high complexity and wanted something controlled and simple.
Of course, you might comment that we do not have the ecosystem like, for example, tools and loaders. In reality, most of langchain tools are just very simple functions wrapped in their classes, you can always quickly look at them and re-implement them easier.

# How it works

We use litellm library to interact with LLMs. 

Here is a simple example of two agents talking to each other.
It is assumed that a typical agent has role, goal and the background story.

```python
from dotenv import load_dotenv

from just_agents.chat_agent import ChatAgent
from just_agents.llm_options import LLAMA3
from loguru import logger
load_dotenv()

customer: ChatAgent = ChatAgent(llm_options = LLAMA3.1, role = "customer at a shop",
                               goal = "Your goal is to order what you want, while speaking concisely and clearly", task="Find the best headphones!")
storekeeper: ChatAgent = ChatAgent(llm_options = LLAMA3.1,
                                  role = "helpful storekeeper", goal="earn profit by selling what customers need", task="sell to the customer")


exchanges: int = 3 # how many times the agents will exchange messages
customer.memory.add_on_message(lambda m: logger.info(f"Customer: {m}") if m.role == "user" else logger.info(f"Storekeeper: {m}"))

customer_reply = "Hi."
for _ in range(exchanges):
    storekeeper_reply = storekeeper.query(customer_reply)
    customer_reply = customer.query(storekeeper_reply)
```

All prompts that we use are stored in yaml files that you can easily overload.

The only complex (but not mandatory) dependency that we use is Mako for prompt templates

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