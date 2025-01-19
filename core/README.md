# just-agents-core

A lightweight, straightforward core library for LLM agents - no over-engineering, just simplicity!

## 🎯 Core Features
- 🪶 Lightweight and simple implementation
- 📝 Easy-to-understand agent interactions
- 🔧 Customizable prompts using YAML files
- 🤖 Support for various LLM models through litellm
- 🔄 Chain of Thought reasoning with function calls

## 🏗️ Core Components

### BaseAgent
A thin wrapper around litellm for basic LLM interactions. Provides:
- Simple prompt management
- Direct LLM communication
- Memory handling

### ChatAgent
The fundamental building block for agent interactions. Here's an example of using multiple chat agents:

```python
from just_agents.base_agent import ChatAgent
from just_agents.llm_options import LLAMA3_3

# Initialize agents with different roles
harris = ChatAgent(
    llm_options=LLAMA3_3, 
    role="You are Kamala Harris in a presidential debate",
    goal="Win the debate with clear, concise responses",
    task="Respond briefly and effectively to debate questions"
)

trump = ChatAgent(
    llm_options=LLAMA3_3,
    role="You are Donald Trump in a presidential debate",
    goal="Win the debate with your signature style",
    task="Respond briefly and effectively to debate questions"
)

moderator = ChatAgent(
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
```

### ChainOfThoughtAgent
Extended agent with reasoning capabilities and function calling:

```python
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent
from just_agents import llm_options

def count_letters(character: str, word: str) -> str:
    """ Returns the number of character occurrences in the word. """
    count = word.count(character)
    return str(count)

# Initialize agent with tools and LLM options
agent = ChainOfThoughtAgent(
    tools=[count_letters],
    llm_options=llm_options.LLAMA3_3
)

# Get result and reasoning chain
result, chain = agent.think("Count the number of occurrences of the letter 'L' in 'HELLO'.")
```

## 📚 Usage
This core package provides the fundamental building blocks for LLM agents. For full usage examples and documentation, please refer to the [main repository](https://github.com/longevity-genie/just-agents).

## 🔧 Installation
```bash
pip install just-agents-core
```
