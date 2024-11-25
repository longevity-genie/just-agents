# just-agents-core

A lightweight, straightforward core library for LLM agents - no over-engineering, just simplicity!

## 🎯 Core Features
- 🪶 Lightweight base agent implementations
- 📝 Simple string-based agent interactions
- 🔧 YAML-based prompt templating
- 🤖 LLM model integration through litellm
- 🔄 Chain of Thought reasoning capabilities

## 🏗️ Core Components

### BaseAgent
A thin wrapper around litellm for basic LLM interactions. Provides:
- Simple prompt management
- Direct LLM communication
- Memory handling

### ChatAgent
The fundamental building block for agent interactions:
```python
from just_agents.simple.chat_agent import ChatAgent
from just_agents.simple.llm_options import LLAMA3_2_VISION

agent = ChatAgent(
    llm_options=LLAMA3_2_VISION,
    role="assistant",
    goal="help the user",
    task="answer questions"
)
```

### ChainOfThoughtAgent
Extended agent with reasoning capabilities and function calling:
```python
from just_agents.patterns.chain_of_throught import ChainOfThoughtAgent

agent = ChainOfThoughtAgent(
    tools=[your_function],
    llm_options=LLAMA3_2_VISION
)
```

## 📚 Usage
This core package is typically used as a dependency by other just-agents packages. For full usage examples and documentation, please refer to the [main repository](https://github.com/longevity-genie/just-agents).

## 🔧 Installation
```bash
pip install just-agents-core
```
