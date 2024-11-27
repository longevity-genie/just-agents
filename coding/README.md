# just-agents-coding

A submodule of just-agents focused on code generation and execution capabilities.

## Overview

just-agents-coding provides secure code execution environments and tools for LLM-powered coding agents. It enables safe code generation and execution through containerized environments, protecting your system while allowing AI agents to write and test code.

## Key Features

- 🔒 Secure code execution through isolated containers
- 🐳 Multiple specialized containers:
  - Standard sandbox for general Python code
  - Biosandbox for bioinformatics tasks
  - Websandbox for web-related code
- 📁 Simple I/O management with mounted `/input` and `/output` directories
- 🔍 Code execution monitoring and logging

## Quick Start

```bash
pip install just-agents-coding
```

## Usage Example

```python
from just_agents.base_agent import BaseAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize agent from configuration
agent = BaseAgent.from_yaml("SimpleCodeAgent", file_path="path/to/coding_agents.yaml")

# Execute code through the agent
result = agent.query("""
Get FGF2 human protein sequence from uniprot using biopython.
As a result, return only the sequence
""")

print(result)
```

## Container Types

### Standard Sandbox
- General Python code execution
- Basic Python packages pre-installed
- Isolated from host system

### Biosandbox
- Specialized for bioinformatics tasks
- Includes common bio packages (Biopython, etc.)
- Safe handling of biological data

### Websandbox
- Web development and testing
- Network access controls
- Common web frameworks available

## Security Features

- Root access disabled in containers
- Resource usage limits
- Network isolation
- Temporary file system
- Controlled package installation

## Documentation

For more detailed documentation and examples, visit:
- [Basic Usage Tutorial](examples/notebooks/03_coding_agent.ipynb)