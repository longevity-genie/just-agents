#!/bin/bash
# Ensure we're in the right micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate just-agents

# Install packages in the correct order
echo "Installing just_agents package..."
cd just_agents
pip install -e .
cd ..

echo "Installing just_agents_coding package..."
cd just_agents_coding
pip install -e .
cd ..

echo "Installing just_agents_web package..."
cd just_agents_web
pip install -e .
cd ..

echo "Installing just_agents_tools package..."
cd just_agents_tools
pip install -e .
cd ..

echo "Installing just_agents_router package..."
cd just_agents_router
pip install -e .
cd ..

echo "Installing just_agents_examples package..."
cd just_agents_examples
pip install -e .
cd ..

echo "Installing root package..."
pip install -e .