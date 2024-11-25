#!/bin/bash
# we assume that the script is running from the root of the project
# like this: ./bin/install_locally.sh

# Ensure we're in the right micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate just-agents

# Install packages in the correct order
echo "Installing core package..."
cd core
pip install -e .
cd ..

echo "Installing coding package..."
cd coding
pip install -e .
cd ..

echo "Installing web package..."
cd web
pip install -e .
cd ..

echo "Installing tools package..."
cd tools
pip install -e .
cd ..

echo "Installing router package..."
cd router
pip install -e .
cd ..

echo "Installing examples package..."
cd examples
pip install -e .
cd ..