#!/usr/bin/env python
"""
Schema Serialization/Deserialization Demo

This script demonstrates how to use the schema serialization and deserialization
functionality in the BaseAgent class to:
1. Create an agent with a Pydantic model as output parser
2. Serialize the agent's configuration to JSON/YAML
3. Deserialize the configuration to recreate the agent
4. Use the agent to get structured outputs
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import json
import yaml
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from just_agents.base_agent import BaseAgent


# Define a sample output schema model
class WeatherResponse(BaseModel):
    """Schema for weather forecast responses"""
    location: str = Field(..., description="The location for the weather forecast")
    current_temp: float = Field(..., description="Current temperature in Celsius")
    conditions: str = Field(..., description="Current weather conditions (e.g., 'sunny', 'cloudy')")
    forecast: List[Dict[str, str]] = Field(..., description="Forecast for the next few days")
    warnings: Optional[List[str]] = Field(None, description="Any weather warnings or alerts")
    
    # In Pydantic v2, we need to provide a model_config
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "location": "New York, NY",
                    "current_temp": 22.5,
                    "conditions": "Partly Cloudy",
                    "forecast": [
                        {"day": "Monday", "conditions": "Sunny", "high": "24°C", "low": "18°C"},
                        {"day": "Tuesday", "conditions": "Rainy", "high": "20°C", "low": "15°C"}
                    ],
                    "warnings": ["Heavy rain expected Tuesday night"]
                }
            ]
        }
    }


def create_agent_with_schema():
    """Create a new agent with the WeatherResponse schema as parser"""
    
    # Create basic LLM options as a dictionary
    options = {
        "model": "gpt-4-turbo-preview",  # Use an appropriate model
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # Create an agent with WeatherResponse as the parser
    agent = BaseAgent(
        llm_options=options,
        system_prompt="You are a helpful weather assistant that provides accurate weather information.",
        parser=WeatherResponse  # Set the parser to our Pydantic model
    )
    
    return agent


def serialize_agent_to_yaml(agent: BaseAgent) -> str:
    """Serialize an agent to YAML format"""
    
    # Convert the agent to a dictionary
    agent_dict = agent.model_dump()
    
    # Convert to YAML
    return yaml.dump(agent_dict, default_flow_style=False)


def serialize_agent_to_json(agent: BaseAgent) -> str:
    """Serialize an agent to JSON format"""
    
    # Convert the agent to a dictionary
    agent_dict = agent.model_dump()
    
    # Convert to JSON
    return json.dumps(agent_dict, indent=2)


def deserialize_agent_from_yaml(yaml_str: str) -> BaseAgent:
    """Deserialize an agent from YAML format"""
    
    # Parse YAML to dictionary
    agent_dict = yaml.safe_load(yaml_str)
    
    # Create agent from dictionary
    return BaseAgent.model_validate(agent_dict)


def deserialize_agent_from_json(json_str: str) -> BaseAgent:
    """Deserialize an agent from JSON format"""
    
    # Parse JSON to dictionary
    agent_dict = json.loads(json_str)
    
    # Create agent from dictionary
    return BaseAgent.model_validate(agent_dict)


def demonstrate_query_structural():
    """Demonstrate using query_structural with the parser"""
    
    # Create an agent with the WeatherResponse parser
    agent = create_agent_with_schema()
    
    # Sample query for weather information
    query = "What's the weather like in San Francisco?"
    
    print(f"Querying agent with: '{query}'")
    print("Using WeatherResponse as the output schema parser")
    
    # This would make an actual API call to the LLM
    # For demo purposes, we'll comment this out
    # result = agent.query_structural(query, parser=WeatherResponse)
    # print("Structured Output:")
    # print(json.dumps(result.model_dump(), indent=2))
    
    print("\nIn a real scenario, this would return a validated WeatherResponse object.")
    print("For demo purposes, we're skipping the actual LLM call.")


def main():
    """Main demo function"""
    print("=== Schema Serialization/Deserialization Demo ===\n")
    
    # Create an agent with a schema
    print("1. Creating agent with WeatherResponse schema...")
    agent = create_agent_with_schema()
    print(f"   Agent created with parser: {agent.parser.__name__ if agent.parser else None}")
    
    # Serialize to YAML
    print("\n2. Serializing agent to YAML...")
    yaml_str = serialize_agent_to_yaml(agent)
    print("   YAML output (excerpt):")
    yaml_lines = yaml_str.split("\n")
    for line in yaml_lines[:20]:  # Show first 20 lines
        if "parser:" in line:
            print(f"   {line}")
            parser_indent = line.index("parser:")
            # Show a few lines of the parser section
            for i in range(yaml_lines.index(line) + 1, min(yaml_lines.index(line) + 10, len(yaml_lines))):
                if yaml_lines[i].startswith(" " * (parser_indent + 2)):
                    print(f"   {yaml_lines[i]}")
                else:
                    break
    print("   ...")
    
    # Serialize to JSON
    print("\n3. Serializing agent to JSON...")
    json_str = serialize_agent_to_json(agent)
    print("   (JSON output omitted for brevity)")
    
    # Deserialize from YAML
    print("\n4. Deserializing agent from YAML...")
    deserialized_agent = deserialize_agent_from_yaml(yaml_str)
    parser_name = deserialized_agent.parser.__name__ if deserialized_agent.parser else None
    print(f"   Agent deserialized with parser: {parser_name}")
    
    # Compare original and deserialized schemas
    if agent.parser and deserialized_agent.parser:
        print("\n5. Comparing original and deserialized schemas...")
        original_fields = set(agent.parser.__class__.model_fields.keys())
        deserialized_fields = set(deserialized_agent.parser.__class__.model_fields.keys())
        print(f"   Original schema fields: {original_fields}")
        print(f"   Deserialized schema fields: {deserialized_fields}")
        print(f"   Fields match: {original_fields == deserialized_fields}")
    
    # Demonstrate query_structural
    print("\n6. Demonstrating query_structural functionality...")
    demonstrate_query_structural()
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 