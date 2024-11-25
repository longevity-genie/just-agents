# Just-Agents Router

The router module in just-agents provides functionality for routing conversations and tasks between different agents. This enables complex multi-agent interactions where specialized agents can handle different aspects of a task.

## Key Components

### SecretaryAgent
The SecretaryAgent acts as a routing coordinator, determining which agent should handle specific requests. As shown in the tests, it maintains a configuration that can be loaded from YAML files and includes:

- Agent descriptions and roles
- Personality traits
- Model configurations
- Routing rules

### Example Usage

Here's a simplified test setup demonstrating the secretary agent configuration:

```python
from just_agents.router.secretary_agent import SecretaryAgent

# Initialize secretary with specific configuration
secretary = SecretaryAgent(
    autoload_from_yaml=False,
    llm_options=your_llm_options,
    extra_dict={
        "personality_traits": "Agent's personality traits go here",
    }
)

# Secretary can update its profile and routing rules
info = secretary.get_info(secretary)
to_populate = secretary.get_to_populate(secretary)
secretary.update_profile(secretary, info, to_populate)

# Save configuration for future use
secretary.save_to_yaml()
```

The secretary can then use this configuration to determine which specialized agent (e.g., a manager, researcher, or technical expert) should handle incoming requests.

## Configuration

Agents can be configured using YAML files, which specify:
- Agent roles and responsibilities
- Routing rules and conditions
- Model preferences
- Custom personality traits

See the tests directory for complete examples of router configuration and usage.
