# JustLocator

A generalized locator pattern that can work with any type of entity, providing registration, lookup, and lifecycle management capabilities.

## Overview

The JustLocator is a type-safe, generalized version of the agent locator pattern that follows the "Just" naming convention. It provides:

- **Generic type support**: Works with any entity type `T`
- **Configurable identifier attributes**: Specify which attribute to use for entity identification
- **Automatic fallback**: Uses class name if the specified attribute doesn't exist
- **Weak reference management**: Automatic cleanup when entities are garbage collected
- **Multiple lookup methods**: By codename, config identifier, class, or custom predicates

## Core Classes

### `EntityIdentifier[T]`

Represents an entity's identifiers:
- `entity_class`: The class type of the entity
- `entity_codename`: Unique system-generated identifier
- `entity_config_identifier`: Dynamic property that returns the entity's config identifier

### `JustLocator[T]`

The main locator class that manages entities of type `T`:
- Generic type parameter `T` for type safety
- Configurable `entity_config_identifier_attr` (defaults to "name")
- Weak reference management for automatic cleanup
- Comprehensive lookup and search methods

### `JustSingletonLocator[T]`

A singleton version of the JustLocator for global entity management.

### `JustAgentsLocator`

A concrete implementation that extends `JustLocator[IAgent]` for agent management:
- Maintains backward compatibility with existing agent locator API
- Uses "shortname" as the entity_config_identifier_attr
- Provides agent-specific methods like `publish_agent()`, `get_agents_by_shortname()`, etc.
- Uses the same `EntityIdentifier[IAgent]` for type consistency

## Key Features

### Configurable Entity Config Identifier

The locator can be configured to use any attribute as the entity config identifier:

```python
# Use 'name' attribute for identification
user_locator = JustLocator[User](entity_config_identifier_attr="name")

# Use 'title' attribute for identification  
product_locator = JustLocator[Product](entity_config_identifier_attr="title")

# Use default 'name' attribute, fall back to class name if not present
service_locator = JustLocator[Service]()
```

### Automatic Fallback

If an entity doesn't have the specified attribute, the locator automatically falls back to using the class name:

```python
class MyEntity:
    def __init__(self, value):
        self.value = value
        # No 'name' attribute

locator = JustLocator[MyEntity]()  # Uses 'name' by default
codename = locator.publish_entity(MyEntity(42))

# Entity will be identified by class name "MyEntity"
entities = locator.get_entities_by_config_identifier("MyEntity")
```

### Type Safety

The locator is fully type-safe with generic type parameters:

```python
user_locator = JustLocator[User]()
users: List[User] = user_locator.get_entities_by_config_identifier("alice")
```

## API Methods

### Registration
- `publish_entity(entity: T) -> str`: Register an entity and return its codename
- `unpublish_entity(entity: T) -> bool`: Remove an entity from the registry
- `unpublish_entity_by_codename(codename: str) -> bool`: Remove by codename
- `unpublish_entities_by_config_identifier(config_id: str) -> int`: Remove by config identifier

### Lookup
- `get_entity_by_codename(codename: str) -> Optional[T]`: Get entity by codename
- `get_entities_by_config_identifier(config_id: str, bounding_class: Optional[Type[T]] = None) -> List[T]`: Get entities by config identifier
- `get_entity_codenames_by_class(bounding_class: Optional[Type[T]] = None) -> List[str]`: Get codenames by class
- `get_entity_codenames_by_config_identifier(config_id: str, bounding_class: Optional[Type[T]] = None) -> List[str]`: Get codenames by config identifier

### Advanced Search
- `arbitrary_search(bounding_class: Type[T], predicate: Callable[[T], bool]) -> List[T]`: Search using custom predicate

### Utilities
- `get_entity_codename(entity: T) -> Optional[str]`: Get codename for an entity instance
- `get_identifier_by_instance(entity: T) -> Optional[EntityIdentifier[T]]`: Get full identifier information

## Usage Examples

### Basic Usage

```python
from just_agents.just_locator import JustLocator

class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

# Create locator for User entities using 'name' attribute
locator = JustLocator[User](entity_config_identifier_attr="name")

# Register users
user1 = User("alice", "alice@example.com")
user2 = User("bob", "bob@example.com")

codename1 = locator.publish_entity(user1)
codename2 = locator.publish_entity(user2)

# Find users
alice = locator.get_entities_by_config_identifier("alice")
bob = locator.get_entity_by_codename(codename2)
```

### Advanced Search

```python
# Find users with specific email domain
gmail_users = locator.arbitrary_search(
    User,
    lambda user: user.email.endswith("@gmail.com")
)

# Get all user codenames
all_codenames = locator.get_entity_codenames_by_class()
```

### Multiple Entity Types

```python
# Different locators for different entity types
user_locator = JustLocator[User](entity_config_identifier_attr="name")
product_locator = JustLocator[Product](entity_config_identifier_attr="title")
service_locator = JustLocator[Service]()  # Uses class name fallback
```

### Agent Locator Usage

```python
from just_agents.just_locator import JustAgentsLocator
from just_agents.interfaces.agent import IAgent

# Agent locator is a singleton with the traditional API
locator = JustAgentsLocator()

# Agents must have a 'shortname' attribute
class MyAgent(IAgent):
    def __init__(self, shortname: str):
        self.shortname = shortname

agent = MyAgent("assistant")
codename = locator.publish_agent(agent)

# Find agents by shortname
assistants = locator.get_agents_by_shortname("assistant")

# Get agent identifier (returns EntityIdentifier[IAgent])
identifier = locator.get_identifier_by_instance(agent)
print(f"Agent codename: {identifier.entity_codename}")
print(f"Agent config identifier: {identifier.entity_config_identifier}")  # shortname
```

## File Structure

```
just-agents/
├── core/just_agents/
│   └── just_locator.py                                    # Main implementation
├── just_agents/examples/just_agents/examples/
│   └── just_locator_example.py                           # Usage examples
└── tests/
    └── test_just_locator.py                              # Unit tests
```

## Benefits Over Original Implementation

1. **Type Safety**: Full generic type support with proper type hints
2. **Flexibility**: Can work with any entity type, not just agents
3. **Configurability**: Customizable attribute names for identification
4. **Robustness**: Automatic fallback to class name when attribute is missing
5. **Consistency**: Same API patterns but with more generic naming
6. **Backward Compatibility**: Original agent locator API is preserved
7. **Extensibility**: Easy to subclass and extend for specific use cases

## Migration Guide

The JustLocator maintains API compatibility while extending functionality:

| Original Agent Locator | JustLocator (Generic) | JustAgentsLocator (Concrete) |
|----------------------|----------------------|----------------------------|
| `publish_agent()` | `publish_entity()` | `publish_agent()` |
| `get_agent_by_codename()` | `get_entity_by_codename()` | `get_agent_by_codename()` |
| `get_agents_by_shortname()` | `get_entities_by_config_identifier()` | `get_agents_by_shortname()` |
| `AgentIdentifier` | `EntityIdentifier[T]` | `EntityIdentifier[IAgent]` |
| `shortname` | `entity_config_identifier` | `entity_config_identifier` |
| `codename` | `entity_codename` | `entity_codename` |
| `agent_class` | `entity_class` | `entity_class` | 