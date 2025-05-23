from typing import Optional, List, Dict, Type, TypeVar, Callable, Set, Generic
import uuid
import threading
from weakref import ReferenceType, WeakSet, ref
from pydantic import BaseModel, Field
from just_agents.just_bus import SingletonMeta
from just_agents.interfaces.agent import IAgent

# Try to import coolname but fall back gracefully if not available
try:
    import coolname
    _COOLNAME_AVAILABLE = True
except ImportError:
    _COOLNAME_AVAILABLE = False

# Type variable for entity types
T = TypeVar('T')

class EntityIdentifier(BaseModel, Generic[T]):
    """
    Pydantic model representing an entity's identifiers.
    
    Attributes:
        entity_class: The class type of the entity
        entity_codename: Unique system-generated identifier
    """
    entity_class: Type[T] = Field(description="Class type of the entity")
    entity_codename: str = Field(description="Unique system-generated identifier")
    
    model_config = {
        "arbitrary_types_allowed": True  # Allow Type[T] to be used
    }
    
    # Use private attributes that are not fields
    def __init__(self, **data):
        super().__init__(**data)
        self._locator: Optional['JustLocator[T]'] = None
        self._entity_config_identifier_attr: str = "name"
    
    def set_locator_reference(self, locator: 'JustLocator[T]', attr_name: str) -> None:
        """Set the locator reference after creation."""
        self._locator = locator
        self._entity_config_identifier_attr = attr_name
    
    @property
    def entity_config_identifier(self) -> str:
        """
        Dynamically get the entity's current config identifier by finding
        the entity instance through the locator.
        
        Returns:
            str: The entity's config identifier
        """
        if self._locator is None:
            raise ValueError("Locator reference not set for this identifier")
            
        # Get the entity instance using our codename
        entity = self._locator.get_entity_by_codename(self.entity_codename)
        if entity is None:
            raise ValueError(f"Entity with codename {self.entity_codename} not found")
        
        return self._locator._get_entity_config_identifier(entity)

class JustLocator(Generic[T]):
    """
    A generic registry for entities of type T.
    Manages the registration and lookup of entities by various discriminators.
    """
    
    # Class variable to store the attribute name used for entity_config_identifier
    entity_config_identifier_attr: str = "name"
    
    def __init__(self, entity_config_identifier_attr: Optional[str] = None) -> None:
        """
        Initialize the entity locator with empty registries.
        
        Args:
            entity_config_identifier_attr: Name of the attribute to use for entity config identifier
        """
        # Override class variable if provided
        if entity_config_identifier_attr is not None:
            self.entity_config_identifier_attr = entity_config_identifier_attr
            
        # Thread safety lock - use RLock to allow re-entrant calls
        self._lock = threading.RLock()
            
        # The only dictionary that uses weak references (central point for GC)
        self._entity_codename_to_instance: Dict[str, ReferenceType[T]] = {}
        
        # Maps entity codenames to their identifiers (using strings only)
        self._entity_codename_to_identifiers: Dict[str, EntityIdentifier[T]] = {}
        
        # Maps entity classes to sets of codenames for efficient lookup by type
        self._entity_codenames_by_class: Dict[Type[T], Set[str]] = {}
        
        # Track if coolname is available for human-readable codenames
        self._use_coolname: bool = _COOLNAME_AVAILABLE
        
        # Callback for weak references when entities are collected
        self._create_cleanup_callback = lambda entity_codename: lambda *args: self._cleanup_entity_codename(entity_codename)
    
    def _get_entity_config_identifier(self, entity: T) -> str:
        """
        Get the entity config identifier from the entity instance.
        
        Args:
            entity: The entity instance
            
        Returns:
            str: The entity config identifier (attribute value or class name)
        """
        if hasattr(entity, self.entity_config_identifier_attr):
            config_id = getattr(entity, self.entity_config_identifier_attr)
            if config_id:  # Make sure it's not None or empty
                return str(config_id)
        
        # Fall back to class name if attribute doesn't exist or is empty
        return entity.__class__.__name__
    
    def _cleanup_entity_codename(self, entity_codename: str) -> None:
        """
        Clean up all references to an entity when it's garbage collected.
        
        Args:
            entity_codename: The unique codename of the entity to clean up
        """
        with self._lock:
            # Remove from codename->instance mapping first, instance is gone
            if entity_codename in self._entity_codename_to_instance:
                del self._entity_codename_to_instance[entity_codename]
            
            # Get the identifier to find class
            if entity_codename in self._entity_codename_to_identifiers:
                identifier = self._entity_codename_to_identifiers[entity_codename]
                entity_class = identifier.entity_class
                
                # Clean up from class mapping
                if entity_class in self._entity_codenames_by_class:  
                    self._entity_codenames_by_class[entity_class].discard(entity_codename)
                    if not self._entity_codenames_by_class[entity_class]:
                        del self._entity_codenames_by_class[entity_class]
            
            # Safely remove from identifier mapping
            if entity_codename in self._entity_codename_to_identifiers:
                del self._entity_codename_to_identifiers[entity_codename]
    
    def publish_entity(self, entity: T) -> str:
        """
        Register an entity with the locator. 
        
        Args:
            entity: The entity instance to register
            
        Returns:
            str: The unique codename assigned to the entity
        """
        with self._lock:
            # Check if entity is already registered by searching for it
            for entity_codename, ref_entity in self._entity_codename_to_instance.items():
                if ref_entity() is entity:
                    return entity_codename
            
            # Generate a unique codename for this entity instance
            entity_codename = self._generate_entity_codename()
            entity_class = type(entity)
            
            # Create and store the entity's identifiers
            identifier = EntityIdentifier[T](
                entity_class=entity_class,
                entity_codename=entity_codename
            )
            # Set the locator reference so the identifier can access the entity
            identifier.set_locator_reference(self, self.entity_config_identifier_attr)
            
            # Update main registries with weak references
            self._entity_codename_to_instance[entity_codename] = ref(entity, self._create_cleanup_callback(entity_codename))
            self._entity_codename_to_identifiers[entity_codename] = identifier
            
            # Update class lookup structure
            if entity_class not in self._entity_codenames_by_class:
                self._entity_codenames_by_class[entity_class] = set()
            
            self._entity_codenames_by_class[entity_class].add(entity_codename)
            
            return entity_codename
    
    def get_entity_codename(self, entity: T) -> Optional[str]:
        """
        Get the codename for an entity instance.
        
        Args:
            entity: The entity instance
            
        Returns:
            Optional[str]: The entity's codename, or None if the entity is not registered
        """
        with self._lock:
            # Defensive copy of items
            items = list(self._entity_codename_to_instance.items())
            
            for entity_codename, ref_entity in items:
                if ref_entity() is entity:
                    return entity_codename
            return None
    
    def get_identifier_by_instance(self, entity: T) -> Optional[EntityIdentifier[T]]:
        """
        Get an entity's identifiers by its instance.
        
        Args:
            entity: The entity instance
            
        Returns:
            Optional[EntityIdentifier[T]]: The entity's identifiers, or None if not registered
        """
        entity_codename = self.get_entity_codename(entity)
        if entity_codename:
            return self._entity_codename_to_identifiers.get(entity_codename)
        return None
    
    def get_entity_codenames_by_class(self, bounding_class: Optional[Type[T]] = None) -> List[str]:
        """
        Get all codenames for entities of a specific class.
        
        Args:
            bounding_class: The class to filter entities by
            
        Returns:
            List[str]: A list of matching entity codenames
        """
        with self._lock:
            # Defensive copy of the keys
            class_keys = list(self._entity_codenames_by_class.keys())
            
            if bounding_class is None:
                bound_classes: List[Type[T]] = class_keys
            else:   
                bound_classes: List[Type[T]] = [
                    bound_type for bound_type in class_keys if issubclass(bound_type, bounding_class)
                ]

            entity_codenames = set()
            for bound_class in bound_classes:
                # Defensive copy of the values
                if bound_class in self._entity_codenames_by_class:  # Extra check in case it was removed
                    entity_codenames.update(set(self._entity_codenames_by_class[bound_class]))
            return list(entity_codenames)

    def get_entity_codenames_by_config_identifier(self, entity_config_identifier: str, bounding_class: Optional[Type[T]] = None) -> List[str]:
        """
        Get codenames of all entities with the given config identifier, optionally filtered by class.
        
        Args:
            entity_config_identifier: The config identifier to match
            bounding_class: If provided, only include entities of this class or its subclasses
            
        Returns:
            List[str]: A list of matching entity codenames
        """
        result: List[str] = []
        entity_codenames = self.get_entity_codenames_by_class(bounding_class)
        
        for entity_codename in entity_codenames:
            entity = self.get_entity_by_codename(entity_codename)
            if entity is None:
                # Skip rather than raise if the entity is gone (defensive)
                continue
            if self._get_entity_config_identifier(entity) == entity_config_identifier:
                result.append(entity_codename)
                
        return result

    def get_entity_by_codename(self, entity_codename: str) -> Optional[T]:
        """
        Get an entity by its codename.
        
        Args:
            entity_codename: The unique codename of the entity
            
        Returns:
            Optional[T]: The entity instance, or None if no entity with that codename exists
        """
        with self._lock:
            ref = self._entity_codename_to_instance.get(entity_codename)
            if ref is not None:
                entity = ref()  # Dereference the weakref
                if entity is not None:
                    return entity
                # If the weak reference is dead, clean up
                self._cleanup_entity_codename(entity_codename)
            return None

    def get_entities_by_config_identifier(self, entity_config_identifier: str, bounding_class: Optional[Type[T]] = None) -> List[T]:
        """
        Get all entities with the given config identifier, optionally filtered by class.

        Args:
            entity_config_identifier: The config identifier to match
            bounding_class: If provided, only include entities of this class or its subclasses
            
        Returns:
            List[T]: A list of matching entity instances
        """
        entity_codenames = self.get_entity_codenames_by_config_identifier(entity_config_identifier, bounding_class)
        entities = [self.get_entity_by_codename(entity_codename) for entity_codename in entity_codenames]
        if None in entities:
            raise ValueError(f"Inconsistent locator state")
        
        return entities
    
    def get_all_entities(self, bounding_class: Optional[Type[T]] = None) -> List[T]:
        """
        Get all entities in the locator, optionally filtered by class.

        Args:
            bounding_class: If provided, only include entities of this class or its subclasses
            
        Returns:
            List[T]: A list of all entity instances
        """
        entity_codenames = self.get_entity_codenames_by_class(bounding_class)
        entities = []
        
        for entity_codename in entity_codenames:
            entity = self.get_entity_by_codename(entity_codename)
            if entity is not None:  # Filter out garbage collected entities
                entities.append(entity)
        
        return entities
    
    def arbitrary_search(self, bounding_class: Type[T], 
                         predicate: Callable[[T], bool]) -> List[T]:
        """
        Search for entities of a specific class that satisfy a predicate function.
        
        Args:
            bounding_class: The class to filter entities by (required)
            predicate: A function that takes an entity instance and returns True if it matches
            
        Returns:
            List[T]: A list of matching entity instances
        """
        # Get a defensive copy of codenames
        entity_codenames = list(self.get_entity_codenames_by_class(bounding_class))

        result: List[T] = []
        
        # Iterate through all codenames and check each entity
        for entity_codename in entity_codenames:
            entity = self.get_entity_by_codename(entity_codename)
            if entity is None:
                # Skip rather than raise if the entity is gone (defensive)
                continue
            if predicate(entity):  
                result.append(entity) 
        
        return result
    
    def unpublish_entity_by_codename(self, entity_codename: str) -> bool:
        """
        Remove an entity from the registry by its codename.
        
        Args:
            entity_codename: The codename of the entity to unregister
            
        Returns:
            bool: True if the entity was successfully unregistered, False if not found
        """
        with self._lock:
            if entity_codename in self._entity_codename_to_instance:
                self._cleanup_entity_codename(entity_codename)
                return True
            return False

    def unpublish_entity(self, entity: T) -> bool:
        """
        Remove an entity from the registry.
        
        Args:
            entity: The entity instance to unregister
            
        Returns:
            bool: True if the entity was successfully unregistered, False if not found
        """
        entity_codename = self.get_entity_codename(entity)
        if not entity_codename:
            return False
            
        return self.unpublish_entity_by_codename(entity_codename)
    
    def unpublish_entities_by_config_identifier(self, entity_config_identifier: str) -> int:
        """
        Remove all entities from the registry by their config identifier.
        
        Args:
            entity_config_identifier: The config identifier of entities to unpublish
            
        Returns:
            int: Number of entities unpublished
        """
        # Note: This method doesn't need additional locking as it calls other
        # methods that are already protected by the RLock (which allows re-entrant calls)
        
        # Get a defensive copy of codenames
        entity_codenames = list(self.get_entity_codenames_by_config_identifier(entity_config_identifier))
        if len(entity_codenames) == 0:
            return 0
        
        count = 0
        for entity_codename in entity_codenames:
            if self.unpublish_entity_by_codename(entity_codename):
                count += 1
        
        # Don't raise an error if we couldn't unpublish all - they might have been
        # garbage collected during iteration
        return count
    
    def _generate_entity_codename(self) -> str:
        """
        Generate a unique codename for an entity.
        
        Uses coolname for human-readable names if available,
        otherwise falls back to UUID strings.
        Ensures the generated codename is not already in use.
        
        Returns:
            str: A unique codename
        """
        if self._use_coolname:
            # Generate a human-readable name like "fancy-snake" using coolname
            entity_codename = coolname.generate_slug(2)  # Two words like "adjective-noun"
        else:
            # Fall back to UUID if coolname isn't available
            entity_codename = str(uuid.uuid4())
        
        # Check if this codename is already in use 
        if entity_codename in self._entity_codename_to_instance:
            # Recursively generate a new one if there's a collision
            return self._generate_entity_codename()
            
        return entity_codename


class JustSingletonLocator(JustLocator[T], metaclass=SingletonMeta):
    """
    A singleton version of the JustLocator.
    """
    pass

# Type variable for agent types with IAgent bound
A = TypeVar('A', bound=IAgent)

class JustAgentsLocator(JustLocator[IAgent], metaclass=SingletonMeta):
    """
    A singleton registry for agents implementing IAgent.
    Manages the registration and lookup of agents by various discriminators.
    """
    def __init__(self) -> None:
        """Initialize the agent locator with empty registries."""
        # Initialize the parent JustLocator with 'shortname' as the config identifier attribute
        super().__init__(entity_config_identifier_attr="shortname")
    
    def _cleanup_codename(self, codename: str) -> None:
        """
        Clean up all references to an agent when it's garbage collected.
        
        Args:
            codename: The unique codename of the agent to clean up
        """
        # Delegate to parent implementation
        self._cleanup_entity_codename(codename)
    
    def publish_agent(self, agent: IAgent) -> str:
        """
        Register an agent with the locator. 
        
        Args:
            agent: The agent instance to register
            
        Returns:
            str: The unique codename assigned to the agent
            
        Raises:
            ValueError: If the agent doesn't have a shortname attribute
        """
        # Check if agent has shortname attribute
        if not hasattr(agent, 'shortname'):
            raise ValueError("Agent must have a 'shortname' attribute to be published")
        
        shortname = getattr(agent, 'shortname')
        if not shortname:
            raise ValueError("Agent's 'shortname' cannot be empty")
            
        # Delegate to parent implementation
        return self.publish_entity(agent)
    
    def get_codename(self, agent: IAgent) -> Optional[str]:
        """
        Get the codename for an agent instance.
        
        Args:
            agent: The agent instance
            
        Returns:
            Optional[str]: The agent's codename, or None if the agent is not registered
        """
        return self.get_entity_codename(agent)
    
    def get_identifier_by_instance(self, agent: IAgent) -> Optional[EntityIdentifier[IAgent]]:
        """
        Get an agent's identifiers by its instance.
        
        Args:
            agent: The agent instance
            
        Returns:
            Optional[EntityIdentifier[IAgent]]: The agent's identifiers, or None if not registered
        """
        return super().get_identifier_by_instance(agent)
    
    def get_codenames_by_class(self, bounding_class: Optional[Type[A]] = None) -> List[str]:
        """
        Get all codenames for agents of a specific class.
        
        Args:
            bounding_class: The class to filter agents by
            
        Returns:
            List[str]: A list of matching agent codenames
        """
        return self.get_entity_codenames_by_class(bounding_class)

    def get_codenames_by_shortname(self, shortname: str, bounding_class: Optional[Type[IAgent]] = None) -> List[str]:
        """
        Get codenames of all agents with the given shortname, optionally filtered by class.
        
        Args:
            shortname: The shortname to match
            bounding_class: If provided, only include agents of this class or its subclasses
            
        Returns:
            List[str]: A list of matching agent codenames
        """
        return self.get_entity_codenames_by_config_identifier(shortname, bounding_class)

    def get_agent_by_codename(self, codename: str) -> Optional[IAgent]:
        """
        Get an agent by its codename.
        
        Args:
            codename: The unique codename of the agent
            
        Returns:
            Optional[IAgent]: The agent instance, or None if no agent with that codename exists
        """
        return self.get_entity_by_codename(codename)

    def get_agents_by_shortname(self, shortname: str, bounding_class: Optional[Type[A]] = None) -> List[A]:
        """
        Get all agents with the given shortname, optionally filtered by class.

        Args:
            shortname: The shortname to match
            bounding_class: If provided, only include agents of this class or its subclasses
            
        Returns:
            List[A]: A list of matching agent instances
        """
        return self.get_entities_by_config_identifier(shortname, bounding_class)
    
    def arbitrary_search(self, bounding_class: Type[A], 
                         predicate: Callable[[A], bool]) -> List[A]:
        """
        Search for agents of a specific class that satisfy a predicate function.
        
        Args:
            bounding_class: The class to filter agents by (required)
            predicate: A function that takes an agent instance and returns True if it matches
            
        Returns:
            List[A]: A list of matching agent instances
        """
        return super().arbitrary_search(bounding_class, predicate)
    
    def unpublish_agent_by_codename(self, codename: str) -> bool:
        """
        Remove an agent from the registry by its codename.
        """
        return self.unpublish_entity_by_codename(codename)

    def unpublish_agent(self, agent: IAgent) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent: The agent instance to unregister
            
        Returns:
            bool: True if the agent was successfully unregistered, False if not found
        """
        return self.unpublish_entity(agent)
    
    
    def unpublish_agents_by_shortname(self, shortname: str) -> int:
        """
        Remove all agents from the registry by their shortname.
        
        Args:
            shortname: The shortname of agents to unpublish
            
        Returns:
            int: Number of agents unpublished
        """
        return self.unpublish_entities_by_config_identifier(shortname)
    
    def _generate_codename(self) -> str:
        """
        Generate a unique codename for an agent.
        
        Uses coolname for human-readable names if available,
        otherwise falls back to UUID strings.
        Ensures the generated codename is not already in use.
        
        Returns:
            str: A unique codename
        """
        return self._generate_entity_codename() 
    
