from typing import Optional, List, Dict, Type, TypeVar, Callable, Set
import uuid
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

# Type variable for agent types with IAgent bound
T = TypeVar('T', bound=IAgent)

class AgentIdentifier(BaseModel):
    """
    Pydantic model representing an agent's identifiers.
    
    Attributes:
        agent_class: The class type of the agent
        codename: Unique system-generated identifier

    """
    agent_class: Type[IAgent] = Field(description="Class type of the agent")
    codename: str = Field(description="Unique system-generated identifier")
    
    model_config = {
        "arbitrary_types_allowed": True  # Allow Type[IAgent] to be used
    }
    
    @property
    def shortname(self) -> str:
        """
        Dynamically get the agent's current shortname by finding
        the agent instance through the locator.
        
        Returns:
            str: The agent's shortname
        """
        locator = JustAgentsLocator()
        
        # Get the agent instance using our codename
        agent = locator.get_agent_by_codename(self.codename)
        if agent is None: #we're somehow in a race condition where the agent is gone but the locator hasn't cleaned up yet
            #locator._cleanup_codename(self.codename) 
            raise ValueError(f"Agent with codename {self.codename} not found")
        
        return getattr(agent, 'shortname')

class JustAgentsLocator(metaclass=SingletonMeta):
    """
    A singleton registry for agents implementing IAgent.
    Manages the registration and lookup of agents by various discriminators.
    """
    def __init__(self) -> None:
        """Initialize the agent locator with empty registries."""
        # The only dictionary that uses weak references (central point for GC)
        self._codename_to_instance: Dict[str, ReferenceType[IAgent]] = {}
        
        # Maps agent codenames to their identifiers (using strings only)
        self._codename_to_identifiers: Dict[str, AgentIdentifier] = {}
        
        # Maps agent classes to sets of codenames for efficient lookup by type
        self._codenames_by_class: Dict[Type[IAgent], Set[str]] = {}
        
        # Track if coolname is available for human-readable codenames
        self._use_coolname: bool = _COOLNAME_AVAILABLE
        
        # Callback for weak references when agents are collected
        self._create_cleanup_callback = lambda codename: lambda *args: self._cleanup_codename(codename)
    
    def _cleanup_codename(self, codename: str) -> None:
        """
        Clean up all references to an agent when it's garbage collected.
        
        Args:
            codename: The unique codename of the agent to clean up
        """
        # Remove from codename->instance mapping first, instance is gone
        if codename in self._codename_to_instance:
            del self._codename_to_instance[codename]
        
        # Get the identifier to find class
        if codename in self._codename_to_identifiers:
            identifier = self._codename_to_identifiers[codename]
            agent_class = identifier.agent_class
            
            # Clean up from class mapping
            if agent_class in self._codenames_by_class:  
                self._codenames_by_class[agent_class].discard(codename)
                if not self._codenames_by_class[agent_class]:
                    del self._codenames_by_class[agent_class]
        
        # Safely remove from identifier mapping
        del self._codename_to_identifiers[codename]
    
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
        # Check if agent is already registered by searching for it
        for codename, ref_agent in self._codename_to_instance.items():
            if ref_agent() is agent:
                return codename
        
        # Extract shortname from agent instance
        if not hasattr(agent, 'shortname'):
            raise ValueError("Agent must have a 'shortname' attribute to be published")
        
        shortname = getattr(agent, 'shortname')
        if not shortname:
            raise ValueError("Agent's 'shortname' cannot be empty")
            
        # Generate a unique codename for this agent instance
        codename = self._generate_codename()
        agent_class = type(agent)
        
        # Create and store the agent's identifiers
        identifier = AgentIdentifier(
            agent_class=agent_class,
            codename=codename
        )
        
        # Update main registries with weak references
        self._codename_to_instance[codename] = ref(agent, self._create_cleanup_callback(codename))
        self._codename_to_identifiers[codename] = identifier
        
        # Update class lookup structure
        if agent_class not in self._codenames_by_class:
            self._codenames_by_class[agent_class] = set()
        
        self._codenames_by_class[agent_class].add(codename)
        
        return codename
    
    def get_codename(self, agent: IAgent) -> Optional[str]:
        """
        Get the codename for an agent instance.
        
        Args:
            agent: The agent instance
            
        Returns:
            Optional[str]: The agent's codename, or None if the agent is not registered
        """
        # Defensive copy of items
        items = list(self._codename_to_instance.items())
        
        for codename, ref_agent in items:
            if ref_agent() is agent:
                return codename
        return None
    
    def get_identifier_by_instance(self, agent: IAgent) -> Optional[AgentIdentifier]:
        """
        Get an agent's identifiers by its instance.
        
        Args:
            agent: The agent instance
            
        Returns:
            Optional[AgentIdentifier]: The agent's identifiers, or None if not registered
        """
        codename = self.get_codename(agent)
        if codename:
            return self._codename_to_identifiers.get(codename)
        return None
    
    def get_codenames_by_class(self, bounding_class: Optional[Type[T]] = None) -> List[str]:
        """
        Get all codenames for agents of a specific class.
        
        Args:
            bounding_class: The class to filter agents by
            
        Returns:
            List[str]: A list of matching agent codenames
        """
        # Defensive copy of the keys
        class_keys = list(self._codenames_by_class.keys())
        
        if bounding_class is None:
            bound_classes: List[Type[T]] = class_keys
        else:   
            bound_classes: List[Type[T]] = [
                bound_type for bound_type in class_keys if issubclass(bound_type, bounding_class)
            ]

        codenames = set()
        for bound_class in bound_classes:
            # Defensive copy of the values
            if bound_class in self._codenames_by_class:  # Extra check in case it was removed
                codenames.update(set(self._codenames_by_class[bound_class]))
        return list(codenames)

    def get_codenames_by_shortname(self, shortname: str, bounding_class: Optional[Type[IAgent]] = None) -> List[str]:
        """
        Get codenames of all agents with the given shortname, optionally filtered by class.
        
        Args:
            shortname: The shortname to match
            bounding_class: If provided, only include agents of this class or its subclasses
            
        Returns:
            List[str]: A list of matching agent codenames
        """
        # Instead of using the shortname mapping, scan all agents
        result: List[str] = []
        codenames = self.get_codenames_by_class(bounding_class)
        
        # Make a defensive copy of identifiers dictionary
        identifiers_snapshot = dict(self._codename_to_identifiers)
        
        for codename in codenames:
            identifier: Optional[AgentIdentifier] = identifiers_snapshot.get(codename)
            if not identifier:
                # Skip rather than raise if the identifier is gone (defensive)
                continue
            if identifier.shortname == shortname:
                result.append(codename)
                
        return result

    def get_agent_by_codename(self, codename: str) -> Optional[IAgent]:
        """
        Get an agent by its codename.
        
        Args:
            codename: The unique codename of the agent
            
        Returns:
            Optional[IAgent]: The agent instance, or None if no agent with that codename exists
        """
        ref = self._codename_to_instance.get(codename)
        if ref is not None:
            agent = ref()  # Dereference the weakref
            if agent is not None:
                return agent
            # If the weak reference is dead, clean up
            self._cleanup_codename(codename)
        return None

    def get_agents_by_shortname(self, shortname: str, bounding_class: Optional[Type[T]] = None) -> List[T]:
        """
        Get all agents with the given shortname, optionally filtered by class.

        Args:
            shortname: The shortname to match
            bounding_class: If provided, only include agents of this class or its subclasses
            
        Returns:
            List[T]: A list of matching agent instances
        """
        codenames = self.get_codenames_by_shortname(shortname, bounding_class)
        agents = [self.get_agent_by_codename(codename) for codename in codenames]
        if None in agents:
            raise ValueError(f"Inconsistent locator state")
        
        return agents
    
    def arbitrary_search(self, bounding_class: Type[T], 
                         predicate: Callable[[T], bool]) -> List[T]:
        """
        Search for agents of a specific class that satisfy a predicate function.
        
        Args:
            bounding_class: The class to filter agents by (required)
            predicate: A function that takes an agent instance and returns True if it matches
            
        Returns:
            List[T]: A list of matching agent instances
        """
        # Get a defensive copy of codenames
        codenames = list(self.get_codenames_by_class(bounding_class))

        result: List[T] = []
        
        # Iterate through all codenames and check each agent
        for codename in codenames:
            agent = self.get_agent_by_codename(codename)
            if agent is None:
                # Skip rather than raise if the agent is gone (defensive)
                continue
            if predicate(agent):  
                result.append(agent) 
        
        return result
    
    def unpublish_agent_by_codename(self, codename: str) -> bool:
        """
        Remove an agent from the registry by its codename.
        """
        if codename in self._codename_to_instance:
            self._cleanup_codename(codename)
            return True
        return False

    def unpublish_agent(self, agent: IAgent) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent: The agent instance to unregister
            
        Returns:
            bool: True if the agent was successfully unregistered, False if not found
        """
        codename = self.get_codename(agent)
        if not codename:
            return False
            
        return self.unpublish_agent_by_codename(codename)
    
    
    def unpublish_agents_by_shortname(self, shortname: str) -> int:
        """
        Remove all agents from the registry by their shortname.
        
        Args:
            shortname: The shortname of agents to unpublish
            
        Returns:
            int: Number of agents unpublished
        """
        # Get a defensive copy of codenames
        codenames = list(self.get_codenames_by_shortname(shortname))
        if len(codenames) == 0:
            return 0
        
        count = 0
        for codename in codenames:
            if self.unpublish_agent_by_codename(codename):
                count += 1
        
        # Don't raise an error if we couldn't unpublish all - they might have been
        # garbage collected during iteration
        return count
    
    def _generate_codename(self) -> str:
        """
        Generate a unique codename for an agent.
        
        Uses coolname for human-readable names if available,
        otherwise falls back to UUID strings.
        Ensures the generated codename is not already in use.
        
        Returns:
            str: A unique codename
        """
        if self._use_coolname:
            # Generate a human-readable name like "fancy-snake" using coolname
            codename = coolname.generate_slug(2)  # Two words like "adjective-noun"
        else:
            # Fall back to UUID if coolname isn't available
            codename = str(uuid.uuid4())
        
        # Check if this codename is already in use 
        if codename in self._codename_to_instance:
            # Recursively generate a new one if there's a collision
            return self._generate_codename()
            
        return codename 
    
