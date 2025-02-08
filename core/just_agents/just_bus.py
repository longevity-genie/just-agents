from typing import Callable, Dict, List, Any, ParamSpec, Protocol

class SingletonMeta(type):
    """
    A metaclass that creates a Singleton base type when called.
    """
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


VariArgs = ParamSpec('VariArgs')
# Signatures for listener templates
class SubscriberCallback(Protocol):
    def __call__(self, event_prefix: str, *args: VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        ...

class JustEventBus(metaclass=SingletonMeta):
    """
    A simple singleton event bus to publish function call results for functions that support it
    Event name can be anything, but suggested use is function names.
    """

    _subscribers: Dict[str, List[SubscriberCallback]]

    def __init__(self):
        # Dictionary of subscription_pattern -> list_of_callbacks (in subscription order)
        self._subscribers = {}

    def subscribe(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Subscribe a callback to a specific event.

        Args:
            event_prefix (str): The event name or pattern to subscribe to. Can be an exact event name 
                like 'mytool.call' or a prefix pattern like 'mytool.*'
            callback (SubscriberCallback): The callback function to be called when matching events are published

        """
        if event_prefix not in self._subscribers:
            self._subscribers[event_prefix] = []

        self._subscribers[event_prefix].append(callback)
        return True

    def unsubscribe(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Unsubscribe a callback from a specific event.
        
        Args:
            event_prefix (str): The event name or pattern to unsubscribe from
            callback (SubscriberCallback): The callback function to remove
            
        Returns:
            bool: True if the callback was successfully removed, False if the event prefix wasn't found
        """
        if event_prefix.endswith('.*'):
            event_prefix = event_prefix[:-2]
        if event_prefix in self._subscribers:
            self._subscribers[event_prefix].remove(callback)
            return True
        return False

    def publish(self, event_name: str, *args, **kwargs) -> bool:
        """Publish an event to all matching subscribers.

        The event will be sent to:
        1. Subscribers that exactly match the event name
        2. Subscribers with prefix patterns (ending in '.*') where the event name starts with the prefix

        Args:
            event_name (str): The name of the event to publish
            *args: Positional arguments to pass to the callbacks
            **kwargs: Keyword arguments to pass to the callbacks

        Returns:
            bool: True if any subscribers received the event, False otherwise
        """
        # (1) Get exact-match subscribers
        exact_callbacks = self._subscribers.get(event_name, [])

        # (2) Find prefix-matching subscribers
        prefix_callbacks : List[SubscriberCallback]= []
        for pattern, callbacks in self._subscribers.items():
            if pattern.endswith('.*'):
                prefix = pattern[:-2]
                if event_name.startswith(prefix):
                    prefix_callbacks.extend(callbacks)

        ## Invoke the callbacks with deduplication ( preserve order of subscription)
        seen = set()
        for cb in exact_callbacks + prefix_callbacks:
            if cb not in seen:
                cb(event_name, *args, **kwargs)
                seen.add(cb)

        return len(seen) > 0






