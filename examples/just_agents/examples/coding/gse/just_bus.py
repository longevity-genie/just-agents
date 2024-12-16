from typing import Callable, Dict, List

class SingletonMeta(type):
    """
    A metaclass that creates a Singleton base type when called.
    """
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

SubscriberCallback = Callable[[...],None]

class JustEventBus(metaclass=SingletonMeta):
    """
    A simple singleton event bus to publish function call results for functions that support it
    Event name can be anything, but suggested use is function names.
    """
    def __init__(self):
        # Type hint explicitly in __init__ to satisfy type checkers
        self._subscribers: Dict[str, List[SubscriberCallback]] = {}

    def subscribe(self, event_name: str, callback: SubscriberCallback):
        """Subscribe a callback to a specific event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)
       # print(f"Subscribed to event '{event_name}': {callback.__name__}")

    def publish(self, event_name: str, *args, **kwargs):
        """Publish an event to all its subscribers."""
        subscribers = self._subscribers.get(event_name, [])
       # print(f"Publishing event '{event_name}' to {len(subscribers)} subscriber(s).")
        for callback in subscribers:
            callback(*args, **kwargs)