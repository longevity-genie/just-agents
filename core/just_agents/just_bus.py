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

SubscriberCallback = Callable[[str,...],None]

class JustEventBus(metaclass=SingletonMeta):
    """
    A simple singleton event bus to publish function call results for functions that support it
    Event name can be anything, but suggested use is function names.
    """

    _subscribers: Dict[str, List[SubscriberCallback]]

    def __init__(self):
        # Dictionary of subscription_pattern -> list_of_callbacks (in subscription order)
        self._subscribers = {}

    def subscribe(self, event_prefix: str, callback: SubscriberCallback):
        """Subscribe a callback to a specific event."""
        """
        Subscribe to an event name or prefix:
          e.g. 'mytool.call' or 'mytool.*'
        """
        if event_prefix not in self._subscribers:
            self._subscribers[event_prefix] = []
        self._subscribers[event_prefix].append(callback)

    def publish(self, event_name: str, *args, **kwargs):
        """
        Publish an event to:
          1) Any exact-match subscriber (same string)
          2) Any prefix subscriber (ends with '.*' and event_name.startswith(prefix))
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






