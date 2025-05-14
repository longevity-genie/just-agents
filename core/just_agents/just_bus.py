from collections import deque
from typing import Callable, Dict, List, Any, ParamSpec, Protocol, Tuple, Deque, Optional, Set

class SingletonMeta(type):
    """
    A metaclass that creates a Singleton base type when called.
    """
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls.__qualname__ not in cls._instances:
            cls._instances[cls.__qualname__] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls.__qualname__]

VariArgs = ParamSpec('VariArgs')

class SubscriberCallback(Protocol):
    """Protocol defining the signature for event subscribers."""
    def __call__(self, event_prefix: str, *args: VariArgs.args, **kwargs: VariArgs.kwargs) -> None:
        ...


class JustEventBus(metaclass=SingletonMeta):
    """
    A simple singleton event bus to publish function call results for functions that support it.
    Event names can be arbitrary (suggested usage is function names).

    This base version has no buffering - events with no subscribers are simply lost.

    Note: This implementation is process-specific. Each forked process will have
    its own independent instance with separate subscribers.
    """
    _subscribers: Dict[str, List[SubscriberCallback]]
    _signature_registry: Dict[str, List[str]]

    def __init__(self) -> None:
        """Initialize the basic event bus."""
        self._subscribers = {}
        self._signature_registry = {}

    @staticmethod
    def extract_function_signature(callback: SubscriberCallback) -> dict[str, str]:
        """
        Extracts signature information from a callback function.
        
        Args:
            callback: The callback function to analyze
            
        Returns:
            A string containing the signature of the function, formatted as "ClassName.method_name" 
            or "__UNBOUND__.method_name" if unbound
        """
        # Extract bound instance information if available
        bound_instance = getattr(callback, '__self__', None)
        
        
        if bound_instance is not None:
            # This is a bound method - extract class and method information
            class_name = type(bound_instance).__name__
            # Get method name (qualname preferably)
            method_name = getattr(callback, '__qualname__', None)
            if method_name is None:
                method_name = getattr(callback, '__name__', str(callback))
                
            # Create a signature for this class+method combination
          
        else:
            # Handle unbound functions
            class_name = "__UNBOUND__"
            method_name = getattr(callback, '__name__', str(callback))

        signature = f"{class_name}.{method_name}"
        return signature

    def _get_signatures_for_prefix(self, event_prefix: str) -> List[str]:
        """Get all signatures for a given event prefix."""
        signatures = []
        for callback in self._subscribers[event_prefix]:
            signature = self.extract_function_signature(callback)
            if signature not in signatures:
                signatures.append(signature)
        return signatures

    def subscribe(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Subscribe a callback to a specific event.

        Args:
            event_prefix: The event name or pattern to subscribe to.
                          For example, 'mytool.call' or 'mytool.*'.
            callback: The callback function to be invoked when matching events are published.

        Returns:
            True indicating successful subscription.
        """
        if event_prefix not in self._subscribers:
            self._subscribers[event_prefix] = []
        if event_prefix not in self._signature_registry:
            self._signature_registry[event_prefix] = []
        signature = self.extract_function_signature(callback)
        if signature not in self._signature_registry[event_prefix]:
            self._signature_registry[event_prefix].append(signature)
        self._subscribers[event_prefix].append(callback)
        
        return True

    def subscribe_unique_by_class(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Subscribe a callback with class-level uniqueness enforcement.
        
        If another method with the same name from the same class is already subscribed 
        to this event, this new subscription will not be added.
        
        Use this method when you want to ensure only one instance of each class
        can subscribe with the same method name (useful for shared tools/adapters).
        
        Args:
            event_prefix: The event name or pattern to subscribe to.
                          For example, 'mytool.call' or 'mytool.*'.
            callback: The callback function to be invoked when matching events are published.
            
        Returns:
            True if subscription was added, False if it already exists.
        """
        signature = self.extract_function_signature(callback)
        if signature not in self._signature_registry[event_prefix]:
            self.subscribe(event_prefix, callback)
            return True
        return False


    def unsubscribe(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Unsubscribe a callback from a specific event.

        Args:
            event_prefix: The event name or pattern to unsubscribe from.
                          Must match exactly the key used during subscription.
            callback: The callback function to remove.

        Returns:
            True if the callback was successfully removed, False otherwise.
        """
        if event_prefix in self._subscribers:
            try:
                self._subscribers[event_prefix].remove(callback)
                self._signature_registry[event_prefix] = self._get_signatures_for_prefix(event_prefix) #rebuild the registry node
                return True
            except ValueError:
                # Callback not found in the list
                return False
        return False

    def publish(self, event_name: str, *args: Any, **kwargs: Any) -> bool:
        """Publish an event to all matching subscribers.

        The event is delivered to:
          1. Subscribers that exactly match the event name.
          2. Subscribers with prefix patterns (ending in '.*') where the event name starts with the prefix.

        Args:
            event_name: The name of the event to publish.
            *args: Positional arguments to pass to the callbacks.
            **kwargs: Keyword arguments to pass to the callbacks.

        Returns:
            True if any subscribers received the event, False otherwise.
        """
        return self._dispatch_event(event_name, *args, **kwargs)

    def _dispatch_event(self, event_name: str, *args: Any, **kwargs: Any) -> bool:
        """Internal method to dispatch an event to matching subscribers.

        Args:
            event_name: The name of the event to dispatch.
            *args: Positional arguments to pass to the callbacks.
            **kwargs: Keyword arguments to pass to the callbacks.

        Returns:
            True if at least one subscriber received the event.
        """
        # Get exact-match subscribers.
        exact_callbacks = self._subscribers.get(event_name, [])

        # Find prefix-matching subscribers.
        prefix_callbacks: List[SubscriberCallback] = []
        for pattern, callbacks in self._subscribers.items():
            if pattern.endswith('.*'):
                prefix = pattern[:-2]
                if event_name.startswith(prefix):
                    prefix_callbacks.extend(callbacks)

        # Invoke callbacks with deduplication (preserving subscription order).
        seen: Set[SubscriberCallback] = set()
        for cb in exact_callbacks + prefix_callbacks:
            if cb not in seen:
                cb(event_name, *args, **kwargs)
                seen.add(cb)
        return len(seen) > 0

class BufferedEventBus(JustEventBus):
    """
    An enhanced event bus that buffers events when no subscribers exist.

    If no subscriber receives a published event, the event is buffered (up to a configurable 
    number of items). Buffer flush is attempted at each publish or subscribe event,
    preserving the order of events.
    """
    _buffer: Deque[Tuple[str, Tuple[Any, ...], Dict[str, Any]]]  # (event_name, args, kwargs)
    _buffer_max_size: int

    def __init__(self, buffer_size: int = 255) -> None:
        """
        Initialize the buffered event bus.

        Args:
            buffer_size: Maximum number of events to keep in the buffer when no subscribers exist.
                         When the buffer is full, the oldest events are dropped.
        """
        super().__init__()
        self._buffer_max_size = buffer_size
        self._buffer = deque(maxlen=self._buffer_max_size)

    def subscribe_unique_by_class(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Subscribe a callback with class-level uniqueness enforcement.
        
        If another method with the same name from the same class is already subscribed 
        to this event, this new subscription will not be added.
        
        Args:
            event_prefix: The event name or pattern to subscribe to.
                          For example, 'mytool.call' or 'mytool.*'.
            callback: The callback function to be invoked when matching events are published.

        Returns:
            True if subscription was added, False if it already exists.
        """
        result = super().subscribe_unique_by_class(event_prefix, callback)
        self._flush_buffer()
        return result 

    def subscribe(self, event_prefix: str, callback: SubscriberCallback) -> bool:
        """Subscribe a callback to a specific event and attempt to flush the buffer.

        Args:
            event_prefix: The event name or pattern to subscribe to.
            callback: The callback function to be invoked when matching events are published.

        Returns:
            True indicating successful subscription.
        """
        result = super().subscribe(event_prefix, callback)
        self._flush_buffer()
        return result


    def publish(self, event_name: str, *args: Any, **kwargs: Any) -> bool:
        """Publish an event to all matching subscribers; buffer it if not delivered.

        The event is delivered to:
          1. Subscribers that exactly match the event name.
          2. Subscribers with prefix patterns (ending in '.*') where the event name starts with the prefix.

        Args:
            event_name: The name of the event to publish.
            *args: Positional arguments to pass to the callbacks.
            **kwargs: Keyword arguments to pass to the callbacks.

        Returns:
            True if any subscribers received the event, False otherwise.
            If no subscriber received the event, it is buffered.
        """
        delivered = super().publish(event_name, *args, **kwargs)
        if not delivered:
            self._buffer.append((event_name, args, kwargs))
        self._flush_buffer()
        return delivered

    def _flush_buffer(self, trim_prefix: Optional[str] = None) -> int:
        """Attempt to flush buffered events by re-publishing them.

        Args:
            trim_prefix: If provided, events with this prefix will be removed without dispatching.

        Returns:
            The number of events removed from the buffer.
        """
        removed_count = 0
        buffer_size = len(self._buffer)
        
        # Process the current events; new events appended during processing will be handled later.
        for _ in range(buffer_size):
            if not self._buffer:
                break

            event_name, args, kwargs = self._buffer.popleft()

            # If trim_prefix is specified and matches, remove this event.
            if trim_prefix and event_name.startswith(trim_prefix):
                removed_count += 1
                continue

            # Try to dispatch the event.
            if not super()._dispatch_event(event_name, *args, **kwargs):
                # If still undeliverable, add it back.
                self._buffer.append((event_name, args, kwargs))
            else:
                removed_count += 1
        
        return removed_count

    def trim_by_prefix(self, prefix: str) -> int:
        """Remove all buffered events with a specific prefix without dispatching them.

        Args:
            prefix: The event name prefix to match for removal.

        Returns:
            The number of events removed from the buffer.
        """
        return self._flush_buffer(trim_prefix=prefix)

class JustToolsBus(JustEventBus):
    """
    A simple singleton tools bus.
    Inherits from JustEventBus with no additional changes.
    """
    pass

class JustLogBus(BufferedEventBus):
    """
    A simple singleton log events bus.
    Inherits from BufferedEventBus with no additional changes.
    """

    @staticmethod
    def log_message(message: str, source: str = 'anonymous', action: str = "log_bus.log_entry", **kwargs: Any) -> None:
        """A helper to log a message from a source via the log bus.

        Args:
            message: The message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus().publish(source, log_message=message, action=action, **kwargs)
        
    @staticmethod
    def trace(message: str, source: str = 'anonymous', action: str = "log_bus.trace", **kwargs: Any) -> None:
        """Log a TRACE level message.
        
        Args:
            message: The trace message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus.log_message(message, source, action, severity="TRACE", **kwargs)
        
    @staticmethod
    def debug(message: str, source: str = 'anonymous', action: str = "log_bus.debug", **kwargs: Any) -> None:
        """Log a DEBUG level message.
        
        Args:
            message: The debug message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus.log_message(message, source, action, severity="DEBUG", **kwargs)
        
    @staticmethod
    def info(message: str, source: str = 'anonymous', action: str = "log_bus.info", **kwargs: Any) -> None:
        """Log an INFO level message.
        
        Args:
            message: The info message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus.log_message(message, source, action, severity="INFO", **kwargs)
        
    @staticmethod
    def warn(message: str, source: str = 'anonymous', action: str = "log_bus.warn", **kwargs: Any) -> None:
        """Log a WARN level message.
        
        Args:
            message: The warning message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus.log_message(message, source, action, severity="WARN", **kwargs)

    @staticmethod
    def warning(message: str, source: str = 'anonymous', action: str = "log_bus.warn", **kwargs: Any) -> None:
        """Log a WARN level message. Alias for JustLogBus.warn."""
        JustLogBus.warn(message, source, action, **kwargs)
    
    @staticmethod
    def error(message: str, source: str = 'anonymous', action: str = "log_bus.error", **kwargs: Any) -> None:
        """Log an ERROR level message.
        
        Args:
            message: The error message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus.log_message(message, source, action, severity="ERROR", **kwargs)
        
    @staticmethod
    def fatal(message: str, source: str = 'anonymous', action: str = "log_bus.fatal", **kwargs: Any) -> None:
        """Log a FATAL level message.
        
        Args:
            message: The fatal error message to log
            source: The source of the log message
            action: The action related to the log message
            **kwargs: Additional parameters to log
        """
        JustLogBus.log_message(message, source, action, severity="FATAL", **kwargs)
    
    @staticmethod
    def mask_api_key(key: Optional[str]) -> str:
        """
        Masks API key by showing first 1/3 of the key and replacing the rest with asterisks.
        
        Args:
            key: The API key to mask
            
        Returns:
            Optional[str]: Masked API key or None if key is None
        """
        if not key:
            return "None"
        
        visible_chars = max(len(key) // 3, 1)  # Show at least 1 character
        masked_length = len(key) - visible_chars
        return key[:visible_chars] + '*' * masked_length

    @staticmethod
    def debug_binding(func: Callable) -> Callable:
        """
        Creates a decorator that logs binding information for methods to help with debugging.
        Shows which instance a method is bound to when called.
        
        Args:
            func: The function or method to decorate
            
        Returns:
            A decorated function that logs binding information
            
        Example:
            class MyClass:
                @JustLogBus.debug_binding
                def my_method(self, arg):
                    return arg
                    
            # Or for callbacks:
            button.on_click(JustLogBus.debug_binding(self.handle_click))
        """
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get binding information
            bound_to = getattr(func, '__self__', None)
            func_name = getattr(func, '__qualname__', func.__name__)
            
            # Log based on binding type
            if bound_to:
                JustLogBus.debug(
                    f"Calling {func_name} bound to {bound_to}",
                    source="debug_binding",
                    action="function.call",
                    bound_instance=str(bound_to),
                    function=func_name
                )
            else:
                # Unbound method case
                if args and isinstance(args[0], object) and not isinstance(args[0], type):
                    JustLogBus.debug(
                        f"Calling {func_name} with self={args[0]}",
                        source="debug_binding",
                        action="function.call",
                        unbound_with_self=str(args[0]),
                        function=func_name
                    )
                else:
                    JustLogBus.debug(
                        f"Calling unbound {func_name}",
                        source="debug_binding",
                        action="function.call",
                        unbound=True,
                        function=func_name
                    )
                
            # Execute the original function
            return func(*args, **kwargs)
        
        return wrapper

