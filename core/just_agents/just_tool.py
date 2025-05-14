from typing import Callable, Optional, List, Dict, Any, Sequence, Union, TypeVar, Type, Tuple, get_origin, get_args
from pydantic import BaseModel, Field, PrivateAttr, create_model
from just_agents.just_bus import JustToolsBus, VariArgs, SubscriberCallback
from importlib import import_module
import inspect

from pydantic import ConfigDict
import sys
import re
from just_agents.just_async import run_async_function_synchronously
from just_agents.just_schema import ModelHelper


# Create a TypeVar for the class
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar('Self', bound='JustTool')

class LiteLLMDescription(BaseModel):

    model_config = ConfigDict(populate_by_name=True)
    name: Optional[str] = Field(..., alias='function', description="The simple name of the function or method (e.g., 'my_function' or 'my_method').")
    description: Optional[str] = Field(None, description="The docstring of the function.")
    parameters: Optional[Dict[str,Any]]= Field(None, description="Parameters of the function.")

class JustTool(LiteLLMDescription):
    package: str = Field(..., description="The name of the module where the function or its containing class is located (e.g., 'my_module').")
    static_class: Optional[str] = Field(None, description="The qualified name of the class if the tool is a static/class method, relative to the package (e.g., 'MyClass' or 'OuterClass.InnerClass').")
    auto_refresh: bool = Field(True, description="Whether to automatically refresh the tool after initialization.")
    max_calls_per_query: Optional[int] = Field(None, ge=1, description="The maximum number of calls to the function per query.")
    is_async: bool = Field(False, description="True if the tool is an async (non-generator) function.")
    model_config = ConfigDict(
        extra="allow",
    )

    _callable: Optional[Callable] = PrivateAttr(default=None)
    """The callable function wrapped with the JustToolsBus callbacks."""
    _raw_callable: Optional[Callable] = PrivateAttr(default=None)
    """The original callable function."""
    _calls_made: int = PrivateAttr(default=0)
    """Counter for tracking how many times this tool has been called."""
    _pydantic_model: Optional[Type[BaseModel]] = PrivateAttr(default=None)
    """The dynamically generated Pydantic model for the tool's parameters."""

    @property
    def remaining_calls(self) -> int:
        """
        Returns the number of remaining calls allowed for this tool.
        
        Returns:
            int: Number of calls remaining, or -1 if unlimited
        """
        if self.max_calls_per_query is None:
            return -1  # Placeholder for infinity
        return max(0, self.max_calls_per_query - self._calls_made)

    def reset(self) -> Self:
        """
        Reset the call counter for this tool.
        """
        self._calls_made = 0
        return self
    
    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized. Refreshes the tools meta-info if auto_refresh is True."""
        super().model_post_init(__context)
        if self.auto_refresh:
            self.refresh()
        if self._callable is None or self._raw_callable is None:
            # Attempt to populate callables if not done by auto_refresh
            # This will raise ImportError if the tool is unresolvable.
            self.get_callable_sync(refresh=False, wrap=True) # Ensures _resolve_callable is attempted

    def _max_calls_wrapper(self, wrapped_function: Callable, tool_name: str) -> Callable:
        """Outer wrapper to enforce max_calls_per_query if set."""
        def __max_calls_final_wrapper(*args: Any, **kwargs: Any) -> Any:
            # This check happens *before* calling the underlying wrapped_function
            if self.max_calls_per_query is not None and self._calls_made >= self.max_calls_per_query:
                error = RuntimeError(f"Maximum number of calls ({self.max_calls_per_query}) reached for {tool_name}")
                # Assuming JustToolsBus is accessible or we publish differently if needed.
                # For simplicity, let's assume bus is available here if events are desired for this specific error.
                # However, the primary error propagation will be the raised RuntimeError.
                JustToolsBus().publish(f"{tool_name}.{id(self)}.error", error=error) # Optional: for consistent eventing
                raise error
            
            # Call the actual wrapped function (simple or parsing logic)
            result = wrapped_function(*args, **kwargs)
            
            # Increment calls *after* the successful execution (or attempt) of the wrapped function
            # The wrapped_function itself will handle its own try/except for tool execution errors
            # and publish .result or .error accordingly.
            # This counter is for the number of *attempts* under the max_calls_per_query limit.
            self._calls_made += 1
            return result
        return __max_calls_final_wrapper

    def _simple_wrapper_logic(self, func_to_run: Callable, tool_name: str) -> Callable:
        """Simple wrapper without dictionary parsing. Max calls check is handled by _max_calls_wrapper."""
        def __simple_wrapper(*args: Any, **kwargs: Any) -> Any:
            bus = JustToolsBus()
            bus.publish(f"{tool_name}.{id(self)}.execute", *args, kwargs=kwargs)
            try:
                # Max calls check and increment removed from here
                if self.is_async:
                    result = run_async_function_synchronously(func_to_run, *args, **kwargs)
                else:
                    result = func_to_run(*args, **kwargs)
                # Call increment removed from here
                bus.publish(f"{tool_name}.{id(self)}.result", result_interceptor=result, kwargs=kwargs)
                return result
            except Exception as e:
                bus.publish(f"{tool_name}.{id(self)}.error", error=e)
                raise e
        return __simple_wrapper

    def _parsing_wrapper_logic(self, func_to_run: Callable, tool_name: str) -> Callable:
        """Wrapper that inspects raw callable and parses string-to-dict for relevant params. Max calls check is handled by _max_calls_wrapper."""
        raw_func_sig = inspect.signature(self._raw_callable)

        def __parsing_wrapper(*args: Any, **kwargs: Any) -> Any:
            bus = JustToolsBus()
            bus.publish(f"{tool_name}.{id(self)}.execute", *args, kwargs=kwargs)
            
            processed_kwargs = {}
            for key, value in kwargs.items():
                param = raw_func_sig.parameters.get(key)
                if param and param.annotation != inspect.Parameter.empty:
                    param_type_origin = get_origin(param.annotation)
                    if (param_type_origin is dict or param_type_origin is Dict) and isinstance(value, str):
                        try:
                            parsed_value = ModelHelper.get_structured_output(value, parser=dict)
                            processed_kwargs[key] = parsed_value
                        except ValueError:
                            processed_kwargs[key] = value 
                    else:
                        processed_kwargs[key] = value
                else:
                    processed_kwargs[key] = value

            try:
                # Max calls check and increment removed from here
                if self.is_async:
                    result = run_async_function_synchronously(func_to_run, *args, **processed_kwargs)
                else:
                    result = func_to_run(*args, **processed_kwargs)
                # Call increment removed from here
                bus.publish(f"{tool_name}.{id(self)}.result", result_interceptor=result, kwargs=processed_kwargs)
                return result
            except Exception as e:
                bus.publish(f"{tool_name}.{id(self)}.error", error=e)
                raise e
        return __parsing_wrapper

    def _wrap_function(self, func_to_run: Callable, tool_name: str) -> Callable:
        """
        Dispatcher: Chooses a core wrapper (simple or parsing) and then optionally applies the max_calls_wrapper.
        The func_to_run is the already-resolved callable.
        """
        core_wrapped_function: Callable
        if self._pydantic_model and ModelHelper.model_has_dict_params(self._pydantic_model):
            core_wrapped_function = self._parsing_wrapper_logic(func_to_run, tool_name)
        else:
            core_wrapped_function = self._simple_wrapper_logic(func_to_run, tool_name)
        
        # Optionally apply the max_calls_per_query wrapper
        if self.max_calls_per_query is not None:
            return self._max_calls_wrapper(core_wrapped_function, tool_name)
        else:
            return core_wrapped_function

    @staticmethod
    def function_to_llm_dict(input_function: Callable) -> Dict[str, Any]:
        """
        Extract function metadata for function calling format, leveraging Pydantic for schema generation.
        
        Args:
            input_function: The function to extract metadata from
            
        Returns:
            Dict with function name, description and parameters
        """
        # Call ModelHelper to get schema and model, then return only the schema part
        schema, _ = ModelHelper.create_tool_schema_from_callable(input_function)
        return schema

    def _resolve_callable(self) -> Callable:
        """
        Helper to import and retrieve the callable based on self.package, self.name, and self.static_class.
        self.name is expected to be the simple function/method name.
        self.package is the module path.
        self.static_class (if provided) is the class name (potentially nested, e.g., 'Outer.Inner').
        """
        try:
            module = import_module(self.package)
        except ImportError as e:
            raise ImportError(f"Could not import module '{self.package}'. Error: {e}") from e

        if self.static_class:
            # Attempt to get a static/class method
            try:
                # Resolve potentially nested class
                current_object = module
                resolved_class_path_upto = module.__name__ # Keep track of how much of the path we successfully resolved
                failed_part = ""

                for class_part in self.static_class.split('.'):
                    try:
                        current_object = getattr(current_object, class_part)
                        resolved_class_path_upto += f".{class_part}"
                    except AttributeError as e:
                        failed_part = class_part
                        # This error means a segment of the static_class path failed.
                        # current_object here is the object *before* failing to find class_part
                        raise ImportError(f"Could not resolve inner class segment '{failed_part}' in '{resolved_class_path_upto}' while trying to find '{self.static_class}' in module '{self.package}'. Error: {e}") from e
                
                class_obj = current_object # This is the fully resolved class
                
                # Now try to get the method from the resolved class
                try:
                    target_method = getattr(class_obj, self.name)
                except AttributeError as e:
                    # This error means the method was not found on the successfully resolved class_obj
                    raise ImportError(f"Could not resolve method '{self.name}' on class '{self.static_class}' in module '{self.package}'. Error: {e}") from e
                
                if callable(target_method):
                    return target_method
                else:
                    raise AttributeError(f"Attribute '{self.name}' on class '{self.static_class}' from module '{self.package}' is not callable.")
            except AttributeError as e: # Should primarily be from the else clause below if logic is flawed
                 # This path should ideally not be hit if the loop for class_part handles its AttributeErrors by raising ImportError.
                 # If it *is* hit, it implies an issue with resolving the class path that wasn't an ImportError from the loop.
                raise ImportError(f"Unexpected error resolving static method '{self.name}' for class '{self.static_class}' in module '{self.package}'. Error: {e}") from e
        else:
            # Attempt to get a regular function from the module
            try:
                target_function = getattr(module, self.name)
                if callable(target_function):
                    return target_function
                else:
                    raise AttributeError(f"Attribute '{self.name}' in module '{self.package}' is not callable.")
            except AttributeError as e:
                raise ImportError(f"Could not resolve function '{self.name}' from module '{self.package}'. Error: {e}") from e

    def get_litellm_description(self) -> Dict[str, Any]:
        """
        Get the LiteLLM compatible function description.
        
        Returns:
            Dictionary with function metadata for LLM function calling
        """
        dump = self.model_dump(
            mode='json',
            by_alias=False,
            exclude_none=True,
            serialize_as_any=False,
            include=set(self.__class__.__bases__[0].model_fields) #Deprecated until v3, blame pydantic for warnings
        )
        return dump

    @classmethod
    def from_callable(cls, input_function: Callable) -> Self:
        """
        Create a JustTool instance from a callable.
        
        Args:
            input_function: Function or static/class method to convert to a JustTool
            
        Returns:
            JustTool instance with function metadata
        """
        module_name = input_function.__module__
        simple_name = input_function.__name__
        qualified_name = input_function.__qualname__

        static_class_name: Optional[str] = None

        # Check if it's a method (static, class, or instance - though we primarily target static/class)
        # simple_name: 'my_method', qualified_name: 'MyClass.my_method' or 'Outer.Inner.my_method'
        # simple_name: 'my_func', qualified_name: 'my_func'
        if '.' in qualified_name and qualified_name.endswith(f'.{simple_name}'):
            # The part before the last dot is the class qualifier.
            static_class_name = qualified_name[:-len(simple_name)-1]

        # Use ModelHelper to get both schema and model
        llm_dict_params, generated_pydantic_model = ModelHelper.create_tool_schema_from_callable(input_function)

        # Determine if the function is an async (non-generator) function
        is_async_regular = inspect.iscoroutinefunction(input_function) and \
                           not inspect.isasyncgenfunction(input_function)

        instance = cls(
            **llm_dict_params, 
            package=module_name,
            static_class=static_class_name,
            is_async=is_async_regular,
        )
        instance._pydantic_model = generated_pydantic_model
        return instance

    def subscribe(self, callback: SubscriberCallback, type: Optional[str]=None) -> bool:
        """
        Subscribe to the JustToolsBus.

        Args:
            callback: Function to call when event occurs
            type: Event type to subscribe to, or None for all events

        Returns:
            Success status of subscription
        """
        bus = JustToolsBus()
        if type is None:
            return bus.subscribe(f"{self.name}.{id(self)}.*", callback)
        else:
            return bus.subscribe(f"{self.name}.{id(self)}.{type}", callback)

    def unsubscribe(self, callback: SubscriberCallback, type: Optional[str]=None) -> bool:
        """
        Unsubscribe from the JustToolsBus.
        
        Args:
            callback: Function to unsubscribe
            type: Event type to unsubscribe from, or None for all events
            
        Returns:
            Success status of unsubscription
        """
        bus = JustToolsBus()
        if type is None:
            return bus.unsubscribe(f"{self.name}.{id(self)}.*", callback)
        else:
            return bus.unsubscribe(f"{self.name}.{id(self)}.{type}", callback)

    def subscribe_to_call(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to the call event.
        
        Args:
            callback (SubscriberCallback): Callback function that takes event_name (str) and *args, kwargs=kwargs
        """
        if not self.subscribe(callback, "execute"):
            raise ValueError(f"Failed to subscribe to {self.name}.execute")

    def subscribe_to_result(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to the result event.
        
        Args:
            callback (SubscriberCallback): Callback function that takes event_name (str) and result_interceptor=result
        """
        if not self.subscribe(callback, "result"):
            raise ValueError(f"Failed to subscribe to {self.name}.result")

    def subscribe_to_error(self, callback: SubscriberCallback) -> None:
        """
        Subscribe to the error event.
        
        Args:
            callback (SubscriberCallback): Callback function that takes event_name (str) and error=exception
        """
        if not self.subscribe(callback, "error"):
            raise ValueError(f"Failed to subscribe to {self.name}.error")

    def refresh(self) -> Self:
        """
        Refresh the JustTool instance to reflect the current state of the actual function.
        Updates package, function name, description, parameters, is_async, and ensures the function is importable.
        
        Returns:
            JustTool: Returns self to allow method chaining or direct appending.
        """
        try:
            # Get the function from the module or class
            func = self._resolve_callable()
            
            # Use ModelHelper to get schema and model
            litellm_description_schema, generated_pydantic_model = ModelHelper.create_tool_schema_from_callable(func)
            
            # Set _raw_callable *before* _wrap_function is called, so _parsing_wrapper_logic can access it.
            self._raw_callable = func 
            self._pydantic_model = generated_pydantic_model

            # Update the description
            self.description = litellm_description_schema.get("description")
            
            # Update parameters
            self.parameters = litellm_description_schema.get("parameters")

            # Update the is_async flag based on the (potentially new) callable
            self.is_async = inspect.iscoroutinefunction(func) and \
                            not inspect.isasyncgenfunction(func)
            
            # Rewrap with the updated callable
            # self.name is the simple name, used for event bus topics
            self._callable = self._wrap_function(func, self.name)
            
            return self
        except ImportError as e: # _resolve_callable will raise ImportError on failure
            # The error message from _resolve_callable provides context
            raise ImportError(f"Error refreshing {self.name} (class: {self.static_class}) from {self.package}: {e}") from e

    def get_callable_sync(self, refresh: bool = False, wrap: bool = True) -> Callable:
        """
        Retrieve the callable function.
        
        Args:
            refresh: If True, the callable is refreshed before returning
            wrap: If True, the callable is wrapped with the JustToolsBus callbacks

        Returns:
            Wrapped callable function if wrap is True, otherwise the raw callable.
            
        Raises:
            ImportError: If function cannot be imported or resolved.
        """
        if refresh:
            self.refresh() # This populates/updates _callable and _raw_callable, or raises error

        # Check if already populated (by previous call, model_post_init, or successful refresh)
        if wrap and self._callable is not None:
            return self._callable
        if not wrap and self._raw_callable is not None:
            return self._raw_callable

        # If not populated, resolve them now by refreshing
        try:
            self.refresh() 
            
            if wrap:
                return self._callable
            else:
                return self._raw_callable
        except ImportError as e: # This ImportError is from _resolve_callable
            # The error message from _resolve_callable is descriptive.
            raise e

    def get_callable(self, refresh: bool = False, wrap: bool = True) -> Callable:
        """Alias for get_callable_sync. Retrieves the callable function."""
        return self.get_callable_sync(refresh=refresh, wrap=wrap)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Allows the JustTool instance to be called like a function.
        
        Args:
            *args: Positional arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function
            
        Returns:
            Result of the wrapped function
        """
        func = self.get_callable(wrap=True) # Now calls the alias
        return func(*args, **kwargs)


class JustPromptTool(JustTool):
    call_arguments: Optional[Dict[str,Any]] = Field(..., description="Input parameters to call the function with.")
    """Input parameters to call the function with."""

JustTools = Union[
    Dict[str, JustTool],  # A dictionary where keys are strings and values are JustTool instances.
    Sequence[
        Union[JustTool, Callable]
    ]  # A sequence (like a list or tuple) containing either JustTool instances or callable objects (functions).
]

JustPromptTools = Union[
    Dict[str, JustPromptTool],  # A dictionary where keys are strings and values are JustPromptTool instances.
    Sequence[
        Union[JustPromptTool, Tuple[Callable, Dict[str,Any]]]
    ]  # A sequence (like a list or tuple) containing either JustPromptTool instances or callable objects (functions) and input parameters.
]