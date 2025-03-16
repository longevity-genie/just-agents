from typing import Callable, Optional, List, Dict, Any, Sequence, Union, TypeVar, Type, Tuple
from pydantic import BaseModel, Field, PrivateAttr
from just_agents.just_bus import JustToolsBus, VariArgs, SubscriberCallback
from importlib import import_module
import inspect
from docstring_parser import parse
from pydantic import ConfigDict
import sys
import re


# Create a TypeVar for the class
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar('Self', bound='JustTool')

class LiteLLMDescription(BaseModel):

    model_config = ConfigDict(populate_by_name=True)
    name: Optional[str] = Field(..., alias='function', description="The name of the function")
    description: Optional[str] = Field(None, description="The docstring of the function.")
    parameters: Optional[Dict[str,Any]]= Field(None, description="Parameters of the function.")

class JustTool(LiteLLMDescription):
    package: str = Field(..., description="The name of the module where the function is located.")
    auto_refresh: bool = Field(True, description="Whether to automatically refresh the tool after initialization.")
    max_calls_per_query: Optional[int] = Field(None, ge=1, description="The maximum number of calls to the function per query.")
    model_config = ConfigDict(
        extra="allow",
    )

    _callable: Optional[Callable] = PrivateAttr(default=None)
    """The callable function wrapped with the JustToolsBus callbacks."""
    _raw_callable: Optional[Callable] = PrivateAttr(default=None)
    """The original callable function."""
    _calls_made: int = PrivateAttr(default=0)
    """Counter for tracking how many times this tool has been called."""

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
            self.get_callable() #populate callables on the instance level if unset

    def _wrap_function(self, func: Callable, name: str) -> Callable:
        """
        Helper to wrap a function with event publishing logic to JustToolsBus.
        """
        def __wrapper(*args: Any, **kwargs: Any) -> Any:
            bus = JustToolsBus()
            bus.publish(f"{name}.{id(self)}.execute", *args, kwargs=kwargs)
            
            try:
                # Check for maximum calls
                if self.max_calls_per_query is not None:
                    if self._calls_made >= self.max_calls_per_query:
                        error = RuntimeError(f"Maximum number of calls ({self.max_calls_per_query}) reached for {name}")
                        bus.publish(f"{name}.{id(self)}.error", error=error)
                        raise error
                
                # Execute function and record call
                result = func(*args, **kwargs)
                self._calls_made += 1
                
                bus.publish(f"{name}.{id(self)}.result", result_interceptor=result, kwargs=kwargs)
                return result
            except Exception as e:
                bus.publish(f"{name}.{id(self)}.error", error=e)
                raise e
        return __wrapper

    @staticmethod
    def function_to_llm_dict(input_function: Callable) -> Dict[str, Any]:
        """
        Extract function metadata for function calling format without external dependencies.
        
        Args:
            input_function: The function to extract metadata from
            
        Returns:
            Dict with function name, description and parameters
        """
        # Map Python types to JSON schema types
        python_to_json_schema_types: Dict[str, str] = {
            str.__name__: "string",
            int.__name__: "integer",
            float.__name__: "number",
            bool.__name__: "boolean",
            list.__name__: "array",
            dict.__name__: "object",
            "NoneType": "null",
        }

        # Extract basic function information
        name: str = input_function.__name__
        docstring: Optional[str] = inspect.getdoc(input_function)
        
        # Parse the docstring using docstring_parser
        if docstring:
            parsed_docstring = parse(docstring)
        else:
            raise ValueError("No docstring found for function")
        
        # Get the function description from the short description or empty string if none
        description: str = parsed_docstring.short_description if parsed_docstring else ""
        #if parsed_docstring and parsed_docstring.long_description:
        #    description += "\n" + parsed_docstring.long_description
        
        # Initialize parameters and required parameters
        parameters: Dict[str, Dict[str, Any]] = {}
        required_params: List[str] = []
        
        # Get function signature information
        param_info = inspect.signature(input_function).parameters

        # Process each parameter
        for param_name, param in param_info.items():
            # Determine parameter type from annotation
            param_type: Optional[str] = None
            if param.annotation != param.empty:
                # Get the type name if it's a class, otherwise use string representation
                type_name = getattr(param.annotation, "__name__", str(param.annotation))
                param_type = python_to_json_schema_types.get(type_name, "string")
            
            param_description: Optional[str] = None
            param_enum: Optional[List[str]] = None

            # Find parameter description from docstring
            if parsed_docstring:
                for param_doc in parsed_docstring.params:
                    if param_doc.arg_name == param_name:
                        param_description = param_doc.description
                        
                        # Check if type is specified in docstring
                        if param_doc.type_name:
                            docstring_type = param_doc.type_name
                            
                            # Handle optional types
                            if "optional" in docstring_type.lower():
                                docstring_type = docstring_type.split(",")[0].strip()
                            
                            # Handle enum-like values in curly braces (e.g. "{option1, option2}")
                            elif "{" in docstring_type and "}" in docstring_type:
                                # Extract content between curly braces
                                match = re.search(r'\{([^}]+)}', docstring_type)
                                if match:
                                    # Split by comma and clean up whitespace
                                    options = [opt.strip() for opt in match.group(1).split(',')]
                                    # Filter out empty strings
                                    options = [opt for opt in options if opt]
                                    if options:
                                        param_enum = options
                                        param_type = "string"  # Enum values are represented as strings
                            
                            # Map to JSON schema type
                            param_type = python_to_json_schema_types.get(docstring_type, "string")

            # Create parameter dictionary
            param_dict: Dict[str, Any] = {
                "type": param_type,
                "description": param_description,
                "enum": param_enum,
            }

            # Filter out None values
            parameters[param_name] = {k: v for k, v in param_dict.items() if v is not None}

            # Check if parameter is required (no default value)
            if param.default == param.empty:
                required_params.append(param_name)

        # Create the final result dictionary
        result: Dict[str, Any] = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
            },
        }

        # Add required parameters if any
        if required_params:
            result["parameters"]["required"] = required_params
        
        # Verify our implementation matches litellm's result (for debugging during refactoring)
        # This can be removed later once the implementation is stable
        # assert result == litellm_result, f"Implementation mismatch with litellm:\nOur result: {result}\nLiteLLM result: {litellm_result}"
        
        return result

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
            input_function: Function to convert to a JustTool
            
        Returns:
            JustTool instance with function metadata
        """
        package = input_function.__module__
        function_name = input_function.__name__

        # Use our own implementation or fallback based on the flag
        litellm_description = cls.function_to_llm_dict(input_function)
        litellm_description['function'] = function_name

        return cls(
            **litellm_description,
            package=package
        )

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
        Updates package, function name, description, parameters, and ensures the function is importable.
        
        Returns:
            JustTool: Returns self to allow method chaining or direct appending.
        """
        try:
            # Get the function from the module
            func = getattr(import_module(self.package), self.name)
            
            # Use our own implementation to get function metadata
            litellm_description = self.function_to_llm_dict(func)
            
            # Update the description
            self.description = litellm_description.get("description")
            
            # Update parameters
            self.parameters = litellm_description.get("parameters")
            
            # Rewrap with the updated callable
            self._callable = self._wrap_function(func, self.name)
            self._raw_callable = func
            
            return self
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error refreshing {self.name} from {self.package}: {e}") from e

    def get_callable(self, refresh: bool = False, wrap: bool = True) -> Callable:
        """
        Retrieve the callable function.
        
        Args:
            refresh: If True, the callable is refreshed before returning
            wrap: If True, the callable is wrapped with the JustToolsBus callbacks

        Returns:
            Wrapped callable function
            
        Raises:
            ImportError: If function cannot be imported
        """
        if refresh:
            self.refresh()
        if self._callable is not None and wrap:
            return self._callable
        if self._callable is not None and not wrap:
            return self._raw_callable
        try:
            self._raw_callable = getattr(import_module(self.package), self.name)
            self._callable = self._wrap_function(self._raw_callable, self.name)
            if wrap:
                return self._callable
            else:
                return self._raw_callable
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error importing {self.name} from {self.package}: {e}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Allows the JustTool instance to be called like a function.
        
        Args:
            *args: Positional arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function
            
        Returns:
            Result of the wrapped function
        """
        func = self.get_callable(wrap=True)
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