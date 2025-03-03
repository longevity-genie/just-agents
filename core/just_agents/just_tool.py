from typing import Callable, Optional, List, Dict, Any, Sequence, Union, Literal, TypeVar
from pydantic import BaseModel, Field, PrivateAttr
from just_agents.just_bus import JustEventBus, VariArgs, SubscriberCallback
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

class JustToolsBus(JustEventBus):
    """
    A simple singleton tools bus.
    Inherits from JustEventBus with no additional changes.
    """
    pass


class LiteLLMDescription(BaseModel):

    model_config = ConfigDict(populate_by_name=True)
    
    name: Optional[str] = Field(..., validation_alias='function', description="The name of the function")
    description: Optional[str] = Field(None, description="The docstring of the function.")
    parameters: Optional[Dict[str,Any]]= Field(None, description="Parameters of the function.")

class JustTool(LiteLLMDescription):
    package: str = Field(..., description="The name of the module where the function is located.")
    auto_refresh: bool = Field(True, description="Whether to automatically refresh the tool after initialization.")
    _callable: Optional[Callable] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized. Refreshes the tools metainfo if auto_refresh is True."""
        super().model_post_init(__context)
        if self.auto_refresh:
            self.refresh()

    @staticmethod
    def _wrap_function(func: Callable, name: str) -> Callable:
        """
        Helper to wrap a function with event publishing logic to JustToolsBus.
        """
        def __wrapper(*args: Any, **kwargs: Any) -> Any:
            bus = JustToolsBus()
            bus.publish(f"{name}.execute", *args, kwargs=kwargs)
            try:
                result = func(*args, **kwargs)
                bus.publish(f"{name}.result", result_interceptor=result, kwargs=kwargs)
                return result
            except Exception as e:
                bus.publish(f"{name}.error", error=e)
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
        # Import docstring_parser instead of relying on NumpyDocString
        
        
        # For validation during refactoring - will compare our result with litellm's
        #from litellm.utils import function_to_dict
        #litellm_result = function_to_dict(input_function) 
        # # TODO: reliably replace huge numpydoc/sphynx deps and litellm call. #decoupling #dependencies

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
        if parsed_docstring and parsed_docstring.long_description:
            description += "\n" + parsed_docstring.long_description
        
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
            param_enum: Optional[str] = None

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
                                match = re.search(r'\{([^}]+)\}', docstring_type)
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
            include=set(super().model_fields)
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
        # Use our own implementation instead of litellm's function_to_dict
        litellm_description = cls.function_to_llm_dict(input_function)
        
        # Get function name from the description
        function_name = input_function.__name__

        wrapped_callable = cls._wrap_function(input_function, function_name)
        
        # Ensure function name is in litellm_description
        litellm_description['function'] = function_name

        return cls(
            **litellm_description,
            package=package,
            _callable=wrapped_callable,
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
            return bus.subscribe(f"{self.name}.*", callback)
        else:
            return bus.subscribe(f"{self.name}.{type}", callback)

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
            return bus.unsubscribe(f"{self.name}.*", callback)
        else:
            return bus.unsubscribe(f"{self.name}.{type}", callback)

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

            return self
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error refreshing {self.name} from {self.package}: {e}") from e

    def get_callable(self, refresh: bool = False) -> Callable:
        """
        Retrieve the callable function.
        
        Args:
            refresh: If True, the callable is refreshed before returning
            
        Returns:
            Wrapped callable function
            
        Raises:
            ImportError: If function cannot be imported
        """
        if refresh:
            self.refresh()
        if self._callable is not None:
            return self._callable
        try:
            func = getattr(import_module(self.package), self.name)
            self._callable = self._wrap_function(func, self.name)
            return self._callable
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
        func = self.get_callable()
        return func(*args, **kwargs)

JustTools = Union[
    Dict[str, JustTool],  # A dictionary where keys are strings and values are JustTool instances.
    Sequence[
        Union[JustTool, Callable]
    ]  # A sequence (like a list or tuple) containing either JustTool instances or callable objects (functions).
]
