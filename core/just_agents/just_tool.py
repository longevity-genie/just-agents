import inspect
import sys
from functools import wraps

from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Dict, Any, Sequence, Union, TypeVar, Type, Tuple, get_origin
from pydantic import ConfigDict, BaseModel, Field, PrivateAttr
from importlib import import_module

from just_agents.data_classes import ToolDefinition
from just_agents.just_async import run_async_function_synchronously
from just_agents.just_schema import ModelHelper
from just_agents.just_bus import JustToolsBus, SubscriberCallback

from just_agents.mcp_client import MCPClient, MCPServerParameters


def max_calls_decorator(tool_instance: 'JustToolBase', max_calls: int, tool_name: str):
    """
    Decorator to limit the number of calls to a function.
    
    Args:
        tool_instance: The tool instance to track calls for
        max_calls: Maximum number of calls allowed
        tool_name: Name of the tool for error messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if max calls reached before calling
            if tool_instance._calls_made >= max_calls:
                error = RuntimeError(f"Maximum number of calls ({max_calls}) reached for {tool_name}")
                JustToolsBus().publish(f"{tool_name}.{id(tool_instance)}.error", error=error)
                raise error
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Increment counter after successful call
            tool_instance._calls_made += 1
            return result
        return wrapper
    return decorator


def event_bus_decorator(tool_instance: 'JustToolBase', tool_name: str):
    """
    Decorator to add event bus publishing and async/sync execution handling to a function.
    
    Publishes execute, result, and error events to the JustToolsBus and handles
    both synchronous and asynchronous function execution.
    
    Args:
        tool_instance: The tool instance for accessing is_async and event publishing
        tool_name: Name of the tool for event publishing
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bus = JustToolsBus()
            bus.publish(f"{tool_name}.{id(tool_instance)}.execute", *args, kwargs=kwargs)
            try:
                if tool_instance.is_async:
                    result = run_async_function_synchronously(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                bus.publish(f"{tool_name}.{id(tool_instance)}.result", result_interceptor=result, kwargs=kwargs)
                return result
            except Exception as e:
                bus.publish(f"{tool_name}.{id(tool_instance)}.error", error=e)
                raise e
        return wrapper
    return decorator


def parsing_wrapper_decorator(tool_instance: 'JustToolBase', tool_name: str):
    """
    Decorator that parses string-to-dict for relevant parameters and adds event publishing.
    
    Args:
        tool_instance: The tool instance for accessing raw callable signature and event publishing
        tool_name: Name of the tool for event publishing
    """
    def decorator(func: Callable) -> Callable:
        raw_func_sig = inspect.signature(tool_instance._raw_callable)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bus = JustToolsBus()
            bus.publish(f"{tool_name}.{id(tool_instance)}.execute", *args, kwargs=kwargs)
            
            # Process kwargs for dict parsing
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
                if tool_instance.is_async:
                    result = run_async_function_synchronously(func, *args, **processed_kwargs)
                else:
                    result = func(*args, **processed_kwargs)
                bus.publish(f"{tool_name}.{id(tool_instance)}.result", result_interceptor=result, kwargs=processed_kwargs)
                return result
            except Exception as e:
                bus.publish(f"{tool_name}.{id(tool_instance)}.error", error=e)
                raise e
        return wrapper
    return decorator


def tool_decorator_composer(tool_instance: 'JustToolBase', tool_name: str):
    """
    A decorator composer that applies the appropriate combination of decorators
    based on the tool instance's configuration.
    
    This is a "decorator for decorators" that intelligently combines:
    - Event bus publishing and execution handling
    - Parameter parsing (if needed)
    - Call count limiting (if configured)
    
    Can be used with @ notation:
    @tool_decorator_composer(my_tool_instance, "my_tool")
    def my_function(param1, param2):
        return param1 + param2
    
    Args:
        tool_instance: The tool instance with configuration
        tool_name: Name of the tool for event publishing
    """
    def decorator(func: Callable) -> Callable:
        # Start with the base function
        wrapped_function = func
        
        # Apply parsing wrapper if needed, otherwise apply event bus wrapper
        if tool_instance._pydantic_model and ModelHelper.model_has_dict_params(tool_instance._pydantic_model):
            wrapped_function = parsing_wrapper_decorator(tool_instance, tool_name)(wrapped_function)
        else:
            wrapped_function = event_bus_decorator(tool_instance, tool_name)(wrapped_function)
        
        # Optionally apply the max_calls decorator
        if tool_instance.max_calls_per_query is not None:
            wrapped_function = max_calls_decorator(tool_instance, tool_instance.max_calls_per_query, tool_name)(wrapped_function)
        
        return wrapped_function
    return decorator


# Create a TypeVar for the class
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar('Self', bound='JustToolBase')

class JustToolBase(ToolDefinition, ABC): #TODO interface
    """
    Abstract base class for all Just tools. Defines common interface and functionality.
    """
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
        if self._callable is None or self._raw_callable is None:

            self.get_callable_sync(refresh=self.auto_refresh, wrap=True)

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

    @abstractmethod
    def _get_raw_function_info(self) -> Tuple[Callable, Dict[str, Any], Optional[Type[BaseModel]]]:
        """
        Abstract method to resolve the raw callable, its schema, and Pydantic model.
        Each concrete tool implementation must define its own way to resolve these.

        Returns:
            A tuple containing:
                - The resolved raw callable function
                - The JSON schema for the tool description
                - The dynamically generated Pydantic model for parameters (or None)
        
        Raises:
            Various exceptions depending on implementation if info cannot be resolved, e.g. ImportError, AttributeError, etc.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
            include=set(ToolDefinition.model_fields.keys())
        )
        return dump

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
        Refresh the tool instance to reflect the current state of the actual function.
        Updates description, parameters, is_async, and ensures the function is importable.
        
        Returns:
            Self: Returns self to allow method chaining or direct appending.
        """
        try:
            # Get the raw callable, its schema, and pydantic model
            # using the concrete implementation's method
            func, tool_description_schema, generated_pydantic_model = self._get_raw_function_info()
            
            self._raw_callable = func
            self._pydantic_model = generated_pydantic_model
            
            # Update the description
            self.description = tool_description_schema.get("description")
            
            # Update parameters
            self.parameters = tool_description_schema.get("parameters")

            # Update the is_async flag based on the (potentially new) callable
            self.is_async = inspect.iscoroutinefunction(func) and \
                            not inspect.isasyncgenfunction(func)
            
            # Rewrap with the updated callable
            # self.name is the simple name, used for event bus topics
            self._callable = tool_decorator_composer(self, self.name)(func)
            
            return self
        except Exception as e:
            # The specific error type and message depends on the concrete implementation
            if hasattr(self, 'static_class'):
                raise type(e)(f"Error refreshing {self.name} (class: {self.static_class}) from {self.package}: {e}") from e
            elif hasattr(self, 'package'):
                raise type(e)(f"Error refreshing {self.name} from {self.package}: {e}") from e
            else:
                raise type(e)(f"Error refreshing {self.name}: {e}") from e

    def get_callable_sync(self, refresh: bool = False, wrap: bool = True) -> Callable:
        """
        Retrieve the callable function.
        
        Args:
            refresh: If True, the callable is refreshed before returning
            wrap: If True, the callable is wrapped with the JustToolsBus callbacks

        Returns:
            Wrapped callable function if wrap is True, otherwise the raw callable.
            
        Raises:
            Exception: If function cannot be resolved based on the specific implementation.
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
        except Exception as e:
            # Propagate the error from the concrete implementation
            raise e

    def get_callable(self, refresh: bool = False, wrap: bool = True) -> Callable:
        """Alias for get_callable_sync. Retrieves the callable function."""
        return self.get_callable_sync(refresh=refresh, wrap=wrap)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Allows the tool instance to be called like a function.
        
        Args:
            *args: Positional arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function
            
        Returns:
            Result of the wrapped function
        """
        func = self.get_callable(wrap=True)
        return func(*args, **kwargs)


class JustImportedTool(JustToolBase):
    """
    A tool that is created from a function imported by module path.
    """
    package: str = Field(..., description="The name of the module where the function or its containing class is located (e.g., 'my_module').")
    static_class: Optional[str] = Field(None, description="The qualified name of the class if the tool is a static/class method, relative to the package (e.g., 'MyClass' or 'OuterClass.InnerClass').")

    def refresh(self) -> 'JustImportedTool':
        """
        Refresh the tool instance to reflect the current state of the actual function.
        Updates description, parameters, is_async, and ensures the function is importable.
        
        Returns:
            Self: Returns self to allow method chaining or direct appending.
        """
        return super().refresh()

    def _get_raw_function_info(self) -> Tuple[Callable, Dict[str, Any], Optional[Type[BaseModel]]]:
        """
        Resolves the raw callable, its schema, and Pydantic model for an imported tool.

        Returns:
            A tuple containing:
                - The resolved raw callable function
                - The JSON schema for the tool description
                - The dynamically generated Pydantic model for parameters (or None)
        """
       
        try:
            module = import_module(self.package)
        except ImportError as e:
            raise ImportError(f"Could not import module '{self.package}'. Error: {e}") from e

        raw_callable: Callable
        if self.static_class:
            try:
                current_object = module
                resolved_class_path_upto = module.__name__
                failed_part = ""

                for class_part in self.static_class.split('.'):
                    try:
                        current_object = getattr(current_object, class_part)
                        resolved_class_path_upto += f".{class_part}"
                    except AttributeError as e:
                        failed_part = class_part
                        raise ImportError(f"Could not resolve inner class segment '{failed_part}' in '{resolved_class_path_upto}' while trying to find '{self.static_class}' in module '{self.package}'. Error: {e}") from e
                
                class_obj = current_object
                try:
                    target_method = getattr(class_obj, self.name)
                    if not callable(target_method):
                        raise AttributeError(f"Attribute '{self.name}' on class '{self.static_class}' from module '{self.package}' is not callable.")
                    raw_callable = target_method
                except AttributeError as e:
                    raise ImportError(f"Could not resolve method '{self.name}' on class '{self.static_class}' in module '{self.package}'. Error: {e}") from e
            except ImportError:
                # Re-raise ImportError as-is
                raise
        else:
            try:
                target_function = getattr(module, self.name)
                if not callable(target_function):
                    raise AttributeError(f"Attribute '{self.name}' in module '{self.package}' is not callable.")
                raw_callable = target_function
            except AttributeError as e:
                raise ImportError(f"Could not resolve function '{self.name}' from module '{self.package}'. Error: {e}") from e
                  
        schema, pydantic_model = ModelHelper.create_tool_schema_from_callable(raw_callable)
        return raw_callable, schema, pydantic_model

    @classmethod
    def from_callable(cls, input_function: Callable) -> Self:
        """
        Create a JustImportedTool instance from a callable.
        
        Args:
            input_function: Function or static/class method to convert to a tool
            
        Returns:
            JustImportedTool instance with function metadata
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
        instance._raw_callable = input_function
        instance._callable = tool_decorator_composer(instance, instance.name)(input_function)
        return instance


class JustMCPToolSetConfig(MCPServerParameters):
    """
    A configuration for a set of MCP tools.
    """
    only_include_tools: Sequence[str] = Field(
        default=(),
        description="The names of the MCP tools to include, if empty includes every tool provided by server listing, otherwise only includes the tools in this list"
        )
    exclude_tools: Sequence[str] = Field(default=(), description="The names of the MCP tools to exclude, or empty for none")
    raise_on_incorrect_names: bool = Field(True, description="If true, raise an error if any tool name in include/exclude is not found in the server listing")

class JustMCPTool(MCPServerParameters, JustToolBase):
    """
    Tool implementation for MCP (Model Context Protocol) tools.
    Allows integration of remote or stdio-based MCP tools into the Just Agents framework.
    """
    _mcp_client: Optional[MCPClient] = PrivateAttr(default=None)
    
    def _ensure_mcp_client_instantiated(self) -> None:
        """Ensures self._mcp_client is instantiated if it's None."""
        if self._mcp_client is None:
            self._mcp_client = MCPClient.get_client_by_inputs(self)

    def refresh(self) -> 'JustMCPTool': 
        """
        Refreshes tool metadata by connecting to MCP endpoint and retrieving tool info. 
        Also resets the client reference
        """
        # Reset client reference - we'll get a fresh one on next use
        self._mcp_client = None
        # Ensure we're connected to fetch tool info
        result = super().refresh()
        self._ensure_mcp_client_instantiated()
        return result
            
    def _get_raw_function_info(self) -> Tuple[Callable, Dict[str, Any], Optional[Type[BaseModel]]]:
        """
        Resolves the raw callable, its schema, and Pydantic model for an MCP tool.
        The schema is fetched directly from the MCP endpoint.
        The callable is dynamically generated to match the MCP tool's signature.
        The Pydantic model is generated from the fetched schema.

        Returns:
            A tuple containing:
                - The dynamically generated raw callable function
                - The JSON schema for the tool description (fetched from MCP)
                - The dynamically generated Pydantic model for parameters (or None)
        
        Raises:
            ImportError: If the tool is not found in MCP or schema cannot be processed.
        """

        tool_mcp_schema = run_async_function_synchronously(self._fetch_tool_info) # This returns the full schema including desc

        # tool_mcp_schema["parameters"] is the part needed for model creation
        parameters_schema = tool_mcp_schema.get("parameters", {})
        pydantic_model: Optional[Type[BaseModel]] = None
        if parameters_schema and parameters_schema.get("properties"):
            pydantic_model = ModelHelper.json_schema_to_base_model(parameters_schema, self.name)

        # The tool_mcp_schema is what _fetch_tool_info returns, which is used here.
        # It contains {"description": ..., "parameters": {"type": "object", "properties": ..., "required": ...}}
        
        # No need for makefun, the raw callable will be _async_invoke_tool.
        # The base class wrappers will handle passing arguments appropriately.
        raw_callable = self._async_invoke_tool

        full_llm_schema = {
            "name": self.name,
            "description": tool_mcp_schema.get("description"),
            "parameters": parameters_schema
        }

        return raw_callable, full_llm_schema, pydantic_model
            
    async def _fetch_tool_info(self) -> Dict[str, Any]:
        """
        Connects to MCP and retrieves information about the specified tool.
        """
        self._ensure_mcp_client_instantiated()
        tool_definitions = await self._mcp_client.list_tools_openai()
        
        for tool_def in tool_definitions:
            if tool_def.name == self.name:
                return tool_def.model_dump()
                
        raise ImportError(f"Tool '{self.name}' not found in MCP")

    
    async def _async_invoke_tool(self, *args, **kwargs) -> Any:
        """
        Asynchronously connects to MCP and invokes the tool with the given parameters.
        """
        self._ensure_mcp_client_instantiated()
        result = await self._mcp_client.invoke_tool(self.name, kwargs) 
        if result.error_code != 0:
            raise ValueError(f"MCP tool error: {result.content}, error code: {result.error_code}")
        
        # Parse the structured JSON response to extract the actual content
        # MCP returns content in the format: {"type":"text","text":"actual_value"}
        try:
            import json
            content_lines = result.content.split('\n')
            if len(content_lines) == 1:
                # Single content item
                parsed_content = json.loads(content_lines[0])
                if isinstance(parsed_content, dict) and "text" in parsed_content:
                    # If the text field contains JSON, try to parse it
                    text_content = parsed_content["text"]
                    try:
                        # Try to parse as JSON in case it's a serialized dict/object
                        return json.loads(text_content)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, return as is
                        return text_content
                else:
                    # Return the parsed content directly (for complex objects)
                    return parsed_content
            else:
                # Multiple content items - extract text from each
                extracted_texts = []
                for line in content_lines:
                    if line.strip():  # Skip empty lines
                        parsed_line = json.loads(line)
                        if isinstance(parsed_line, dict) and "text" in parsed_line:
                            text_content = parsed_line["text"]
                            try:
                                # Try to parse as JSON
                                extracted_texts.append(json.loads(text_content))
                            except (json.JSONDecodeError, TypeError):
                                extracted_texts.append(text_content)
                        else:
                            extracted_texts.append(parsed_line)
                # If we have only one item, return it directly, otherwise return list
                return extracted_texts[0] if len(extracted_texts) == 1 else extracted_texts
        except (json.JSONDecodeError, KeyError, TypeError):
            # If parsing fails, return the raw content
            return result.content
    
    @classmethod
    def from_mcp_endpoint(cls, name: str, endpoint: str, **kwargs) -> 'JustMCPTool':
        """
        Create a JustMCPTool instance from an MCP SSE endpoint.
        
        Args:
            name: The name of the MCP tool
            endpoint: The MCP endpoint URL
            **kwargs: Additional parameters to pass to JustMCPTool constructor
            
        Returns:
            JustMCPTool instance
        """
        instance = cls(
            name=name,
            mcp_sse_endpoint=endpoint,
            **kwargs
        )
        
        return instance.refresh()
    
    @classmethod
    def from_mcp_stdio(cls, name: str, command: str, args: Optional[List[str]] = None, **kwargs) -> 'JustMCPTool':
        """
        Create a JustMCPTool instance from stdio command.
        
        Args:
            name: The name of the MCP tool
            command: The command to run
            args: Command arguments (optional)
            **kwargs: Additional parameters to pass to JustMCPTool constructor
            
        Returns:
            JustMCPTool instance
        """
        stdio_command = [command]
        if args:
            stdio_command.extend(args)
        
        instance = cls(
            name=name,
            mcp_stdio_command=stdio_command,
            **kwargs
        )
        
        return instance.refresh()


# For backward compatibility, JustTool becomes an alias to JustImportedTool
# This allows all existing code using JustTool to continue working
JustTool = JustImportedTool

class JustPromptTool(JustImportedTool):
    call_arguments: Optional[Dict[str,Any]] = Field(..., description="Input parameters to call the function with.")
    """Input parameters to call the function with."""

JustTools = Union[
    Dict[str, Union[JustImportedTool, JustMCPTool, JustMCPToolSetConfig]],  # A dictionary where keys are strings and values are JustToolBase instances.
    Sequence[
        Union[JustImportedTool, JustMCPTool, JustMCPToolSetConfig, Callable]
    ]  # A sequence (like a list or tuple) containing either JustToolBase instances or callable objects (functions).
]

JustPromptTools = Union[
    Dict[str, JustPromptTool],  # A dictionary where keys are strings and values are JustPromptTool instances.
    Sequence[
        Union[JustPromptTool, Tuple[Callable, Dict[str,Any]]]
    ]  # A sequence (like a list or tuple) containing either JustPromptTool instances or callable objects (functions) and input parameters.
]

class JustToolFactory:
    """
    A factory class for creating JustTool instances from various input types.
    This encapsulates the logic for converting callables, sequences, and dictionaries
    into the appropriate JustTool or JustPromptTool instances.
    """

    @staticmethod
    def create_tool(
            item: Union[Callable, JustToolBase, Dict[str, Any]],
            type_hint: Type[JustToolBase] = JustImportedTool
        ) -> JustToolBase:
        """
        Create a single tool from a callable, JustToolBase instance, or dictionary with tool parameters.

        Args:
            item: The callable function, JustToolBase instance, or dictionary with tool parameters
            type_hint: The type of tool to create if item is a callable or dictionary (defaults to JustImportedTool)

        Returns:
            JustToolBase: The created tool instance

        Raises:
            TypeError: If the item is neither a callable, JustToolBase instance, nor a valid dictionary
        """
        if isinstance(item, JustToolBase):
            if item.auto_refresh:
                item.refresh()
            return item
        elif callable(item):
            if type_hint == JustImportedTool or type_hint == JustTool:
                return JustTool.from_callable(item)
            else:
                # Use the from_callable of the specified type if it exists
                if hasattr(type_hint, 'from_callable') and callable(getattr(type_hint, 'from_callable')):
                    return type_hint.from_callable(item)
                # Otherwise, create a JustTool and cast it
                tool = JustTool.from_callable(item)
                # This assumes the type_hint is a subclass of JustToolBase with similar fields
                return type_hint(**tool.model_dump())
        elif isinstance(item, dict):
            # Heuristically determine the class based on dictionary keys
            tool_class = JustToolBase #abstract, will raise if not overridden.
            
            # Check if we have minimum required fields for any tool
            if not 'name' in item:
                raise ValueError("Function name must be provided in the dictionary")

            # Check for MCP tool vs ImportedTool
            if 'mcp_sse_endpoint' in item or 'mcp_stdio_command' in item:
                tool_class = JustMCPTool
            elif 'package' in item or 'static_class' in item:
                if issubclass(type_hint, JustImportedTool) or issubclass(type_hint, JustPromptTool): 
                   #use type_hint if it's a subclass of JustImportedTool or JustPromptTool
                   tool_class = type_hint
                else:
                   tool_class = JustImportedTool

                # JustPromptTool may have call_arguments
                if 'call_arguments' in item:
                    if issubclass(type_hint, JustPromptTool):
                        tool_class = type_hint
                    else:
                        tool_class = JustPromptTool
            
            try:
                # Create and return the tool instance
                tool_instance = tool_class(**item)
                return tool_instance
            except Exception as e:
                raise ValueError(f"Failed to create tool from dictionary: {e}")
        else:
            raise TypeError("Item must be a callable, JustToolBase instance, or a dictionary with tool parameters.")

    @staticmethod
    def create_prompt_tool(
            item: Union[JustPromptTool, Tuple[Callable, Dict[str, Any]], Dict[str, Any]],
        ) -> JustPromptTool:
        """
        Create a single prompt tool from a JustPromptTool instance, a (callable, params) tuple,
        or a dictionary with tool parameters including call_arguments.

        Args:
            item: The JustPromptTool instance, (callable, params) tuple, or dictionary

        Returns:
            JustPromptTool: The created prompt tool instance

        Raises:
            TypeError: If the item is not a valid input type
            ValueError: If the input parameters are not JSON serializable or missing required fields
        """
        if isinstance(item, JustPromptTool):
            prompt_tool = item
        elif isinstance(item, tuple) and len(item) == 2 and callable(item[0]) and isinstance(item[1], dict):
            func, input_params = item
            # Ensure input parameters are JSON serializable
            try:
                import json
                json.dumps(input_params)
            except (TypeError, OverflowError):
                raise ValueError(f"Input parameters for {func.__name__} must be JSON serializable")

            tool = JustTool.from_callable(func)
            prompt_tool = JustPromptTool(
                **tool.model_dump(),
                call_arguments=input_params
            )
        elif isinstance(item, dict):
            return JustToolFactory.create_tool(item, type_hint=JustPromptTool)
        else:
            raise TypeError(
                "Item must be a JustPromptTool instance, a (callable, input_params) tuple, or a dictionary with 'call_arguments'.")

        prompt_tool.max_calls_per_query = None  # Force disable
        if prompt_tool.auto_refresh:
            prompt_tool = prompt_tool.refresh()
        return prompt_tool

    @classmethod
    def create_tools_dict(
            cls,
            tools: JustTools,
            type_hint: Type[JustToolBase] = JustImportedTool
        ) -> Dict[str, JustToolBase]:
        """
        Process a JustTools input (dict or sequence) to create a dictionary of tool instances.

        Args:
            tools: Input JustTools (Dict[str, JustToolBase] or Sequence[Union[JustToolBase, Callable, Dict[str, Any]]])
            type_hint: The type of tool to create for callable items (defaults to JustImportedTool)

        Returns:
            Dict[str, JustToolBase]: Dictionary mapping tool names to tool instances

        Raises:
            TypeError: If tools is not a dictionary or sequence, or if items in a sequence are invalid
        """
        if not tools:
            return {}
        
        new_tools = {}
        if isinstance(tools, dict):
            # If it's already a dictionary, ensure all values are proper tools
            for name, item in tools.items():
                if isinstance(item, JustToolBase):
                    new_tools[name] = item
                elif isinstance(item, JustMCPToolSetConfig):
                    # Handle MCP tool set config - generate multiple tools
                    mcp_tools = cls.create_tools_from_mcp(item)
                    new_tools.update(mcp_tools)
                elif isinstance(item, Callable):
                    try:
                        tool = cls.create_tool(item, type_hint=type_hint)
                        new_tools[tool.name] = tool
                    except (TypeError, ValueError) as e:
                        raise TypeError(f"Invalid item for key '{name}' in tools dictionary: {e}")
                else:
                    raise TypeError(f"Unsupported item type for key '{name}' in tools dictionary: {type(item)}")
            return new_tools
        elif isinstance(tools, Sequence):
            for item in tools:
                try:
                    if isinstance(item, JustMCPToolSetConfig):
                        # Handle MCP tool set config - generate multiple tools
                        mcp_tools = cls.create_tools_from_mcp(item)
                        new_tools.update(mcp_tools)
                    else:
                        tool = cls.create_tool(item, type_hint=type_hint)
                        new_tools[tool.name] = tool
                except (TypeError, ValueError) as e:
                    raise TypeError(f"Invalid item in tools sequence: {e}")
        else:
            raise TypeError("The 'tools' input must be a sequence or a valid dictionary of JustToolBase or JustMCPToolSetConfig instances.")

        return new_tools

    @classmethod
    def create_prompt_tools_dict(
            cls,
            prompt_tools: JustPromptTools
        ) -> Dict[str, JustPromptTool]:
        """
        Process a JustPromptTools input (dict or sequence) to create a dictionary of prompt tool instances.

        Args:
            prompt_tools: Input JustPromptTools (Dict[str, JustPromptTool] or 
                         Sequence[Union[JustPromptTool, Tuple[Callable, Dict[str, Any]], Dict[str, Any]]])

        Returns:
            Dict[str, JustPromptTool]: Dictionary mapping tool names to prompt tool instances

        Raises:
            TypeError: If prompt_tools is not a dictionary or sequence, or if items in a sequence are invalid
        """
        if not prompt_tools:
            return {}

        if isinstance(prompt_tools, dict):
            # If it's already a dictionary, ensure all values are proper prompt tools
            new_prompt_tools = {}
            for name, item in prompt_tools.items():
                if isinstance(item, JustPromptTool):
                    new_prompt_tools[name] = item
                else:
                    # Try to convert non-tool items to prompt tools
                    try:
                        prompt_tool = cls.create_prompt_tool(item)
                        new_prompt_tools[prompt_tool.name] = prompt_tool
                    except (TypeError, ValueError) as e:
                        raise TypeError(f"Invalid item for key '{name}' in prompt_tools dictionary: {e}")
            return new_prompt_tools
        elif not isinstance(prompt_tools, Sequence):
            raise TypeError(
                "The 'prompt_tools' input must be a sequence of JustPromptTool instances, (callable, input_params) tuples, or valid dictionaries.")

        new_prompt_tools = {}
        for item in prompt_tools:
            try:
                prompt_tool = cls.create_prompt_tool(item)
                new_prompt_tools[prompt_tool.name] = prompt_tool
            except (TypeError, ValueError) as e:
                raise TypeError(f"Invalid item in prompt_tools sequence: {e}")

        return new_prompt_tools

    @classmethod
    def list_mcp_tools(
            cls,
            sse_endpoint: Optional[str] = None,
            stdio_command: Optional[List[str]] = None
        ) -> Dict[str, ToolDefinition]:
        """
        List available MCP tools as a dictionary of ToolDefinition objects.
        
        Args:
            sse_endpoint: The SSE endpoint URL for MCP connection
            stdio_command: The stdio command and arguments for MCP connection
            
        Returns:
            Dict[str, ToolDefinition]: Dictionary mapping tool names to ToolDefinition objects
            
        Raises:
            ValueError: If neither sse_endpoint nor stdio_command is provided
        """

        if not sse_endpoint and not stdio_command:
            raise ValueError("Either sse_endpoint or stdio_command must be provided")
        
        # Get the MCP client
        server_params = MCPServerParameters(mcp_sse_endpoint=sse_endpoint, mcp_stdio_command=stdio_command)
        client = MCPClient.get_client_by_inputs(server_params)
        
        # Create and run the coroutine to list tools
        async def get_tools():
            async with client:
                tool_definitions = await client.list_tools_openai()
                return {tool.name: tool for tool in tool_definitions}
            
        # Run the coroutine using run_async_function_synchronously
        return run_async_function_synchronously(get_tools)

    @classmethod
    def get_mcp_tool_by_name(
            cls,
            tool_name: str,
            sse_endpoint: Optional[str] = None,
            stdio_command: Optional[List[str]] = None
        ) -> Optional[ToolDefinition]:
        """
        Get a specific MCP tool by name as a ToolDefinition object.
        
        Args:
            tool_name: The name of the tool to retrieve
            sse_endpoint: The SSE endpoint URL for MCP connection
            stdio_command: The stdio command and arguments for MCP connection
            
        Returns:
            Optional[ToolDefinition]: The tool as a ToolDefinition object, or None if not found
            
        Raises:
            ValueError: If neither sse_endpoint nor stdio_command is provided
        """
        if not sse_endpoint and not stdio_command:
            raise ValueError("Either sse_endpoint or stdio_command must be provided")
        
        # Get the MCP client
        server_params = MCPServerParameters(mcp_sse_endpoint=sse_endpoint, mcp_stdio_command=stdio_command)
        client = MCPClient.get_client_by_inputs(server_params)
        
        # Create and run the coroutine to get the tool
        async def get_tool():
            async with client:
                return await client.get_tool_openai_by_name(tool_name)
            
        # Run the coroutine using run_async_function_synchronously
        return run_async_function_synchronously(get_tool)

    @classmethod
    def create_tools_from_mcp(
            cls,
            config: JustMCPToolSetConfig
        ) -> Dict[str, JustMCPTool]:
        """
        Creates a dictionary of JustMCPTool instances from an MCP server configuration.

        Args:
            config: JustMCPToolSetConfig with server parameters and tool filtering options

        Returns:
            Dict[str, JustMCPTool]: Dictionary mapping tool names to JustMCPTool instances

        Raises:
            ValueError: If neither mcp_sse_endpoint nor mcp_stdio_command is provided in config
            ValueError: If raise_on_incorrect_names is True and include/exclude contains names not in available tools
        """
        # Get the available tools from the MCP server
        tool_defs = cls.list_mcp_tools(sse_endpoint=config.mcp_sse_endpoint, 
                                      stdio_command=config.mcp_stdio_command)
        
        # Convert to a set for easier filtering
        available_tools = set(tool_defs.keys())
        
        # Check if include/exclude names are valid
        include_set = set(config.only_include_tools)
        exclude_set = set(config.exclude_tools)
        
        if include_set and config.raise_on_incorrect_names:
            missing = include_set - available_tools
            if missing:
                missing_str = ", ".join(missing)
                raise ValueError(f"Requested tools not available from MCP server: {missing_str}")
        
        if exclude_set and config.raise_on_incorrect_names:
            missing = exclude_set - available_tools
            if missing:
                missing_str = ", ".join(missing)
                raise ValueError(f"Excluded tools not found in MCP server: {missing_str}")
        
        # Determine the tools to create
        tools_to_create = set()
        if include_set:
            # Only include specified tools that are available
            tools_to_create = include_set.intersection(available_tools)
        else:
            # Include all available tools
            tools_to_create = available_tools
        
        # Apply exclude filter
        tools_to_create -= exclude_set
        
        # Create tool instances
        tool_dict = {}
        for tool_name in tools_to_create:
            if config.mcp_sse_endpoint:
                tool = JustMCPTool.from_mcp_endpoint(
                    name=tool_name,
                    endpoint=config.mcp_sse_endpoint
                )
            else:
                tool = JustMCPTool.from_mcp_stdio(
                    name=tool_name,
                    command=config.mcp_stdio_command[0],
                    args=config.mcp_stdio_command[1:] if len(config.mcp_stdio_command) > 1 else None
                )
            
            tool_dict[tool_name] = tool
        
        return tool_dict

