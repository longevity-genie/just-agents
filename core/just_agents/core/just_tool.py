from typing import Callable, Optional, List, Dict, Any, Sequence, Union, Literal

from litellm.utils import function_to_dict
from pydantic import BaseModel, Field, PrivateAttr
import importlib
import inspect

FunctionParamFields=Literal["kind","default","type_annotation"]
FunctionParams = List[Dict[str, Dict[FunctionParamFields,Optional[str]]]]

class LiteLLMDescription(BaseModel, populate_by_name=True):
    name: Optional[str] = Field(..., alias='function', description="The name of the function")
    description: Optional[str] = Field(None, description="The docstring of the function.")
    parameters: Optional[Dict[str,Any]]= Field(None, description="Parameters of the function.")

class JustTool(LiteLLMDescription):
    package: str = Field(..., description="The name of the module where the function is located.")
    auto_refresh: bool = Field(True, description="Whether to automatically refresh the tool after initialization.")
    arguments: Optional[FunctionParams] = Field(
         None, description="List of parameters with their details.", exclude=True
    )
    _callable: Optional[Callable] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """Called after the model is initialized. Refreshes the tools metainfo if auto_refresh is True."""
        super().model_post_init(__context)
        if self.auto_refresh:
            self.refresh()

    def get_litellm_description(self) -> Dict[str, Any]:
        dump = self.model_dump(
            mode='json',
            by_alias=False,
            exclude_none=True,
            serialize_as_any=False,
            include=set(super().model_fields)
        )
        return dump

    @classmethod
    def from_callable(cls, input_function: Callable) -> 'JustTool':
        """Create a JustTool instance from a callable."""
        package = input_function.__module__
        litellm_description = function_to_dict(input_function)
        arguments = cls._extract_parameters(input_function)
        return cls(
            **litellm_description,
            package=package,
            arguments=arguments,
            _callable=input_function,
        )

    @staticmethod
    def _extract_parameters(func: Callable) -> List[Dict[str, Any]]:
        """Extract parameters from the function's signature."""
        signature = inspect.signature(func)
        parameters = []
        for name, param in signature.parameters.items():
            param_info = {
                'kind': str(param.kind),
                'default': str(param.default) if param.default != param.empty else None,
                'type_annotation': str(param.annotation) if param.annotation != param.empty else None
            }
            parameters.append({ name: param_info})
        return parameters

    def refresh(self)->'JustTool':
        """
        Refresh the JustTool instance to reflect the current state of the actual function.
        Updates package, function name, description, parameters, and ensures the function is importable.
        Returns:
            JustTool: Returns self to allow method chaining or direct appending.
        """
        try:
            # Import the module
            package = importlib.import_module(self.package)
            # Get the function from the module
            func = getattr(package, self.name)
            # Update LiteLLM description
            litellm_description = LiteLLMDescription (**function_to_dict(func))
            # Update the description
            self.description = litellm_description.description
            # Update parameters
            self.parameters= litellm_description.parameters
            self.arguments = self._extract_parameters(func)
            # Update the cached callable
            self._callable = func

            return self  # Return self to allow chaining or direct appending
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error refreshing {self.name} from {self.package}: {e}") from e

    def get_callable(self, refresh: bool = False) -> Callable:
        """
        Retrieve the callable function.
        If refresh is True, the callable is refreshed before returning.
        """
        if refresh:
            self.refresh()
        if self._callable is not None:
            return self._callable
        try:
            package = importlib.import_module(self.package)
            func = getattr(package, self.name)
            self._callable = func  # Cache the callable
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error importing {self.name} from {self.package}: {e}")

    def __call__(self, *args, **kwargs):
        """Allows the JustTool instance to be called like a function."""
        func = self.get_callable()
        return func(*args, **kwargs)

JustTools = Union[
    Dict[str, JustTool],  # A dictionary where keys are strings and values are JustTool instances.
    Sequence[
        Union[JustTool, Callable]
    ]  # A sequence (like a list or tuple) containing either JustTool instances or callable objects (functions).
]
# Although an internal dictionary representation is preferable, a list representation of tools can be handled and converted to a dictionary using validation.
