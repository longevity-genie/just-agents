from typing import Callable, Optional, List, Dict, Any, Self
from pydantic import BaseModel, Field, PrivateAttr
import importlib
import inspect

class JustTool(BaseModel):
    package: str = Field(..., description="The name of the module where the function is located.")
    function: str = Field(..., description="The name of the function.")
    description: Optional[str] = Field(None, description="The docstring of the function.")
    auto_refresh: bool = Field(True, description="Whether to automatically refresh the tool after initialization.")
    parameters: Optional[List[Dict[str, Any]]] = Field(
         None, description="List of parameters with their details."
    )
    _callable: Optional[Callable] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """Called after the model is initialized. Refreshes the tool if auto_refresh is True."""
        if self.auto_refresh:
            self.refresh()

    @classmethod
    def from_callable(cls, func: Callable) -> 'JustTool':
        """Create a JustTool instance from a callable."""
        package = func.__module__
        function = func.__name__
        description = func.__doc__
        parameters = cls._extract_parameters(func)
        return cls(
            package=package,
            function=function,
            description=description,
            parameters=parameters,
            _callable=func,
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

    def refresh(self)->Self:
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
            func = getattr(package, self.function)
            # Update the description
            self.description = func.__doc__
            # Update parameters
            self.parameters = self._extract_parameters(func)
            # Update the cached callable
            self._callable = func
            return self  # Return self to allow chaining or direct appending
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error refreshing {self.function} from {self.package}: {e}")

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
            func = getattr(package, self.function)
            self._callable = func  # Cache the callable
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error importing {self.function} from {self.package}: {e}")

    def __call__(self, *args, **kwargs):
        """Allows the JustTool instance to be called like a function."""
        func = self.get_callable()
        return func(*args, **kwargs)
