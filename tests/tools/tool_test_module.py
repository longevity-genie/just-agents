"""Test module for JustTool tests."""

from typing import List, Union, Dict
from pydantic import BaseModel

def regular_function(x: int, y: int) -> int:
    """A regular function."""
    return x + y

class TopLevelClass:
    """A top-level class."""
    @staticmethod
    def static_method_top(a: str, b: str = "default") -> str:
        """A static method in a top-level class."""
        return f"{a}-{b}"

    class NestedClass:
        """A nested class."""
        @staticmethod
        def static_method_nested(flag: bool) -> str:
            """A static method in a nested class."""
            return "NestedTrue" if flag else "NestedFalse"

        class DeeperNestedClass:
            """A deeper nested class."""
            @staticmethod
            def static_method_deeper(val: float) -> float:
                """A static method in a deeper nested class."""
                return val * 2.0

class SimpleModel(BaseModel):
    """A simple Pydantic model for testing."""
    name: str
    age: int
    score: float
    is_active: bool

def type_tester_function(
    s_arg: str,
    i_arg: int,
    f_arg: float,
    b_arg: bool,
    l_arg: list,
    composite_list_arg: List[Union[str, bool, int, float]],
    dict_arg: Dict[str, Union[str, int, bool, float]],
    model_arg: SimpleModel
    ):
    """Tests argument type handling.

    Accepts various argument types and returns a dictionary detailing
    the types and values of the received arguments.

    Args:
        s_arg: A string argument.
        i_arg: An integer argument.
        f_arg: A float argument.
        b_arg: A boolean argument.
        l_arg: A list argument (generic list, typically implies list of strings for LLM).
        composite_list_arg: A list argument with mixed scalar types.
        dict_arg: A dictionary argument with mixed scalar types as values.
        model_arg: A Pydantic model argument with basic type fields.

    Returns:
        A dictionary containing the type and value of each argument.
        The dictionary includes keys for the type of each argument
        (e.g., "s_arg_type"), the types of items within the lists
        (e.g., "l_items_type", "composite_list_items_type"),
        and the value of each argument (e.g., "s_arg_value").
    """
    # Validate the model_arg if it's not None
    model_instance = SimpleModel.model_validate(model_arg) if model_arg is not None else None
    
    return {
        "s_arg_type": type(s_arg).__name__,
        "i_arg_type": type(i_arg).__name__,
        "f_arg_type": type(f_arg).__name__,
        "b_arg_type": type(b_arg).__name__,
        "l_arg_type": type(l_arg).__name__,
        "l_items_type": [type(i).__name__ for i in l_arg] if l_arg else [],
        "composite_list_arg_type": type(composite_list_arg).__name__,
        "composite_list_items_type": [type(i).__name__ for i in composite_list_arg] if composite_list_arg else [],
        "dict_arg_type": type(dict_arg).__name__,
        "d_items_type": {k: type(v).__name__ for k, v in dict_arg.items()} if dict_arg else {},
        "model_arg_type": type(model_instance).__name__ if model_instance is not None else None,
        "model_arg_fields": {k: type(v).__name__ for k, v in model_instance.model_dump().items()} if model_instance is not None else None,
        "s_arg_value": s_arg,
        "i_arg_value": i_arg,
        "f_arg_value": f_arg,
        "b_arg_value": b_arg,
        "l_arg_value": l_arg,
        "composite_list_arg_value": composite_list_arg,
        "dict_arg_value": dict_arg,
        "model_arg_value": model_instance.model_dump() if model_instance is not None else None
    } 