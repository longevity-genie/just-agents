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


class StatefulPDFReader:
    """A stateful class that demonstrates bound instance methods."""
    
    def __init__(self, content: str):
        """Initialize with PDF content."""
        self.content = content
        self.pages = content.split('\n---\n')  # Split by page separator
        self.current_page = 0
        self.total_pages = len(self.pages)
    
    def get_current_page(self) -> str:
        """Get the current page content."""
        if self.current_page < self.total_pages:
            return f"Page {self.current_page + 1}/{self.total_pages}: {self.pages[self.current_page]}"
        return "No more pages."
    
    def get_next_page(self) -> str:
        """Move to next page and return its content."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            return self.get_current_page()
        return "Already at last page."
    
    def get_previous_page(self) -> str:
        """Move to previous page and return its content."""
        if self.current_page > 0:
            self.current_page -= 1
            return self.get_current_page()
        return "Already at first page."
    
    def jump_to_page(self, page_number: int) -> str:
        """Jump to a specific page number (1-based)."""
        if 1 <= page_number <= self.total_pages:
            self.current_page = page_number - 1
            return self.get_current_page()
        return f"Invalid page number. Available pages: 1-{self.total_pages}"
    
    @staticmethod
    def create_sample_pdf() -> 'StatefulPDFReader':
        """Create a sample PDF reader with test content."""
        sample_content = "First page content with magic word: Abra\n---\nSecond page content with magic word: Shwabra\n---\nThird page content with magic word: Kadabra!"
        return StatefulPDFReader(sample_content)
    
    @staticmethod
    def create_spell_pdf() -> 'StatefulPDFReader':
        """Create a PDF with spell components for testing agent comprehension."""
        spell_content = (
            "Ancient Spell Components - Page 1 of 3\n"
            "The first incantation word is: Abra\n"
            "This word must be spoken first to begin the spell.\n"
            "---\n"
            "Magical Enhancement - Page 2 of 3\n"
            "The second power word is: Shwabra\n"
            "This amplifies the spell's magical energy.\n"
            "---\n"
            "Final Invocation - Page 3 of 3\n"
            "The concluding word is: Kadabra!\n"
            "Speak this last to complete the ancient spell."
        )
        return StatefulPDFReader(spell_content)


class SimpleCounterTool:
    """A simple counter tool for testing transient behavior."""
    
    def __init__(self, initial_value: int = 0):
        """Initialize counter with initial value."""
        self.count = initial_value
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter by specified amount."""
        self.count += amount
        return self.count
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter by specified amount."""
        self.count -= amount
        return self.count
    
    def get_count(self) -> int:
        """Get current counter value."""
        return self.count
    
    def reset(self) -> int:
        """Reset counter to zero."""
        self.count = 0
        return self.count


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