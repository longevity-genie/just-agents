from typing import  Any, List, Union, Dict, Annotated, TypeVar, Iterable
from just_agents.data_classes import Message
from pydantic import BaseModel
from pydantic import AfterValidator
from pydantic.functional_serializers import PlainSerializer
from pyrsistent import PVector, pvector
######### Common ###########

MessageDict = Dict[str, Any] # Plain python dictionary with str keys matching LLM chat completion message structure

MessageVariant = Union[      # Variant type to handle different message representations
    str,                     # Raw string, 'user' role implied
    MessageDict,             # Plain python dictionary message representation
    Message,                 # Support seamless API request Pydantic Message class as argument
    BaseModel,               # Or a Pydantic BaseModel for structured output
]

SupportedMessages = Union[
    MessageVariant,          # A single [message]
    List[MessageVariant],    # Or a list of messages
    PVector[MessageVariant], # Or a pvector of messages
]

# Generic PVector field alias with Pydantic v2 validators/serializers
# - Schema/JSON: represented as a List[T]
# - Runtime: stored as pyrsistent PVector[T]
T = TypeVar('T')

def _to_pvector_after(value: Any) -> PVector[Any]:
    """
    Convert validated list-like values to PVector. Accepts list, tuple, iterable, PVector or None.
    """
    if value is None:
        return pvector()
    if isinstance(value, PVector):
        return value
    if isinstance(value, list):
        return pvector(value)
    if isinstance(value, tuple):
        return pvector(list(value))
    if isinstance(value, Iterable):
        return pvector(list(value))
    return pvector([value])

def _pvector_to_list(value: Any) -> List[Any]:
    """
    Serialize PVector (or list-like) to plain list for JSON/YAML.
    """
    if value is None:
        return []
    if isinstance(value, PVector):
        return list(value)
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]

# Use List[T] as the schema-visible type, and convert to PVector at runtime
PVectorField = Annotated[
    List[T],
    AfterValidator(_to_pvector_after),
    PlainSerializer(_pvector_to_list, return_type=list, when_used='always'),
]
