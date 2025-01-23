from typing import  Any, List, Union, Dict
from just_agents.data_classes import Message

######### Common ###########

MessageDict = Dict[str, Any] # Plain python dictionary with str keys matching LLM chat completion message structure

MessageVariant = Union[      # Variant type to handle different message representations
    str,                     # Raw string, 'user' role implied
    MessageDict,             # Plain python dictionary message representation
    Message                  # Support seamless API request Pydantic Message class as argument
]

SupportedMessages = Union[
    MessageVariant,          # A single [message]
    List[MessageVariant]     # Or a list of messages
]
