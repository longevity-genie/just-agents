from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Type, ClassVar, Tuple, Sequence

from just_agents.just_agent import JustAgent
from just_agents.just_yaml import JustYaml as Yml


class JustAgentProfileDescriptions(BaseModel):
    """
    A static class containing descriptions for the JustAgentProfile fields.
    """

    SERVICE_FIELDS: ClassVar[Tuple[str, ...]] = ('profile_type', 'extras')
    """Service fields to be removed form output"""

    TO_REFRESH: ClassVar[Tuple[str, ...]] = ('shortname', 'description')

    Descriptions: ClassVar[Dict[str, str]] = {
        "profile_type": "A marker to discern versions of this class when loading from configuration YAML files.",
        "shortname": "A short name used as a section name in YAML storage. Must be alphanumeric, underscores, or dashes.",
        "description": "Short description of what the agent does.",
        "role": "Role of the agent.",
        "goal": "Goal of the agent.",
        "task": "Tasks of the agent.",
        "expertise_domain": "Agent's field of expertise.",
        "limitations": "Agent's known limitations.",
        "system_prompt": "System prompt of the agent.",
        "backstory": "Backstory of the agent.",
        "tools": "An array, list of the tools available to the agent",
        "preferred_model_name": "The name of the preferred model to use for inference",
        "knowledge_sources": "An array, list of external knowledge sources the agent is capable of accessing, e.g. databases, APIs etc",
        "extras": "Fallback container to store fields that don't fit the model."
    }

    @classmethod
    def validate_keys_match(cls, model_fields: set):
        """
        Validates that the keys in the Descriptions dictionary match the model fields.
        """
        description_keys = set(cls.Descriptions.keys())
        if model_fields != description_keys:
            missing_in_descriptions = model_fields - description_keys
            missing_in_fields = description_keys - model_fields
            raise ValueError(f"Mismatch between model fields and Descriptions keys. Missing in Descriptions: {missing_in_descriptions}. Missing in fields: {missing_in_fields}.")

class JustAgentProfile(BaseModel, extra="allow"):
    """
    A Pydantic model representing an agent profile with extended attributes.
    Instances of this class behave like strings based on the 'description' field,
    allowing seamless replacement of a string property with this extended record.
    """

    DescriptionsClass : ClassVar[Type[BaseModel]] = JustAgentProfileDescriptions
    """A variable to store description Class type matching this class"""

    TO_RENAME : ClassVar[str] = "RenameMe"
    """A marker of stub shortname"""

    profile_type: str = Field(None)
    """A marker to discern versions of this class when loading from configuration YAML files."""

    shortname: str = Field(default=TO_RENAME)
    """A short name used as a section name in YAML storage. Must be alphanumeric, underscores, or dashes."""

    description: str = Field(default="Generic all-purpose AI agent")
    """Short description of what the agent does."""

    role: Optional[str] = None
    """Role of the agent."""

    goal: Optional[str] = None
    """Goal of the agent."""

    task: Optional[str] = None
    """Tasks of the agent."""

    expertise_domain: Optional[str] = None
    """Agent's field of expertise."""

    limitations: Optional[str] = None
    """Agent's known limitations."""

    system_prompt: Optional[str] = None
    """System prompt of the agent."""

    backstory: Optional[str] = None
    """Backstory of the agent."""

    preferred_model_name: Optional[str] = None
    """The name of the preferred model to use for inference"""

    tools: Optional[List[str]] = None
    """List of the tools available to the agent"""

    knowledge_sources: Optional[List[str]] = None
    """List of external knowledge sources the agent is capable of accessing, e.g., databases, APIs, etc."""

    extras: Optional[dict] = None
    """Fallback container to store fields that don't fit the model."""

    @field_validator('shortname', mode="before")
    def validate_shortname(cls, value):
        """
        Validates the 'shortname' field to ensure it contains only allowed characters.
        Regex pattern equivalent: /^[a-zA-Z0-9_\-]+$/
        """
        allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        if not set(value).issubset(allowed_characters):
            raise ValueError("Shortname can only contain alphanumeric characters, underscores, and dashes.")
        return value

    @classmethod
    def validate_keys_match(cls):
        """
        Validates that the keys in the Descriptions dictionary match the model fields.
        """
        cls.DescriptionsClass.validate_keys_match(set(cls.model_fields.keys()))

    @classmethod
    def get_full_class_path(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def __init__(self, **data):
        """
        Custom initializer to handle extra fields by storing them in the 'extras' attribute.
        """
        super().__init__(**data)
        self.validate_keys_match()
        # Collect extra fields into 'extras' attribute
        if not self.profile_type:
            self.profile_type = self.get_full_class_path()
        self.extras = getattr(self, '__pydantic_extra__', {}) or {}


    def __str__(self):
        """
        Returns the 'description' field when the instance is converted to a string.
        """
        return self.description

    def __repr__(self):
        """
        Returns a string representation of the instance.
        """
        return f"{self.__class__.__name__}({self.description!r})"

    def __add__(self, other):
        """
        Supports concatenation with other strings.
        """
        return self.description + other

    def __radd__(self, other):
        """
        Supports concatenation when the instance is on the right side.
        """
        return other + self.description

    def __len__(self):
        """
        Returns the length of the 'description' string.
        """
        return len(self.description)

    def __contains__(self, item):
        """
        Checks if a substring is contained in the 'description'.
        """
        return item in self.description

    def __getitem__(self, index):
        """
        Supports indexing and slicing on the 'description' string.
        """
        return self.description[index]

    def __eq__(self, other):
        """
        Compares the instance with another string or object.
        """
        if isinstance(other, str):
            return self.description == other
        return super().__eq__(other)

    def __hash__(self):
        """
        Returns the hash of the 'description' string.
        """
        return hash(self.description)

    @classmethod
    def from_strings(cls, shortname: str, description: str):
        """
        Creates an instance from two strings: shortname and description.

        Args:
            shortname (str): The short name for the agent profile.
            description (str): The description of the agent.

        Returns:
            AgentProfile: A new instance of AgentProfile.
        """
        return cls(shortname=shortname, description=description)

    # duplicated logic from just-agent, instead of importing to keep class self-contained
    @classmethod
    def from_yaml(cls, section_name: str, file_path: Path = Yml.DEFAULT_CONFIG_PATH,
                  parent_section: str = Yml.DEFAULT_AGENT_PROFILES_SECTION):
        """
        Creates an instance from a YAML file path, section name, and parent section name.

        Args:
            file_path (Path): The path to the YAML file.
            section_name (str): The section name (shortname) in the YAML file.
            parent_section (str): The parent section name in the YAML file.

        Returns:
            AgentProfile: A new instance of AgentProfile.

        Raises:
            ValueError: If the specified section is not found in the YAML file.
        """
        section_data = Yml.read_yaml_data(file_path, section_name, parent_section)
        return cls.model_validate(section_data)

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]):
        """
        Constructor from JSON data. Populates fields that are present in the class,
        and stores any extra fields in 'extras'.

        Args:
            json_data (Dict[str, Any]): The JSON data as a dictionary.

        Returns:
            AgentProfile: A new instance of AgentProfile.
        """
        return cls(**json_data)

    def save_to_yaml(self, file_path: Path = Yml.DEFAULT_CONFIG_PATH,
                     parent_section: str = Yml.DEFAULT_AGENT_PROFILES_SECTION):
        """
        Saves the instance's data to a YAML file under the specified parent section and section name (shortname).
        If the file exists, it appends or updates the existing data.

        Args:
            file_path (Path): The path to the YAML file.
            parent_section (str): The parent section name in the YAML file.
        """
        section_data = self.model_dump(exclude_none=True)
        Yml.save_to_yaml(file_path, section_data, self.shortname, parent_section)

    def model_dump_with_extras(self) -> Dict[str, Any]:
        """
        Serializes the instance to a JSON-compatible dictionary, flattening the 'extras' back into the main dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the instance, including extra fields.
        """
        data = self.model_dump(exclude_none=True)
        # Remove the 'extras' key to prevent duplication
        data.pop('extras', None)
        if self.extras:
            data.update(self.extras)
        return data

    def model_dump_for_llm(
            self,
            max_prompt_length: int = 0,
            exclude_list: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Serializes the instance to a dictionary ready for use by language models, optionally truncating the system_prompt.

        Args:
            max_prompt_length (int): Maximum allowed length for the system_prompt. If the system_prompt exceeds this length,
                                     it will be excluded from the output. Set to -1 to always include.
            exclude_list (Optional[Sequence[str]]): Fields to exclude from the output, defaults to SERVICE_FIELDS.

        Returns:
            Dict[str, Any]: A dictionary representation of the instance for language model use.
        """
        if exclude_list is None:
            exclude_list = self.DescriptionsClass.SERVICE_FIELDS

        data = self.model_dump_with_extras()
        for key, value in list(data.items()):
            if key in exclude_list:
                data.pop(key, None)
            if value in (None, "", [], {}):
                data.pop(key, None) # remove empty fields

        if 0 <= max_prompt_length < len(self.system_prompt):
            data.pop('system_prompt', None)
        return data

    def to_populate(
            self,
            refresh: bool = False,
            exclude_list: Optional[Sequence[str]] = None,
            refresh_list: Optional[Sequence[str]] = None,
            extra_list: Dict[str,str] = None
    ) -> Dict[str, str]:
        """
        Generates a dictionary with keys corresponding to fields that are None (except 
        'profile_type' and 'extras'), and values set to descriptions from the 
        DescriptionsClass.

        Args:
            refresh (bool): If True, includes 'shortname' and 'description' in the 
            dictionary.
            exclude_list (Optional[Sequence[str]]): Fields to exclude from the dictionary, defaults to SERVICE_FIELDS.
            refresh_list (Optional[Sequence[str]]): Fields to refresh if the 'refresh' flag is set, defaults to TO_REFRESH.
            extra_list (Dict[str,str]): Extra fields to populate, defaults to empty list. Must be supplied with descriptions

        Returns:
            Dict[str, str]: A dictionary of fields to populate, with descriptions 
            as values.
        """
        if exclude_list is None:
            exclude_list = self.DescriptionsClass.SERVICE_FIELDS
        if refresh_list is None:
            refresh_list = self.DescriptionsClass.TO_REFRESH

        fields_to_populate = {
            field: self.DescriptionsClass.Descriptions[field]
            for field, value in self.__dict__.items()
            if value is None and field not in exclude_list
        }

        if refresh:
            for field in refresh_list:
                if field not in self.model_fields.keys():
                    continue
                fields_to_populate[field] = self.DescriptionsClass.Descriptions[field]


        if extra_list:
            for field, value in list(extra_list.items()):
                fields_to_populate[field] = value

        return fields_to_populate

    def update(self, new_data: 'JustAgentProfile', overwrite: bool = False) -> None:
        """
        Updates the fields of the current instance with the values from another JustAgentProfile instance.

        Args:
            new_data (JustAgentProfile): The instance from which to update the current instance's fields.
            overwrite (bool): If True, existing fields are overwritten with new values.
        """
        for field, value in new_data.__dict__.items():
            if (value
                and field not in self.DescriptionsClass.SERVICE_FIELDS
                and field in self.model_fields.keys() #extras are processed separately
            ):
                if getattr(self, field, None) is None or overwrite:
                    setattr(self, field, value)
        # Update extras separately to avoid overwriting the entire dictionary
        if new_data.extras:
            if self.extras is None:
                self.extras = {}
            self.extras.update(new_data.extras)
