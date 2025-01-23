import yaml
import importlib
from typing import Optional, Dict, Any, ClassVar, Sequence, Union, Set, Type, TypeVar
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError
from collections.abc import MutableMapping, MutableSequence
from pydantic import ConfigDict
import sys

# Create a TypeVar for the class
if sys.version_info >= (3, 11):
    from typing import Self
    SelfType = Self
else:
    SelfType = TypeVar('SelfType', bound='JustSerializable')

class JustYaml:
    """
    A utility static class for reading and saving data to YAML files.

    Methods:
        read_yaml_data(file_path: Path, section_name: str, parent_section: str = DEFAULT_AGENT_PROFILES_SECTION) -> Dict:
            Reads data from a given section within a YAML file.

        save_to_yaml(file_path: Path, section_data: Dict, section_name: str, parent_section: str = DEFAULT_AGENT_PROFILES_SECTION) -> None:
            Updates a section within a YAML file with new data.
    """

    @staticmethod
    def str_presenter(dumper, data):
        """
        An override to write multiline strings (like prompts!) to yaml in scalar form that removes trailing spaces
        see https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        pyyaml bug: https://github.com/yaml/pyyaml/issues/121
        """
        if len(data.splitlines()) > 1 or '\n' in data:  # check for multiline string
            text_list = [line.rstrip() for line in data.splitlines()]
            fixed_data = "\n".join(text_list)
            return dumper.represent_scalar('tag:yaml.org,2002:str', fixed_data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    @staticmethod
    def read_yaml_data(
            file_path: Path,
            section_name: str,
            parent_section: str = None
    ) -> Dict:
        """
        Reads data from a given section within a YAML file.

        Args:
            file_path (Path): The path to the YAML file.
            section_name (str): The name of the section to read.
            parent_section (str): The parent section name. Defaults to 'agent_profiles'.

        Returns:
            Dict: The data from the specified section.

        Raises:
            ValueError: If the section or parent section is not found in the YAML file.
        """
        if file_path.exists():
            with file_path.open('r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader) or {}
        else:
            raise FileNotFoundError(
                f"File '{file_path}' not found."
            )
        try:
            # Retrieve the data for the specified section
            if parent_section:
                return data[parent_section][section_name]
            else:
                return data[section_name]
        except KeyError as e:
            raise KeyError(
                f"Section '{section_name}' under parent section '{parent_section}' not found in '{file_path}'"
            ) from e

    @staticmethod
    def read_yaml_data_safe(
            file_path: Path,
            section_name: str,
            parent_section: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Reads data from a given section within a YAML file.

        Args:
            file_path (Path): The path to the YAML file.
            section_name (str): The name of the section to read.
            parent_section (str): The parent section name. Defaults to 'agent_profiles'.

        Returns:
            Optional[Dict]: The data from the specified section, or None if not found or error occurred while reading.
        """
        if file_path.exists():
            with file_path.open('r') as f:
                data = yaml.safe_load(f) or {}
        else:
            return None
        try:
            # Retrieve the data for the specified section
            if parent_section:
                if parent_section in data:
                    if section_name and section_name in data[parent_section]:
                        return data[parent_section][section_name]
            elif section_name and section_name in data:
                return data[section_name]
            elif not section_name and data:
                return data
            else:
                return None
        except KeyError:
            return None

    @staticmethod
    def save_to_yaml(
            file_path: Path,
            section_data: Dict,
            section_name: str,
            parent_section: str = None
    ) -> None:
        """
        Updates a section within a YAML file with new data.

        Args:
            file_path (Path): The path to the YAML file.
            section_data (Dict): The data to be saved in the specified section.
            section_name (str): The name of the section to update.
            parent_section (str): The parent section name.

        Returns:
            None
        """

        data = {}
        # Check if the YAML file exists and load existing data
        if file_path.exists():
            with file_path.open('r') as f:
                existing_data = dict(yaml.safe_load(f) or {})
                data.update(existing_data)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the parent section exists
        if parent_section:
            if parent_section not in data:
                data[parent_section] = {}

        # Update the data with the current instance's data
        if parent_section:
            data[parent_section][section_name] = section_data
        else:
            data[section_name] = section_data

        # Write the updated data back to the YAML file
        with file_path.open('w') as f:
            yaml.safe_dump(data, f)
            #yaml.safe_dump(data, f)


# configure YAML to use fixed representer:
yaml.add_representer(str, JustYaml.str_presenter)
# to use with safe_dump:
yaml.representer.SafeRepresenter.add_representer(str, JustYaml.str_presenter)

class JustSerializable(BaseModel):
    """
    Pydantic2 wrapper class that implements semi-automated YAML and JSON serialization and deserialization

    Constants:
        DEFAULT_CONFIG_PATH (Path): Default path to the configuration YAML file.
        DEFAULT_PARENT_SECTION (str): Default parent section name.
        DEFAULT_SECTION_NAME (str): Default section name to use when none is provided.

    """
    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        validate_assignment=True,
        populate_by_name=True
    )
    DEFAULT_CONFIG_PATH : ClassVar[Path] = Path('./config/default_config.yaml')
    DEFAULT_PARENT_SECTION : ClassVar[Optional[str]] = None
    DEFAULT_SECTION_NAME : ClassVar[Optional[str]] = "Agent" #'RenameMe'
    #MODULE_DIR : ClassVar[Path] = Path(os.path.abspath(os.path.dirname(__file__)))

    config_path : Optional[Path] = Field(None,exclude=True)
    config_parent_section: Optional[str] = Field(None,exclude=True)

    shortname: str = Field(
        DEFAULT_SECTION_NAME,
        description="A short name used as a section name in YAML storage. Must be alphanumeric, underscores, or dashes.",
        alias="name",
        exclude=True)
    """A short name used as a section name in YAML storage. Must be alphanumeric, underscores, or dashes."""

    class_hierarchy: str = Field(
        None,
        exclude=True,
        description="Information about object's parent classes.")
    """Information about object's parent classes."""

    class_qualname: str = Field(
        None,
        description="A marker to discern versions of this class when loading from configuration YAML files")
    """A marker to discern versions of this class when loading from configuration YAML files."""

    extras: Optional[dict] = Field(
        None,
        exclude=True)
    """Fallback container to store fields that don't fit the model."""

    def model_post_init(self, __context: Any) -> None:
        """
        Custom initializer to handle extra fields by storing them in the 'extras' attribute.
        """
        super().model_post_init(__context)
        if self.class_qualname is None:
            self.class_qualname = self.get_full_class_path()
        if self.class_hierarchy is None:
            self.class_hierarchy = self.get_class_hierarchy()
        if self.shortname == self.DEFAULT_SECTION_NAME:
            self.shortname = self.__class__.__name__
        # Collect extra fields into 'extras' attribute
        extra = getattr(self, '__pydantic_extra__', {}) or {}
        if self.extras is None:
            self.extras = extra
        else:
            self.extras.update(extra)

    @classmethod
    def get_full_class_path(cls) -> str:
        """
        Returns the full class path including the module and the qualified name of the class.
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    @classmethod
    def get_class_hierarchy(cls) -> str:
        """
        Extracts the class hierarchy up to the root and represents it as a string.

        Args:
            cls: The class for which to extract the inheritance chain.

        Returns:
            str: The inheritance chain as a string, e.g., "root.BaseModel.ClassA.ClassC".
        """
        # Initialize the hierarchy list with the name of the current class
        hierarchy = []
        current_class = cls

        while current_class is not None:
            hierarchy.append(current_class.__name__)
            if current_class.__bases__:
                current_class = current_class.__bases__[0]
            else:
                current_class = None

        hierarchy.reverse()
        return '.'.join(hierarchy)

    @classmethod
    @field_validator('shortname', mode="before")
    def validate_shortname(cls, value: str) -> str:
        """
        Validates the 'shortname' field to ensure it contains only allowed characters.
        Regex pattern equivalent: '/^[a-zA-Z0-9_-]+$/'
        """
        allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        if not set(value).issubset(allowed_characters):
            raise ValueError("Shortname can only contain alphanumeric characters, underscores, and dashes.")
        return value

    @classmethod
    def from_json(cls, json_data: Dict[str, Any], qualname_check: bool = True) -> 'JustSerializable':
        """
        Constructor from JSON data. Populates fields that are present in the class,
        and stores any extra fields in 'extras'.

        Args:
          json_data (Dict[str, Any]): The JSON data as a dictionary.
          qualname_check (bool): If True, checks if the class_qualname matches.

        Returns:
          JustSerializable: A new instance of JustYamlSerializable.

        Raises:
          ValidationError: If class_qualname does not match when qualname_check is True.
        """

        instance = cls.model_validate(json_data)
        class_qualname = cls.get_full_class_path()
        if qualname_check and class_qualname != instance.class_qualname:
            raise ValidationError(f"Field class_qualname mismatch:'{instance.class_qualname}', self:'{class_qualname}'")
        return instance

    @staticmethod
    def update_config_data(
            config_data: dict,
            section_name: str,
            parent_section: Optional[str],
            file_path: Path,
            class_hint: Optional[str] = None
    ) -> dict:
        """
        Link YAML configuration file, sections and class info to an instance
        """
        if not config_data.get("config_path"):
            config_data.update({"config_path": file_path})
        if not config_data.get("config_parent_section") and parent_section is not None: # '' is a valid value
            config_data.update({"config_parent_section": parent_section})
        config_data.update({"shortname": section_name})
        if not config_data.get("class_qualname") and class_hint:
            config_data.update({"class_qualname": class_hint})
        return config_data

    @classmethod
    def from_yaml(cls, section_name: str,
                  parent_section: str = None,
                  file_path: Path = None,
    ) -> SelfType:
        """
        Creates an instance from a YAML file path, section name, and parent section name.

        Args:
            section_name (str): The section name (shortname) in the YAML file.
            parent_section (str): The parent section name in the YAML file.
            file_path (Path): The path to the YAML file.

        Returns:
            JustSerializable: A new instance of JustYamlSerializable.

        Raises:
            ValueError: If the specified section is not found in the YAML file.
        """
        if not file_path:
            file_path = cls.DEFAULT_CONFIG_PATH
        if parent_section is None:
            parent_section = cls.DEFAULT_PARENT_SECTION
        if section_name is None:
            section_name = cls.DEFAULT_SECTION_NAME
        section_data = JustYaml.read_yaml_data(
            file_path,
            section_name,
            parent_section,
        )
        section_data = cls.update_config_data(section_data, section_name, parent_section, file_path)
        return cls.model_validate(section_data)

    @staticmethod
    def from_yaml_auto(section_name: str,
                       parent_section: Optional[str],
                       file_path: Path,
                       class_hint: Optional[Union[Type|str]] = None,
                       ) -> Any:
        """
        Creates an instance from a YAML file.

        This function reads configuration data from a specified YAML file, section name,
        and parent section name. If the configuration data contains a `class_qualname` field,
        it dynamically imports and instantiates the corresponding class. Otherwise, returns None.

        Args:
            section_name (str): The section name in the YAML file.
            parent_section (Optional[str]): The parent section name in the YAML file.
            file_path (Path): The path to the YAML file.
            class_hint (Optional[str]): Attempt instantiation with this class_qualname if not specified in schema

        Returns:
            Any: An instance of the dynamically imported class if `class_qualname` is found in the
                 configuration data; otherwise, returns None.
        """
        if isinstance(class_hint, type):
            class_hint : str = f"{class_hint.__module__}.{class_hint.__qualname__}"
        config_data = JustYaml.read_yaml_data_safe(
            file_path,
            section_name,
            parent_section,
        )
        if config_data is None:
            return None
        config_data = JustSerializable.update_config_data(config_data, section_name, parent_section, file_path, class_hint=class_hint)
        class_qualname = config_data.get("class_qualname")
        if class_qualname:
            try:
                # Splits into `module.submodule` and `ClassName` for dynamic import
                module_name, class_name = class_qualname.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                # Dynamic instantiation of `Child class` or whatever class is specified
                instance = cls.from_json(config_data)
                return instance
            except Exception as e:
                raise ValueError(f"Exception occurred: {str(e)}") from e
        else:
            return None


    def to_json(
            self,
            include_extras: bool = True,
            include: Optional[Set[str]] = None,
            exclude: Optional[Set[str]] = None,
            by_alias: bool = True,
            exclude_none: bool = True,
            serialize_as_any: bool = True,
            exclude_defaults: bool = True, 
            exclude_unset: bool = True
    ) -> Dict[str, Any]:
        """
        Serializes the instance to a JSON-compatible dictionary, flattening the 'extras' back into the main dictionary.
        See pydantic documentation of the arguments, they are directly mapped to model_dump

        Args:
            include_extras (bool): Whether to exclude Extra fields of Pydantic model.
            include (Optional[Set[str]]): Set of fields to include in the output, acts as Allowlist
            exclude (Optional[Set[str]]): Set of fields to exclude from the output, acts as Deny-list
            by_alias (bool): Whether to use the field's alias (if defined) in the output.
            exclude_none (bool): Whether to exclude fields with None values from the output.
            serialize_as_any (bool): Whether to serialize values by their types.
            exclude_defaults (bool): Whether to exclude fields with the default values from the output.
            exclude_unset (bool): Whether to exclude fields that were not explicitly set during instance creation from the output.

        Returns:
            Dict[str, Any]: A dictionary representation of the instance, including extra fields.
        """
        data = self.model_dump(
            mode='json',
            by_alias=by_alias,
            exclude_none=exclude_none,
            include=include,
            exclude=exclude,
            serialize_as_any=serialize_as_any,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
        )
        # Flatten Extras
        if include_extras and self.extras:
            data.update(self.extras)
        # Recreate dict with only public fields
        # public_data = {k: v for k, v in data.items() if not k.startswith('_')}
        return data

    def to_json_inclusive(
            self,
            include_extras: bool = True,
            include_list: Optional[Sequence[str]] = None,
            exclude_list: Optional[Sequence[str]] = None,
            by_alias: bool = True,
            exclude_none: bool = True,
            serialize_as_any: bool = True,

    ) -> Dict[str, Any]:
        """
        Serializes the instance to a JSON-compatible dictionary, changes the behavior of to override include

        Args:
            include_extras (bool): Whether to exclude Extra fields of Pydantic model.
            include_list (Optional[Sequence[str]]): List of fields to force-include in the output in addition to default set
            exclude_list (Optional[Sequence[str]]): Set of fields to exclude from the output.
            by_alias (bool): Whether to use the field's alias (if defined) in the output.
            exclude_none (bool): Whether to exclude fields with None values from the output.
            serialize_as_any (bool): Whether to serialize values by their types.

        Returns:
            Dict[str, Any]: A dictionary representation of the instance, including extra fields.
        """

        available_fields = frozenset(self.model_fields.keys())

        if exclude_list is not None:
            exclude_set = set(exclude_list) & available_fields
        else:
            exclude_set = set()

        if include_list is not None:
            include_set = set(include_list) & available_fields - exclude_set
        else:
            include_set = set()

        data = self.to_json(
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude=exclude_set,
            include_extras=include_extras,
            serialize_as_any=serialize_as_any,
        )

        for field in include_set:
            value = getattr(self, field, None)
            if value is not None or not exclude_none:
                data[field] = value

        return data

    def save_to_yaml(
            self,
            section_name: str = None,
            parent_section: str = None,
            file_path: Path = None,
            include_extras: bool = True,
            include: Optional[Set[str]] = None,
            exclude: Optional[Set[str]] = None,
            by_alias: bool = True,
            exclude_none: bool = True,
            serialize_as_any: bool = True,
            exclude_defaults: bool = True, 
            exclude_unset: bool = False
    ):
        """
        Saves the instance's data to a YAML file under the specified parent section and section name (shortname).
        If the file exists, it appends or updates the existing data.

        Args:
            section_name (str): The  section name in the YAML file, default to shortname
            parent_section (str): The parent section name in the YAML file.
            file_path (Path): The path to the YAML file.
            include_extras (bool): Whether to exclude Extra fields of Pydantic model.
            See pydantic documentation of the following flags:
            include (Optional[Set[str]]): Set of fields to include in the output, acts as Allowlist
            exclude (Optional[Set[str]]): Set of fields to exclude from the output, acts as Deny-list
            by_alias (bool): Whether to use the field's alias (if defined) in the output.
            exclude_none (bool): Whether to exclude fields with None values from the output.
            serialize_as_any (bool): Whether to serialize values by their types.
            exclude_defaults (bool): Whether to exclude fields with the default values from the output.
            exclude_unset (bool): Whether to exclude unset fields from the output.
        """

        if not file_path:
            file_path = self.config_path or self.DEFAULT_CONFIG_PATH #set configured or default
        if parent_section is None:
            parent_section = self.config_parent_section
        if not section_name:
            section_name = self.shortname #Set configured

        section_data = self.to_json(
            include_extras=include_extras,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_none=exclude_none,
            serialize_as_any=serialize_as_any,
            exclude_defaults=exclude_defaults, 
            exclude_unset=exclude_unset
        )
        JustYaml.save_to_yaml(file_path, section_data, section_name or self.shortname, parent_section)

    def update(self, new_data: Dict, overwrite: bool = False, strict: bool = False) -> None:
        """
        Updates the fields of the current instance with the values from another JustAgentProfile instance.

        Args:
            new_data (Dict): The instance from which to update the current instance's fields.
            overwrite (bool): If True, existing fields are overwritten with new values.
            strict (bool): If True, existing fields are overwritten with new values.
        Raises:
          ValidationError: If class_qualname does not match between objects.
        """

        if new_data is None:
            return
        else:
            new_data.pop("class_qualname", None)
            #if new_qualname and new_qualname != self.class_qualname:
            #    raise ValidationError(f"Field class_qualname mismatch:'{new_qualname}', self:'{self.class_qualname}'")
            if strict:
                self.validate_keys_match(new_data)

        for field_name, field_info in self.model_fields.items():
            self_value = getattr(self, field_name, None)
            new_value : Optional[Any] = new_data.pop( field_name, None )

            if new_value is None: #there is a value to set
                continue

            if not self_value or overwrite:
                setattr(self, field_name, new_value)  #set or overwrite
            else: # self_value is set, overwrite unset, update if extension is possible
                if isinstance(self_value, MutableMapping):
                    self_value.update(new_value)
                    setattr(self, field_name, self_value)
                elif isinstance(self_value, MutableSequence):
                    self_value.extend(new_value)
                    setattr(self, field_name, self_value)
                elif isinstance(self_value, JustSerializable): #recurse
                    self_value.update(new_value.model_dump(), overwrite)
                else:
                    pass #extension not possible

        if new_data: #extra data remaining
            if self.extras is None:
                self.extras = new_data
            else:
                self.extras.update(new_data)

    def update_from_yaml(self, overwrite: bool = False):
        """
        Update instance fields from linked YAML configuration
        """
        profile  = self.from_yaml_auto(
            self.shortname,
            parent_section=self.config_parent_section,
            file_path=self.config_path,
        )
        # Loaded some data, parameters set in init take precedence
        if profile and isinstance(profile, self.__class__):
            self.update(
                profile.to_json(
                    by_alias=False
                ),
                overwrite=overwrite
            )

    def validate_keys_match(self, instance: Union[BaseModel, Dict, Sequence]):
        """
        Dynamic validation of effective model fields compatibility between two instances.
        """
        if isinstance(instance, BaseModel):
            model_fields = set(instance.model_fields.keys())
        elif isinstance(instance, dict):
            model_fields=set(instance.keys())
        elif isinstance(instance, Sequence):
            model_fields=set(instance)
        else:
            raise TypeError(f"Unsupported type: {str(instance)}")
        this_fields = set(self.model_fields.keys())

        if model_fields.issubset(this_fields):
            missing_in_this = model_fields - this_fields
            missing_in_instance = this_fields - model_fields
            raise ValueError(f"Mismatch between model fields! Missing in Self: {missing_in_this}. In instance: {missing_in_instance}.")

    def fields_to_populate(
            self,
            include_nonnull: bool = False,
            include_list: Optional[Set[str]] = None,
            exclude_list: Optional[Set[str]] = None,
            extra_list: Dict[str, str] = None
    ) -> Dict[str, str]:
        """
        Lists the fields of the model that are suitable for being populated by language models.

        Args:
            include_nonnull (bool): If True, include fields that are already set (non-null) in the output.
            include_list (Optional[Set[str]]): A set of field names to force include in the output, when Description is available for the field.
            exclude_list (Optional[Set[str]]): A set of field names to additionally exclude from the output, include_list takes precedence.
            extra_list (Dict[str, str]): Extra fields to populate, with field names as keys and descriptions as values.

        Returns:
            Dict[str, str]: A dictionary of fields to populate, with descriptions as values.
        """
        # Set of settable fields with descriptions
        available_fields = {
            field_name for field_name, field_info in self.model_fields.items()
            if field_info.description and not field_info.frozen
        }

        excluded_fields = {
            field_name for field_name, field_info in self.model_fields.items()
            if field_info.exclude
        }

        # Extend exclude set of field names that are already set (nonnull values) if include_nonnull is False
        if not include_nonnull:
            excluded_fields.update(set({
                field_name for field_name in available_fields
                if getattr(self, field_name, None) is not None
            }))

        # Exclude set of field names that were explicitly marked for exclusion
        if exclude_list:
            excluded_fields.update(set(exclude_list))

        # Reduce exclude set by field names that are in include list
        if include_list:
            excluded_fields.difference_update(set(include_list))

        # Calculate eligible fields: all described fields minus frozen, nonnull (if applicable), excluded, except included
        eligible_fields = (available_fields - excluded_fields)

        # Create the final field_list dictionary using the eligible fields
        fields_to_populate = {field_name: self.model_fields[field_name].description for field_name in eligible_fields}

        # Top up with extras if requested
        if extra_list:
            for field, value in list(extra_list.items()):
                if field not in self.extras:
                    fields_to_populate[field] = value

        return fields_to_populate

