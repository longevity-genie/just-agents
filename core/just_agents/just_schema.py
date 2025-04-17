from typing import Type, TypeVar, Any, Dict, Optional, Union, get_origin, get_args, List, Literal
from pydantic import BaseModel, Field, create_model, ConfigDict
import re
import ast
import json

from just_agents.web.chat_ui import ModelConfig

T = TypeVar('T', bound=BaseModel)
ConfigDictExtra = Literal["ignore", "allow", "forbid"]

class ModelHelper:
    """
    Utility class with static methods for working with Pydantic models.
    """
    
    @staticmethod
    def extract_common_fields(selected_class: Type[T], instance: BaseModel) -> T:
        """
        Trims and typecasts an instance of a class to only include the fields of the selected class.

        Args:
            selected_class: The class type to trim to.
            instance: The instance of the class to be trimmed.
        
        Returns:
            An instance of the selected class populated with the relevant fields from the provided object.
        """
        # Extract only the fields defined in the base class
        base_fields = {field: getattr(instance, field) for field in selected_class.model_fields}

        # Instantiate and return the base class with these fields
        return selected_class(**base_fields)

    @staticmethod
    def trim_to_parent(instance: T) -> BaseModel:
        """
        Trims an instance of a derived class to only include the fields of its direct parent class.
        
        Args:
            instance: The instance of the derived class to be trimmed.
        
        Returns:
            An instance of the parent class populated with the relevant fields from the derived class.
        """
        # Get the direct parent class of the instance
        parent_class = instance.__class__.__bases__[0]

        # Instantiate and return the parent class with these fields
        return ModelHelper.extract_common_fields(parent_class, instance)

    @staticmethod
    def make_all_fields_required(
        model_class: Type[BaseModel],
        cache: Optional[Dict[Type[BaseModel], Type[BaseModel]]] = None
    ) -> Type[BaseModel]:
        """
        Creates a new Pydantic model class where all fields are required (recursively).
        
        Args:
            model_class: The original Pydantic model class to transform
            cache: A dictionary to cache already processed models to avoid infinite recursion
            
        Returns:
            A new Pydantic model class with all fields marked as required
        """
        # Initialize cache to avoid processing the same model twice (prevents infinite recursion)
        if cache is None:
            cache = {}
        
        # If we've already processed this model, return the cached version
        if model_class in cache:
            return cache[model_class]
        
        # Create a temporary placeholder to handle recursive references
        new_class_name = f"Required{model_class.__name__}"
        cache[model_class] = None  # Will be replaced with actual implementation
        
        # Process all fields
        new_fields: Dict[str, tuple] = {}
        
        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation
            
            # Process the type to make nested models required
            processed_type = ModelHelper.process_field_type(field_type, cache)
            
            # Create a Field without default value to make it required
            field_kwargs = {
                # Preserve field metadata
                "description": field_info.description,
                "title": field_info.title,
                # Remove defaults to make field required
            }
            
            if field_info.json_schema_extra:
                field_kwargs["json_schema_extra"] = field_info.json_schema_extra
                
            # Add the processed field to the new fields dictionary
            new_fields[field_name] = (processed_type, Field(**field_kwargs))
        
        # Create the new model with required fields
        model_config = getattr(model_class, "model_config", None)
        
        new_model = create_model(
            new_class_name,
            __config__=model_config,
            **new_fields
        )
        
        # Store the created model in cache
        cache[model_class] = new_model
        
        return new_model

    @staticmethod
    def process_field_type(
        field_type: Any, 
        cache: Dict[Type[BaseModel], Type[BaseModel]]
    ) -> Any:
        """
        Process field types recursively, handling container types like List, Dict, Union, etc.
        
        Args:
            field_type: The type annotation to process
            cache: Cache of already processed models
            
        Returns:
            The processed type with nested models made required
        """
        # Handle None/Optional types by unwrapping them
        origin = get_origin(field_type)
        args = get_args(field_type)
        
        # Handle Union types (including Optional which is Union[T, None])
        if origin is Union:
            # Process each union member type
            processed_args = [
                ModelHelper.process_field_type(arg, cache) for arg in args 
                if arg is not type(None)  # Remove None from Union to make required
            ]
            
            # If only one type remains, return it directly
            if len(processed_args) == 1:
                return processed_args[0]
            # Otherwise create a new Union with the processed types
            return Union[tuple(processed_args)]
        
        # Handle container types like List, Dict, etc.
        elif origin is not None and args:
            # Process container element types
            processed_args = tuple(ModelHelper.process_field_type(arg, cache) for arg in args)
            # Check if the origin type supports subscription
            if hasattr(origin, '__getitem__'):
                return origin[processed_args]
            # For types that don't support subscription, return the original type
            return field_type
        
        # Handle Pydantic models - recursively make their fields required
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return ModelHelper.make_all_fields_required(field_type, cache)
        
        # Return unmodified type for basic types
        return field_type
    
    @staticmethod
    def clean_fallback_result(raw: str) -> str:
        """
        Remove any markdown code fences from the fallback JSON response.
        This regex checks if the entire string is enclosed within triple-backticks with an
        optional "json" language tag, and if so, returns just the content.
        
        Args:
            raw: The raw string that might contain markdown code fences
            
        Returns:
            Cleaned string with markdown code fences removed if present
        """
        raw = raw.strip()
        pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
        match = pattern.match(raw)
        if match:
            return match.group(1)
        return raw
    
    @staticmethod
    def get_structured_output(
        raw_response: Union[dict, str, Any],
        parser: Type[BaseModel] = BaseModel
    ) -> Union[dict, BaseModel]:
        """
        Parse the response according to the provided parser.
        Attempts multiple parsing strategies in the following order:
        1. Direct dict usage if already a dict
        2. Standard json parsing
        3. Clean markdown code fences if present and retry parsing
        4. AST literal eval as final fallback
        
        Args:
            raw_response: The raw response from the LLM
            parser: The Pydantic model class to validate against or dict
            
        Returns:
            Either a dictionary or validated Pydantic model
            
        Raises:
            ValueError: If parsing fails with all available methods
        """
        # If already a dict, no parsing needed
        if isinstance(raw_response, dict):
            return raw_response if parser == dict else parser.model_validate(raw_response)

        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        parsing_errors = []
        
        # Try standard json first on the raw response
        try:
            response_dict = json.loads(raw_response)
            return response_dict if parser == dict else parser.model_validate(response_dict)
        except json.JSONDecodeError as e:
            parsing_errors.append(f"Standard JSON parsing failed: {str(e)}")
            
            # Only clean markdown code blocks if initial parsing failed
            cleaned_response = ModelHelper.clean_fallback_result(raw_response)

            # Check for and fix common double-escaping issues
            if '\\\\' in cleaned_response:
                cleaned_response = cleaned_response.replace('\\\\', '\\')
                
            # Try parsing the cleaned response
            try:
                response_dict = json.loads(cleaned_response)
                return response_dict if parser == dict else parser.model_validate(response_dict)
            except json.JSONDecodeError as e:
                parsing_errors.append(f"Cleaned JSON parsing failed: {str(e)}")

            # Final fallback: AST literal eval
            try:
                response_dict = ast.literal_eval(cleaned_response)
                if isinstance(response_dict, dict):
                    return response_dict if parser == dict else parser.model_validate(response_dict)
                parsing_errors.append("AST parsing succeeded but result was not a dict")
            except (ValueError, SyntaxError) as e:
                parsing_errors.append(f"AST literal_eval parsing failed: {str(e)}")

        # If all parsing attempts failed, raise detailed error
        raise ValueError(
            f"Failed to parse response using multiple methods:\n"
            f"{chr(10).join(parsing_errors)}\n"
            f"Original response:\n{raw_response}\n"
            f"Cleaned response:\n{cleaned_response}"
        )
    
    @staticmethod
    def get_response_schema(parser: Type[BaseModel]) -> dict:
        """
        Extract JSON schema from a Pydantic model for use with LLM response formatting.
        
        Args:
            parser: A Pydantic model class to extract schema from
            
        Returns:
            A dictionary representing the JSON schema
        """
        # Get the JSON schema from the Pydantic model
        schema = parser.model_json_schema()
        
        def clean_schema_recursively(schema_obj):
            """Recursively remove Pydantic-specific fields that cause issues with LLM APIs"""
            if isinstance(schema_obj, dict):
                # Remove problematic fields at current level
                for field in ["default", "title", "$defs"]:
                    if field in schema_obj:
                        del schema_obj[field]
                
                # Handle nested properties
                if "properties" in schema_obj:
                    for prop_name, prop_value in schema_obj["properties"].items():
                        clean_schema_recursively(prop_value)
                
                # Handle items in arrays
                if "items" in schema_obj:
                    clean_schema_recursively(schema_obj["items"])
                
                # Handle additional properties
                if "additionalProperties" in schema_obj and isinstance(schema_obj["additionalProperties"], dict):
                    clean_schema_recursively(schema_obj["additionalProperties"])
        
        # Clean the schema recursively
        clean_schema_recursively(schema)
        
        # Make sure required fields are specified for OpenAI models
        if "properties" in schema:
            # If required field doesn't exist, create it
            if "required" not in schema:
                schema["required"] = []
            
            # Get all property names
            all_properties = list(schema["properties"].keys())
            
            # For OpenAI models, we need to include all properties in the required array
            # even if they're optional in the Pydantic model
            for field_name in all_properties:
                if field_name not in schema["required"]:
                    schema["required"].append(field_name)
        
        return schema

    @staticmethod
    def infer_type_from_value(value: Any) -> Type:
        """
        Infers the appropriate Python/Pydantic type from a value.
        
        Args:
            value: The value to infer a type from. Can be a Python value or a string
                  representing a type annotation (e.g., 'str', 'int', 'Optional[str]')
            
        Returns:
            The inferred type
        """
        # Basic type mapping for string notations
        type_mapping = {
            'str': str, 'string': str,
            'int': int, 'integer': int,
            'float': float,
            'bool': bool, 'boolean': bool,
            'list': List[Any],
            'dict': Dict[str, Any],
            'any': Any,
            'none': type(None)
        }
        
        # Handle string type notation (e.g., "str", "Optional[int]")
        if isinstance(value, str):
            # Normalize the input string
            normalized_value = value.strip()
            lower_normalized_value = normalized_value.lower()
            
            # 1. Check for basic types
            if lower_normalized_value in type_mapping:
                return type_mapping[lower_normalized_value]
            
            # Helper functions for parsing complex type strings
            def extract_bracket_content(s, start_pos, open_bracket='[', close_bracket=']'):
                """Extract content between balanced brackets from the given position."""
                if start_pos >= len(s) or s[start_pos] != open_bracket:
                    return None, start_pos
                
                bracket_level = 0
                start = start_pos
                for i in range(start_pos, len(s)):
                    if s[i] == open_bracket:
                        bracket_level += 1
                    elif s[i] == close_bracket:
                        bracket_level -= 1
                        if bracket_level == 0:
                            # Found the matching closing bracket
                            return s[start + 1:i], i + 1
                
                # Unbalanced brackets
                return None, start_pos
            
            def split_top_level_commas(s):
                """Split string by commas, but only at the top level (not inside brackets)."""
                result = []
                bracket_level = 0
                start = 0
                
                for i, char in enumerate(s):
                    if char == '[':
                        bracket_level += 1
                    elif char == ']':
                        bracket_level -= 1
                    elif char == ',' and bracket_level == 0:
                        result.append(s[start:i].strip())
                        start = i + 1
                
                # Add the last part
                if start < len(s):
                    result.append(s[start:].strip())
                
                return result
            
            # 2. Handle complex type annotations by checking prefixes
            
            # Optional[T]
            if normalized_value.lower().startswith('optional['):
                content, _ = extract_bracket_content(normalized_value, normalized_value.find('['))
                if content:
                    inner_type = ModelHelper.infer_type_from_value(content)
                    return Optional[inner_type]
            
            # Union[T1, T2, ...]
            if normalized_value.lower().startswith('union['):
                content, _ = extract_bracket_content(normalized_value, normalized_value.find('['))
                if content:
                    type_names = split_top_level_commas(content)
                    union_types = [ModelHelper.infer_type_from_value(name) for name in type_names]
                    valid_types = [t for t in union_types if t is not None]
                    if valid_types:
                        return Union[tuple(valid_types)] if len(valid_types) > 1 else valid_types[0]
            
            # List[T]
            if normalized_value.lower().startswith('list['):
                content, _ = extract_bracket_content(normalized_value, normalized_value.find('['))
                if content:
                    item_type = ModelHelper.infer_type_from_value(content)
                    return List[item_type]
            
            # Dict[K, V]
            if normalized_value.lower().startswith('dict['):
                content, _ = extract_bracket_content(normalized_value, normalized_value.find('['))
                if content:
                    parts = split_top_level_commas(content)
                    if len(parts) == 2:
                        key_type = ModelHelper.infer_type_from_value(parts[0])
                        value_type = ModelHelper.infer_type_from_value(parts[1])
                        return Dict[key_type, value_type]
            
            # 3. Check if it looks like a class name (PascalCase)
            if normalized_value and normalized_value[0].isupper() and not re.search(r'[\s\[\])(,]', normalized_value):
                # Assume it might be a Pydantic model or other class name, return Any
                return Any

            # 4. Default: treat as normal string value
            return str
        
        # Handle non-string Python values
        if value is None:
            return Any
        elif isinstance(value, bool):
            return bool
        elif isinstance(value, int):
            return int
        elif isinstance(value, float):
            return float
        elif isinstance(value, list):
            return List[ModelHelper.infer_type_from_value(value[0])] if value else List[Any]
        elif isinstance(value, dict):
            if not value:  # Empty dict
                return Dict[str, Any]
                
            # Check if it looks like a schema definition (keys and values are strings)
            if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                # If it's a schema-like dictionary with string values
                val_types = {ModelHelper.infer_type_from_value(v) for v in value.values()}
                if len(val_types) == 1 and next(iter(val_types)) != str:
                    return Dict[str, next(iter(val_types))]
                return Dict[str, Any]
            
            # Try to create a nested model from the dict structure
            try:
                nested_model_name = f"Nested{hash(frozenset(value.keys())) % 10000}"
                return ModelHelper.create_model_from_flat_yaml(nested_model_name, value)
            except Exception:
                return Dict[str, Any]
        
        # Default: use the runtime type
        return type(value)
    
    @staticmethod
    def create_model_from_yaml(
        model_name: str, 
        yaml_data: Dict[str, Any],
        optional_fields: bool = True,
        extra: ConfigDictExtra = "ignore"
    ) -> Type[BaseModel]:
        """
        Creates a Pydantic model dynamically from YAML data by inferring types from values.
        
        Args:
            model_name: Name for the created model class
            yaml_data: Dictionary with field definitions from YAML
            optional_fields: Whether fields should be optional (can be None)
            extra: Behavior for extra fields ("ignore", "allow", or "forbid")
            
        Returns:
            A new Pydantic model class with the inferred field types
            
        Example YAML structure:
            parser:
              - name: "John"
              - age: 12
              - score: 0.8
              - participant: true
              - undefined_property
            
            parser_python_like:
              - name: str
              - age: int
              - score: float
              - maybe_name: Optional[str]
              - id_or_name: Union[int, str]
        """
        field_definitions = {}
        
        # Extract field definitions
        for item in yaml_data:
            if isinstance(item, dict):
                # Handle key-value entries
                for field_name, default_value in item.items():
                    field_type = ModelHelper.infer_type_from_value(default_value)
                    
                    # For string type notation, we use the type but don't set a default value
                    if isinstance(default_value, str) and (
                        default_value.lower() in [
                            'str', 'string', 'int', 'integer', 'float', 'bool', 'boolean', 
                            'list', 'dict', 'any', 'none'
                        ] or 
                        re.match(r'^(?:optional|union|list)\[[\w\s,]+]$', default_value, re.IGNORECASE)
                    ):
                        if optional_fields:
                            field_definitions[field_name] = (Optional[field_type], None)
                        else:
                            field_definitions[field_name] = (field_type, ...)
                    else:
                        # Normal case with a concrete value
                        if optional_fields:
                            field_definitions[field_name] = (Optional[field_type], default_value)
                        else:
                            field_definitions[field_name] = (field_type, default_value)
            elif isinstance(item, str):
                # Handle bare field names without values (undefined_property)
                if optional_fields:
                    field_definitions[item] = (Optional[Any], None)
                else:
                    field_definitions[item] = (Any, ...)
        
        # Create model config
        model_config: ConfigDict = {"extra": extra}
        
        # Create and return model
        return create_model(
            model_name,
            __config__=model_config,
            **field_definitions
        )
    
    @staticmethod
    def create_model_from_flat_yaml(
        model_name: str,
        yaml_data: Dict[str, Any],
        optional_fields: bool = True,
        extra: ConfigDictExtra = "ignore"
    ) -> Type[BaseModel]:
        """
        Creates a Pydantic model from a flat YAML dictionary (not list-based).
        
        Args:
            model_name: Name for the created model class
            yaml_data: Dictionary mapping field names to values or type annotations
            optional_fields: Whether fields should be optional
            extra: Behavior for extra fields
            
        Returns:
            A new Pydantic model class
        
        Example structure:
            parser:
                name: "John"      # Value-based type inference
                age: 12           # Will become int
                score: 0.8        # Will become float
                participant: true # Will become bool
                
            parser_python_like:
                name: str         # Type notation-based inference  
                age: int          # Will use the specified type
                score: float      # Will use the specified type
                maybe_name: Optional[str]  # Will be an optional string
                id_or_name: Union[int, str]  # Will accept either int or str
        """
        field_definitions = {}
        
        for field_name, default_value in yaml_data.items():
            field_type = ModelHelper.infer_type_from_value(default_value)
            
            # For string type notation, we use the type but don't set a default value
            if isinstance(default_value, str) and (
                default_value.lower() in [
                    'str', 'string', 'int', 'integer', 'float', 'bool', 'boolean', 
                    'list', 'dict', 'any', 'none'
                ] or 
                re.match(r'^(?:optional|union|list)\[[\w\s,]+]$', default_value, re.IGNORECASE)
            ):
                if optional_fields:
                    field_definitions[field_name] = (Optional[field_type], None)
                else:
                    field_definitions[field_name] = (field_type, ...)
            else:
                # Normal case with a concrete value
                if optional_fields:
                    field_definitions[field_name] = (Optional[field_type], default_value)
                else:
                    field_definitions[field_name] = (field_type, default_value)
        
        # Create model config
        model_config : ConfigDict = {"extra": extra}
        
        # Create and return model
        return create_model(
            model_name,
            __config__=model_config,
            **field_definitions
        )

    @staticmethod
    def model_instance_from_yaml(
        model_name: str,
        yaml_data: Dict[str, Any],
        instance_data: Optional[Dict[str, Any]] = None,
        optional_fields: bool = True,
        extra: ConfigDictExtra = "ignore"
    ) -> BaseModel:
        """
        Creates a Pydantic model from YAML data and returns an instance of it.
        
        This is a convenience method that combines model creation and instantiation
        in a single step. 
        
        Args:
            model_name: Name for the created model class
            yaml_data: YAML data (list or dict) to infer model structure from
            instance_data: Optional data to use when instantiating the model
                           (defaults to using yaml_data values)
            optional_fields: Whether fields should be optional
            extra: Behavior for extra fields
            
        Returns:
            An instance of the created model
            
        Example:
            yaml_data = [{"name": "John"}, {"age": 30}, "email"]
            user = ModelHelper.model_instance_from_yaml("User", yaml_data)
        """
        # Create model based on data structure
        if isinstance(yaml_data, dict):
            model_class = ModelHelper.create_model_from_flat_yaml(
                model_name, 
                yaml_data, 
                optional_fields=optional_fields,
                extra=extra
            )
        else:
            raise ValueError(f"Unexpected data format: {type(yaml_data)}")
        
        # Create instance with provided data or inferred values
        if instance_data is not None:
            return model_class(**instance_data)
        else:
            # Convert YAML definition to instance data
            if isinstance(yaml_data, list):
                data = {}
                for item in yaml_data:
                    if isinstance(item, dict):
                        data.update(item)
                    elif isinstance(item, str):
                        # For bare field names, we default to None
                        data[item] = None
                return model_class(**data)
            else:
                # For flat dict structure, use as is
                return model_class(**yaml_data)

    @staticmethod
    def serialize_model_schema(model_class: Type[BaseModel], simplify: bool = True) -> Dict[str, Any]:
        """
        Serializes a Pydantic model class into a dictionary representation of field types.
        This is essentially the reverse of create_model_from_flat_yaml.
        
        Args:
            model_class: The Pydantic model class to serialize
            simplify: Whether to simplify complex type annotations to more readable strings
                      (e.g., "Optional[str]" instead of actual type objects)
        
        Returns:
            A dictionary where keys are field names and values are type representations
            
        Example:
            ```python
            class User(BaseModel):
                name: str
                age: int
                email: Optional[str] = None
                
            schema = ModelHelper.serialize_model_schema(User)
            # Result: {'name': 'str', 'age': 'int', 'email': 'Optional[str]'}
            ```
        """
        # Get the model's JSON schema
        json_schema = model_class.model_json_schema()
        
        # Extract field information
        result = {}
        
        # Process properties from the schema
        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation
            
            if simplify:
                # Get a simplified string representation of the type
                type_str = ModelHelper.simplify_type_annotation(field_type)
                result[field_name] = type_str
            else:
                # Use the actual type object
                result[field_name] = field_type
                
        return result
    
    @staticmethod
    def simplify_type_annotation(type_annotation: Any) -> str:
        """
        Converts complex type annotations to simplified string representations.
        
        Args:
            type_annotation: A type annotation object (e.g., str, Optional[str], etc.)
            
        Returns:
            A simplified string representation of the type
            
        Examples:
            str -> "str"
            Optional[str] -> "Optional[str]"
            Union[str, int] -> "Union[str, int]"
            List[str] -> "List[str]"
        """
        # Handle None type
        if type_annotation is type(None):
            return "None"
            
        # Handle simple types
        if type_annotation is str:
            return "str"
        elif type_annotation is int:
            return "int"
        elif type_annotation is float:
            return "float"
        elif type_annotation is bool:
            return "bool"
        elif type_annotation is Any:
            return "Any"
            
        # Handle complex types
        origin = get_origin(type_annotation)
        args = get_args(type_annotation)
        
        if origin is Union:
            # Filter out NoneType for Optional/Union handling
            non_none_args = [arg for arg in args if arg is not type(None)]
            
            # Check if it was Optional (Union[T, None])
            if len(args) == 2 and type(None) in args:
                 # Format as Optional[T]
                 other_type = non_none_args[0]
                 return f"Optional[{ModelHelper.simplify_type_annotation(other_type)}]"
            # Check if it was Union[..., None] -> format as Optional[Union[...]]
            elif type(None) in args and len(non_none_args) > 1:
                 union_args_str = [ModelHelper.simplify_type_annotation(arg) for arg in non_none_args]
                 return f"Optional[Union[{', '.join(union_args_str)}]]"
            # Regular Union without None
            elif len(non_none_args) > 0:
                arg_strings = [ModelHelper.simplify_type_annotation(arg) for arg in non_none_args]
                # If only one type left after removing None, it's not really a Union
                if len(arg_strings) == 1:
                    return arg_strings[0]
                return f"Union[{', '.join(arg_strings)}]"
            else: # Should not happen unless Union[None] which is None
                return "None"

        elif origin is list or origin is List:
            if args:
                return f"List[{ModelHelper.simplify_type_annotation(args[0])}]"
            return "List[Any]"
            
        elif origin is dict or origin is Dict:
            if len(args) == 2:
                key_type_str = ModelHelper.simplify_type_annotation(args[0])
                value_type_str = ModelHelper.simplify_type_annotation(args[1])
                return f"Dict[{key_type_str}, {value_type_str}]"
            return "Dict[str, Any]"
        
        # For Pydantic models, return their name
        if isinstance(type_annotation, type) and issubclass(type_annotation, BaseModel):
            return type_annotation.__name__
            
        # Default: use the string representation of the type
        type_repr = str(type_annotation)
        # Clean up common prefixes
        type_repr = type_repr.replace("typing.", "")
        type_repr = re.sub(r"<class '(__main__|builtins)\.(\w+)'>", r"\2", type_repr)
        return type_repr
