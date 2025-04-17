import pytest
from typing import List, Dict, Optional, Union, Any, get_origin, get_args
from pydantic import BaseModel
import yaml
import tempfile
import os
import re

from just_agents.just_schema import ModelHelper


# Define test models with various type annotations
class SimpleModel(BaseModel):
    """A simple model with basic types"""
    name: str
    age: int
    score: float
    is_active: bool


class ComplexModel(BaseModel):
    """A model with more complex types"""
    id: int
    name: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    parent: Optional[SimpleModel] = None
    scores: Dict[str, float] = {}
    status: Union[str, int] = "active"


class NestedModel(BaseModel):
    """A model with nested structure"""
    title: str
    details: ComplexModel
    alternatives: Optional[List[SimpleModel]] = None


# --- New Convoluted Models ---
class VeryNestedItem(BaseModel):
    item_id: str
    properties: Optional[Dict[str, Union[int, str, bool]]] = None


class DeepStructure(BaseModel):
    level1_id: str
    level2_list: List[Optional[VeryNestedItem]] = []
    level3_dict: Dict[str, List[Union[SimpleModel, ComplexModel]]] = {}


class UnionWithModel(BaseModel):
    identifier: Union[str, int, SimpleModel]
    extra_data: Any = None


class OptionalComplexCollection(BaseModel):
    collection_name: str
    items: Optional[List[Dict[str, Union[int, str, SimpleModel]]]] = None


class ModelWithAny(BaseModel):
    explicit_any: Any
    dict_with_any_val: Dict[str, Any]
    list_of_any: List[Any]


# Create test model instances directly in a function (not a fixture)
def get_test_model_instances():
    """Create and return test model instances."""
    simple = SimpleModel(name="John Doe", age=30, score=98.6, is_active=True)
    complex_instance = ComplexModel(id=123, name="Complex Object", tags=["test", "demo"], metadata={"source": "test_suite", "version": 1}, scores={"math": 95.0, "science": 88.5}, parent=simple)
    nested = NestedModel(title="Nested Example", details=complex_instance, alternatives=[SimpleModel(name="Alt 1", age=25, score=85.5, is_active=False), SimpleModel(name="Alt 2", age=35, score=92.0, is_active=True)])
    very_nested_1 = VeryNestedItem(item_id="vn1", properties={"size": 10, "color": "red"})
    very_nested_2 = VeryNestedItem(item_id="vn2")
    deep = DeepStructure(level1_id="deep1", level2_list=[very_nested_1, None, very_nested_2], level3_dict={"groupA": [simple, complex_instance], "groupB": [SimpleModel(name="GroupB Simple", age=40, score=70.0, is_active=True)]})
    union_model_str = UnionWithModel(identifier="string_id", extra_data=123)
    union_model_int = UnionWithModel(identifier=999, extra_data=True)
    union_model_model = UnionWithModel(identifier=SimpleModel(name="Union Simple", age=50, score=99.0, is_active=False))
    optional_collection = OptionalComplexCollection(collection_name="Mixed Collection", items=[{"item1": 100, "item2": "abc"}, {"nested_model": SimpleModel(name="Collection Simple", age=1, score=10.0, is_active=True)}])
    model_any = ModelWithAny(explicit_any=42, dict_with_any_val={"key1": "string", "key2": 100, "key3": None}, list_of_any=[True, 3.14, "hello", simple])

    return [
        ("simple", simple),
        ("complex", complex_instance),
        ("nested", nested),
        ("deep", deep),
        ("union_str", union_model_str),
        ("union_int", union_model_int),
        ("union_model", union_model_model),
        ("optional_collection", optional_collection),
        ("model_any", model_any),
        ("very_nested", very_nested_1)
    ]


# --- Shared helper functions for type checking ---
def is_semantically_equivalent(type_str1, type_str2):
    """Check if two type strings are semantically equivalent, handling Any and Union edge cases."""
    # Normalize strings
    s1 = type_str1.lower().replace(" ", "")
    s2 = type_str2.lower().replace(" ", "")
    
    # 1. Direct equality check
    if s1 == s2:
        return True
        
    # 2. Union[Any, Any, ...] === Any
    if "union" in s1 and "any" in s1 and all(part.strip() == "any" for part in 
                                          re.findall(r'union\[(.*?)\]', s1)[0].split(',')) and s2 == "any":
        return True
    if "union" in s2 and "any" in s2 and all(part.strip() == "any" for part in 
                                          re.findall(r'union\[(.*?)\]', s2)[0].split(',')) and s1 == "any":
        return True
        
    # 3. Union normalization - order of types in union doesn't matter
    union_pattern1 = re.search(r'union\[(.*?)\]', s1)
    union_pattern2 = re.search(r'union\[(.*?)\]', s2)
    if union_pattern1 and union_pattern2:
        types1 = sorted(t.strip() for t in union_pattern1.group(1).split(','))
        types2 = sorted(t.strip() for t in union_pattern2.group(1).split(','))
        if types1 == types2:
            return True
            
    # Helper function to normalize Any expressions recursively
    def normalize_any_expressions(s):
        """Replace all Union[Any, Any, ...] with Any recursively in nested type expressions"""
        # First normalize any Union[Any, Any] -> Any
        def replace_union_any(match):
            union_content = match.group(1)
            if all(part.strip() == "any" for part in union_content.split(',')):
                return "any"
            return f"union[{union_content}]"
        
        # Apply replacements repeatedly until no more changes
        prev_s = ""
        while prev_s != s:
            prev_s = s
            s = re.sub(r'union\[((?:any,\s*)*any)\]', replace_union_any, s)
        
        return s
    
    # Apply normalization to both strings
    norm_s1 = normalize_any_expressions(s1)
    norm_s2 = normalize_any_expressions(s2)
    
    # Try direct comparison after normalization
    if norm_s1 == norm_s2:
        return True
            
    # 4. List[Union[Any, Any]] === List[Any]
    list_pattern1 = re.search(r'list\[(.*?)\]', norm_s1)
    list_pattern2 = re.search(r'list\[(.*?)\]', norm_s2)
    if list_pattern1 and list_pattern2:
        inner1 = list_pattern1.group(1)
        inner2 = list_pattern2.group(1)
        
        # Recursively check inner types
        return is_semantically_equivalent(inner1, inner2)
            
    # 5. Dict[Key, Union[Any, Any]] === Dict[Key, Any]
    dict_pattern1 = re.search(r'dict\[(.*?),\s*(.*?)\]', norm_s1)
    dict_pattern2 = re.search(r'dict\[(.*?),\s*(.*?)\]', norm_s2)
    if dict_pattern1 and dict_pattern2:
        key1, val1 = dict_pattern1.groups()
        key2, val2 = dict_pattern2.groups()
        
        # Keys must match
        if key1.lower() != key2.lower():
            return False
            
        # Recursively check value types
        return is_semantically_equivalent(val1, val2)
    
    # Not equivalent
    return False

def get_all_model_names():
    """Get all test model names to replace with Any."""
    all_model_classes = [SimpleModel, ComplexModel, NestedModel, VeryNestedItem, 
                         DeepStructure, UnionWithModel, OptionalComplexCollection, ModelWithAny]
    return list(set(cls.__name__ for cls in all_model_classes))

def check_for_model_in_type(type_to_check):
    """Check if a type annotation contains any BaseModel references."""
    if isinstance(type_to_check, type) and issubclass(type_to_check, BaseModel):
        return True
    origin = get_origin(type_to_check)
    if origin is not None:
        args = get_args(type_to_check)
        return any(check_for_model_in_type(arg) for arg in args)
    return False

def replace_model_names_with_any(type_str, model_names):
    """Replace model class names with 'Any' in a type string."""
    sorted_names = sorted(model_names, key=len, reverse=True)
    result = type_str
    for name in sorted_names:
        result = re.sub(r'\b' + re.escape(name) + r'\b', 'Any', result)
    return result


# --- Test Pydantic -> Schema Dict -> Pydantic round trip ---
@pytest.mark.parametrize("model_name, original_model", get_test_model_instances())
def test_model_roundtrip(model_name, original_model):
    """Test the round trip from Pydantic model -> schema dict -> new model (one model per test)"""
    print(f"Testing round trip for: {model_name} ({original_model.__class__.__name__})")
    
    # Step 1: Serialize model schema to dict
    schema_dict = ModelHelper.serialize_model_schema(original_model.__class__)
    print(f"  Serialized Schema: {schema_dict}")
    
    # Step 2: Create a new model class from the schema
    new_model_class_name = f"New{original_model.__class__.__name__}_{model_name}" # More unique name
    new_model_class = ModelHelper.create_model_from_flat_yaml(
        new_model_class_name,
        schema_dict,
        optional_fields=False # Match original model's requirement structure
    )
    
    all_model_names_in_test = get_all_model_names()

    # Step 3: Verify the new model class has the same field structure
    for field_name, field_info in original_model.__class__.model_fields.items():
        assert field_name in new_model_class.model_fields, f"{model_name}: Field {field_name} missing in recreated model"
        
        orig_type = field_info.annotation
        new_type = new_model_class.model_fields[field_name].annotation
        
        orig_type_str = ModelHelper.simplify_type_annotation(orig_type)
        new_type_str = ModelHelper.simplify_type_annotation(new_type)
        
        contains_model_ref = check_for_model_in_type(orig_type)

        if contains_model_ref:
            expected_new_type_str = replace_model_names_with_any(orig_type_str, all_model_names_in_test)
            print(f"  Field '{field_name}': Original Type: '{orig_type_str}', New Type: '{new_type_str}', Expected New (Model->Any): '{expected_new_type_str}'")
            
            # Use semantic equivalence instead of exact string matching
            assert is_semantically_equivalent(new_type_str, expected_new_type_str), \
                f"{model_name} - Field {field_name}: Expected '{expected_new_type_str}' (or equivalent) " \
                f"after round-trip for model ref, got '{new_type_str}'"
        else:
            print(f"  Field '{field_name}': Original Type: '{orig_type_str}', New Type: '{new_type_str}'")
            assert orig_type_str == new_type_str, \
                f"{model_name} - Field {field_name}: Expected {orig_type_str}, got {new_type_str}"

    # Step 4: Create an instance of the new model with the same data
    try:
        instantiation_data = original_model.model_dump() 
        new_instance = new_model_class(**instantiation_data)
        assert new_instance is not None
        print(f"  Successfully created instance of {new_model_class_name}")
        assert new_instance.model_dump() == original_model.model_dump()

    except Exception as e:
        pytest.fail(f"{model_name}: Failed to create/validate new model instance: {e}", pytrace=True)


# --- Test serialization of complex type annotations ---
TYPE_SIMPLIFICATION_TEST_CASES = [
    # Simple types
    (str, "str"),
    (int, "int"),
    (float, "float"),
    (bool, "bool"),
    (Any, "Any"),
    (type(None), "None"),
    # Optional types
    (Optional[str], "Optional[str]"),
    (Optional[int], "Optional[int]"),
    # Container types
    (List[str], "List[str]"),
    (Dict[str, int], "Dict[str, int]"),
    (List[Any], "List[Any]"),
    (Dict[str, Any], "Dict[str, Any]"),
    # Nested types
    (List[Optional[str]], "List[Optional[str]]"),
    (Dict[str, List[int]], "Dict[str, List[int]]"),
    # Union types
    (Union[str, int], "Union[str, int]"),
    # Nested union types
    (Optional[Union[str, int]], "Optional[Union[str, int]]"),
    # Model types
    (SimpleModel, "SimpleModel"),
    (Optional[SimpleModel], "Optional[SimpleModel]"),
    # New complex types from models
    (List[Optional[VeryNestedItem]], "List[Optional[VeryNestedItem]]"),
    (Dict[str, List[Union[SimpleModel, ComplexModel]]], "Dict[str, List[Union[SimpleModel, ComplexModel]]]"),
    (Union[str, int, SimpleModel], "Union[str, int, SimpleModel]"),
    (Optional[List[Dict[str, Union[int, str, SimpleModel]]]], "Optional[List[Dict[str, Union[int, str, SimpleModel]]]]")
]

@pytest.mark.parametrize("input_type, expected_string", TYPE_SIMPLIFICATION_TEST_CASES)
def test_simplify_type_annotation(input_type, expected_string):
    """Test serializing a single complex type annotation to a string."""
    assert ModelHelper.simplify_type_annotation(input_type) == expected_string


# --- Test YAML -> Pydantic -> YAML round trip ---
def test_yaml_to_model_to_yaml_roundtrip():
    """Test the round trip from YAML -> Pydantic model -> YAML"""
    yaml_schema = """
    model_name: ComplexTestModelFromYAML
    schema:
      name: str
      age: int
      email: Optional[str]
      tags: List[str]
      scores: Dict[str, float]
      status: Union[str, int]
      maybe_nested: Optional[List[Dict[str, Union[int, bool]]]]
      union_with_model: Union[str, int, SimpleModel]
      any_field: Any
      list_of_any: List[Any]
      optional_list_optional_item: Optional[List[Optional[str]]]
    """
    yaml_data = yaml.safe_load(yaml_schema)
    model_name = yaml_data["model_name"]
    schema_dict = yaml_data["schema"]
    
    model_class = ModelHelper.create_model_from_flat_yaml(
        model_name, schema_dict, optional_fields=True
    )
    new_schema_dict = ModelHelper.serialize_model_schema(model_class)
    
    all_model_names_in_test = get_all_model_names()

    for field_name, type_str in schema_dict.items():
        assert field_name in new_schema_dict, f"Field {field_name} missing in serialized schema"
        
        # Normalize original type string from YAML
        expected_type_str = type_str.replace(" ", "")
        actual_type_str = new_schema_dict[field_name].replace(" ", "")
        
        # Check if the original type string contains a model name
        contains_model_ref_in_yaml = any(re.search(r'\b' + re.escape(name) + r'\b', expected_type_str) for name in all_model_names_in_test)
        
        if contains_model_ref_in_yaml:
             # Expect model names to become 'Any'
             expected_type_str = replace_model_names_with_any(expected_type_str, all_model_names_in_test)
        
        # Because we used optional_fields=True, wrap expected type in Optional if not already
        if not expected_type_str.lower().startswith("optional["):
            # Handle case like Union[str, int] becoming Optional[Union[str, int]]
            if expected_type_str.lower().startswith("union["):
                 expected_type_str = f"Optional[{expected_type_str}]"
            else:
                 expected_type_str = f"Optional[{expected_type_str}]"

        print(f"  Field '{field_name}': Original YAML Type: '{type_str}', Expected Normalized: '{expected_type_str}', Actual Serialized: '{actual_type_str}'")
        assert is_semantically_equivalent(expected_type_str, actual_type_str), \
            f"Field {field_name}: Expected '{expected_type_str}' (or equivalent) from YAML, got '{actual_type_str}'"


# --- Test saving/loading to actual YAML file ---
def test_yaml_file_roundtrip():
    """Test saving a model schema to YAML file and loading it back"""
    class UserProfileForFile(BaseModel):
        id: int; username: str; email: Optional[str] = None; roles: List[Union[str, int]] = []; preferences: Dict[str, Any] = {}; last_login: Optional[str] = None

    schema_dict = ModelHelper.serialize_model_schema(UserProfileForFile)
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".yaml", delete=False, encoding='utf-8') as tmp:
        file_path = tmp.name
        try:
            yaml_content = {"model_name": "UserProfileForFile", "schema": schema_dict}
            yaml.dump(yaml_content, tmp); tmp.flush()
            with open(file_path, "r", encoding='utf-8') as f: loaded_yaml = yaml.safe_load(f)
            
            loaded_schema = loaded_yaml["schema"]
            new_model_class = ModelHelper.create_model_from_flat_yaml(
                loaded_yaml["model_name"], loaded_schema, optional_fields=True
            )
            
            for field_name, type_str in schema_dict.items():
                assert field_name in new_model_class.model_fields
                new_type = new_model_class.model_fields[field_name].annotation
                new_type_str = ModelHelper.simplify_type_annotation(new_type)
                expected_type_str = type_str
                if not expected_type_str.lower().startswith("optional["):
                     expected_type_str = f"Optional[{expected_type_str}]"
                assert is_semantically_equivalent(expected_type_str, new_type_str), \
                   f"Field {field_name}: Expected '{expected_type_str}' (or equivalent), got '{new_type_str}'"
            
            test_data = {"id": 123, "username": "testuser", "email": "test@example.com", "roles": ["admin", 1], "preferences": {"theme": "dark", "notifications": True}, "last_login": "2023-01-01T10:00:00Z"}
            new_instance = new_model_class(**test_data)
            assert new_instance.id == 123 and new_instance.roles == ["admin", 1]
        finally:
            if os.path.exists(file_path): os.unlink(file_path) 