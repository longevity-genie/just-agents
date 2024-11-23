import os
import random
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, TypeVar, Type, cast
import importlib.resources as resources
from dotenv import load_dotenv
import importlib
from typing import Callable
from pydantic import BaseModel
from deprecated import deprecated


class SchemaValidationError(ValueError):
    pass

VALIDATION_EXCEPTIONS = ["options"]
VALIDATION_EXTRAS = ["package", "function"]

def resolve_agent_schema(agent_schema: str | Path | dict):
    """
    Resolve the agent schema from a string, path (to yaml file) or dict.
    """
    if isinstance(agent_schema, str):
        agent_schema = Path(agent_schema)
    if isinstance(agent_schema, Path):
        if not agent_schema.exists():
            raise ValueError(
                f"In constructor agent_schema path  does not exist: ({str(agent_schema.absolute())})!")
        with open(agent_schema) as f:
            agent_schema = yaml.safe_load(f)
    if not isinstance(agent_schema, dict):
        raise ValueError(
            f"In constructor agent_schema parameter should be string, Path or dict!")

    return agent_schema

def resolve_and_validate_agent_schema(agent_schema: str | Path | dict | None, default_file_name: str):
    reference_schema = resolve_agent_schema(Path(Path(__file__).parent, "config", default_file_name))
    if agent_schema is None:
        return reference_schema

    agent_schema = resolve_agent_schema(agent_schema)
    validate_schema(reference_schema, agent_schema)

    return agent_schema


def create_fields_set(source: dict[str, Any], fields_set: set[str]):
    for key in source:
        fields_set.add(key)
        if (key not in VALIDATION_EXCEPTIONS) and isinstance(source[key], dict):
            create_fields_set(source[key], fields_set)


def validate_schema(reference: dict[str, Any], schema: dict[str, Any]):
    reference_set: set[str] = set(VALIDATION_EXTRAS)
    schema_set: set[str] = set()
    create_fields_set(reference, reference_set)
    create_fields_set(schema, schema_set)
    error_fields = []
    for field in schema_set:
        if field not in reference_set and field not in VALIDATION_EXCEPTIONS:
            error_fields.append(field)

    if len(error_fields) > 0:
        raise SchemaValidationError(f" Fields {error_fields} not exists in yaml schema. Choose from {reference_set}")


def resolve_llm_options(agent_schema: dict, llm_options: dict):
    if llm_options is None:
        llm_options = agent_schema.get("options", None)
    if llm_options is None:
        raise ValueError(
            "llm_options should not be None. You should pass it through llm_options or agent_schema['options'].")

    return llm_options


def resolve_system_prompt(agent_schema: dict):
    system_prompt = agent_schema.get("system_prompt", None)
    system_prompt_path = agent_schema.get("system_prompt_path", None)
    if (system_prompt is not None) and (system_prompt_path is not None):
        raise ValueError("You should use only one of system_prompt or system_prompt_path not both together.")
    if system_prompt_path is not None:
        path = Path(system_prompt_path)
        if path.exists():
            system_prompt = path.read_text(encoding="utf8")
    return system_prompt


def resolve_tools(agent_schema: dict) -> list[Callable]:
    function_list:list[Callable] = []
    tools = agent_schema.get('tools', None)
    if tools is None:
        return None
    for entry in tools:
        package_name: str = entry['package']
        function_name: str = entry['function']
        try:
            # Dynamically import the package
            package = importlib.import_module(package_name)
            # Get the function from the package
            func = getattr(package, function_name)
            function_list.append(func)
        except (ImportError, AttributeError) as e:
            print(f"Error importing {function_name} from {package_name}: {e}")

    return function_list


def rotate_env_keys() -> str:
    load_dotenv()
    keys = []
    index = 1

    while True:
        # Determine the environment variable name
        key_name = 'KEY' if index == 0 else f'KEY_{index}'
        key_value = os.getenv(key_name)

        # Break the loop if the key is not found
        if key_value is None:
            break

        # Add the found key to the list
        keys.append(key_value)
        index += 1

    # Raise an error if no keys are found
    if not keys:
        raise ValueError("No keys found in environment variables")

    # Randomly choose one of the available keys
    return random.choice(keys)

def load_config(resource: str, package: str = "just_agents.config") -> Dict[str, Any]:
    """
    :rtype: yaml config
    """
    if Path(resource).exists():
        return yaml.safe_load(Path(resource).open("r"))
    in_config = Path("config") / resource
    if in_config.exists():
        return yaml.safe_load( (Path("config") / resource).open("r") )
    else:
        # Load from package resources
        with resources.open_text(package, 'agent_prompts.yaml') as file:
            return yaml.safe_load(file)


@deprecated(version='0.3.0', reason="You should use native serialization of BaseAgent or its descendants")
def build_agent(agent_schema: str | Path | dict):
    from just_agents.simple.cot_agent import ChainOfThoughtAgent
    from just_agents.simple.llm_session import LLMSession
    agent_schema = resolve_agent_schema(agent_schema)
    class_name = agent_schema.get("class", None)
    if class_name is None or class_name == "LLMSession":
        return LLMSession(agent_schema=agent_schema)
    elif class_name == "ChainOfThoughtAgent":
        return ChainOfThoughtAgent(agent_schema=agent_schema)


######### Pydantic models ###########

BaseT = TypeVar('BaseT', bound=BaseModel)
def extract_common_fields(selected_class: Type[BaseT], instance: BaseModel) -> BaseT:
    """
    Trims and typecasts an instance of a class to only include the fields of the selected class.

    :param selected_class: The class type to trim to.
    :param instance: The instance of the class to be trimmed.
    :return: An instance of the selected class populated with the relevant fields from the provided object.
    """
    # Extract only the fields defined in the base class
    base_fields = {field: getattr(instance, field) for field in selected_class.model_fields}

    # Instantiate and return the base class with these fields
    return selected_class(**base_fields)

def trim_to_parent(instance: BaseT) -> BaseModel:
    """
    Trims an instance of a derived class to only include the fields of its direct parent class.
    :param instance: The instance of the derived class to be trimmed.
    :return: An instance of the parent class populated with the relevant fields from the derived class.
    """
    # Get the direct parent class of the instance
    parent_class = type(instance).__class__.__bases__[0]

    # Instantiate and return the parent class with these fields
    parent_instance = extract_common_fields(parent_class, instance)
    return cast(parent_class, parent_instance)

