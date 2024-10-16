import os
import random

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.resources as resources
from dotenv import load_dotenv
import importlib
from typing import Callable
from litellm import Message, ModelResponse, completion
import copy

#
# class RotateKeys():
#     keys:list[str]
#
#     def __init__(self, file_path:str):
#         with open(file_path) as f:
#             text = f.read().strip()
#             self.keys = text.split("\n")
#
#     def __call__(self, *args, **kwargs):
#         return random.choice(self.keys)
#
#     def remove(self, key:str):
#         self.keys.remove(key)
#
#     def len(self):
#         return len(self.keys)



def resolve_agent_schema(agent_schema: str | Path | dict | None, class_name: str, default_file_name: str):
    if agent_schema is None:
        agent_schema = Path(Path(__file__).parent, "config", default_file_name)
    if isinstance(agent_schema, str):
        agent_schema = Path(agent_schema)
    if isinstance(agent_schema, Path):
        if not agent_schema.exists():
            raise ValueError(
                f"In {class_name} constructor agent_schema path is not exists: ({str(agent_schema)})!")
        with open(agent_schema) as f:
            agent_schema = yaml.full_load(f)
    if not isinstance(agent_schema, dict):
        raise ValueError(
            f"In {class_name} constructor agent_schema parameter should be None, string, Path or dict!")

    return agent_schema


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