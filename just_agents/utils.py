import os
import random

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.resources as resources
from dotenv import load_dotenv
import copy

class RotateKeys():
    keys:list[str]
    def __init__(self, file_path:str):
        with open(file_path) as f:
            self.keys = f.readlines()
    def __call__(self, *args, **kwargs):
        return random.choice(self.keys).strip()


def prepare_options(options:dict[str, any]):
    res = options.copy()
    key_getter = res.pop("key_getter", None)
    if key_getter is not None:
        res["api_key"] = key_getter()
    return res


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