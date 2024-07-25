import os
import random

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.resources as resources
from dotenv import load_dotenv
from litellm import Message, ModelResponse, completion


class RotateKeys():
    keys:list[str]

    def __init__(self, file_path:str):
        with open(file_path) as f:
            text = f.read().strip()
            self.keys = text.split("\n")

    def __call__(self, *args, **kwargs):
        return random.choice(self.keys)

    def remove(self, key:str):
        self.keys.remove(key)

    def len(self):
        return len(self.keys)


def rotate_completion(messages: list[dict], options: dict[str, str], stream: bool, remove_key_on_error: bool = True, max_tries: int = 2) -> ModelResponse:
    opt = options.copy()
    key_getter: RotateKeys = opt.pop("key_getter", None)
    backup_opt: dict = opt.pop("backup_options", None)
    if key_getter is not None:
        if max_tries < 1:
            max_tries = key_getter.len()
        else:
            if remove_key_on_error:
                max_tries = min(max_tries, key_getter.len())
        last_exception = None
        for _ in range(max_tries):
            opt["api_key"] = key_getter()
            try:
                response = completion(messages=messages, stream=stream, **opt)
                return response
            except Exception as e:
                last_exception = e
                if remove_key_on_error:
                    key_getter.remove(opt["api_key"])
        if backup_opt:
            return completion(messages=messages, stream=stream, **backup_opt)
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Run out of tries to execute completion. Check your keys! Keys {key_getter.len()} left.")
    else:
        return completion(messages=messages, stream=stream, **opt)


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