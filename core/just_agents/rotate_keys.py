import random
import copy

class RotateKeys():
    """
    RotateKeys is a class that rotates through a list of keys.
    """
    keys:list[str]

    def __init__(self, keys: list[str]):
        self.keys = keys

    @classmethod
    def from_path(cls, file_path: str):
        with open(file_path) as f:
            text = f.read().strip()
            keys = text.split("\n")
        return cls(keys)

    @classmethod
    def from_list(cls, keys: list[str]):
        return cls(keys)

    @classmethod
    def from_env(cls, env_var: str):
        import os
        all_keys = []
        
        # Get the base environment variable
        keys_str = os.getenv(env_var)
        if not keys_str:
            # Check for additional numbered variables even if base is missing
            pass # Allow falling through to numbered check
            #raise ValueError(f"Environment variable {env_var} not found")
        else:
            all_keys.extend([k.strip() for k in keys_str.split(",")])

        # Check for additional numbered variables
        counter = 1
        while True:
            numbered_env = f"{env_var}_{counter}"
            keys_str = os.getenv(numbered_env)
            if not keys_str:
                break # No more numbered vars
            all_keys.extend([k.strip() for k in keys_str.split(",")])
            counter += 1

        if not all_keys: # Raise error only if no keys were found at all
            raise ValueError(f"No keys found in environment variable {env_var} or its numbered variants.")

        return cls(all_keys)

    def __call__(self, *args, **kwargs):
        if not self.keys:
            raise IndexError("No keys available in RotateKeys instance.")
        return random.choice(self.keys)

    def remove(self, key:str):
        try:
            self.keys.remove(key)
        except ValueError:
            pass # Key might have already been removed or wasn't present

    def len(self):
        return len(self.keys)

    def __deepcopy__(self, memo):
        # Create a new instance with a deep copy of the keys list
        new_keys = copy.deepcopy(self.keys, memo)
        new_instance = self.__class__(new_keys)
        memo[id(self)] = new_instance  # Store the new instance in memo
        return new_instance