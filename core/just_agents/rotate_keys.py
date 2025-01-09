import random

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
            raise ValueError(f"Environment variable {env_var} not found")
        all_keys.extend([k.strip() for k in keys_str.split(",")])
        
        # Check for additional numbered variables
        counter = 1
        while True:
            numbered_env = f"{env_var}_{counter}"
            keys_str = os.getenv(numbered_env)
            if not keys_str:
                break
            all_keys.extend([k.strip() for k in keys_str.split(",")])
            counter += 1
        
        return cls(all_keys)

    def __call__(self, *args, **kwargs):
        return random.choice(self.keys)

    def remove(self, key:str):
        self.keys.remove(key)

    def len(self):
        return len(self.keys)