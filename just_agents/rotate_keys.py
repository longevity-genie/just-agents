import random

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