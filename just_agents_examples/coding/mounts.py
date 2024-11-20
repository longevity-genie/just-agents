import os
from docker.types import Mount
from pathlib import Path

coding_examples_dir = Path(__file__).parent.absolute()
assert coding_examples_dir.exists(), f"Examples directory {str(coding_examples_dir)} does not exist, check the current working directory"

output_dir = coding_examples_dir / "output"
input_dir =  coding_examples_dir / "input"
output_dir.mkdir(parents=True, exist_ok=True)
input_dir.mkdir(parents=True, exist_ok=True)
os.chmod(output_dir, 0o777)
os.chmod(input_dir, 0o777)

def make_mounts():
    assert coding_examples_dir.exists(), f"Examples directory {str(coding_examples_dir)} does not exist, check the current working directory"
    return [
        Mount(target="/input", source=str(input_dir.absolute()), type="bind"),
        Mount(target="/output", source=str(output_dir.absolute()), type="bind")
    ]