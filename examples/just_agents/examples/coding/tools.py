import re
from just_agents.just_bus import JustEventBus

import requests

from llm_sandbox.micromamba import MicromambaSession

from pathlib import Path
from just_agents.examples.coding.mounts import make_mounts, input_dir, output_dir

"""
Tools for running code in sandboxed environment that also mounts input and output directories.
"""
CODE_OK : str = "Code syntax is correct"

##HELPER TOOLS##

def submit_code(code: str, filename: str)-> str:
    """
    Validates the syntax of a Python code string and submits the code for future processing if correct

    Attempts to compile the provided code to check for syntax errors. Code is not executed at this step.
    Returns a success message if valid or an error message with details if invalid.

    Parameters
    ----------
    code : str
        Python code to validate.
    filename : str
        Filename to include in error messages for context.

    Returns
    -------
    str
        'Code syntax is correct' if valid, or an error message if syntax errors are found.
    """
    result = validate_python_code_syntax(code, filename)
    # Publish the validation result
    event_bus = JustEventBus()
    event_bus.publish("submit_code", code, filename, result)
    return result

def submit_console_output(output: str, append :bool = True)-> bool:
    """
    Submits console output for further recording and analysis

    Parameters
    ----------
    output : str
        Python code to validate.
    append : bool
        Filename to include in error messages for context.

    Returns
    -------
    bool
        True denotes successful submission.
    """
    # Publish the validation result
    try:
        event_bus = JustEventBus()
        event_bus.publish("submit_console_output", output, append)
    except Exception as e:
        return False
    return True

def validate_python_code_syntax(code: str, filename: str)-> str:
    """
    Validates the syntax of a Python code string.
    """
    try:
        # Compile the code string to check for syntax errors
        compiled_code = compile(code, f"/example/{filename}", "exec")
        return (CODE_OK)
    except SyntaxError as e:
        return (f"Syntax error in code: {e}")


def download_file(source_url: str, file_name: str) -> bool:
    """ Download file from source_url and save it to '/input' folder with file_name that available mount for runtime. """
    input_path = input_dir / file_name
    try:
        print(f"Downloading: {source_url} to {input_path}")
        response = requests.get(source_url, stream=True)
        response.raise_for_status()

        with open(input_path, 'wb') as file:  # Remove encoding for binary write
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"File downloaded successfully and saved to {input_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False




def ugly_log(text: str, folder: Path, name: str = "code", ext: str = "py"):
    """
    Logs code to a numbered file in the specified folder.
    Creates code_1.py if no files exist, otherwise increments the number.
    
    Args:
        text: str - The code text to log
        folder: Path - The folder where to save the file


    It is a temporal functin that will be removed soon.
    For logging purposesonly
    """
    # Create folder if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)
    
    # Find all existing code files
    existing_files = list(folder.glob(f"{name}_*.{ext}"))
    
    if not existing_files:
        # No files exist, create code_1.py
        new_file = folder / f"{name}_1.{ext}"
    else:
        # Find the highest number
        numbers = [int(f.stem.split('_')[1]) for f in existing_files]
        next_num = max(numbers) + 1
        new_file = folder / f"{name}_{next_num}.{ext}"
    
    # Write the code to the new file
    with new_file.open('w', encoding='utf-8') as f:
        f.write(text)

def run_bash_command(command: str) -> str:
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    mounts = make_mounts()
    try:
        with MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main",
                               lang="python",
                               keep_template=True,
                               verbose=True,
                               mounts=mounts) as session:
            result = session.execute_command(command=command)
            ugly_log(command, output_dir, "bash", "sh")
            return result
    except Exception as e:
        print(f"Error executing bash command: {e}")
        return str(e)

def run_python_code(code: str) -> str:
    """
    code: str # python code to run in micromamba environment
    """
    mounts = make_mounts()
    try:
        with MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main",
                               lang="python",
                               keep_template=True,
                               verbose=True,
                               mounts=mounts) as session:
            result = session.run(code)
            ugly_log(code, output_dir, "code", "py")
            return result
    except Exception as e:
        print(f"Error executing Python code: {e}")
        return str(e)


def write_thoughts_and_results(name: str, thoughts: str, result: str):
    """
    Write thoughts and results to a file in the output directory
    """
    print("Thoughts: ", thoughts)
    print("Result: ", result)
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    where = output_dir / f"thoughts_{name}.txt"
    # Write thoughts and results to the output file
    with where.open("w", encoding="utf-8") as f:
        f.write("Thoughts:\n")
        f.write(thoughts)
        f.write("\n\nResult:\n")
        f.write(result)
    
    print(f"Thoughts and results have been written to {where}")


def amino_match_endswith(text, ending):
    # Define the regex pattern for an amino acid sequence, with > optional
    fasta_pattern = r">?(?>[^\n]*?\n)?([ACDEFGHIKLMNPQRSTVWY\n]{4,})"

    # Find all matches
    matches = re.findall(fasta_pattern, text,  re.MULTILINE)

    # Check that there is exactly one match
    if len(matches) != 1:
        return False

    # Remove newlines from the sequence and check the ending
    sequence = "".join(matches[0].splitlines())

    return sequence.endswith(ending.upper())


#TEMPORAL FUNCTIONS for webscrapper, will be removed soon

def execute_bash_command(command: str) -> str:
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    mounts = make_mounts()
    try:
        with MicromambaSession(image="ghcr.io/longevity-genie/just-agents/websandbox:main", 
                               lang="python", 
                               keep_template=True, 
                               verbose=True,
                               mounts=mounts) as session:
            result = session.execute_command(command=command)
            ugly_log(command, output_dir, "bash", "sh")
            return result
    except Exception as e:
        print(f"Error executing bash command: {e}")
        return str(e)
    

def execute_python_code(code: str) -> str:
    """
    code: str # python code to run in micromamba environment
    """
    mounts = make_mounts()
    try:
        with MicromambaSession(image="ghcr.io/longevity-genie/just-agents/websandbox:main", 
                               lang="python", 
                               keep_template=True, 
                               verbose=True,
                               mounts=mounts) as session:
            result = session.run(code)
            ugly_log(code, output_dir, "code", "py")
            return result
    except Exception as e:
        print(f"Error executing Python code: {e}")
        return str(e)