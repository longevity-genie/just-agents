import os
import io 
import re
import contextlib
import subprocess

def get_next_filename(directory: str, prefix: str, extension: str) -> str:
    """
    Returns the next filename in sequence in the specified directory.

    Parameters:
    directory (str): The directory where files are stored.
    prefix (str): The prefix of the filename.
    extension (str): The file extension.

    Returns:
    str: The next filename in sequence.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Enumerate existing files to determine the next filename
    existing_files = os.listdir(directory)
    pattern = re.compile(rf'{re.escape(prefix)}_(\d+)\.{re.escape(extension)}$')
    numbers = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]
    next_num = max(numbers, default=0) + 1

    filename = os.path.join(directory, f'{prefix}_{next_num}.{extension}')
    return filename

def run_python_code(code: str):
    """
    Executes Python code, writes it to the 'code' folder, and returns the output or the value of 'result'.

    Parameters:
    code (str): Python code as a string.

    Returns:
    str: The output, the value of 'result', or an error message.
    """
    filename = get_next_filename('code', 'code', 'py')

    # Write the code to the file
    with open(filename, 'w') as file:
        file.write(code)

    # Prepare to capture output and errors
    output = io.StringIO()
    error = io.StringIO()
    local_vars = {}

    try:
        # Execute the code
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            exec(code, {}, local_vars)
        # Get printed output
        printed_output = output.getvalue().strip()
        # Check if 'result' is in local variables
        if 'result' in local_vars:
            result_value = local_vars['result']
            # Append 'result' to the printed output
            if printed_output:
                return f"{printed_output}\n{result_value}"
            else:
                return str(result_value)
        else:
            # Return printed output if any, or "No output."
            return printed_output if printed_output else "No output."
    except Exception as e:
        return f"Error: {str(e)}"

def execute_bash(command: str):
    """
    Executes a Bash command and writes it to the 'bash' folder.

    Parameters:
    command (str): Bash command as a string.

    Returns:
    str: The output or error message from executing the command.
    """
    filename = get_next_filename('bash', 'bash', 'sh')

    # Write the command to the file
    with open(filename, 'w') as file:
        file.write(command)

    # Execute the command
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip() if result.stdout else "No output."
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

def clean_folder(folder: str):
    """
    Deletes all files in the specified folder without removing the folder itself.

    Parameters:
    folder (str): The path to the folder to clean.

    Returns:
    str: A message indicating the result of the cleaning operation.
    """
    if not os.path.exists(folder):
        return f"Folder '{folder}' does not exist."

    # Iterate through all files in the folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                return f"Skipping non-file item: {file_path}"
        except Exception as e:
            return f"Error deleting {file_path}: {str(e)}"
    return f"All files in '{folder}' have been cleaned."


