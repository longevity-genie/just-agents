from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import ConsoleOutput

def run_bash_command(command: str):
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.execute_command(command=command)
        return result

def validate_python_code_syntax(code: str, filename: str)-> str:
    """
    code: str # python code to validate
    filename: str # a filename to use in error messages
    """
    try:
        # Compile the code string to check for syntax errors
        compiled_code = compile(code, f"/example/{filename}", "exec")
        return ("Code syntax is correct")
    except SyntaxError as e:
        return (f"Syntax error in code: {e}")

def save_text_to_runtime(text: str, filename: str):
    """
    text: str # ptext to be saved
    filename: str # a filename to use i
    """
    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", lang="python", keep_template=True, verbose=True) as session:

        text_file = f"/tmp/{filename}"
        dest_file = f"/example/{filename}"
        with open(text_file, "w") as f:
            f.write(text)
        result: ConsoleOutput = session.copy_to_runtime(src=text_file, dest=dest_file)
        return result


def run_python_code(code: str):
    """
    code: str # python code to run in micromamba environment
    """
    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.run(code)
        return result
    
def copy_from_container(src: str, dest: str):
    """
    src: str # path to file in runtime
    dest: str # path to file in host
    """
    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.copy_from_runtime(src=src, dest=dest)
        return result


def copy_files_to_runtime(src: str, dest: str):
    """
    src: str # path to file in host
    dest: str # path to file in runtime
    """
    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.copy_to_runtime(src=src, dest=dest)
        return result