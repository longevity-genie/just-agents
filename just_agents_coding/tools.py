from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.docker import ConsoleOutput

def run_bash_command(command: str):
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    with  MicromambaSession(image="ghcr.io/longevity-genie/just-agents/biosandbox:main", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.execute_command(command=command)
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